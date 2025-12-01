import sys
import json
import argparse
import torch
import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# ==========================================
# 1. 保留 ExamChatTSDataset：透传参数并读取 Category
# ==========================================
class ExamChatTSDataset(ChatTSDataset):
    def __init__(self, jsonl_path, *args, **kwargs):
        # 透传 kwargs (如 patch_stride) 给父类
        super().__init__(jsonl_path, *args, **kwargs)
        
        self.raw_categories = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.raw_categories.append(rec.get("category", "Uncategorized"))
        except Exception as e:
            print(f"[Warning] Failed to load raw categories: {e}")
            self.raw_categories = ["Uncategorized"] * len(self.records)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if idx < len(self.raw_categories):
            item["category"] = self.raw_categories[idx]
        else:
            item["category"] = "Uncategorized"
        return item

def exam_collate_fn(batch):
    batch_dict = chatts_collate_fn(batch)
    batch_dict["categories"] = [item["category"] for item in batch]
    return batch_dict

# ==========================================
# 2. Embedding 构建 (保持不变)
# ==========================================
def build_chatts_embeddings_for_inference(model, input_text, ts_list, device):
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    num_markers = len(text_parts) - 1
    timeseries_list = list(ts_list)
    
    # 填充缺失的时间序列占位符
    if len(timeseries_list) < num_markers:
        for _ in range(num_markers - len(timeseries_list)):
            timeseries_list.append(torch.zeros(model.config.seq_len, model.config.input_channels, device=device))
    elif len(timeseries_list) > num_markers:
        timeseries_list = timeseries_list[:num_markers]
        
    segment_embeds, segment_masks = [], []
    tokenizer = model.tokenizer.tokenizer
    target_dtype = next(model.llm.parameters()).dtype
    
    # 1. Prefix Text
    if text_parts[0]:
        tokens = tokenizer(text_parts[0], return_tensors="pt", add_special_tokens=False).to(device)
        segment_embeds.append(model.llm.embed(tokens.input_ids)[0])
        segment_masks.append(tokens.attention_mask[0])
    
    # 2. TS + Middle Text
    for idx, ts in enumerate(timeseries_list):
        ts = ts.to(device)
        
        # --- 显式统计量文本 ---
        if ts.numel() > 0:
            ts_mean = ts.mean().item()
            ts_std = ts.std().item()
            ts_min = ts.min().item()
            ts_max = ts.max().item()
        else:
            ts_mean = ts_std = ts_min = ts_max = 0.0
            
        stats_str = f" [Stats: mean={ts_mean:.2f}, std={ts_std:.2f}, min={ts_min:.2f}, max={ts_max:.2f}] "
        
        stats_tokens = tokenizer(stats_str, return_tensors="pt", add_special_tokens=False).to(device)
        segment_embeds.append(model.llm.embed(stats_tokens.input_ids)[0])
        segment_masks.append(stats_tokens.attention_mask[0])
        
        # --- 时间序列 Embedding ---
        ts_batch = ts.unsqueeze(0)
        ts_tokens = model.ts_model._process_single_channel(ts_batch)
        
        if ts_tokens.dtype != target_dtype: 
            ts_tokens = ts_tokens.to(dtype=target_dtype)

        ts_emb = ts_tokens[0]
        
        segment_embeds.append(ts_emb)
        segment_masks.append(torch.ones(ts_emb.shape[0], device=device, dtype=torch.long))
        
        # 添加 SEP token
        if idx < len(timeseries_list) - 1:
            sep_embed = model.sep_token
            if sep_embed.dtype != target_dtype: 
                sep_embed = sep_embed.to(dtype=target_dtype)
            segment_embeds.append(sep_embed)
            segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
            
        # Middle Text
        txt_idx = idx + 1
        if txt_idx < len(text_parts) and text_parts[txt_idx]:
            tokens = tokenizer(text_parts[txt_idx], return_tensors="pt", add_special_tokens=False).to(device)
            segment_embeds.append(model.llm.embed(tokens.input_ids)[0])
            segment_masks.append(tokens.attention_mask[0])
            
    # 3. 最终组装
    if segment_embeds:
        full_emb = torch.cat(segment_embeds, dim=0)
        full_msk = torch.cat(segment_masks, dim=0)
        return full_emb.unsqueeze(0), full_msk.unsqueeze(0)
    
    return torch.empty(1,0,model.config.llm_embed_dim).to(device), torch.empty(1,0).to(device)

# ==========================================
# 3. 生成预测 (ChatML + 截断)
# ==========================================
def generate_predictions(model, dataloader, device, output_file, rank, max_samples=None):
    model.eval()
    results = []
    tokenizer = model.tokenizer.tokenizer
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    iterator = tqdm(dataloader, desc=f"Rank {rank} Inference") if rank == 0 else dataloader
    sample_count = 0

    with torch.no_grad():
        for batch in iterator:
            if max_samples and sample_count >= max_samples: break
            
            input_texts = batch["input_texts"]
            ts_lists = batch["timeseries_lists"]
            gts = batch["output_texts"]
            cats = batch["categories"]
            
            for i in range(len(input_texts)):
                if max_samples and sample_count >= max_samples: break
                
                # [Qwen ChatML 格式适配]
                full_input = f"<|im_start|>user\n{input_texts[i]}<|im_end|>\n<|im_start|>assistant\n"
                
                try:
                    embeds, mask = build_chatts_embeddings_for_inference(model, full_input, ts_lists[i], device)
                    
                    outs = model.llm.model.generate(
                        inputs_embeds=embeds, 
                        attention_mask=mask,
                        max_new_tokens=128, 
                        min_new_tokens=1,   
                        pad_token_id=tokenizer.pad_token_id, 
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                    pred = tokenizer.decode(outs[0], skip_special_tokens=True)
                    
                    # 截断逻辑
                    stop_words = ["<|im_end|>", "<|im_start|>", "user", "model", "Assistant:"]
                    for stop_word in stop_words:
                        idx = pred.find(stop_word)
                        if idx != -1:
                            pred = pred[:idx]
                    pred = pred.strip()
                    
                    results.append({
                        "ground_truth": gts[i],
                        "prediction": pred,
                        "category": cats[i],
                        "full_input_text": input_texts[i] 
                    })
                    sample_count += 1
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    temp_file = f"{output_file}.rank{rank}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")

# ==========================================
# 4. 正则表达式判断逻辑 (替代 AI Judge)
# ==========================================
def extract_answer_option(text):
    """
    从文本中提取选项字母 (A, B, C, D)。
    策略：
    1. 优先匹配行首或字符串开头的单个字母。
    2. 其次匹配 'Answer: A' 或 'Option A' 等模式。
    3. 最后查找任何独立的 A-D 字符，取第一个。
    """
    if not text:
        return "None"
    
    text = text.strip().upper()
    
    # 1. 严格匹配开头 (例如 "A", "A.", "A)", "A:")
    match = re.match(r'^([A-D])([.,:;)]|$)', text)
    if match:
        return match.group(1)
        
    # 2. 匹配 "Answer is A", "Option: B" 等
    match = re.search(r'(?:ANSWER|OPTION|CHOICE)\s*[:\-\s]*([A-D])\b', text)
    if match:
        return match.group(1)
        
    # 3. 最后的保底：查找文本中第一个出现的独立 A-D 字符
    # \b 确保不是单词的一部分 (如 "BAD" 中的 B, A, D)
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[0]
        
    return "None"

def evaluate_exam_results_with_regex(results_list):
    total_stats = {"correct": 0, "total": 0}
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    print(f"\n>>> Starting Regex Evaluation on {len(results_list)} samples...")
    for i, item in enumerate(tqdm(results_list, desc="Regex Judging")):
        gt = item.get("ground_truth", "").strip()
        pred = item.get("prediction", "").strip()
        cat = item.get("category", "Uncategorized")
        
        # 提取 Ground Truth 和 Prediction 中的选项
        gt_opt = extract_answer_option(gt)
        pred_opt = extract_answer_option(pred)
        
        # 判断是否正确
        is_correct_val = 1 if (gt_opt != "None" and gt_opt == pred_opt) else 0
        
        item["judge_score"] = is_correct_val
        item["judge_type"] = "regex"
        item["extracted_gt"] = gt_opt
        item["extracted_pred"] = pred_opt
        
        total_stats["total"] += 1
        cat_stats[cat]["total"] += 1
        if is_correct_val == 1:
            total_stats["correct"] += 1
            cat_stats[cat]["correct"] += 1

    print(f"\n{'='*25} Regex Judge Report {'='*25}")
    print(f"{'Category':<35} | {'Acc':<8} | {'Correct':<8} | {'Total':<8}")
    print("-" * 75)

    forced_order = [
        "Pattern Recognition",
        "Noise Understanding",
        "Anomaly Detection", 
        "Similarity Analysis",
        "Causality Analysis"
    ]
    
    printed_categories = set()
    for cat in forced_order:
        target_key = cat
        # 处理拼写错误的特殊情况
        if cat not in cat_stats and "Anolmaly Detection" in cat_stats and cat == "Anomaly Detection":
            target_key = "Anolmaly Detection"

        if target_key in cat_stats:
            s = cat_stats[target_key]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            print(f"{target_key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")
            printed_categories.add(target_key)
    
    remaining_cats = sorted([c for c in cat_stats.keys() if c not in printed_categories])
    for cat in remaining_cats:
        s = cat_stats[cat]
        acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
        print(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    print("-" * 75)
    total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
    print(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
    print("=" * 75 + "\n")
    return results_list

# ==========================================
# 5. 主程序 (Stage 2 专用)
# ==========================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized(): dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="Test Stage 2 (Alignment) Model - Multi GPU with Regex Judge")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to exam jsonl file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 2 checkpoint (e.g., chatts_stage2_aligned.pth)")
    parser.add_argument("--output-file", type=str, default="chatts_stage2_exam_results.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    # parser.add_argument("--judge-model-path", type=str, required=True) # 已移除
    parser.add_argument("--num-gen-samples", type=int, default=100)
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # === Phase 1: Generation using Stage 2 Model ===
    
    # 配置与 Stage 2 训练时一致
    config = CROMEConfig(
        input_channels=1,
        llm_embed_dim=get_llm_embed_dim(args.llm_model_path),
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=f"cuda:{local_rank}",
        llm_dtype="bfloat16",
    )
    
    # 初始化模型（LLM 为 Frozen，不含 LoRA）
    model = StatBypassCROMETS1(config).to(device)
        
    if rank == 0: print(f">>> Loading Stage 2 Checkpoint: {args.checkpoint}")
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    ds = ExamChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, 
        split="val",
        split_ratio=0.0, # 使用全量数据测试
        patch_stride=args.patch_stride 
    )
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, collate_fn=exam_collate_fn)
    
    max_samples = (args.num_gen_samples + world_size - 1) // world_size if args.num_gen_samples > 0 else None
    if args.num_gen_samples == -1: max_samples = None

    generate_predictions(model, loader, device, args.output_file, rank, max_samples)
    
    if dist.is_initialized(): dist.barrier()
    
    # === Phase 2: Evaluation using Regex ===
    
    if rank == 0:
        print(f">>> Merging results from all {world_size} ranks...")
        merged_results = []
        for r in range(world_size):
            fname = f"{args.output_file}.rank{r}"
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as f:
                    merged_results.extend([json.loads(line) for line in f])
                os.remove(fname)
        
        print(f">>> Total Samples Generated: {len(merged_results)}")
        
        # 直接使用正则进行评估，无需卸载模型
        final_scored_results = evaluate_exam_results_with_regex(merged_results)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_scored_results: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f">>> Final results saved to {args.output_file}")
            
    cleanup_distributed()

if __name__ == "__main__":
    main()