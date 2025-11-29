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

# 引入 PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# ==========================================
# 1. 修复 Dataset：透传参数并读取 Category
# ==========================================
class ExamChatTSDataset(ChatTSDataset):
    def __init__(self, jsonl_path, *args, **kwargs):
        # [Corrected] 用户确认底层支持 patch_stride，直接透传 kwargs
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
# 2. Embedding 构建 (ChatML 适配版)
# ==========================================
def build_chatts_embeddings_for_inference(model, input_text, ts_list, device):
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    num_markers = len(text_parts) - 1
    timeseries_list = list(ts_list)
    
    if len(timeseries_list) < num_markers:
        for _ in range(num_markers - len(timeseries_list)):
            timeseries_list.append(torch.zeros(model.config.seq_len, model.config.input_channels, device=device))
    elif len(timeseries_list) > num_markers:
        timeseries_list = timeseries_list[:num_markers]
        
    segment_embeds, segment_masks = [], []
    tokenizer = model.tokenizer.tokenizer
    target_dtype = next(model.llm.parameters()).dtype
    
    # 1. Prefix
    if text_parts[0]:
        # Qwen 格式下，Prompt 开头通常已有 <|im_start|>，这里 add_special_tokens=False 更稳妥
        # 但如果 tokenizer 配置为自动加 BOS 且不冲突，也可以 True。这里为了安全设为 False
        tokens = tokenizer(text_parts[0], return_tensors="pt", add_special_tokens=False).to(device)
        segment_embeds.append(model.llm.embed(tokens.input_ids)[0])
        segment_masks.append(tokens.attention_mask[0])
    
    # 2. TS + Middle Text
    for idx, ts in enumerate(timeseries_list):
        ts = ts.to(device).unsqueeze(0)
        # 兼容 LoRA: 调用内部 ts_model 方法
        stat_token, ts_tokens = model.ts_model._process_single_channel(ts)
        
        if stat_token.dtype != target_dtype: stat_token = stat_token.to(dtype=target_dtype)
        if ts_tokens.dtype != target_dtype: ts_tokens = ts_tokens.to(dtype=target_dtype)

        ts_emb = torch.cat([stat_token[0], ts_tokens[0]], dim=0)
        segment_embeds.append(ts_emb)
        segment_masks.append(torch.ones(ts_emb.shape[0], device=device, dtype=torch.long))
        
        if idx < len(timeseries_list) - 1:
            sep_embed = model.sep_token
            if sep_embed.dtype != target_dtype: sep_embed = sep_embed.to(dtype=target_dtype)
            segment_embeds.append(sep_embed)
            segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
            
        txt_idx = idx + 1
        if txt_idx < len(text_parts) and text_parts[txt_idx]:
            tokens = tokenizer(text_parts[txt_idx], return_tensors="pt", add_special_tokens=False).to(device)
            segment_embeds.append(model.llm.embed(tokens.input_ids)[0])
            segment_masks.append(tokens.attention_mask[0])
            
    if segment_embeds:
        full_emb = torch.cat(segment_embeds, dim=0)
        full_msk = torch.cat(segment_masks, dim=0)
        return full_emb.unsqueeze(0), full_msk.unsqueeze(0)
    
    return torch.empty(1,0,model.config.llm_embed_dim).to(device), torch.empty(1,0).to(device)

def compute_test_loss(model, dataloader, device, rank):
    model.eval()
    return 0.0

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
                
                # [FIX] 切换为 Qwen ChatML 格式
                # input_texts[i] 已经包含了 <ts><ts/>
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
                        # Qwen 的 EOS 通常是 <|im_end|>
                        # bos_token_id=tokenizer.bos_token_id, 
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                    pred = tokenizer.decode(outs[0], skip_special_tokens=True)
                    
                    # [FIX] Qwen 适配截断逻辑
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
# 4. 独立 AI 裁判逻辑 (加载新模型)
# ==========================================
def run_ai_judge(model, tokenizer, input_text, gt, pred, device):
    clean_input = input_text.replace("<ts><ts/>", "").strip()
    prompt = (
        "You are an exam grader.\n"
        "Your task is to determine if the Candidate's Prediction matches the Ground Truth for the given Question.\n\n"
        f"--- Question & Options ---\n{clean_input}\n\n"
        f"--- Ground Truth ---\n{gt}\n\n"
        f"--- Candidate Prediction ---\n{pred}\n\n"
        "Question: Is the Candidate Prediction correct? \n"
        "The prediction is correct if it matches the Ground Truth option (e.g., A, B, C, D) or meaning.\n"
        "Output ONLY '1' for Correct or '0' for Incorrect.\n"
        "Answer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id
        )
    judge_output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    if "1" in judge_output: return 1
    elif "0" in judge_output: return 0
    else:
        if gt.strip().upper() in pred.strip().upper(): return 1
        return 0

def evaluate_exam_results_with_judge_model(results_list, judge_model_path, device):
    print(f"\n>>> Loading Judge Model from: {judge_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(judge_model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    total_stats = {"correct": 0, "total": 0}
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    print(f"\n>>> Starting AI Evaluation on {len(results_list)} samples...")
    for i, item in enumerate(tqdm(results_list, desc="AI Judging")):
        gt = item.get("ground_truth", "").strip()
        pred = item.get("prediction", "").strip()
        cat = item.get("category", "Uncategorized")
        input_text = item.get("full_input_text", "")
        
        is_correct_val = run_ai_judge(model, tokenizer, input_text, gt, pred, device)
        item["judge_score"] = is_correct_val
        item["judge_model"] = judge_model_path 
        
        total_stats["total"] += 1
        cat_stats[cat]["total"] += 1
        if is_correct_val == 1:
            total_stats["correct"] += 1
            cat_stats[cat]["correct"] += 1

    print(f"\n{'='*25} AI Judge Report {'='*25}")
    print(f"Judge Model: {judge_model_path}")
    print(f"{'Category':<35} | {'Acc':<8} | {'Correct':<8} | {'Total':<8}")
    print("-" * 75)

    # [MODIFIED] 定义强制排序列表
    forced_order = [
        "Pattern Recognition",
        "Noise Understanding",
        "Anomaly Detection", 
        "Similarity Analysis",
        "Causality Analysis"
    ]
    
    printed_categories = set()

    # 1. 先打印强制排序的类别
    for cat in forced_order:
        # 简单容错：如果 JSONL 数据里真的是 "Anolmaly" (Typo)，这里做一个映射处理
        target_key = cat
        if cat not in cat_stats and "Anolmaly Detection" in cat_stats and cat == "Anomaly Detection":
            target_key = "Anolmaly Detection"

        if target_key in cat_stats:
            s = cat_stats[target_key]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            # 打印时如果原本是错别字，这里还是打印数据里的 key (target_key)，或者你可以强制显示 cat (修正后的拼写)
            print(f"{target_key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")
            printed_categories.add(target_key)
    
    # 2. 打印剩下的类别 (防止有漏网之鱼，按字母顺序)
    remaining_cats = sorted([c for c in cat_stats.keys() if c not in printed_categories])
    for cat in remaining_cats:
        s = cat_stats[cat]
        acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
        print(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    print("-" * 75)
    
    # 3. 最后打印 OVERALL
    total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
    print(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
    print("=" * 75 + "\n")
    return results_list

# ==========================================
# 5. 主程序
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
    parser = argparse.ArgumentParser(description="Test Stage 3 (ChatTS Instruct) - Multi GPU with External AI Judge")
    parser.add_argument("--jsonl-path", type=str, default="chatts_data.jsonl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="chatts_stage3_test_results.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    parser.add_argument("--judge-model-path", type=str, required=True, help="Path to the external AI Judge model")
    parser.add_argument("--calc-loss", action="store_true")
    parser.add_argument("--num-gen-samples", type=int, default=100)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Phase 1: Generation
    config = CROMEConfig(
        input_channels=1,
        llm_embed_dim=get_llm_embed_dim(args.llm_model_path),
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=f"cuda:{local_rank}",
        llm_dtype="bfloat16",
        use_stats_projector=True
    )
    model = StatBypassCROMETS1(config).to(device)
    
    if args.use_lora and PEFT_AVAILABLE:
        if rank == 0: print(">>> Applying LoRA...")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model.llm.model = get_peft_model(model.llm.model, peft_config)
        
    if rank == 0: print(f">>> Loading Checkpoint: {args.checkpoint}")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    
    # [Corrected] 传入 patch_stride
    ds = ExamChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, 
        split="val",
        split_ratio=0.0, 
        patch_stride=args.patch_stride 
    )
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, collate_fn=exam_collate_fn)
    
    max_samples = (args.num_gen_samples + world_size - 1) // world_size if args.num_gen_samples > 0 else None
    if args.num_gen_samples == -1: max_samples = None

    generate_predictions(model, loader, device, args.output_file, rank, max_samples)
    
    if dist.is_initialized(): dist.barrier()
    
    # Phase 2: Evaluation
    if rank == 0:
        print(f">>> Merging results from all {world_size} ranks...")
        merged_results = []
        for r in range(world_size):
            fname = f"{args.output_file}.rank{r}"
            if os.path.exists(fname):
                with open(fname, 'r') as f:
                    merged_results.extend([json.loads(line) for line in f])
                os.remove(fname)
        
        print(f">>> Total Samples Generated: {len(merged_results)}")
        print(">>> Unloading subject model to free GPU memory for Judge...")
        del model
        torch.cuda.empty_cache()
        
        final_scored_results = evaluate_exam_results_with_judge_model(merged_results, args.judge_model_path, device)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_scored_results: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f">>> Final results saved to {args.output_file}")
            
    cleanup_distributed()

if __name__ == "__main__":
    main()