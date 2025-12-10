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
# 1. 保留 ExamChatTSDataset
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
# 2. [已删除] build_chatts_embeddings_for_inference
# 逻辑已移至 model.py 的 prepare_multimodal_embeds
# ==========================================

def compute_test_loss(model, dataloader, device, rank):
    model.eval()
    return 0.0

# ==========================================
# 3. 生成预测 (修改后：直接调用 model.generate)
# ==========================================
def generate_predictions(model, dataloader, device, output_file, rank, max_samples=None, enable_cot=False):
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
                
                original_input = input_texts[i]
                
                # --- Prompt 构造策略 ---
                if enable_cot:
                    cot_suffix = "\nPlease answer the question and provide the correct option letter, e.g., A), B), C), D), and option content at the end of your answer. All information need to answer the question is given. If you are unsure, please provide your best guess."
                    full_input = f"<|im_start|>user\n{original_input}{cot_suffix}<|im_end|>\n<|im_start|>assistant\n"
                    max_new_tokens = 2048 
                    stop_prefix = None
                else:
                    full_input = f"<|im_start|>user\n{original_input}<|im_end|>\n<|im_start|>assistant\nThe answer is option"
                    max_new_tokens = 16 
                    stop_prefix = "The answer is option" 

                try:
                    # ========================================================
                    # [关键修改]：直接调用 model.generate
                    # 此时会自动处理 Text-Guided, Stats 位置等所有逻辑
                    # ========================================================
                    outs = model.generate(
                        input_texts=[full_input],       # 传入列表
                        timeseries_lists=[ts_lists[i]], # 传入列表的列表
                        
                        # 生成参数透传给底层 LLM
                        max_new_tokens=max_new_tokens, 
                        min_new_tokens=1,   
                        pad_token_id=tokenizer.pad_token_id, 
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        repetition_penalty=1.1
                    )
                    
                    pred = tokenizer.decode(outs[0], skip_special_tokens=True)
                    
                    # --- 后处理 ---
                    stop_words = ["<|im_end|>", "<|im_start|>"]
                    for stop_word in stop_words:
                        idx = pred.find(stop_word)
                        if idx != -1:
                            pred = pred[:idx]
                    
                    if stop_prefix and stop_prefix in pred:
                        pred = pred.split(stop_prefix)[-1]
                    
                    pred = pred.strip()
                    
                    results.append({
                        "ground_truth": gts[i],
                        "prediction": pred,
                        "category": cats[i],
                        "mode": "cot" if enable_cot else "direct"
                    })
                    sample_count += 1
                except Exception as e:
                    print(f"[Rank {rank}] Gen Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    temp_file = f"{output_file}.rank{rank}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")

# ==========================================
# 4. 正则表达式判断逻辑
# ==========================================
def extract_answer_option(text):
    if not text: return "None"
    text = text.strip()
    
    # [新增] 如果包含 <think>，尝试去掉思考过程，只看最后的部分
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # 1. 优先匹配明确的结束语
    match = re.search(r'The answer is\s*[:\-\s]*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. 匹配 "Option A" 这种格式
    match = re.search(r'Option\s*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 3. 最后的兜底：找文本中最后出现的 A-D 选项
    # (风险：可能会匹配到思考过程中的选项，但对于 CoT 来说通常最后一句是结论)
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches: return matches[-1]
    
    return "None"

def evaluate_exam_results_with_regex(results_list):
    total_stats = {"correct": 0, "total": 0}
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    none_indices = []
    
    print(f"\n>>> Starting Regex Evaluation on {len(results_list)} samples...")
    for i, item in enumerate(tqdm(results_list, desc="Regex Judging")):
        gt = item.get("ground_truth", "").strip()
        pred = item.get("prediction", "").strip()
        cat = item.get("category", "Uncategorized")
        
        gt_opt = extract_answer_option(gt)
        pred_opt = extract_answer_option(pred)
        
        if pred_opt == "None":
            record_id = item.get("id", i) 
            none_indices.append(record_id)
        
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
        "Pattern Recognition", "Noise Understanding", "Anomaly Detection", 
        "Similarity Analysis", "Causality Analysis"
    ]
    
    printed_categories = set()
    for cat in forced_order:
        target_key = cat
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
    
    print("-" * 75)
    print(f"Failed Extraction (None) Count : {len(none_indices)}")
    if len(none_indices) > 0:
        display_ids = none_indices[:50]
        suffix = "..." if len(none_indices) > 50 else ""
        print(f"Sample Indices with 'None'     : {display_ids} {suffix}")
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
    parser.add_argument("--num-gen-samples", type=int, default=100)
    
    # [新增] Stage 2 可能也需要启用 CoT
    parser.add_argument("--enable-cot", action="store_true", 
                        help="Enable Chain-of-Thought reasoning.")

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
    
    # === [修改重点] 加载权重并打印缺失/冲突 key ===
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # 尝试加载，允许不匹配 (strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if rank == 0:
        # 智能过滤：忽略我们已知不需要加载的 key
        # 1. 忽略 LLM 自身的 key (因为它们是 Frozen 的，通常不保存在 checkpoint 里)
        real_missing = [k for k in missing_keys if not k.startswith("llm.model")]
        # 2. 忽略 mask_token 和 head (来自 Stage 1 预训练任务，推理时不需要)
        real_missing = [k for k in real_missing if "mask_token" not in k and "head." not in k]
        
        # 3. 忽略 pos_encoding (因为我们现在用 RoPE，旧权重里可能有残留的 pos_encoding)
        real_unexpected = [k for k in unexpected_keys if "pos_encoding" not in k]

        print("\n" + "="*50)
        print(f"Weight Loading Report for {args.checkpoint}")
        print("="*50)
        
        if len(real_missing) > 0:
            print(f"!!! MISSING KEYS ({len(real_missing)}) [Critical Warning] !!!")
            # 只打印前 20 个，避免刷屏
            for k in real_missing[:20]:
                print(f"  - {k}")
            if len(real_missing) > 20:
                print(f"  ... and {len(real_missing) - 20} more.")
        else:
            print(">>> No critical missing keys.")

        if len(real_unexpected) > 0:
            print(f"\n!!! UNEXPECTED KEYS ({len(real_unexpected)}) [Check if Architecture Changed] !!!")
            for k in real_unexpected[:20]:
                print(f"  + {k}")
            if len(real_unexpected) > 20:
                print(f"  ... and {len(real_unexpected) - 20} more.")
        else:
            print(">>> No unexpected keys.")
        print("="*50 + "\n")
    # ========================================================
    
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

    if rank == 0:
        mode_str = "Chain-of-Thought" if args.enable_cot else "Direct Answer"
        print(f">>> Inference Mode: {mode_str}")

    generate_predictions(
        model, loader, device, args.output_file, rank, 
        max_samples=max_samples,
        enable_cot=args.enable_cot
    )
    
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