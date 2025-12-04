import sys
import json
import argparse
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import transformers.utils
from typing import TypedDict

# --- 新增的 import ---
from transformers.cache_utils import DynamicCache  # <--- 必须引入这个

# =================================================================
# Monkey Patch 区域 (修复 transformers 4.57+ 兼容性)
# =================================================================

# 1. 修复 LossKwargs (继承 TypedDict)
if not hasattr(transformers.utils, "LossKwargs"):
    print("[Warning] 'LossKwargs' not found. Applying TypedDict Monkey Patch...")
    class LossKwargs(TypedDict): 
        pass
    transformers.utils.LossKwargs = LossKwargs

# 2. 修复 DynamicCache.seen_tokens (由 get_seq_length 替代)
if not hasattr(DynamicCache, "seen_tokens"):
    print("[Warning] 'DynamicCache.seen_tokens' not found. Applying Monkey Patch...")
    
    @property
    def seen_tokens(self):
        # 新版本 transformers 使用 get_seq_length() 获取长度
        return self.get_seq_length()
    
    DynamicCache.seen_tokens = seen_tokens
# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 复用 Dataset 逻辑 (假设 src.crome_ts 存在且兼容)
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# ==========================================
# 1. Dataset 定义 (保持不变)
# ==========================================
class ExamChatTSDataset(ChatTSDataset):
    def __init__(self, jsonl_path, *args, **kwargs):
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
# 2. 推理生成 (适配 ChatTS Multimodal)
# ==========================================
def generate_predictions_chatts(model, tokenizer, processor, dataloader, device, output_file, rank, max_samples=None, enable_cot=False):
    model.eval()
    results = []
    
    iterator = tqdm(dataloader, desc=f"Rank {rank} Inference") if rank == 0 else dataloader
    sample_count = 0

    with torch.no_grad():
        for batch in iterator:
            if max_samples and sample_count >= max_samples: break
            
            # 获取原始数据
            # 注意：ChatTS通常处理单条或List，这里我们为了安全逐条处理(Batch Size=1)
            # 如果 DataLoader batch_size > 1，下面的逻辑需要调整为循环处理 batch 中的每一项
            
            input_texts = batch["input_texts"]   # list of strings
            ts_lists = batch["timeseries_lists"] # list of list of tensors
            gts = batch["output_texts"]
            cats = batch["categories"]
            
            for i in range(len(input_texts)):
                if max_samples and sample_count >= max_samples: break
                
                raw_text = input_texts[i]
                
                # ChatTS 要求的 <ts><ts/> 标记已在 raw_text 中 (Dataset逻辑决定)
                # 我们需要将 Tensor 转为 Numpy 传给 Processor
                ts_tensor_list = ts_lists[i] 
                ts_numpy_list = [t.cpu().numpy().flatten() for t in ts_tensor_list] 
                # 注意：ChatTS 示例中 np.sin(...) 是一维或简单的 shape，
                # 根据模型具体要求，这里可能需要 flatten 或者保持 shape。
                # 假设 ChatTSDataset 出来的 tensor 维度符合模型要求。

                # --- 构造 Prompt (Chat Template) ---
                if enable_cot:
                    # CoT 模式
                    user_content = raw_text + "\nLet's analyze step by step. At the end, state the answer as 'The answer is <Option>'."
                    # ChatTS 标准模版
                    full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{user_content}<|im_end|><|im_start|>assistant\n"
                    
                    max_new_tokens = 512
                    stop_prefix = None
                else:
                    # Direct 模式
                    # 在 assistant 标签后预填 "The answer is option" 引导模型
                    user_content = raw_text
                    full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{user_content}<|im_end|><|im_start|>assistant\nThe answer is option"
                    
                    max_new_tokens = 16
                    stop_prefix = "The answer is option"

                try:
                    # --- Processor 处理 ---
                    # text 需要是 List[str], timeseries 需要是 List[np.array]
                    # 注意：如果一条文本里有多个 <ts>，timeseries 参数通常传入包含所有TS的列表
                    # 某些 processor 实现可能要求嵌套 list，这里按最通用标准实现
                    
                    inputs = processor(
                        text=[full_prompt], 
                        timeseries=ts_numpy_list, 
                        padding=True, 
                        return_tensors="pt"
                    )
                    
                    # 移动到 GPU
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                    # --- 生成 ---
                    outs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False, # 为了评测确定性，通常 greedy search
                        repetition_penalty=1.1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=False
                    )
                    
                    # 解码
                    # input_ids 长度
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = outs[0][input_len:]
                    pred = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # --- 后处理 ---
                    # 清理 Direct 模式下的引导词
                    if stop_prefix and stop_prefix in pred:
                        pred = pred.split(stop_prefix)[-1]
                    
                    pred = pred.strip()
                    
                    results.append({
                        "ground_truth": gts[i],
                        "prediction": pred,
                        "category": cats[i],
                        "full_input_text": full_prompt,
                        "mode": "cot" if enable_cot else "direct"
                    })
                    sample_count += 1
                    
                except Exception as e:
                    print(f"[Rank {rank}] Error processing sample: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    # 保存临时文件
    temp_file = f"{output_file}.rank{rank}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")

# ==========================================
# 3. 正则表达式判断逻辑 (复用原脚本)
# ==========================================
def extract_answer_option(text):
    if not text: return "None"
    text = text.strip()
    match = re.search(r'The answer is\s*[:\-\s]*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches: return matches[-1]
    match = re.match(r'^([A-D])([.,:;)]|$)', text.upper())
    if match: return match.group(1)
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
            none_indices.append(i)
        
        is_correct_val = 1 if (gt_opt != "None" and gt_opt == pred_opt) else 0
        
        item["judge_score"] = is_correct_val
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
    
    # 强制排序以便观察
    forced_order = ["Pattern Recognition", "Noise Understanding", "Anomaly Detection", "Similarity Analysis", "Causality Analysis"]
    printed = set()
    for cat in forced_order:
        key = cat if cat in cat_stats else ("Anolmaly Detection" if "Anolmaly Detection" in cat_stats and cat=="Anomaly Detection" else None)
        if key and key in cat_stats:
            s = cat_stats[key]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            print(f"{key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")
            printed.add(key)
            
    for cat in sorted(cat_stats.keys()):
        if cat not in printed:
            s = cat_stats[cat]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            print(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    print("-" * 75)
    total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
    print(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
    print(f"Failed Extraction (None) Count : {len(none_indices)}")
    print("=" * 75 + "\n")
    return results_list

# ==========================================
# 4. 主程序
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
    parser = argparse.ArgumentParser(description="Test ChatTS (Multimodal) - Multi GPU with Regex Judge")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to exam jsonl file")
    # 默认路径改为 ChatTS
    parser.add_argument("--llm-model-path", type=str, default="bytedance-research/ChatTS-14B", help="Path to ChatTS model")
    parser.add_argument("--output-file", type=str, default="chatts_exam_results.jsonl")
    parser.add_argument("--num-gen-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--enable-cot", action="store_true", help="Enable Chain-of-Thought reasoning.")
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # === Phase 1: Generation using ChatTS ===
    
    if rank == 0:
        print(f">>> Loading ChatTS Model: {args.llm_model_path}")
        mode_str = "Chain-of-Thought (Analysis First)" if args.enable_cot else "Direct Answer (Constraint Decoding)"
        print(f">>> Inference Mode: {mode_str}")
    
    # 1. 加载 Model, Tokenizer, Processor
    # ChatTS 需要 trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path, 
        torch_dtype=torch.float16, # ChatTS 推荐 float16
        device_map=device,
        trust_remote_code=True,
        # attn_implementation="eager" # 如果是 V100 等老卡，可能需要取消注释这一行
    )
    
    processor = AutoProcessor.from_pretrained(args.llm_model_path, trust_remote_code=True, tokenizer=tokenizer)
    
    # 2. 加载数据 (BS=1 以便于单独处理多模态输入)
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

    # 3. 推理
    generate_predictions_chatts(
        model, tokenizer, processor, loader, device, args.output_file, rank, 
        max_samples=max_samples,
        enable_cot=args.enable_cot
    )
    
    if dist.is_initialized(): dist.barrier()
    
    # === Phase 2: Evaluation ===
    
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
        
        final_scored_results = evaluate_exam_results_with_regex(merged_results)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_scored_results: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f">>> Final results saved to {args.output_file}")
            
    cleanup_distributed()

if __name__ == "__main__":
    main()