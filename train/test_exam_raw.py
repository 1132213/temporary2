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

# 复用 Dataset 逻辑
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# ==========================================
# 1. Dataset 定义 (保持一致)
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
# 2. 核心：将时序数据转换为文本字符串 (保持不变)
# ==========================================
def serialize_timeseries_to_text(input_text, ts_list):
    """
    将 <ts><ts/> 替换为文本形式的数值序列。
    例如: "Look at <ts><ts/>..." -> "Look at [Series: 0.12, 0.34, ...]..."
    """
    ts_marker = "<ts><ts/>"
    parts = input_text.split(ts_marker)
    
    result_text = parts[0]
    
    # 遍历每个部分进行拼接
    for i in range(len(ts_list)):
        if i < len(parts) - 1:
            # 获取对应的时序数据
            ts_tensor = ts_list[i] # [Seq_Len, 1]
            
            # 转为 numpy 并展平
            vals = ts_tensor.flatten().cpu().numpy()
            
            # 为了防止 Prompt 过长，可以进行降采样或截断
            max_points = 256 
            if len(vals) > max_points:
                # 简单均匀采样
                indices = np.linspace(0, len(vals)-1, max_points, dtype=int)
                vals = vals[indices]
            
            # 格式化为字符串，保留2位小数
            ts_str = ", ".join([f"{v:.2f}" for v in vals])
            ts_text_repr = f" [Series Values: {ts_str}] "
            
            result_text += ts_text_repr + parts[i+1]
    
    # 兜底拼接
    if len(ts_list) < len(parts) - 1:
        for k in range(len(ts_list), len(parts)-1):
            result_text += " [Missing Series] " + parts[k+1]
            
    return result_text

# ==========================================
# 3. 推理生成 (使用 Raw LLM)
# ==========================================
def generate_predictions_raw(model, tokenizer, dataloader, device, output_file, rank, max_samples=None):
    model.eval()
    results = []
    
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
                
                # 1. 文本化时序数据
                raw_input = input_texts[i]
                text_with_ts = serialize_timeseries_to_text(raw_input, ts_lists[i])
                
                # 2. 构建 Prompt (使用 Chat Template)
                messages = [
                    {"role": "user", "content": text_with_ts}
                ]
                
                try:
                    # 尝试应用 Chat Template
                    text_prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                except Exception:
                    # 如果模型没有 template (base model)，回退到简单拼接
                    text_prompt = f"User: {text_with_ts}\nAssistant:"

                inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
                
                try:
                    outs = model.generate(
                        **inputs,
                        max_new_tokens=128, 
                        min_new_tokens=1,   
                        do_sample=False,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # 解码并提取新生成的 token
                    generated_ids = outs[0][inputs.input_ids.shape[1]:]
                    pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    results.append({
                        "ground_truth": gts[i],
                        "prediction": pred,
                        "category": cats[i],
                        "full_input_text": raw_input # 保留原始含标记的文本方便回顾
                    })
                    sample_count += 1
                except Exception as e:
                    print(f"[Rank {rank}] Error: {e}")
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
        
        gt_opt = extract_answer_option(gt)
        pred_opt = extract_answer_option(pred)
        
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

    print(f"\n{'='*25} Regex Judge Report (Raw LLM) {'='*25}")
    print(f"Subject Model: Raw LLM (Serialized TS)")
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
    parser = argparse.ArgumentParser(description="Test Raw LLM (Baseline) - Multi GPU with Regex Judge")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to exam jsonl file")
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B", help="Path to the RAW LLM")
    # parser.add_argument("--judge-model-path", type=str, required=True) # 已移除
    parser.add_argument("--output-file", type=str, default="chatts_raw_llm_exam_results.jsonl")
    parser.add_argument("--num-gen-samples", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=8)
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # === Phase 1: Generation using Raw LLM ===
    
    if rank == 0:
        print(f">>> Loading RAW LLM: {args.llm_model_path}")
        print(">>> Note: Time series will be serialized to text strings.")
    
    # 1. 加载 Tokenizer 和 Raw Model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device,
        trust_remote_code=True
    )
    
    # 2. 加载数据
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
    generate_predictions_raw(model, tokenizer, loader, device, args.output_file, rank, max_samples)
    
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