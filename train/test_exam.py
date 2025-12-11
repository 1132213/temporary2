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

# === PEFT 库支持 ===
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
# 1. Dataset & Collate
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

def compute_test_loss(model, dataloader, device, rank):
    model.eval()
    return 0.0

# ==========================================
# 3. 生成预测 (支持 CoT, Direct, Ablation)
# ==========================================
def generate_predictions(
    model, dataloader, device, output_file, rank, 
    max_samples=None, enable_cot=False,
    mask_query=False, mask_detail=False, mask_text_stats=False
):
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
                
                # --- Prompt 构造 ---
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
                    # [关键修改]：将 mask 参数传递给 model.generate
                    # ========================================================
                    outs = model.generate(
                        input_texts=[full_input],
                        timeseries_lists=[ts_lists[i]],
                        
                        # 透传 Ablation 参数
                        mask_query=mask_query,
                        mask_detail=mask_detail,
                        mask_text_stats=mask_text_stats,
                        
                        # 生成参数
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
                        "mode": "cot" if enable_cot else "direct",
                        "ablation": f"mask_q={mask_query},mask_d={mask_detail},mask_s={mask_text_stats}"
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
    
    for cat in forced_order:
        target_key = cat
        if cat not in cat_stats and "Anolmaly Detection" in cat_stats and cat == "Anomaly Detection":
            target_key = "Anolmaly Detection"

        if target_key in cat_stats:
            s = cat_stats[target_key]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            print(f"{target_key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")
    
    remaining_cats = sorted([c for c in cat_stats.keys() if c not in forced_order and c != "Anolmaly Detection"])
    for cat in remaining_cats:
        s = cat_stats[cat]
        acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
        print(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    print("-" * 75)
    total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
    print(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
    print("-" * 75)
    print(f"Failed Extraction (None) Count : {len(none_indices)}")
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
    parser = argparse.ArgumentParser(description="Test Stage 3 (ChatTS Instruct) - Multi GPU with Regex Judge")
    parser.add_argument("--jsonl-path", type=str, default="chatts_data.jsonl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="chatts_stage3_test_results.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B")
    parser.add_argument("--num-gen-samples", type=int, default=746)
    
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    parser.add_argument("--enable-cot", action="store_true", 
                        help="Enable Chain-of-Thought reasoning.")
    
    # [新增] 消融参数
    parser.add_argument("--mask-query", action="store_true", help="[Ablation] Mask out Q-Former query tokens.")
    parser.add_argument("--mask-detail", action="store_true", help="[Ablation] Mask out Detail projection tokens.")
    parser.add_argument("--mask-text-stats", action="store_true", help="[Ablation] Mask out explicit text statistics (mean/std/etc).")
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
    )
    model = StatBypassCROMETS1(config).to(device)
    
    if args.use_lora and PEFT_AVAILABLE:
        if rank == 0: print(">>> Applying LoRA for Inference...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=True, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model.llm.model = get_peft_model(model.llm.model, peft_config)
        
    if rank == 0: print(f">>> Loading Checkpoint: {args.checkpoint}")
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    # 捕获返回值 msg
    msg = model.load_state_dict(state_dict, strict=False)
    
    if rank == 0:
        print(f">>> Weights Loaded.")
        print(f"    - Missing Keys: {len(msg.missing_keys)}")
        print(f"    - Unexpected Keys: {len(msg.unexpected_keys)}")
        
        if len(msg.missing_keys) > 0:
            print(f"    ! Warning: First 5 missing keys: {msg.missing_keys[:5]}")
        if len(msg.unexpected_keys) > 0:
            print(f"    ! Warning: First 5 unexpected keys: {msg.unexpected_keys[:5]}")
    
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

    if rank == 0:
        mode_str = "Chain-of-Thought" if args.enable_cot else "Direct Answer"
        ablation_str = []
        if args.mask_query: ablation_str.append("Mask Query")
        if args.mask_detail: ablation_str.append("Mask Detail")
        if args.mask_text_stats: ablation_str.append("Mask Text Stats")
        ablation_info = ", ".join(ablation_str) if ablation_str else "None"
        
        print(f">>> Inference Mode: {mode_str}")
        print(f">>> Ablation Mode: {ablation_info}")

    generate_predictions(
        model, loader, device, args.output_file, rank, 
        max_samples=max_samples,
        enable_cot=args.enable_cot,
        mask_query=args.mask_query,
        mask_detail=args.mask_detail,
        mask_text_stats=args.mask_text_stats
    )
    
    if dist.is_initialized(): dist.barrier()
    
    # Phase 2: Evaluation
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