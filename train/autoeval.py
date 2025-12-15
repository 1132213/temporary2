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

# 尝试导入 PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# 路径设置 (确保能找到 src)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# ==========================================
# 1. Dataset & Collate (复用原逻辑)
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
        # 补齐 category
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
# 2. 核心评测逻辑 (可被 import)
# ==========================================
def extract_answer_option(text):
    if not text: return "None"
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    
    # 1. 优先匹配 "The answer is X"
    match = re.search(r'The answer is\s*[:\-\s]*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # 2. 匹配 "Option X"
    match = re.search(r'Option\s*([A-D])', text, re.IGNORECASE)
    if match: return match.group(1).upper()

    # 3. 兜底：找最后一个 A-D
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches: return matches[-1]
    
    return "None"

def evaluate_exam_results_with_regex(results_list):
    """统计并打印结果"""
    total_stats = {"correct": 0, "total": 0}
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    none_indices = []
    
    # 统计逻辑
    for i, item in enumerate(results_list):
        gt = item.get("ground_truth", "").strip()
        pred = item.get("prediction", "").strip()
        cat = item.get("category", "Uncategorized")
        
        gt_opt = extract_answer_option(gt)
        pred_opt = extract_answer_option(pred)
        
        if pred_opt == "None":
            # 优先使用 id，如果没有则用索引
            none_indices.append(item.get("id", i))
        
        is_correct_val = 1 if (gt_opt != "None" and gt_opt == pred_opt) else 0
        
        item["judge_score"] = is_correct_val
        total_stats["total"] += 1
        cat_stats[cat]["total"] += 1
        if is_correct_val == 1:
            total_stats["correct"] += 1
            cat_stats[cat]["correct"] += 1

    # 打印报表
    print(f"\n{'='*25} Regex Judge Report {'='*25}")
    print(f"{'Category':<35} | {'Acc':<8} | {'Correct':<8} | {'Total':<8}")
    print("-" * 75)
    
    forced_order = ["Pattern Recognition", "Noise Understanding", "Anomaly Detection", "Similarity Analysis", "Causality Analysis"]
    for cat in forced_order:
        target_key = cat if cat in cat_stats else ("Anolmaly Detection" if "Anolmaly Detection" in cat_stats and cat == "Anomaly Detection" else None)
        if target_key and target_key in cat_stats:
            s = cat_stats[target_key]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            print(f"{target_key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    remaining = sorted([c for c in cat_stats.keys() if c not in forced_order and c != "Anolmaly Detection"])
    for cat in remaining:
        s = cat_stats[cat]
        acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
        print(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

    print("-" * 75)
    total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
    print(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
    print("=" * 75)
    
    return results_list

def run_internal_eval(model, tokenizer, args, device, rank, world_size):
    """
    内部接口：供 sft.py 或 main 使用。
    model: 必须是已经 .to(device) 的模型实例（如果是DDP，建议传入 model.module，或者确保 model 有 generate 方法）
    """
    # 1. 准备数据
    ds = ExamChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, 
        split="val",
        split_ratio=0.0, 
        patch_stride=args.patch_stride 
    )
    
    # 支持分布式推理
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, collate_fn=exam_collate_fn)
    
    # 2. 推理循环
    results = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    iterator = tqdm(loader, desc=f"Rank {rank} Eval") if rank == 0 else loader
    
    # 确定生成参数
    max_new_tokens = 2048 if args.enable_cot else 16
    stop_prefix = None if args.enable_cot else "The answer is option"
    
    model.eval()
    
    # 确保使用 base model 进行 generate (防止 DDP wrapper 报错)
    # 如果传入的是 DDP wrapper，通常通过 model.module 访问
    inference_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for batch in iterator:
            # 这里的 batch size 为 1
            input_texts = batch["input_texts"]
            ts_lists = batch["timeseries_lists"]
            gts = batch["output_texts"]
            cats = batch["categories"]
            # [修改] 获取样本 ID，用于后续去重
            sample_ids = batch["sample_idxs"]
            
            # 构造 Prompt
            if args.enable_cot:
                cot_suffix = "\nPlease answer the question and provide the correct option letter..."
                full_input = f"<|im_start|>user\n{input_texts[0]}{cot_suffix}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_input = f"<|im_start|>user\n{input_texts[0]}<|im_end|>\n<|im_start|>assistant\nThe answer is option"

            try:
                outs = inference_model.generate(
                    input_texts=[full_input],
                    timeseries_lists=[ts_lists[0]],
                    mask_query=getattr(args, 'mask_query', False),
                    mask_detail=getattr(args, 'mask_detail', False),
                    mask_text_stats=getattr(args, 'mask_text_stats', False),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    repetition_penalty=1.1
                )
                
                pred = tokenizer.decode(outs[0], skip_special_tokens=True)
                
                # 清洗结果
                for stop_word in ["<|im_end|>", "<|im_start|>"]:
                    if stop_word in pred: pred = pred.split(stop_word)[0]
                if stop_prefix and stop_prefix in pred:
                    pred = pred.split(stop_prefix)[-1]
                
                results.append({
                    "id": sample_ids[0],  # [修改] 保存 ID
                    "ground_truth": gts[0],
                    "prediction": pred.strip(),
                    "category": cats[0]
                })
            except Exception as e:
                print(f"[Rank {rank}] Error: {e}")
    
    # 3. 结果汇总 (通过文件系统)
    # 每个 rank 写自己的结果
    temp_file = f"{args.output_file}.rank{rank}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for res in results: f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    # [修改] 显式指定 device_ids 以防止 warning 和潜在死锁
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
        
    # 4. Rank 0 负责合并和评分
    final_scored = []
    if rank == 0:
        merged_results = {} # [修改] 使用字典进行 ID 去重
        for r in range(world_size):
            fname = f"{args.output_file}.rank{r}"
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            # [修改] 利用 ID 去重 (Padding 的数据 ID 会重复，后者覆盖前者无所谓)
                            if 'id' in item:
                                merged_results[item['id']] = item
                            else:
                                # 兼容旧格式（虽不太可能）
                                merged_results[str(item)] = item
                os.remove(fname)
        
        # 转回列表并排序，保证确定性
        final_list = list(merged_results.values())
        # 如果 ID 是数字，尝试按数字排序，否则按字符串
        try:
            final_list.sort(key=lambda x: int(x['id']))
        except:
            final_list.sort(key=lambda x: str(x.get('id', 0)))

        print(f">>> [AutoEval] Merged {len(final_list)} unique samples.")
        final_scored = evaluate_exam_results_with_regex(final_list)
        
        # 写入最终文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_scored: f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # [修改] 增加第二次同步，确保 Ranks 1-3 等待 Rank 0 写完文件后再退出
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    return final_scored

# ==========================================
# 3. 单独运行时入口
# ==========================================
def main():
    # 默认使用 0,1 卡
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        
    parser = argparse.ArgumentParser(description="AutoEval Standalone")
    parser.add_argument("--jsonl-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="eval_result.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, required=True)
    parser.add_argument("--enable-cot", action="store_true")
    # LoRA / Ablation args...
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--mask-query", action="store_true")
    parser.add_argument("--mask-detail", action="store_true")
    parser.add_argument("--mask-text-stats", action="store_true")
    
    args = parser.parse_args()
    
    # 初始化分布式
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 兼容 torch.nn.DataParallel 或单卡逻辑
        rank = 0
        world_size = 1
        local_rank = 0 
        
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 加载模型
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
    
    # LoRA
    if args.use_lora and PEFT_AVAILABLE:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=True, 
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model.llm.model = get_peft_model(model.llm.model, peft_config)
        
    # 加载权重
    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # 运行评测
    run_internal_eval(model, model.tokenizer.tokenizer, args, device, rank, world_size)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()