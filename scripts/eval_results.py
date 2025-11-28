import argparse
import json
import torch
import re
import numpy as np
import os
import math
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

# --- 分布式辅助函数 ---
def setup_distributed():
    """初始化分布式环境，支持 torchrun 或单机多卡"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        print(f"[Init] Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        return rank, world_size, local_rank
    else:
        print("[Init] 未检测到分布式环境，使用单卡模式")
        return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_score(response_text):
    """从LLM的回复中提取分数 (0-10)。"""
    match = re.search(r'(?:Score|Rating|分(?:数)?)\s*[:：]\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    if match: return float(match.group(1))
    
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', response_text)
    if match: return float(match.group(1))
        
    clean_text = re.sub(r'[^\d\.]', '', response_text)
    if clean_text:
        try:
            score = float(clean_text)
            if 0 <= score <= 10: return score
        except ValueError: pass
            
    numbers = re.findall(r'\b([0-9]|10)\b', response_text)
    if numbers: return float(numbers[0])
        
    return None

def build_eval_prompt(input_text, ground_truth, prediction):
    prompt = (
        "You are an expert time series analyst acting as a judge. \n"
        "Your task is to evaluate the quality of a model's prediction based on the User's Input and the Ground Truth.\n\n"
        "Evaluation Criteria:\n"
        "1. **Instruction Following**: Does the prediction directly answer the question asked in the User Input?\n"
        "2. **Accuracy**: Does the prediction correctly identify trends, anomalies, or values as described in the Ground Truth?\n"
        "3. **Completeness**: Did the model miss any major features mentioned in the Ground Truth?\n\n"
        "Please rate the prediction on a scale of 0 to 10.\n"
        "Output ONLY the numeric score (e.g., 8). Do not provide explanations.\n\n"
        f"--- User Input (Question/Instruction) ---\n{input_text}\n\n"
        f"--- Ground Truth ---\n{ground_truth}\n\n"
        f"--- Prediction ---\n{prediction}\n\n"
        "Score:"
    )
    return prompt

def build_reasoning_prompt(input_text, ground_truth, prediction, score):
    prompt = (
        "You are an expert time series analyst.\n"
        f"The model's prediction below received a low score ({score}/10).\n\n"
        f"--- User Input (Question) ---\n{input_text}\n\n"
        f"--- Ground Truth ---\n{ground_truth}\n\n"
        f"--- Prediction ---\n{prediction}\n\n"
        "Task: Briefly explain why this prediction is poor. \n"
        "Consider: Did it fail to answer the specific question? Did it hallucinate features? Is the trend wrong?\n"
        "Response format:\n"
        "- Reason 1\n"
        "- Reason 2\n"
        "..."
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 Results using LLM (Multi-GPU)")
    parser.add_argument("--input-file", type=str, default="/root/emhua/btwu/temporary2/chatts_stage3_test_results.jsonl")
    parser.add_argument("--output-file", type=str, default="stage3_eval_scores.jsonl")
    parser.add_argument("--bad-case-file", type=str, default="stage3_bad_cases_analysis.jsonl")
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Qwen14B")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # 1. 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 2. 加载模型 (每个进程加载到自己的GPU)
    if rank == 0:
        print(f">>> Loading Evaluator Model from {args.llm_model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device  # 直接指定当前进程的 device
    )
    model.eval()

    # 3. 读取并切分数据
    all_results = []
    # 所有进程都读取文件（假设文件不大），如果文件巨大建议只在 rank0 读取后 scatter，或者使用 seek
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    if args.max_samples:
        all_results = all_results[:args.max_samples]

    # --- 数据切分逻辑 ---
    total_samples = len(all_results)
    chunk_size = math.ceil(total_samples / world_size)
    start_idx = rank * chunk_size
    end_idx = min((rank + 1) * chunk_size, total_samples)
    
    local_data = all_results[start_idx:end_idx]
    
    if rank == 0:
        print(f">>> Total samples: {total_samples}. Distributing across {world_size} GPUs.")
    print(f">>> Rank {rank} processing samples {start_idx} to {end_idx} (Count: {len(local_data)})")

    scored_results = []
    bad_cases = []
    local_scores = []
    
    # 4. 执行推理 (只处理 local_data)
    # 只有 rank 0 显示详细进度条，其他简单显示
    iterator = tqdm(local_data, desc=f"Rank {rank} Evaluating") if len(local_data) > 0 else []
    
    for idx, item in enumerate(iterator):
        input_text = item.get("input", item.get("input_text", "")) 
        gt = item.get("ground_truth", "")
        pred = item.get("prediction", "")
        
        # 评分
        score_prompt = build_eval_prompt(input_text, gt, pred)
        inputs = tokenizer(score_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids, 
                max_new_tokens=10, 
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        score_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        score = parse_score(score_text)
        
        final_score = score if score is not None else 0.0
        
        # --- 修改开始：重构字典，将 score 放在第一位 ---
        result_item = {"score": final_score}
        # 将原始 item 中的其他字段加入，跳过 score 以防重复（虽然输入里一般没有）
        for k, v in item.items():
            if k != "score":
                result_item[k] = v
        # 记录原始的 LLM 回复
        result_item["eval_response"] = score_text
        
        scored_results.append(result_item)
        # --- 修改结束 ---

        if score is not None:
            local_scores.append(score)

        # 低分分析
        if final_score <= 6.0:
            sample_id = item.get('id', item.get('sample_id', f"r{rank}_idx{idx}"))
            reason_prompt = build_reasoning_prompt(input_text, gt, pred, final_score)
            reason_inputs = tokenizer(reason_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                reason_output_ids = model.generate(
                    reason_inputs.input_ids, 
                    max_new_tokens=256, 
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            analysis_text = tokenizer.decode(reason_output_ids[0][reason_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # --- 修改开始：Bad Case 记录也将 score 放在第一位 ---
            bad_case_record = {
                "score": final_score,
                "id": sample_id,
                "input": input_text,
                "ground_truth": gt,
                "prediction": pred,
                "failure_analysis": analysis_text
            }
            # --- 修改结束 ---
            
            bad_cases.append(bad_case_record)
            # 更新主结果中的 analysis 字段
            result_item["failure_analysis"] = analysis_text

    # 5. 保存分片结果
    # 为了避免通信开销，每个 Rank 保存自己的结果到临时文件，最后由 Rank 0 合并
    temp_output_file = f"{args.output_file}.rank{rank}"
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        for item in scored_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    temp_bad_case_file = f"{args.bad_case_file}.rank{rank}"
    if bad_cases:
        with open(temp_bad_case_file, 'w', encoding='utf-8') as f:
            for item in bad_cases:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 等待所有进程完成
    if dist.is_initialized():
        dist.barrier()

    # 6. 合并结果 (仅 Rank 0)
    if rank == 0:
        print(f">>> Merging results from all ranks...")
        final_results = []
        final_bad_cases = []
        final_scores_list = []

        # 合并结果文件
        for r in range(world_size):
            fname = f"{args.output_file}.rank{r}"
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as f:
                    for line in f:
                        res = json.loads(line)
                        final_results.append(res)
                        if res.get("score") is not None:
                            final_scores_list.append(res["score"])
                os.remove(fname) # 删除临时文件

            # 合并 Bad Case
            bname = f"{args.bad_case_file}.rank{r}"
            if os.path.exists(bname):
                with open(bname, 'r', encoding='utf-8') as f:
                    for line in f:
                        final_bad_cases.append(json.loads(line))
                os.remove(bname)

        # 写入最终文件
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        if final_bad_cases:
            with open(args.bad_case_file, 'w', encoding='utf-8') as f:
                for item in final_bad_cases:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 统计报告
        if final_scores_list:
            scores_np = np.array(final_scores_list)
            avg_score = np.mean(scores_np)
            bins = [0, 2, 4, 6, 8, 10]
            hist, _ = np.histogram(scores_np, bins=bins)
            
            print("\n" + "="*50)
            print("Evaluation Report (Merged)")
            print(f"Total Processed: {len(final_results)}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Bad Cases (<=6): {len(final_bad_cases)}")
            print(f"Distribution: {hist}")
            print(f"Results saved to: {args.output_file}")
            print("="*50)
    
    cleanup_distributed()

if __name__ == "__main__":
    main()