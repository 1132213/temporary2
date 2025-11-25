import argparse
import json
import torch
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
# import matplotlib.pyplot as plt

def parse_score(response_text):
    """
    从LLM的回复中提取分数 (0-10)。
    优先寻找明确的数字格式，如 "Score: 8", "8/10", 或行首/行尾的单纯数字。
    """
    # 1. 尝试匹配 "Score: X" 或 "Rating: X"
    match = re.search(r'(?:Score|Rating|分(?:数)?)\s*[:：]\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # 2. 尝试匹配 "X/10"
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', response_text)
    if match:
        return float(match.group(1))
        
    # 3. 如果回复很短（例如就是一个数字），直接提取
    # 移除所有非数字字符（保留小数点）
    clean_text = re.sub(r'[^\d\.]', '', response_text)
    if clean_text:
        try:
            score = float(clean_text)
            if 0 <= score <= 10:
                return score
        except ValueError:
            pass
            
    # 4. 暴力搜索第一个出现的 0-10 之间的整数
    numbers = re.findall(r'\b([0-9]|10)\b', response_text)
    if numbers:
        return float(numbers[0])
        
    return None

def build_eval_prompt(ground_truth, prediction):
    """
    构造评估 Prompt。
    """
    prompt = (
        "You are an expert time series analyst acting as a judge. \n"
        "Your task is to evaluate the quality of a model's prediction compared to the ground truth analysis.\n\n"
        "Evaluation Criteria:\n"
        "1. **Accuracy of Trends & Seasonality**: Does the prediction correctly identify the period (e.g., 48, 96) and general trend?\n"
        "2. **Anomaly Detection**: Does the prediction correctly identify the location (timestep) and direction (spike/dip) of anomalies? Close approximations (within +/- 10 steps) are acceptable.\n"
        "3. **Completeness**: Did the model miss any major features mentioned in the ground truth?\n\n"
        "Please rate the prediction on a scale of 0 to 10, where 0 is completely wrong and 10 is a perfect match.\n"
        "Output ONLY the numeric score (e.g., 8). Do not provide explanations.\n\n"
        f"--- Ground Truth ---\n{ground_truth}\n\n"
        f"--- Prediction ---\n{prediction}\n\n"
        "Score:"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 Results using LLM")
    parser.add_argument("--input-file", type=str, default="stage3_results.jsonl", help="包含预测结果的JSONL文件")
    parser.add_argument("--output-file", type=str, default="stage3_eval_scores.jsonl", help="输出包含分数的JSONL文件")
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B", help="用于评估的LLM路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None, help="仅评估前N个样本用于调试")
    args = parser.parse_args()

    print(f">>> Loading Evaluator Model from {args.llm_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device
    )
    model.eval()

    print(f">>> Reading results from {args.input_file}...")
    results = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if args.max_samples:
        results = results[:args.max_samples]

    scored_results = []
    scores = []
    
    print(f">>> Starting Evaluation on {len(results)} samples...")
    
    for item in tqdm(results, desc="Evaluating"):
        gt = item.get("ground_truth", "")
        pred = item.get("prediction", "")
        
        # 构造输入
        prompt = build_eval_prompt(gt, pred)
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        # 生成评分
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids, 
                max_new_tokens=10, # 只需要一个数字，不用生成太长
                do_sample=False,   # 确定性输出
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码并提取只属于“回答”的部分
        # model.generate 会返回 input + output，所以要切片
        generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 解析分数
        score = parse_score(generated_text)
        
        # 记录
        item["eval_response"] = generated_text
        item["score"] = score if score is not None else 0.0 # 解析失败默认0分，避免统计报错
        
        scored_results.append(item)
        if score is not None:
            scores.append(score)

    # --- 保存结果 ---
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in scored_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f">>> Scored results saved to {args.output_file}")

    # --- 统计分析 ---
    if not scores:
        print("No valid scores extracted.")
        return

    scores_np = np.array(scores)
    avg_score = np.mean(scores_np)
    median_score = np.median(scores_np)
    std_dev = np.std(scores_np)
    
    # 分段统计
    bins = [0, 2, 4, 6, 8, 10]
    hist, _ = np.histogram(scores_np, bins=bins)
    
    print("\n" + "="*50)
    print("Evaluation Report")
    print("="*50)
    print(f"Total Samples: {len(scores)}")
    print(f"Average Score: {avg_score:.2f} / 10.0")
    print(f"Median Score:  {median_score:.2f}")
    print(f"Std Deviation: {std_dev:.2f}")
    print("-" * 20)
    print("Score Distribution:")
    print(f"  [0-2):  {hist[0]} samples (Bad)")
    print(f"  [2-4):  {hist[1]} samples (Poor)")
    print(f"  [4-6):  {hist[2]} samples (Fair)")
    print(f"  [6-8):  {hist[3]} samples (Good)")
    print(f"  [8-10]: {hist[4]} samples (Excellent)")
    print("="*50)

if __name__ == "__main__":
    main()