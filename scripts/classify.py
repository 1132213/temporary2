import argparse
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 定义类别列表
CATEGORIES = [
    "Pattern Recognition",   # 模式识别
    "Noise Understanding",   # 噪声理解
    "Anomaly Detection",     # 异常检测
    "Similarity Analysis",   # 相似性分析
    "Causality Analysis"     # 因果关系分析
]

def clean_text(text):
    """移除 <ts><ts/> 标记，保留纯文本指令"""
    if not text:
        return ""
    # 移除标记
    text = text.replace("<ts><ts/>", "")
    # 移除多余空格
    text = " ".join(text.split())
    return text

def build_classification_prompt(question):
    """构建分类 Prompt"""
    categories_str = "\n".join([f"- {c}" for c in CATEGORIES])
    
    prompt = (
        "You are an expert in time series analysis. Your task is to classify the user's question into EXACTLY ONE of the following categories:\n\n"
        f"{categories_str}\n\n"
        "Rules:\n"
        "1. Output ONLY the category name.\n"
        "2. Do not explain your reasoning.\n"
        "3. If the question involves multiple aspects, choose the most dominant one.\n\n"
        f"User Question: \"{question}\"\n\n"
        "Category:"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Classify ChatTS questions using an LLM.")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to the ChatTS format .jsonl file")
    parser.add_argument("--llm-model-path", type=str, default="/mnt/shared-storage-user/dllm-share/Models/Qwen2.5-7B-Instruct", help="Path to the LLM for classification")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process")
    args = parser.parse_args()

    # 1. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    # 2. 加载模型和 Tokenizer
    print(f">>> Loading LLM from: {args.llm_model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 3. 读取数据
    print(f">>> Reading data from: {args.jsonl_path}")
    items = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if args.max_samples:
        items = items[:args.max_samples]
    
    print(f">>> Total samples to classify: {len(items)}")

    # 统计计数器
    stats = {cat: 0 for cat in CATEGORIES}
    stats["Uncategorized"] = 0

    # 4. 逐条处理
    print("\n" + "="*60)
    print(f"{'ID':<5} | {'Category':<25} | {'Question Snippet'}")
    print("-" * 60)

    for i, item in enumerate(tqdm(items, desc="Classifying")):
        # 获取并清洗文本
        raw_input = item.get("input", "")
        clean_input = clean_text(raw_input)
        
        if not clean_input:
            print(f"{i:<5} | {'SKIPPED (Empty)':<25} | -")
            continue

        # 构建 Prompt
        # 适配 Chat 模板 (如果模型支持 apply_chat_template)
        messages = [{"role": "user", "content": build_classification_prompt(clean_input)}]
        
        try:
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # 回退到普通拼接
            text_input = build_classification_prompt(clean_input)

        inputs = tokenizer(text_input, return_tensors="pt").to(device)

        # 推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20, # 分类名很短
                do_sample=False,   # 贪婪搜索，保证确定性
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码并提取回复
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 后处理：简单的文本匹配以防 LLM 废话
        matched_category = "Uncategorized"
        for cat in CATEGORIES:
            # 忽略大小写匹配
            if cat.lower() in generated_text.lower():
                matched_category = cat
                break
        
        stats[matched_category] += 1
        
        # 打印单条结果 (截断过长的文本以便显示)
        display_text = clean_input[:50].replace("\n", " ") + "..." if len(clean_input) > 50 else clean_input
        # 为了不破坏进度条，使用 tqdm.write 或者直接 print
        # 这里用 tqdm 格式化输出可能有点乱，所以简单打印，或者你可以注释掉 tqdm
        # print(f"{i:<5} | {matched_category:<25} | {display_text}")

    # 5. 输出最终统计报告
    print("\n" + "="*40)
    print(">>> CLASSIFICATION STATISTICS")
    print("="*40)
    total = sum(stats.values())
    for cat in CATEGORIES:
        count = stats[cat]
        percent = (count / total * 100) if total > 0 else 0
        print(f"{cat:<25}: {count:<5} ({percent:.1f}%)")
    
    print("-" * 40)
    print(f"{'Uncategorized':<25}: {stats['Uncategorized']:<5}")
    print(f"{'TOTAL':<25}: {total:<5}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()