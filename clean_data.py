import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def clean_jsonl(input_path, output_path, std_threshold=1e-4):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found.")
        return

    valid_count = 0
    total_count = 0
    removed_stats = {
        "nan_inf": 0,
        "constant": 0,  # 标准差过小
        "too_short": 0, # 长度不符合预期（可选）
        "extreme": 0    # 包含极端异常值
    }

    print(f">>> Scanning {input_file}...")
    
    # 使用 buffer 写入，提高 IO 效率
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                item = json.loads(line)
                
                # 1. 获取时序数据
                # 注意：根据您的描述，数据直接在 'input_ts' 字段中，是 list
                ts_data = item.get("input_ts", [])
                
                # 转为 numpy 数组进行计算
                ts_array = np.array(ts_data, dtype=np.float32)
                
                # --- 检查 1: 长度是否合法 ---
                # 如果您要求固定长度（如 1024），可以在这里过滤
                if len(ts_array) < 16: # 至少要能切出一个 Patch
                    removed_stats["too_short"] += 1
                    continue
                    
                # --- 检查 2: NaN / Inf ---
                if np.isnan(ts_array).any() or np.isinf(ts_array).any():
                    removed_stats["nan_inf"] += 1
                    continue
                
                # --- 检查 3: 常数序列 (标准差极小) ---
                # 这是导致 RevIN 崩溃的主要原因
                std_val = np.std(ts_array)
                if std_val < std_threshold:
                    removed_stats["constant"] += 1
                    continue

                # --- 检查 4: 极端异常值 (可选) ---
                # 例如：剔除绝对值超过 1e8 的物理上不可能的数值
                if np.max(np.abs(ts_array)) > 1e8:
                    removed_stats["extreme"] += 1
                    continue

                # --- 通过检查，写入新文件 ---
                # 直接写入原行，避免 json.dumps 的额外开销（除非你想修改数据）
                fout.write(line + '\n')
                valid_count += 1
                
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON at line {total_count}")
                continue
            except Exception as e:
                print(f"Error processing line {total_count}: {e}")
                continue

    # --- 输出报告 ---
    print("\n" + "="*40)
    print(f"Data Cleaning Report")
    print("="*40)
    print(f"Input File:   {input_path}")
    print(f"Output File:  {output_path}")
    print(f"Total Lines:  {total_count}")
    print(f"Valid Lines:  {valid_count} ({valid_count/total_count*100:.2f}%)")
    print(f"Removed:      {total_count - valid_count}")
    print("-" * 20)
    print("Removal Reasons:")
    print(f"  - NaN / Inf:        {removed_stats['nan_inf']}")
    print(f"  - Constant (Std~0): {removed_stats['constant']}")
    print(f"  - Too Short:        {removed_stats['too_short']}")
    print(f"  - Extreme Values:   {removed_stats['extreme']}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean JSONL time series data")
    parser.add_argument("--input", type=str, required=True, help="Path to dirty .jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to save clean .jsonl file")
    parser.add_argument("--std-threshold", type=float, default=1e-4, help="Threshold for constant sequence detection")
    
    args = parser.parse_args()
    
    clean_jsonl(args.input, args.output, args.std_threshold)