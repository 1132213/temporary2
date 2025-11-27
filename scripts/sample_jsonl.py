#!/usr/bin/env python3
"""
随机抽取 JSONL 文件中的 N 条记录并保存为新文件。

用法:
    python sample_jsonl.py input.jsonl output.jsonl 1000
"""

import argparse
import json
import random
import sys
from pathlib import Path


def sample_jsonl(input_path: Path, output_path: Path, n: int, seed: int | None = None) -> None:
    """
    从 input_path 中随机抽取 n 条记录写入 output_path。

    参数:
        input_path:  输入 .jsonl 文件路径
        output_path: 输出 .jsonl 文件路径
        n:           需要抽取的条数
        seed:        随机种子（可选，用于复现）
    """
    if seed is not None:
        random.seed(seed)

    # 先快速统计总行数
    total = 0
    with input_path.open("rt", encoding="utf-8") as f:
        for _ in f:
            total += 1

    if n > total:
        print(f"警告：请求抽取 {n} 条，但文件只有 {total} 条，将返回全部数据。", file=sys.stderr)
        n = total

    # 生成要抽取的行号（0-based）
    chosen_lines = set(random.sample(range(total), n))

    with input_path.open("rt", encoding="utf-8") as fin, output_path.open("wt", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx in chosen_lines:
                # 简单验证 JSON 合法性（可选）
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"跳过非法 JSON 行 {idx + 1}: {e}", file=sys.stderr)
                    continue
                fout.write(line)

    print(f"已抽取 {n} 条记录 -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="随机抽取 JSONL 中的 N 条记录")
    parser.add_argument("input", type=Path, help="输入 .jsonl 文件")
    parser.add_argument("output", type=Path, help="输出 .jsonl 文件")
    parser.add_argument("num", type=int, help="需要抽取的条数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可选）")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.num <= 0:
        print("错误：抽取条数必须 > 0", file=sys.stderr)
        sys.exit(1)

    sample_jsonl(args.input, args.output, args.num, args.seed)


if __name__ == "__main__":
    main()