#!/usr/bin/env python
"""
将 GiftEval 数据集中的原始时间序列切分成固定长度的滑动窗口，
并生成可供 `train/pretrain_encoder.py` 使用的 JSONL 索引文件。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, Tuple, List

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


def iter_gifteval_dirs(root: Path) -> Iterator[Tuple[str, str, Path]]:
    """遍历 GiftEval 目录，返回 (dataset_name, freq_name, path) 三元组。"""
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for freq_dir in sorted(dataset_dir.iterdir()):
            if not freq_dir.is_dir():
                continue
            if (freq_dir / "dataset_info.json").exists():
                yield dataset_dir.name, freq_dir.name, freq_dir


def sanitize_name(text: str | None) -> str:
    """将 item_id 等文本转换为安全的文件名片段。"""
    if not text:
        return "series"
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", text)
    safe = safe.strip("_")
    return safe or "series"


def target_to_2d_array(target) -> np.ndarray:
    """
    将 GiftEval 的 target 字段转换为 [time, channels] 形状。
    GiftEval 中多变量序列通常是 [channels, time]，需要转置。
    """
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2:
        arr = arr.T  # GiftEval 的多变量格式：每个 channel 一行
    else:
        arr = arr.reshape(arr.shape[0], -1).T
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def ensure_single_channel(arr: np.ndarray) -> np.ndarray:
    """
    将输入数组转换为单通道格式 [time]。
    多通道序列沿通道维取均值，再压缩为一维。
    """
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim >= 2:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32)


def generate_windows(
    arr: np.ndarray,
    seq_len: int,
    stride: int,
) -> List[np.ndarray]:
    """对单条序列切片并返回窗口数组列表（仅内存，不落地 .npy）。"""
    num_steps = arr.shape[0]
    if num_steps < seq_len:
        return []

    windows: list[np.ndarray] = []
    for start in range(0, num_steps - seq_len + 1, stride):
        window = arr[start : start + seq_len].astype(np.float32)
        windows.append(ensure_single_channel(window))
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="GiftEval 滑动窗口生成脚本")
    parser.add_argument(
        "--gift-root",
        type=str,
        default="/root/emhua/btwu/timedataset/GiftEval",
        help="GiftEval 根目录（包含各个子数据集）",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="/root/emhua/btwu/CROME2/data/gifteval_windows.jsonl",
        help="输出 JSONL 索引文件路径",
    )
    parser.add_argument("--seq-len", type=int, default=1024, help="窗口长度")
    parser.add_argument("--stride", type=int, default=512, help="滑动窗口步长")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="限制写入 JSONL 的窗口条数（默认不限）",
    )
    args = parser.parse_args()

    gift_root = Path(args.gift_root)
    jsonl_path = Path(args.jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    total_windows = 0
    total_series = 0

    with jsonl_path.open("w", encoding="utf-8") as writer:
        series_counter = 0
        for dataset_name, freq_name, path in iter_gifteval_dirs(gift_root):
            ds = load_from_disk(str(path))
            iterable = tqdm(
                ds,
                desc=f"{dataset_name}-{freq_name}",
                leave=False,
            )
            for row in iterable:
                series_counter += 1
                total_series += 1

                item_id = sanitize_name(row.get("item_id"))
                arr = target_to_2d_array(row["target"])

                windows = generate_windows(
                    arr,
                    seq_len=args.seq_len,
                    stride=args.stride,
                )

                for window_idx, window in enumerate(windows):
                    if args.max_samples is not None and total_windows >= args.max_samples:
                        break
                    record = {
                        "input_ts": window.tolist(),
                        "ts_shape": list(window.shape),
                        "ts_length": int(window.shape[0]),
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_windows += 1
                if args.max_samples is not None and total_windows >= args.max_samples:
                    break
            if args.max_samples is not None and total_windows >= args.max_samples:
                break

    print(
        f"Finished. Processed {total_series} series -> {total_windows} windows. "
        f"JSONL saved to {jsonl_path.resolve()}"
    )


if __name__ == "__main__":
    main()

