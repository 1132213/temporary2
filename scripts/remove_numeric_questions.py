#!/usr/bin/env python3
"""
Filter ChatTS jsonl files by removing prompts that mention concrete numeric thresholds.
Supports Multi-GPU data parallelism.
python scripts/remove_numeric_questions.py \
    /root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/train_cleaned.jsonl \
    --model-name /root/emhua/btwu/Qwen14B \
    --gpus "4,5,6,7" \
    --output-name new_cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import os
import time
from typing import Iterable, Tuple
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置启动方法为 spawn，这是 CUDA 多进程必须的
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

PROMPT_TEMPLATE = """You are a strict data cleaner. Answer with ONLY \"YES\" or \"NO\".
Question: Does the following user prompt explicitly request analysis about specific
numeric thresholds, ranges, or points (for example: \"below -10\", \"point 76\",
\"values greater than 0.4\")? If the prompt references any concrete numeric targets
or bounds, respond YES. Otherwise respond NO.

Prompt:
{prompt}

Answer (YES or NO):"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove jsonl entries that talk about specific numeric values."
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=pathlib.Path,
        help="One or more jsonl files to clean.",
    )
    parser.add_argument(
        "--model-name",
        default="/root/emhua/btwu/Llama-3.2-3B",
        help="HF model name or local path. Must be a Llama 3B instruct checkpoint.",
    )
    parser.add_argument(
        "--output-name",
        default="train_cleaned.jsonl",
        help="Filename (not path) for cleaned output placed next to each source file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Number of tokens to sample from the classifier head.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation (0 = greedy).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype to load the model with.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). Default is '0'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return None
    return getattr(torch, name)


def load_model_on_device(model_name: str, dtype_name: str, device: str):
    """
    Load model specifically on the target device (avoiding device_map='auto' which might spread layers).
    """
    logging.info(f"Loading tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    dtype = resolve_dtype(dtype_name)
    logging.info(f"Loading model {model_name} on {device} (dtype={dtype_name})")
    
    # 直接加载到指定设备，不使用 device_map="auto" 以免占用其他卡资源
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer, model


@torch.inference_mode()
def classify_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int,
    temperature: float,
    device: str
) -> bool:
    """Return True if prompt mentions concrete numbers (should drop)."""
    question = PROMPT_TEMPLATE.format(prompt=prompt.strip())
    
    # 显式指定 device
    inputs = tokenizer(question, return_tensors="pt").to(device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip().lower()
    
    if answer.startswith("yes"):
        return True
    if answer.startswith("no"):
        return False
    # Logging inside worker process might be messy, but kept for consistency
    # logging.warning("Ambiguous classifier answer '%s'; defaulting to keep entry.", answer)
    return False


def worker_main(rank: int, gpu_id: str, args, in_queue: mp.Queue, out_queue: mp.Queue):
    """
    Worker process function.
    Loads model on specific GPU and consumes data from in_queue.
    """
    # 配置 Logging (多进程中需重新配置)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=f"[GPU-{gpu_id}] %(asctime)s %(levelname)s %(message)s",
    )
    
    device = f"cuda:{gpu_id}"
    try:
        tokenizer, model = load_model_on_device(args.model_name, args.dtype, device)
        logging.info(f"Worker {rank} ready on {device}")

        while True:
            item = in_queue.get()
            if item is None:
                # 收到结束信号
                break
            
            idx, raw_line, record = item
            
            prompt = record.get("input") or record.get("instruction") or ""
            should_drop = False
            
            if prompt:
                should_drop = classify_prompt(
                    prompt,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device
                )
            
            # 将结果发回主进程：(index, raw_line, record, should_drop)
            out_queue.put((idx, raw_line, record, should_drop))
            
    except Exception as e:
        logging.error(f"Worker {rank} crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logging.info(f"Worker {rank} finished.")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[Main] %(asctime)s %(levelname)s %(message)s",
    )

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    num_workers = len(gpu_list)
    logging.info(f"Starting parallel processing with {num_workers} workers on GPUs: {gpu_list}")

    for src in args.sources:
        if not src.exists():
            logging.error("Source file %s does not exist, skipping.", src)
            continue
        
        dst = src.with_name(args.output_name)
        logging.info("Processing %s -> %s", src, dst)

        # 1. 统计总行数以便显示进度条
        total_lines = 0
        with src.open("r", encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
        
        if total_lines == 0:
            logging.warning("File is empty.")
            continue

        # 2. 启动 Workers
        in_queue = mp.Queue(maxsize=num_workers * 100) # 限制队列大小，防止内存爆掉
        out_queue = mp.Queue()
        processes = []
        
        for i, gpu_id in enumerate(gpu_list):
            p = mp.Process(
                target=worker_main,
                args=(i, gpu_id, args, in_queue, out_queue)
            )
            p.start()
            processes.append(p)

        # 3. 启动一个线程或直接在主循环中推送数据？
        # 为了简单，我们在主线程推送数据，因为读取比推理快得多，但为了防止阻塞写出，
        # 我们最好先由主线程读取一部分，或者使用非阻塞方式。
        # 最好的方式：另外起一个线程读文件推入队列，主线程负责从 out_queue 拿结果写文件。
        
        import threading
        def producer(src_path, q, n_workers):
            with src_path.open("r", encoding="utf-8") as fin:
                for idx, line in enumerate(fin):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        q.put((idx, line, rec))
                    except json.JSONDecodeError:
                        # 错误行直接忽略或原样处理，这里选择原样通过（作为 keep 处理）
                        # 为简化逻辑，这里仅打印日志，不推入队列（即丢弃坏行）
                        logging.warning(f"Skipping bad json at line {idx}")
            
            # 发送结束信号
            for _ in range(n_workers):
                q.put(None)
        
        producer_thread = threading.Thread(target=producer, args=(src, in_queue, num_workers))
        producer_thread.start()

        # 4. 主循环：收集结果并写入
        # 注意：多进程处理后，结果顺序可能会乱。如果顺序不重要，直接写。
        # 如果顺序重要，需要缓存。通常清洗数据可以容忍乱序，但为了友好，我们这里直接追加写。
        
        kept = 0
        removed = 0
        
        with dst.open("w", encoding="utf-8") as fout:
            pbar = tqdm(total=total_lines, desc=f"Cleaning {src.name}", unit="rec")
            processed_count = 0
            
            while processed_count < total_lines:
                # 注意：如果Producer线程因只有坏行而提前结束，这里需要更健壮的退出机制
                # 但这里假设 total_lines 是准确的。
                # 更安全的做法是计数 workers 的结束信号，但这里用 total_lines + timeout 做简单处理
                
                try:
                    # 阻塞等待结果
                    res = out_queue.get(timeout=600) # 10分钟没有结果则超时
                    processed_count += 1
                    pbar.update(1)
                    
                    idx, raw_line, record, should_drop = res
                    
                    if should_drop:
                        removed += 1
                    else:
                        fout.write(raw_line) # 这里我们写回原始行，或者 json.dumps(record)
                        kept += 1
                        
                except Exception: # Queue Empty or Timeout
                    # 检查生产者是否活著，如果生产者死了且队列空了，就退出
                    if not producer_thread.is_alive() and out_queue.empty():
                        break
            
            pbar.close()

        # 等待所有进程结束
        producer_thread.join()
        for p in processes:
            p.join()

        logging.info(
            "Finished %s: kept %d | removed %d | output %s",
            src,
            kept,
            removed,
            dst,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())