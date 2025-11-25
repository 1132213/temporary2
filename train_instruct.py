import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# 引入模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn
from test.common import set_seed

def compute_loss(model, series, prefixes, suffixes, device):
    """
    封装 Loss 计算逻辑，供 Train 和 Eval 共用
    """
    # 获取模型输出
    model_out = model(series, prefixes, suffixes)
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits
    
    # 计算 Suffix 部分的 Logits 起始位置
    # Suffix Start = Prefix长度 + Stat(1) + TS(N)
    prefix_width = model_out["prefix_mask"].shape[1]
    ts_width = 1 + model_out["ts_tokens"].shape[1]
    start_idx = prefix_width + ts_width
    
    # 截取 Suffix 对应的 Logits
    suffix_logits = logits[:, start_idx:, :]
    
    # 获取 Suffix 的真实 Token IDs (Label)
    # 注意：这里使用了 padding=True，这在 batch_size > 1 时是必要的
    suffix_labels = model.tokenizer.tokenizer(
        suffixes, return_tensors="pt", padding=True
    ).input_ids.to(device)
    
    # 对齐长度 (以较短者为准，防止 Shape Mismatch)
    # 理论上如果不截断，应该用 Attention Mask 处理 Padding Loss，这里简化处理直接截断
    min_len = min(suffix_logits.shape[1], suffix_labels.shape[1])
    suffix_logits = suffix_logits[:, :min_len, :]
    suffix_labels = suffix_labels[:, :min_len]
    
    # Causal LM Loss: Shift Logits & Labels
    # Pred[t] 预测 Label[t+1]
    shift_logits = suffix_logits[..., :-1, :].contiguous()
    shift_labels = suffix_labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    return loss

def evaluate(model, val_loader, device):
    """
    执行验证循环，计算 Validation Loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for series, prefixes, suffixes in val_loader:
            series = series.to(device)
            loss = compute_loss(model, series, prefixes, suffixes, device)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    return avg_loss

def train(args):
    # 设置日志：同时输出到控制台和文件
    script_dir = Path(__file__).parent
    log_filename = script_dir / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_filename}")
    logger.info("=" * 60)
    logger.info("训练参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    set_seed(args.seed)

    # 若指定 GPU ID，则优先使用 `cuda:{gpu_id}` 形式
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None and ":" not in args.device:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)

    # 1. 配置与初始化
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim, 
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16"
    )
    
    logger.info(">>> Initializing Model...")
    model = StatBypassCROMETS1(config).to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f">>> Trainable Parameters: {len(trainable_params)} tensors")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 2. 数据加载
    logger.info(f">>> Loading Dataset: {args.jsonl_path}...")
    train_ds = JSONLInstructDataset(
        args.jsonl_path, args.seq_len, args.input_channels, split="train"
    )
    val_ds = JSONLInstructDataset(
        args.jsonl_path, args.seq_len, args.input_channels, split="val"
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=instruct_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=instruct_collate_fn)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. 训练循环
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss_sum = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for series, prefixes, suffixes in progress:
            series = series.to(device)
            
            loss = compute_loss(model, series, prefixes, suffixes, device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            progress.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # --- Validation Phase ---
        logger.info(f"Epoch {epoch+1} [Eval] Computing Validation Loss...")
        avg_val_loss = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model based on VAL LOSS
        if avg_val_loss < best_val_loss:
            logger.info(f">>> Val Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving Model...")
            best_val_loss = avg_val_loss
            model_path = script_dir / "crome_instruct_best.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f">>> Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID，例如 0/1")
    
    args = parser.parse_args()
    train(args)