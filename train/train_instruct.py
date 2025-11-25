import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from datetime import datetime

# 引入项目模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn
from test.common import set_seed

def compute_loss(model, series, prefixes, suffixes, device):
    """
    计算 Causal LM Loss
    """
    model_out = model(series, prefixes, suffixes)
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits
    
    prefix_width = model_out["prefix_mask"].shape[1]
    ts_width = 1 + model_out["ts_tokens"].shape[1]
    start_idx = prefix_width + ts_width
    
    suffix_logits = logits[:, start_idx:, :]
    
    suffix_labels = model.tokenizer.tokenizer(
        suffixes, return_tensors="pt", padding=True
    ).input_ids.to(device)
    
    min_len = min(suffix_logits.shape[1], suffix_labels.shape[1])
    suffix_logits = suffix_logits[:, :min_len, :]
    suffix_labels = suffix_labels[:, :min_len]
    
    shift_logits = suffix_logits[..., :-1, :].contiguous()
    shift_labels = suffix_labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    return loss

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for series, prefixes, suffixes in val_loader:
            series = series.to(device)
            loss = compute_loss(model, series, prefixes, suffixes, device)
            total_loss += loss.item()   
    return total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

def train(args):
    # 日志设置
    script_dir = Path(__file__).parent
    log_filename = script_dir / f"instruct_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_filename}")
    
    set_seed(args.seed)
    device = torch.device(args.device)

    # 1. 初始化配置
    # 注意：Stage 3 可以选择解冻 Encoder (freeze_patch_encoder=False) 进行微调，
    # 也可以保持冻结 (True) 仅微调 Adapter。这里通过 args 控制。
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim, 
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=args.device,
        llm_dtype="bfloat16",
        freeze_patch_encoder=args.freeze_encoder, # <--- 通过参数控制
        use_stats_projector=True
    )
    
    logger.info(">>> Initializing Model for Instruction Tuning...")
    model = StatBypassCROMETS1(config).to(device)
    
    # 2. 加载 Stage 2 (Alignment) 的权重
    # 这是关键步骤，接续上一阶段的成果
    if Path(args.stage2_checkpoint).exists():
        logger.info(f">>> Loading Stage 2 Aligned Weights from {args.stage2_checkpoint}...")
        state_dict = torch.load(args.stage2_checkpoint, map_location=device)
        # strict=False 因为配置可能微调（比如 requires_grad 属性），但权重键值应该匹配
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        logger.warning("!!! Stage 2 checkpoint not found! Training from scratch (NOT RECOMMENDED). !!!")

    # 3. 确定可训练参数
    # 再次确认冻结状态，防止 Config 没覆盖到位
    if args.freeze_encoder:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f">>> Trainable Parameters: {len(trainable_params)} tensors")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 4. 数据加载
    logger.info(f">>> Loading Dataset: {args.jsonl_path}...")
    train_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="train")
    val_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=instruct_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=instruct_collate_fn)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 5. 训练循环
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss_sum = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Instruct]")
        
        for series, prefixes, suffixes in progress:
            series = series.to(device)
            loss = compute_loss(model, series, prefixes, suffixes, device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0) # 梯度裁剪
            optimizer.step()
            
            train_loss_sum += loss.item()
            progress.set_postfix(loss=loss.item())
            
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        logger.info(f"Epoch {epoch+1} [Eval] Computing Validation Loss...")
        avg_val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            logger.info(f">>> Val Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving Model...")
            best_val_loss = avg_val_loss
            model_path = script_dir / "crome_instruct_best.pth"
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    parser.add_argument("--stage2-checkpoint", type=str, default="crome_stage2_aligned.pth", help="上一阶段的权重路径")
    parser.add_argument("--freeze-encoder", action="store_true", help="是否在微调阶段继续冻结 PatchTST Encoder")
    
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5) # 微调阶段学习率通常比对齐阶段小
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    train(args)