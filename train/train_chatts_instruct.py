import sys
from pathlib import Path

# 路径设置：确保能找到 src 和 test 包
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 引入项目模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn
from test.common import set_seed
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
from datetime import datetime
import math

def compute_chatts_loss(model, batch, device):
    """
    计算 ChatTS 格式的 Causal LM Loss
    """
    input_texts = batch["input_texts"]
    timeseries_lists = batch["timeseries_lists"]
    output_texts = batch["output_texts"]
    
    # 使用 forward_chatts 方法
    model_out = model.forward_chatts(
        input_texts=input_texts,
        timeseries_lists=timeseries_lists,
        output_texts=output_texts,
        llm_kwargs={}
    )
    
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits  # [B, Total_L, Vocab]
    
    # 获取输出文本的标签
    suffix_labels = model.tokenizer.tokenizer(
        output_texts, return_tensors="pt", padding=True
    ).input_ids.to(device)
    
    # 计算输出文本在完整序列中的起始位置
    batch_size = logits.shape[0]
    suffix_mask_lengths = model_out["suffix_mask_lengths"]
    
    # 计算每个样本的 suffix 起始位置
    losses = []
    for i in range(batch_size):
        # 获取该样本的有效长度（排除padding）
        valid_len = model_out["attention_mask"][i].sum().item()
        suffix_len = suffix_mask_lengths[i]
        
        if suffix_len == 0:
            continue
        
        # suffix 的起始位置
        suffix_start = int(valid_len - suffix_len)
        
        # 提取该样本的 logits 和 labels
        sample_logits = logits[i, suffix_start:suffix_start+suffix_len, :]  # [suffix_len, Vocab]
        sample_labels = suffix_labels[i, :suffix_len]  # [suffix_len]
        
        # Causal LM: 预测下一个 token
        if sample_logits.shape[0] > 1:
            shift_logits = sample_logits[:-1, :]  # [suffix_len-1, Vocab]
            shift_labels = sample_labels[1:]  # [suffix_len-1]
            
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return torch.stack(losses).mean()

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_chatts_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0

def train_chatts_instruct(args):
    # 日志设置
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 创建log和model文件夹
    log_dir = project_root / "log"
    model_dir = project_root / "model"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"chatts_instruct_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_filename}")
    
    set_seed(args.seed)
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None and ":" not in args.device:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)

    # 1. 初始化配置
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    # ChatTS 格式中每个时间序列都是单通道
    input_channels = 1
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim, 
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16",
        freeze_patch_encoder=args.freeze_encoder, # <--- 通过参数控制
        use_stats_projector=True,
    )
    
    logger.info(">>> Initializing Model for ChatTS Instruction Tuning...")
    model = StatBypassCROMETS1(config).to(device)
    
    # 2. 加载 Stage 2 (Alignment) 的权重
    # 处理相对路径和绝对路径
    stage2_path = Path(args.stage2_checkpoint)
    if not stage2_path.is_absolute():
        # 如果是相对路径，尝试在model文件夹中查找
        if not stage2_path.exists():
            model_path = model_dir / stage2_path.name
            if model_path.exists():
                stage2_path = model_path
            else:
                # 尝试在项目根目录查找
                root_path = project_root / stage2_path
                if root_path.exists():
                    stage2_path = root_path
    
    if stage2_path.exists():
        logger.info(f">>> Loading Stage 2 Aligned Weights from {stage2_path}...")
        state_dict = torch.load(str(stage2_path), map_location=device)
        # strict=False 因为配置可能微调（比如 requires_grad 属性），但权重键值应该匹配
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        logger.warning(f"!!! Stage 2 checkpoint not found at {stage2_path}! Training from scratch (NOT RECOMMENDED). !!!")

    # 3. 确定可训练参数
    # 再次确认冻结状态，防止 Config 没覆盖到位
    if args.freeze_encoder:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = False
    
    # 冻结 LLM
    for p in model.llm.parameters():
        p.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f">>> Trainable Parameters: {len(trainable_params)} tensors")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 4. 数据加载
    logger.info(f">>> Loading ChatTS Dataset from {args.jsonl_path}...")
    train_ds = ChatTSDataset(
        args.jsonl_path, args.seq_len, input_channels, 
        split="train"
    )
    val_ds = ChatTSDataset(
        args.jsonl_path, args.seq_len, input_channels, 
        split="val"
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=chatts_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=chatts_collate_fn)
    
    # 计算 steps_per_epoch
    accumulation_steps = args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    
    logger.info(f">>> OneCycleLR Steps per epoch: {steps_per_epoch} (Loader len: {len(train_loader)}, Accum: {accumulation_steps})")
    logger.info(f">>> Gradient Accumulation Steps: {accumulation_steps}")
    logger.info(f">>> Effective Batch Size: {args.batch_size * accumulation_steps}")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=25.0
    )
    
    # 5. 训练循环
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss_sum = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [ChatTS-Instruct]")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            loss = compute_chatts_loss(model, batch, device) / accumulation_steps
            loss.backward()
            
            train_loss_sum += loss.item() * accumulation_steps
            
            # 梯度累积更新
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            # 在进度条中同时显示 Loss 和 LR
            progress.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.2e}")
        
        # 处理剩余梯度
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        logger.info(f"Epoch {epoch+1} [Eval] Computing Validation Loss...")
        avg_val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            logger.info(f">>> Val Loss Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving Model...")
            best_val_loss = avg_val_loss
            model_path = model_dir / "chatts_instruct_best.pth"
            torch.save(model.state_dict(), model_path)
            logger.info(f">>> Model saved to: {model_path}")
    
    logger.info(">>> ChatTS Instruction Tuning Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatTS 格式数据 - Stage 3 指令微调")
    parser.add_argument("--jsonl-path", type=str, 
                        default="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/train_cleaned.jsonl",
                        help="ChatTS 格式的 JSONL 数据文件路径")
    parser.add_argument("--stage2-checkpoint", type=str, 
                        default="model/chatts_stage2_aligned.pth", 
                        help="Stage 2 对齐阶段的权重路径")
    parser.add_argument("--freeze-encoder", action="store_true", 
                        help="是否在微调阶段继续冻结 PatchTST Encoder")
    
    parser.add_argument("--seq-len", type=int, default=512, help="时间序列长度")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="学习率（微调阶段通常比对齐阶段小）") 
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None, help="指定使用的GPU卡号")
    
    args = parser.parse_args()
    train_chatts_instruct(args)

