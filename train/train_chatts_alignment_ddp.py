import sys
import os
from pathlib import Path

# 禁用 tokenizers 并行以避免多进程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
from datetime import datetime
import math
import os

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到分布式环境变量，使用单卡模式")
        rank = 0
        world_size = 1
        local_rank = 0
    
    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_chatts_loss(model, batch, device):
    """
    计算 ChatTS 格式的 Causal LM Loss (修正版：解决 EOS 对齐问题)
    """
    input_texts = batch["input_texts"]
    timeseries_lists = batch["timeseries_lists"]
    output_texts = batch["output_texts"]
    
    # 获取 tokenizer
    # DDP 模式下需要通过 module 访问
    if isinstance(model, DDP):
        tokenizer = model.module.tokenizer.tokenizer
    else:
        tokenizer = model.tokenizer.tokenizer
        
    # === 核心修正 1: 构造带有 EOS 的文本 ===
    # 确保模型输入和标签都包含 EOS Token，防止“不会闭嘴”和 Loss 对齐错误
    eos_token = tokenizer.eos_token
    output_texts_with_eos = [str(text) + eos_token for text in output_texts]
    
    # === 核心修正 2: 将带 EOS 的文本传给 Forward ===
    actual_model = model.module if isinstance(model, DDP) else model
    model_out = actual_model.forward_chatts(
        input_texts=input_texts,
        timeseries_lists=timeseries_lists,
        output_texts=output_texts_with_eos,  # <--- 使用带 EOS 的文本
        llm_kwargs={}
    )
    
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits  # [B, Total_L, Vocab]
    
    # === 核心修正 3: 生成标签 (保持默认行为) ===
    # tokenizer 默认行为 (add_special_tokens=True) 会添加 BOS
    # 这样 Input = [BOS, ..., EOS], Label = [BOS, ..., EOS]
    # 错位预测时刚好对齐
    suffix_labels = tokenizer(
        output_texts_with_eos, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True 
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
        
        # 提取 logits 和 labels
        sample_logits = logits[i, suffix_start:suffix_start+suffix_len, :]  # [suffix_len, Vocab]
        sample_labels = suffix_labels[i, :suffix_len]  # [suffix_len]
        
        # 对齐截断 (防止 tokenizer padding 导致的微小长度差异)
        min_len = min(sample_logits.shape[0], sample_labels.shape[0])
        sample_logits = sample_logits[:min_len]
        sample_labels = sample_labels[:min_len]
        
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

def evaluate(model, val_loader, device, rank):
    """
    验证函数 - 支持分布式
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_chatts_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    # 在分布式环境下同步所有进程的损失
    if dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        num_batches = int(loss_tensor[1].item())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def train_chatts_alignment_ddp(args, rank, world_size, local_rank):
    """
    多卡分布式训练主函数
    """
    # 设置设备
    device = torch.device(f"cuda:{local_rank}")
    
    # --- 日志设置（只在 rank 0 上设置）---
    logger = None
    if rank == 0:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # 创建log和model文件夹
        log_dir = project_root / "log"
        model_dir = project_root / "model"
        log_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)
        
        log_filename = log_dir / f"chatts_alignment_ddp_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"日志文件保存在: {log_filename}")
        logger.info(f"分布式训练: Rank {rank}/{world_size}, Local Rank: {local_rank}")
    
    set_seed(args.seed + rank)  # 每个进程使用不同的随机种子

    # 1. 初始化配置
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    if rank == 0 and logger:
        logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    # ChatTS 格式中每个时间序列都是单通道
    input_channels = 1
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim, 
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=f"cuda:{local_rank}",  # 每个进程使用自己的 GPU
        llm_dtype="bfloat16",
        freeze_patch_encoder=True,
        use_stats_projector=True,
        epsilon=1e-5,
    )
    
    if rank == 0 and logger:
        logger.info(">>> Initializing Model for ChatTS Alignment...")
    
    model = StatBypassCROMETS1(config).to(device)

    # 2. 加载预训练的权重 (Stage 1) - 所有进程都需要加载
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / "model"
    
    pretrained_path = Path(args.pretrained_encoder_path)
    if not pretrained_path.is_absolute():
        if not pretrained_path.exists():
            model_path = model_dir / pretrained_path.name
            if model_path.exists():
                pretrained_path = model_path
            else:
                root_path = project_root / pretrained_path
                if root_path.exists():
                    pretrained_path = root_path
    
    if pretrained_path.exists():
        if rank == 0 and logger:
            logger.info(f">>> Loading Pre-trained Weights from {pretrained_path}...")
        
        checkpoint = torch.load(str(pretrained_path), map_location=device)
        
        def fix_state_dict_keys(state_dict):
            """修复预训练权重的键名，使其与当前模型匹配"""
            fixed_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith('encoder.encoder.'):
                    new_k = k.replace('encoder.encoder.', 'encoder.')
                elif k.startswith('encoder.embedding.'):
                    new_k = k.replace('encoder.embedding.', 'embedding.')
                elif k in ['mask_token', 'head.weight', 'head.bias'] or k.startswith('pos_encoding.'):
                    continue
                fixed_dict[new_k] = v
            return fixed_dict
        
        # 适配新的保存格式
        if "state_dict" in checkpoint:
            encoder_dict = {k.replace("encoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
            if encoder_dict:
                encoder_dict = fix_state_dict_keys(encoder_dict)
                msg = model.ts_model.shape_encoder.load_state_dict(encoder_dict, strict=False)
                if rank == 0 and logger:
                    logger.info(f"Loaded Shape Encoder from state_dict: {msg}")
            else:
                if rank == 0 and logger:
                    logger.warning("!!! No encoder weights found in state_dict prefix 'encoder.' !!!")
        elif "encoder" in checkpoint:
            encoder_dict = fix_state_dict_keys(checkpoint["encoder"])
            msg = model.ts_model.shape_encoder.load_state_dict(encoder_dict, strict=False)
            if rank == 0 and logger:
                logger.info(f"Loaded Shape Encoder from dict: {msg}")
        else:
            fixed_checkpoint = fix_state_dict_keys(checkpoint)
            msg = model.ts_model.shape_encoder.load_state_dict(fixed_checkpoint, strict=False)
            if rank == 0 and logger:
                logger.info(f"Loaded Shape Encoder (Direct): {msg}")
    else:
        if rank == 0 and logger:
            logger.warning(f"!!! Pre-trained checkpoint not found at {pretrained_path}! Using random initialization. !!!")

    # 3. 冻结参数策略
    for name, param in model.named_parameters():
        if "llm" in name:
            param.requires_grad = False
        elif "shape_encoder" in name:
            param.requires_grad = False
        elif "preprocessor" in name:
            param.requires_grad = False
        else:
            # 训练：Q-Former, Adapter, Projectors
            param.requires_grad = True
    
    # 4. 使用 DDP 包装模型
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True  # 因为某些参数可能不参与梯度计算
        )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0 and logger:
        logger.info(f">>> Stage 2 Trainable Parameters: {len(trainable_params)} tensors")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 5. 数据加载 - 使用 DistributedSampler
    if rank == 0 and logger:
        logger.info(f">>> Loading ChatTS Dataset from {args.jsonl_path}...")
    
    train_ds = ChatTSDataset(args.jsonl_path, args.seq_len, input_channels, split="train")
    val_ds = ChatTSDataset(args.jsonl_path, args.seq_len, input_channels, split="val")
    
    # 使用分布式采样器
    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        seed=args.seed
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # 如果使用 sampler，则不能 shuffle
        collate_fn=chatts_collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        shuffle=False,
        collate_fn=chatts_collate_fn,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    # 计算 steps_per_epoch
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    
    if rank == 0 and logger:
        logger.info(f">>> OneCycleLR Steps per epoch: {steps_per_epoch} (Loader len: {len(train_loader)}, Accum: {args.gradient_accumulation_steps})")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1,
        div_factor=25.0
    )
    
    # 6. 训练循环
    best_val_loss = float("inf")
    accumulation_steps = args.gradient_accumulation_steps
    
    if rank == 0 and logger:
        logger.info(f">>> Gradient Accumulation Steps: {accumulation_steps}")
        logger.info(f">>> Effective Batch Size: {args.batch_size * accumulation_steps * world_size}")
    
    for epoch in range(args.epochs):
        # 设置 epoch（对于 DistributedSampler 很重要）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        train_loss_sum = 0
        
        # 只在 rank 0 上显示进度条
        if rank == 0:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [ChatTS-Align-DDP]")
        else:
            progress = train_loader
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            # 计算 loss
            loss = compute_chatts_loss(model, batch, device)
            
            # === DDP 安全的 NaN 处理 ===
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.warning(f"!!! NaN detected at Epoch {epoch+1} Step {step}. Zeroing loss.")
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 归一化 (Gradient Accumulation)
            loss = loss / accumulation_steps
            loss.backward()
            
            train_loss_sum += loss.item() * accumulation_steps
            
            # 梯度累积更新
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 只在 rank 0 上更新进度条
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.2e}")
        
        # 处理剩余梯度
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 同步所有进程的训练损失
        avg_train_loss = train_loss_sum / len(train_loader)
        if dist.is_initialized():
            train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = train_loss_tensor.item()
        
        # Validation
        avg_val_loss = evaluate(model, val_loader, device, rank)
        
        if rank == 0 and logger:
            logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 只在 rank 0 上保存模型
        if rank == 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 根据 model_suffix 参数构建模型文件名
            if args.model_suffix:
                model_filename = f"chatts_stage2_aligned_ddp_{args.model_suffix}.pth"
            else:
                model_filename = "chatts_stage2_aligned_ddp.pth"
            save_path = model_dir / model_filename
            
            # 获取实际模型（去除 DDP wrapper）
            model_to_save = model.module if isinstance(model, DDP) else model
            
            # 只保存非 LLM 的部分
            full_state_dict = model_to_save.state_dict()
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if "llm.model" not in k}
            
            torch.save(filtered_state_dict, save_path)
            if logger:
                logger.info(f">>> Best ChatTS Alignment Model (Filtered LLM) saved to: {save_path}")
        
        # 同步所有进程（确保所有进程在同一 epoch）
        if dist.is_initialized():
            dist.barrier()

    if rank == 0 and logger:
        logger.info(">>> ChatTS Alignment Training Completed!")

def main():
    parser = argparse.ArgumentParser(description="ChatTS 格式数据 - Stage 2 多卡对齐训练")
    parser.add_argument("--jsonl-path", type=str, 
                        default="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/train_cleaned.jsonl",
                        help="ChatTS 格式的 JSONL 数据文件路径")
    parser.add_argument("--pretrained-encoder-path", type=str, 
                        default="model/patchtst_pretrained_full_3b.pth", 
                        help="Stage 1预训练权重路径")
    parser.add_argument("--seq-len", type=int, default=512, help="时间序列长度")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    parser.add_argument("--batch-size", type=int, default=4, help="每个GPU的批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率") 
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 的工作进程数")
    parser.add_argument("--model-suffix", type=str, default="", 
                        help="模型文件名的后缀，例如：1st，则模型名为 chatts_stage2_aligned_ddp_1st.pth")
    
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    try:
        train_chatts_alignment_ddp(args, rank, world_size, local_rank)
    finally:
        # 清理分布式环境
        cleanup_distributed()

if __name__ == "__main__":
    main()