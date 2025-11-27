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
from torch.utils.data import DataLoader
from torch.optim import AdamW
# 引入 OneCycleLR
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
from datetime import datetime
import math # 引入 math 用于向上取整

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

def train_chatts_alignment(args):
    # --- 日志设置 ---
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 创建log和model文件夹
    log_dir = project_root / "log"
    model_dir = project_root / "model"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"chatts_alignment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        freeze_patch_encoder=True,  # 初始冻结，后续通过 param.requires_grad 细粒度控制
        use_stats_projector=True,
        epsilon=1e-5,
    )
    
    logger.info(">>> Initializing Model for ChatTS Alignment...")
    model = StatBypassCROMETS1(config).to(device)

    # 2. 加载预训练的权重 (Stage 1)
    # 处理相对路径和绝对路径
    pretrained_path = Path(args.pretrained_encoder_path)
    if not pretrained_path.is_absolute():
        # 如果是相对路径，尝试在model文件夹中查找
        if not pretrained_path.exists():
            model_path = model_dir / pretrained_path.name
            if model_path.exists():
                pretrained_path = model_path
            else:
                # 尝试在项目根目录查找
                root_path = project_root / pretrained_path
                if root_path.exists():
                    pretrained_path = root_path
    
    if pretrained_path.exists():
        logger.info(f">>> Loading Pre-trained Weights from {pretrained_path}...")
        checkpoint = torch.load(str(pretrained_path), map_location=device)
        
        # 修复键名函数：处理预训练权重的键名前缀问题
        def fix_state_dict_keys(state_dict):
            """修复预训练权重的键名，使其与当前模型匹配"""
            fixed_dict = {}
            for k, v in state_dict.items():
                new_k = k
                # 修复 encoder.encoder.layers.X -> encoder.layers.X
                if k.startswith('encoder.encoder.'):
                    new_k = k.replace('encoder.encoder.', 'encoder.')
                # 修复 encoder.embedding -> embedding
                elif k.startswith('encoder.embedding.'):
                    new_k = k.replace('encoder.embedding.', 'embedding.')
                # 跳过不需要的键（mask_token, pos_encoding, head 等）
                elif k in ['mask_token', 'head.weight', 'head.bias'] or k.startswith('pos_encoding.'):
                    continue
                fixed_dict[new_k] = v
            return fixed_dict
        
        # 适配新的保存格式 (state_dict)
        if "state_dict" in checkpoint:
            # 过滤出 encoder 相关的权重
            encoder_dict = {k.replace("encoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
            if encoder_dict:
                # 修复键名
                encoder_dict = fix_state_dict_keys(encoder_dict)
                msg = model.ts_model.shape_encoder.load_state_dict(encoder_dict, strict=False)
                logger.info(f"Loaded Shape Encoder from state_dict: {msg}")
            else:
                logger.warning("!!! No encoder weights found in state_dict prefix 'encoder.' !!!")
        # 兼容旧格式
        elif "encoder" in checkpoint:
            encoder_dict = fix_state_dict_keys(checkpoint["encoder"])
            msg = model.ts_model.shape_encoder.load_state_dict(encoder_dict, strict=False)
            logger.info(f"Loaded Shape Encoder from dict: {msg}")
        else:
            # 尝试直接加载（假设全是 encoder）
            fixed_checkpoint = fix_state_dict_keys(checkpoint)
            msg = model.ts_model.shape_encoder.load_state_dict(fixed_checkpoint, strict=False)
            logger.info(f"Loaded Shape Encoder (Direct): {msg}")
    else:
        logger.warning(f"!!! Pre-trained checkpoint not found at {pretrained_path}! Using random initialization. !!!")

    # 3. 冻结参数策略
    # 冻结 LLM 和 Encoder
    for name, param in model.named_parameters():
        if "llm" in name:
            param.requires_grad = False
        elif "shape_encoder" in name:
            param.requires_grad = False
        elif "preprocessor" in name: # Time2Vec/PosEncoding 也不要动
            param.requires_grad = False
        else:
            # 训练：Q-Former, Adapter, Projectors
            param.requires_grad = True
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f">>> Stage 2 Trainable Parameters: {len(trainable_params)} tensors")
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 4. 数据加载
    logger.info(f">>> Loading ChatTS Dataset from {args.jsonl_path}...")
    train_ds = ChatTSDataset(args.jsonl_path, args.seq_len, input_channels, split="train")
    val_ds = ChatTSDataset(args.jsonl_path, args.seq_len, input_channels, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=chatts_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=chatts_collate_fn)
    
    # === 修复: 计算正确的 steps_per_epoch ===
    # 如果 len(loader) 不能被 accumulation_steps 整除，最后会多出一个 step
    # 使用 math.ceil 向上取整
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    
    logger.info(f">>> OneCycleLR Steps per epoch: {steps_per_epoch} (Loader len: {len(train_loader)}, Accum: {args.gradient_accumulation_steps})")

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=0.1, # 10% Warmup
        div_factor=25.0
    )
    
    # 5. 训练循环
    best_val_loss = float("inf")
    accumulation_steps = args.gradient_accumulation_steps
    logger.info(f">>> Gradient Accumulation Steps: {accumulation_steps}")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [ChatTS-Align]")
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            # 计算 loss
            loss = compute_chatts_loss(model, batch, device) / accumulation_steps
            loss.backward()
            
            train_loss_sum += loss.item() * accumulation_steps
            
            # 梯度累积更新
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0) # 梯度裁剪
                optimizer.step()
                scheduler.step() # Step-level update
                optimizer.zero_grad()
                
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.2e}")
        
        # 处理剩余梯度 (Last step of epoch)
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step() # 这里的 step 就是上面 calculation 漏掉的那个
            optimizer.zero_grad()
        
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        avg_val_loss = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = model_dir / "chatts_stage2_aligned.pth"
            
            # 只保存非 LLM 的部分
            full_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if "llm.model" not in k}
            
            torch.save(filtered_state_dict, save_path)
            logger.info(f">>> Best ChatTS Alignment Model (Filtered LLM) saved to: {save_path}")

    logger.info(">>> ChatTS Alignment Training Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatTS 格式数据 - Stage 2 对齐训练")
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
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, 
                        help="梯度累积步数，实际批次大小 = batch_size × gradient_accumulation_steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率") 
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None)
    
    args = parser.parse_args()
    train_chatts_alignment(args)

