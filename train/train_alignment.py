import sys
from pathlib import Path

# 路径设置：确保能找到 src 和 test 包
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 引入项目模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn
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

def compute_loss(model, series, prefixes, suffixes, device):
    """
    计算 Causal LM Loss (Next Token Prediction)
    """
    # 获取模型输出
    model_out = model(series, prefixes, suffixes)
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits
    
    # Suffix Start = Prefix长度 + Stat(1) + TS(N)
    prefix_width = model_out["prefix_mask"].shape[1]
    ts_width = 1 + model_out["ts_tokens"].shape[1] # 1 for stat_token
    start_idx = prefix_width + ts_width
    
    # 截取 Suffix 对应的 Logits
    suffix_logits = logits[:, start_idx:, :]
    
    # 获取 Suffix 的真实 Token IDs
    suffix_labels = model.tokenizer.tokenizer(
        suffixes, return_tensors="pt", padding=True
    ).input_ids.to(device)
    
    # 对齐长度
    min_len = min(suffix_logits.shape[1], suffix_labels.shape[1])
    suffix_logits = suffix_logits[:, :min_len, :]
    suffix_labels = suffix_labels[:, :min_len]
    
    # Causal LM Loss: Shift Logits & Labels
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

def train_alignment(args):
    # --- 日志设置 ---
    script_dir = Path(__file__).parent
    log_filename = script_dir / f"alignment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    
    set_seed(args.seed)
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None and ":" not in args.device:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)

    # 1. 初始化配置
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
        llm_dtype="bfloat16",
        freeze_patch_encoder=True,  # 初始冻结，后续通过 param.requires_grad 细粒度控制
        use_stats_projector=True,
        epsilon=1e-5
    )
    
    logger.info(">>> Initializing Model for Alignment...")
    model = StatBypassCROMETS1(config).to(device)

    # 2. 加载预训练的权重 (Stage 1)
    # 这里的路径应该是 "patchtst_pretrained_full.pth"
    if Path(args.pretrained_encoder_path).exists():
        logger.info(f">>> Loading Pre-trained Weights from {args.pretrained_encoder_path}...")
        checkpoint = torch.load(args.pretrained_encoder_path, map_location=device)
        
        # 适配新的保存格式 (state_dict)
        if "state_dict" in checkpoint:
            # 过滤出 encoder 相关的权重
            encoder_dict = {k.replace("encoder.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
            if encoder_dict:
                msg = model.ts_model.shape_encoder.load_state_dict(encoder_dict, strict=False)
                logger.info(f"Loaded Shape Encoder from state_dict: {msg}")
            else:
                logger.warning("!!! No encoder weights found in state_dict prefix 'encoder.' !!!")
        # 兼容旧格式
        elif "encoder" in checkpoint:
            msg = model.ts_model.shape_encoder.load_state_dict(checkpoint["encoder"], strict=True)
            logger.info(f"Loaded Shape Encoder from dict: {msg}")
        else:
            # 尝试直接加载（假设全是 encoder）
            msg = model.ts_model.shape_encoder.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded Shape Encoder (Direct): {msg}")
    else:
        logger.warning("!!! Pre-trained checkpoint not found! Using random initialization. !!!")

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
    train_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="train")
    val_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=instruct_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=instruct_collate_fn)
    
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
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Align]")
        
        optimizer.zero_grad()
        
        for step, (series, prefixes, suffixes) in enumerate(progress):
            series = series.to(device)
            
            # 计算 loss
            loss = compute_loss(model, series, prefixes, suffixes, device) / accumulation_steps
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
            save_path = script_dir / "crome_stage2_aligned.pth"
            
            # 只保存非 LLM 的部分
            full_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if "llm.model" not in k}
            
            torch.save(filtered_state_dict, save_path)
            logger.info(f">>> Best Alignment Model (Filtered LLM) saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    parser.add_argument("--pretrained-encoder-path", type=str, default="patchtst_pretrained_full.pth")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, 
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-3) 
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None)
    
    args = parser.parse_args()
    train_alignment(args)