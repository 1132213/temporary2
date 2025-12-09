import sys
import math
from pathlib import Path

# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.crome_ts.model import CROMEConfig, PatchTSTEncoder, RevIN, get_llm_embed_dim
# 注意：不再导入 Time2Vec
from src.crome_ts.data_instruct import JSONLInstructDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR 
import argparse
from tqdm import tqdm
import logging
from datetime import datetime

# (旧的 PE 类保留也没关系，只要不调用即可)
class FixedPositionalEncoding(nn.Module):
    """标准的正弦位置编码 (不可学习)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, Max_Len, D]
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [Batch, Seq_Len, D]
        return x + self.pe[:, :x.size(1), :]

class PatchTSTForMaskedPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 归一化
        self.revin = RevIN(config.epsilon)
        
        # 2. [修改] 移除外部位置编码，改用 Encoder 内部的 RoPE
        # self.pos_encoding = FixedPositionalEncoding(config.patch_embedding_dim, max_len=5000)
        
        # 3. 编码器
        self.encoder = PatchTSTEncoder(config, config.input_channels)
        
        # 4. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.patch_embedding_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # 5. 重建头
        self.head = nn.Linear(
            config.patch_embedding_dim, 
            config.patch_len * config.input_channels
        )

    def forward(self, x, mask_ratio=0.4, alpha=0.5):
        """
        alpha: MAE Loss 的权重 (0.0 表示只用 MSE, 1.0 表示只用 MAE)
               建议初始尝试 0.5 或 0.3
        """
        B, L, C = x.shape
        
        # --- 1. 归一化 ---
        x_norm, _ = self.revin(x)
        
        # --- 2. Patch Embedding ---
        patches = self.encoder.embedding(x_norm)
        B, N, D = patches.shape
        
        # --- 3. 生成 Mask (保持不变) ---
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1) 
        
        # --- 4. 应用 Mask Token (保持不变) ---
        x_input = patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
        x_input[mask_expanded] = self.mask_token.expand_as(x_input)[mask_expanded]
        
        # --- 5. Forward (RoPE) ---
        x_out = x_input
        for block in self.encoder.blocks:
            x_out = block(x_out, self.encoder.rotary_emb)
        
        latent = self.encoder.norm(x_out)
        
        # --- 6. 计算 Loss (修改部分) ---
        pred_patches = self.head(latent)
        
        target_patches = x_norm.unfold(1, self.config.patch_len, self.config.patch_stride)
        target_patches = target_patches.contiguous().view(B, N, -1)
        
        # 仅在被 Mask 的位置计算 Loss
        mask_float = mask.float().unsqueeze(-1) # [B, N, 1]
        
        # A. MSE Loss (原本的)
        loss_mse = (pred_patches - target_patches) ** 2
        loss_mse = (loss_mse * mask_float).sum() / (mask_float.sum() * pred_patches.shape[-1] + 1e-6)
        
        # B. MAE Loss (新增，用于拉升幅值)
        loss_mae = torch.abs(pred_patches - target_patches)
        loss_mae = (loss_mae * mask_float).sum() / (mask_float.sum() * pred_patches.shape[-1] + 1e-6)

        # C. 组合 Loss
        # 建议 alpha 取 0.3 - 0.5，既保留 MSE 的平滑性，又利用 MAE 提振幅值
        loss = (1 - alpha) * loss_mse + alpha * loss_mae

        # --- D. (进阶可选) 频域一致性 Loss ---
        # 如果你的数据周期性极强，且上面的 Hybrid Loss 效果不够，可以取消下面的注释
        # pred_fft = torch.fft.rfft(pred_patches, dim=-1)
        # target_fft = torch.fft.rfft(target_patches, dim=-1)
        # loss_fft = torch.abs(pred_fft - target_fft).mean()
        # loss += 0.1 * loss_fft 
        
        return loss

    # 为了可视化脚本兼容，添加 forward_viz 接口
    def forward_viz(self, x, mask_ratio=0.4):
        B, L, C = x.shape
        x_norm, _ = self.revin(x)
        patches = self.encoder.embedding(x_norm)
        B, N, D = patches.shape
        
        # Masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, len_keep:], 1)
        
        x_input = patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
        x_input[mask_expanded] = self.mask_token.expand_as(x_input)[mask_expanded]
        
        # Add PE (Removed for RoPE)
        x_final = x_input
        
        # Forward (Updated)
        x_out = x_final
        for block in self.encoder.blocks:
            x_out = block(x_out, self.encoder.rotary_emb)
        latent = self.encoder.norm(x_out)
        
        pred_patches = self.head(latent).view(B, N, self.config.patch_len, C)
        
        # Unpatchify
        recon_series = torch.zeros_like(x_norm)
        count_map = torch.zeros_like(x_norm)
        mask_map_series = torch.zeros(B, L, device=x.device)
        stride = self.config.patch_stride
        patch_len = self.config.patch_len
        
        for n in range(N):
            start = n * stride
            end = start + patch_len
            if end > L: break
            recon_series[:, start:end, :] += pred_patches[:, n, :, :]
            count_map[:, start:end, :] += 1.0
            if mask[0, n]: mask_map_series[:, start:end] = 1.0
            
        recon_series = recon_series / (count_map + 1e-6)
        recon_series_denorm = self.revin(recon_series, mode='denorm')
        return x, recon_series_denorm, mask_map_series
        # return x_norm, recon_series, mask_map_series

def collate_fn_ts_only(batch):
    series = torch.stack([item["series"] for item in batch])
    return series

def train_encoder(args):
    # --- 日志设置 ---
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 创建log和model文件夹
    log_dir = project_root / "log"
    model_dir = project_root / "model"
    log_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"encoder_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_filename}")
    
    device = torch.device(args.device)
    logger.info(f">>> Using Device: {device}")
    
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        epsilon=1e-5
    )
    
    logger.info(">>> Stage 1: Initializing Pre-training (RoPE Enhanced)...")
    model = PatchTSTForMaskedPretraining(config).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f">>> Model Params: {params / 1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    dataset = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_ts_only,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        steps_per_epoch=len(dataloader), 
        epochs=args.epochs,
        pct_start=0.1
    )
    
    logger.info(f">>> Start Training for {args.epochs} epochs...")
    model.train()
    best_loss = float('inf')
    
    # 权重保存路径
    if args.model_suffix:
        model_filename = f"encoder_{args.model_suffix}.pth"
    else:
        model_filename = "encoder.pth"
    best_save_path = model_dir / model_filename
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for series in pbar:
            series = series.to(device)
            
            optimizer.zero_grad()
            loss = model(series, mask_ratio=0.4, alpha=0.5)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存完整模型状态
            torch.save(model.state_dict(), best_save_path)
            logger.info(f">>> [New Best] Saved to {best_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="/root/emhua/btwu/CROME2/data/gifteval_windows1.jsonl")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf", 
                        help="LLM模型路径，用于获取embed_dim")
    parser.add_argument("--epochs", type=int, default=10) 
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--model-suffix", type=str, default="", 
                        help="模型文件名的后缀")
    args = parser.parse_args()
    
    if args.gpu_id: args.device = f"cuda:{args.gpu_id}"
    train_encoder(args)