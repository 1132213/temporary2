import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import argparse

# --- 1. 路径与环境设置 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.crome_ts.model import CROMEConfig, PatchTSTEncoder, RevIN, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn

# --- 2. 模型定义 (同步 FixedPositionalEncoding 版本) ---

class FixedPositionalEncoding(nn.Module):
    """标准的正弦位置编码 (不可学习)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PatchTSTForMaskedPretraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 归一化
        self.revin = RevIN(config.epsilon)
        
        # 2. 固定位置编码
        self.pos_encoding = FixedPositionalEncoding(config.patch_embedding_dim, max_len=5000)
        
        # 3. 编码器
        self.encoder = PatchTSTEncoder(config, config.input_channels)
        
        # 4. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.patch_embedding_dim))
        
        # 5. 重建头
        self.head = nn.Linear(config.patch_embedding_dim, config.patch_len * config.input_channels)

    def forward_viz(self, x, mask_ratio=0.4):
        """
        可视化专用前向传播
        """
        B, L, C = x.shape
        
        # 1. 归一化 (Ground Truth)
        x_norm, stats = self.revin(x)
        
        # 2. Patch Embedding
        patches = self.encoder.embedding(x_norm) # [B, N, D]
        B, N, D = patches.shape
        
        # 3. 生成 Mask
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        # 如果 mask_ratio 为 0，这里 len_keep == N，不会进行 scatter，mask 全为 False
        if len_keep < N:
            mask.scatter_(1, ids_shuffle[:, len_keep:], 1) # 1=Masked
        
        # 4. 替换内容 (Masking)
        x_input = patches.clone()
        if mask_ratio > 0:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
            x_input[mask_expanded] = self.mask_token.expand_as(x_input)[mask_expanded]
        
        # 5. 注入位置信息 (Add PE)
        x_final = self.pos_encoding(x_input)
        
        # 6. Encoder Forward
        latent = self.encoder.encoder(x_final)
        
        # 7. 重建 Patch
        pred_patches = self.head(latent).view(B, N, self.config.patch_len, C)
        
        # 8. Unpatchify (拼接回序列)
        recon_series = torch.zeros_like(x_norm)
        count_map = torch.zeros_like(x_norm)
        
        # --- 核心修改：准确计算盲区 ---
        # visible_count 记录每个时间点被多少个“未Mask”的 Patch 覆盖
        visible_count = torch.zeros(B, L, device=x.device)
        
        stride = self.config.patch_stride
        patch_len = self.config.patch_len
        
        for n in range(N):
            start = n * stride
            end = start + patch_len
            if end > L: break
            
            # 累加预测值
            recon_series[:, start:end, :] += pred_patches[:, n, :, :]
            count_map[:, start:end, :] += 1.0
            
            # 如果当前 Patch 是可见的 (mask=0)，则该区域可见度 +1
            if not mask[0, n]:
                visible_count[:, start:end] += 1.0
            
        recon_series = recon_series / (count_map + 1e-6)
        
        # 只有当 visible_count == 0 时，才是真正的“盲区” (完全没有上下文)
        real_blind_spot = (visible_count == 0).float()
        
        return x_norm, recon_series, real_blind_spot

# --- 3. 可视化工具函数 ---
def plot_sample(x_norm, recon, mask_map, save_path, title):
    plt.figure(figsize=(12, 6))
    
    time_steps = np.arange(len(x_norm))
    
    # 绘制灰色背景表示“真正的盲区” (仅当 mask_map 中存在 > 0 的值时)
    if np.any(mask_map > 0):
        plt.fill_between(time_steps, x_norm.min(), x_norm.max(), 
                         where=(mask_map > 0), 
                         color='gray', alpha=0.2, label='True Blind Spot')
    
    # 绘制真实值
    plt.plot(x_norm, label='Ground Truth', color='blue', linewidth=2, alpha=0.8)
    
    # 绘制重建值
    plt.plot(recon, label='Reconstruction', color='orange', linestyle='--', linewidth=1.5)
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# --- 4. 主流程 ---
def main(args):
    device = torch.device(args.device)
    
    # 1. 配置
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    print(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        epsilon=1e-5
    )
    
    # 2. 加载模型
    print(f">>> Loading model from {args.checkpoint}...")
    model = PatchTSTForMaskedPretraining(config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "state_dict" in checkpoint:
        try:
            model.load_state_dict(checkpoint["state_dict"])
            print(">>> Successfully loaded FULL model (Encoder + Head).")
        except Exception as e:
            print(f"!!! Error loading state_dict: {e}")
            model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif "encoder" in checkpoint:
        print("!!! Warning: Checkpoint contains 'encoder' only. Head might be random.")
        model.encoder.load_state_dict(checkpoint["encoder"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 3. 加载数据
    print(f">>> Loading Validation Set from {args.jsonl_path}...")
    val_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="val")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=instruct_collate_fn)
    
    # 4. 推理与可视化
    print(f">>> Visualizing {args.num_samples} samples (Masked & Unmasked)...")
    output_dir = Path("vis_results")
    output_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i, (series, _, _) in enumerate(val_loader):
            if i >= args.num_samples:
                break
            
            series = series.to(device)
            
            # --- 场景 1: 带 Mask (Mask Ratio = 0.4) ---
            gt, pred, mask_map = model.forward_viz(series, mask_ratio=0.4)
            
            gt_np = gt[0, :, 0].cpu().numpy()
            pred_np = pred[0, :, 0].cpu().numpy()
            mask_np = mask_map[0, :].cpu().numpy()
            
            plot_sample(gt_np, pred_np, mask_np, 
                        output_dir / f"masked_sample_{i}.png", 
                        title=f"Sample {i}: Masked Reconstruction (Ratio=0.4)")
            
            # --- 场景 2: 无 Mask (Pure Reconstruction, Ratio = 0.0) ---
            # 直接使用相同的 series 进行纯重建
            _, pred_clean, mask_clean = model.forward_viz(series, mask_ratio=0.0)
            
            pred_clean_np = pred_clean[0, :, 0].cpu().numpy()
            mask_clean_np = mask_clean[0, :].cpu().numpy() # 应该是全0
            
            plot_sample(gt_np, pred_clean_np, mask_clean_np, 
                        output_dir / f"recon_sample_{i}.png",
                        title=f"Sample {i}: Pure Reconstruction (No Mask)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="/root/emhua/btwu/CROME2/data/gifteval_windows1.jsonl")
    parser.add_argument("--checkpoint", type=str, default="patchtst_pretrained_full.pth")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=str, default=None)
    
    args = parser.parse_args()
    if args.gpu_id: args.device = f"cuda:{args.gpu_id}"
    
    main(args)