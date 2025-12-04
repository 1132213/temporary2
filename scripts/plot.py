import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# --- 1. 环境设置：确保能导入项目模块 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, PatchTSTEncoder, RevIN, get_llm_embed_dim

# --- 2. 定义 Stage 1 模型结构 (必须与训练时一致) ---
# 这些类直接复制自 test_encoder.py，因为 Stage 1 的 Head 不在主 model.py 中

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
        
        # 3. 编码器 (从 src 导入)
        self.encoder = PatchTSTEncoder(config, config.input_channels)
        
        # 4. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.patch_embedding_dim))
        
        # 5. 重建头 (Stage 1 特有)
        self.head = nn.Linear(config.patch_embedding_dim, config.patch_len * config.input_channels)

    def forward_viz(self, x, mask_ratio=0.0):
        """
        可视化专用前向传播
        返回: (归一化后的真值, 重建值, Mask矩阵)
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
        if len_keep < N:
            mask.scatter_(1, ids_shuffle[:, len_keep:], 1) # 1=Masked
        
        # 4. 替换内容 (Masking)
        x_input = patches.clone()
        if mask_ratio > 0:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D)
            x_input[mask_expanded] = self.mask_token.expand_as(x_input)[mask_expanded]
        
        # 5. 注入位置信息
        x_final = self.pos_encoding(x_input)
        
        # 6. Encoder Forward
        latent = self.encoder.encoder(x_final)
        
        # 7. 重建 Patch
        pred_patches = self.head(latent).view(B, N, self.config.patch_len, C)
        
        # 8. Unpatchify (拼接回序列)
        recon_series = torch.zeros_like(x_norm)
        count_map = torch.zeros_like(x_norm)
        
        stride = self.config.patch_stride
        patch_len = self.config.patch_len
        
        for n in range(N):
            start = n * stride
            end = start + patch_len
            if end > L: break
            
            recon_series[:, start:end, :] += pred_patches[:, n, :, :]
            count_map[:, start:end, :] += 1.0
            
        recon_series = recon_series / (count_map + 1e-6)
        
        # 为了可视化 Mask 区域，将其扩展回原序列长度
        mask_map_series = torch.zeros(B, L, device=x.device)
        for n in range(N):
            if mask[0, n]:
                start = n * stride
                end = start + patch_len
                if end <= L:
                    mask_map_series[:, start:end] = 1.0
        
        return x_norm, recon_series, mask_map_series

# --- 3. 绘图函数 ---
def process_and_plot(jsonl_path, sample_idx, checkpoint, output_file, config, device, mask_ratio=0.0):
    # A. 加载模型
    print(f">>> Loading Stage 1 Encoder from {checkpoint}...")
    model = PatchTSTForMaskedPretraining(config).to(device)
    
    if Path(checkpoint).exists():
        state_dict = torch.load(checkpoint, map_location=device)
        # 兼容不同的保存格式
        if "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"], strict=False)
        elif "encoder" in state_dict:
            print("Warning: Checkpoint only has encoder. Head will be random initialized!")
            model.encoder.load_state_dict(state_dict["encoder"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: Checkpoint {checkpoint} not found.")
        return

    model.eval()

    # B. 读取数据
    target_record = None
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                try:
                    target_record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error: Decode failed at line {i}")
                    return
                break
    
    if target_record is None:
        print(f"Error: Sample {sample_idx} not found.")
        return

    # 获取时序列表
    timeseries_list = target_record.get("timeseries") or target_record.get("input_ts")
    if timeseries_list is None:
        print("Error: No timeseries data found.")
        return
    # 如果是单条数据，转为列表
    if not isinstance(timeseries_list[0], list):
        timeseries_list = [timeseries_list]

    num_series = len(timeseries_list)
    print(f">>> Found {num_series} time series in sample {sample_idx}. Mask Ratio: {mask_ratio}")

    # C. 绘图准备
    fig_height = max(4, 3 * num_series)
    fig, axes = plt.subplots(num_series, 1, figsize=(12, fig_height), sharex=False)
    if num_series == 1: axes = [axes]

    # D. 逐条处理与绘制
    for i, ts_data in enumerate(timeseries_list):
        ax = axes[i]
        
        # 1. 预处理数据
        arr = np.array(ts_data).flatten()
        # 截断或填充以匹配 Patch
        # 为了简单，直接转 Tensor，让 forward_viz 处理
        # 注意：模型需要 Input [B, L, C]
        ts_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        # 2. 运行模型
        with torch.no_grad():
            # x_norm: 归一化后的真值
            # recon: 重建值
            # mask_map: 哪里被 Mask 了 (1=Masked)
            gt, pred, mask_map = model.forward_viz(ts_tensor, mask_ratio=mask_ratio)
        
        # 转 Numpy
        gt_np = gt[0, :, 0].cpu().numpy()
        pred_np = pred[0, :, 0].cpu().numpy()
        mask_np = mask_map[0, :].cpu().numpy()
        
        # 3. 绘制
        time_steps = np.arange(len(gt_np))
        
        # 画 Mask 区域 (灰色背景)
        if np.any(mask_np > 0):
            ax.fill_between(time_steps, gt_np.min(), gt_np.max(), 
                            where=(mask_np > 0), 
                            color='gray', alpha=0.2, label='Masked Region')
        
        # 画曲线
        ax.plot(gt_np, label='Ground Truth (Norm)', color='#1f77b4', linewidth=2, alpha=0.7)
        ax.plot(pred_np, label='Reconstruction', color='#ff7f0e', linestyle='--', linewidth=1.5)
        
        # 统计指标 (MSE)
        mse = np.mean((gt_np - pred_np)**2)
        ax.set_title(f"Series {i+1} | MSE: {mse:.4f}", fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 4. 标题与保存
    input_text = target_record.get("input", "")[:100].replace("\n", " ")
    plt.suptitle(f"Reconstruction (Ratio={mask_ratio}) - Sample {sample_idx}\n{input_text}", fontsize=12)
    plt.tight_layout()
    
    # 确保目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">>> Saved plot to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize reconstruction from Stage 1 Encoder.")
    parser.add_argument("--jsonl-path", type=str, default="/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.json")
    parser.add_argument("--checkpoint", type=str, default="/mnt/shared-storage-user/huaermo/code/test_wbt/temporary2/model/encoder.pth", 
                        help="Path to Stage 1 checkpoint (Must contain encoder weights)")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-file", type=str, default="recon_plot.png")
    parser.add_argument("--mask-ratio", type=float, default=0.0, 
                        help="Mask ratio for visualization (0.0 = Full Reconstruction, 0.4 = Masked Autoencoding)")
    
    # 模型参数 (必须与训练时一致)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/mnt/shared-storage-user/dllm-share/Models/Qwen2.5-3B-Instruct")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构造 Config
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    config = CROMEConfig(
        input_channels=1,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        epsilon=1e-3 # 保持与最新 SFT 设置一致，或者用 1e-5 取决于 Stage 1 训练时的设置
    )
    
    process_and_plot(
        args.jsonl_path, 
        args.sample_idx, 
        args.checkpoint, 
        args.output_file, 
        config, 
        device,
        mask_ratio=args.mask_ratio
    )

if __name__ == "__main__":
    main()