from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class InstructionRefiner(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.embed_dim = config.llm_embed_dim
        self.num_queries = config.num_task_queries
        
        # 1. 可学习的任务查询向量 (Task Queries)
        # 形状: [1, Num_Queries, LLM_Dim]
        self.task_queries = nn.Parameter(
            torch.randn(1, self.num_queries, self.embed_dim)
        )
        nn.init.normal_(self.task_queries, std=0.02)
        
        # 2. Cross-Attention 层
        # Query = Task Queries
        # Key/Value = Raw Instruction Embeddings
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4, # 4头足以处理语义聚类
            batch_first=True,
            dropout=config.proj_dropout
        )
        
        # 3. LayerNorm (稳定训练)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, instruction_embeds: Tensor) -> Tensor:
        if instruction_embeds is None:
            return None

        B = instruction_embeds.shape[0]
        ref = instruction_embeds  # ⭐ dtype / device 的唯一权威来源

        # 1. Queries 对齐
        queries = self.task_queries.expand(B, -1, -1).to(
            dtype=ref.dtype,
            device=ref.device
        )

        # 2. Cross-Attention 参数对齐
        if self.cross_attn.in_proj_weight.dtype != ref.dtype:
            self.cross_attn = self.cross_attn.to(
                dtype=ref.dtype,
                device=ref.device
            )

        # 3. LayerNorm 参数对齐（🔥 这次报错的根因）
        if self.norm.weight.dtype != ref.dtype:
            self.norm = self.norm.to(
                dtype=ref.dtype,
                device=ref.device
            )

        # 4. Cross Attention
        task_embeds, _ = self.cross_attn(
            query=queries,
            key=ref,
            value=ref
        )

        # 5. Norm
        return self.norm(task_embeds)




class CNNDetailEncoder(nn.Module):
    def __init__(self, input_channels, patch_embedding_dim, dropout=0.1):
        super().__init__()
        
        # 隐藏层维度
        hidden_dim = 64 
        
        # 第一层投影
        self.first_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        
        # 定义 3 个标准的残差块
        # 坚持 Kernel=3, Dilation=1 (不膨胀), Padding=1 (保持长度)
        # 这种结构只关注 "t" 时刻及其左右邻居，非常符合 "Micro" 的定义
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(3) # 堆叠 3 层，足以提取复杂的微观特征
        ])
        
        # 最终投影
        self.final_proj = nn.Conv1d(hidden_dim, patch_embedding_dim, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Seq_Len]
        x = self.first_conv(x)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual # ResNet 风格连接，训练更稳定
            
        x = self.final_proj(x)
        return x.transpose(1, 2) # [B, L, D]
class FusionGate(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        # [修改] 输出维度从 1 改为 patch_embedding_dim (例如 512)
        self.proj = nn.Linear(config.llm_embed_dim, config.patch_embedding_dim)
        
    def forward(self, instruction_embeds):
        if instruction_embeds is None:
            return 0.5 
        
        pooled = instruction_embeds.mean(dim=1)
        pooled = pooled.to(self.proj.weight.dtype)
        
        # [修改] 输出形状为 [B, 1, D]
        # 这样 Gate 可以在不同的特征通道上分别决定“听宏观的”还是“听微观的”
        gate = torch.sigmoid(self.proj(pooled)).unsqueeze(1) 
        return gate
# ---2. 特征融合门控 ---
# class FusionGate(nn.Module):
#     """
#     根据指令语义，动态决定关注宏观趋势还是微观细节。
#     """
#     def __init__(self, config: CROMEConfig):
#         super().__init__()
#         # 输入是 LLM 的 Instruction Embedding
#         self.proj = nn.Linear(config.llm_embed_dim, 1)
        
#     def forward(self, instruction_embeds):
#         # instruction_embeds: [B, L_text, D_llm]
#         if instruction_embeds is None:
#             # 如果没有指令，默认 50/50 混合
#             return 0.5 
        
#         # 对文本序列取均值作为句子表示
#         # [B, L, D] -> [B, D]
#         pooled = instruction_embeds.mean(dim=1)
#         pooled = pooled.to(self.proj.weight.dtype)
        
#         # Sigmoid 映射到 [0, 1]
#         # output: [B, 1, 1] 用于广播
#         gate = torch.sigmoid(self.proj(pooled)).unsqueeze(1)
#         return gate


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        # 预计算频率
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        freqs = torch.outer(t, inv_freq)
        # Different from some implementations, we concat in last dim to match (d/2) pair
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        # x: [Batch, Heads, Seq_Len, Head_Dim]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len + 128, device=x.device)
            
        return (
            self.cos_cached[..., :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[..., :seq_len, :].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [Batch, Heads, Seq_Len, Head_Dim]
    # cos, sin: [1, 1, Seq_Len, Head_Dim]
    # 确保 cos/sin 与 q/k 广播兼容
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# === 支持 RoPE 的自定义 Transformer Block ===

class RoPESelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rotary_emb):
        B, L, D = x.shape
        # [B, L, 3*D] -> [B, L, 3, Heads, Head_Dim] -> [3, B, Heads, L, Head_Dim]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = rotary_emb(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention: [B, Heads, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Output: [B, L, D]
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class RoPETransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPESelfAttention(dim, num_heads, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, rotary_emb):
        # Pre-Norm 结构，更稳定
        x = x + self.attn(self.norm1(x), rotary_emb)
        x = x + self.mlp(self.norm2(x))
        return x
        
@dataclass
class CROMEConfig:
    """
    全局配置。
    """
    input_channels: int
    llm_embed_dim: int
    patch_len: int = 16
    patch_stride: int = 8
    patch_embedding_dim: int = 512
    patch_num_heads: int = 8
    patch_num_layers: int = 4
    freeze_patch_encoder: bool = False
    query_tokens: int = 64
    num_task_queries: int = 8
    adapter_hidden_dim: int = 256
    fuse_mode: str = "add"
    epsilon: float = 1e-4
    proj_dropout: float = 0.0
    # LLM 接口
    llm_model_path: str = "/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"
    llm_dtype: str = "bfloat16"
    llm_device_map: str = "auto"


def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if not hasattr(torch, name):
        raise ValueError(f"无法解析 dtype: {name}")
    return getattr(torch, name)


def get_llm_embed_dim(llm_model_path: str) -> int:
    try:
        config = AutoConfig.from_pretrained(llm_model_path)
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        elif hasattr(config, 'd_model'):
            return config.d_model
        elif hasattr(config, 'n_embd'):
            return config.n_embd
        else:
            raise ValueError(f"无法从模型配置中获取embed_dim。模型路径: {llm_model_path}")
    except Exception as e:
        raise RuntimeError(f"加载模型配置失败: {e}。模型路径: {llm_model_path}")

class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dtype != torch.float32:
            pass
        orig_dtype = x.dtype
        x_fp32 = x.float()
        
        mu = x_fp32.mean(dim=1, keepdim=True)
        sigma = x_fp32.std(dim=1, keepdim=True).clamp_min(self.eps)
        
        x_norm = (x_fp32 - mu) / sigma
        x_norm = x_norm.to(dtype=orig_dtype)
        stats = torch.stack((mu.squeeze(1), sigma.squeeze(1)), dim=-1).to(dtype=orig_dtype)
        return x_norm, stats


class FixedSinePositionalEncoding(nn.Module):
    """标准固定正弦位置编码。"""
    def __init__(self, dim: int, scale: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
            * (-math.log(self.scale) / self.dim)
        )
        pe = torch.zeros(length, self.dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        cos_slots = self.dim // 2
        if cos_slots > 0:
            pe[:, 1::2] = torch.cos(position * div_term[:cos_slots])
        return pe.to(dtype=dtype)


class InputPreprocessor(nn.Module):
    """模块 I：输入预处理 + 去量纲 + 时间编码。"""
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(config.epsilon)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, l, c = x.shape
        x_norm, stats = self.revin(x)
        return x_norm,stats


class PatchEmbedding(nn.Module):
    def __init__(self, config: CROMEConfig, in_dim: int):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride
        self.project = nn.Linear(self.patch_len * in_dim, config.patch_embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, _, c = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        n = patches.shape[1]
        patches = patches.contiguous().view(b, n, self.patch_len * c)
        return self.project(patches)


class PatchTSTEncoder(nn.Module):
    """模块 II：基于 RoPE 的 PatchTST 编码器。"""
    def __init__(self, config: CROMEConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # 1. Patch Embedding (保持不变)
        self.embedding = PatchEmbedding(config, input_dim)
        
        # 2. RoPE 生成器 (替代原有的 FixedSinePositionalEncoding)
        # head_dim = embedding_dim / num_heads
        head_dim = config.patch_embedding_dim // config.patch_num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim)

        # 3. Transformer Blocks (替代原有的 nn.TransformerEncoder)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                dim=config.patch_embedding_dim,
                num_heads=config.patch_num_heads,
                mlp_ratio=4.0,
                dropout=0.1 # 如果 config 有 dropout 参数可替换
            )
            for _ in range(config.patch_num_layers)
        ])
        
        # 4. Final Norm (Pre-Norm 结构通常需要在最后加一层 Norm)
        self.norm = nn.LayerNorm(config.patch_embedding_dim)

        if config.freeze_patch_encoder:
            self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [Batch, Seq_Len, Channels]
        
        # 1. Patchify & Project -> [Batch, Num_Patches, Patch_Dim]
        emb = self.embedding(x) 
        
        # 2. Forward through Blocks with RoPE
        # 注意：不再需要 self.pos_encoding(emb) 的加法操作
        
        x_out = emb
        for block in self.blocks:
            x_out = block(x_out, self.rotary_emb)
        
        x_out = self.norm(x_out)

        if self.config.freeze_patch_encoder:
            # 返回原始 emb 和编码后的特征
            return emb, x_out.detach()
        else:
            return emb, x_out

class QFormerLayer(nn.Module):
    """
    Q-Former 的单层 Block。
    执行顺序：Self-Attn (Query) -> Cross-Attn (Text) -> Cross-Attn (Time Series) -> FFN
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        dim = config.patch_embedding_dim
        num_heads = config.patch_num_heads
        dropout = getattr(config, "dropout", 0.1)
        
        # 1. Self-Attention: Query Tokens 内部交互
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # 2. Text Cross-Attention: 文本引导 (Instruction)
        self.text_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        # 文本投影层：将 LLM 维度的文本映射到 Q-Former 维度
        self.text_proj = nn.Linear(config.llm_embed_dim, dim) 
        
        # 3. Time-Series Cross-Attention: 时序特征提取
        self.ts_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        
        # 4. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm4 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 保存最后一次的 Attention 权重用于可视化
        self.last_text_attn_weights = None
        self.last_ts_attn_weights = None

    def forward(self, queries, text_embeds, ts_embeds):
        """
        queries: [Batch, Num_Queries, Dim]
        text_embeds: [Batch, Text_Len, LLM_Dim]
        ts_embeds: [Batch, Num_Patches, Dim]
        """
        # 1. Self-Attention (Q=K=V=Queries)
        # Query 之间相互交流，整合上一层提取到的信息
        q_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout(q_out))
        
        # 2. Text Interaction (文本引导)
        if text_embeds is not None:
            text_embeds_input = text_embeds.to(self.text_proj.weight.dtype)
            
            # 投影文本特征到当前维度
            text_kv = self.text_proj(text_embeds_input).to(queries.dtype)
            
            # Cross-Attention: Query 关注 Text
            text_out, text_attn = self.text_attn(queries, text_kv, text_kv)
            queries = self.norm2(queries + self.dropout(text_out))
            
            self.last_text_attn_weights = text_attn.detach().cpu()
        
        # 3. TS Extraction (时序提取)
        ts_out, ts_attn = self.ts_attn(queries, ts_embeds, ts_embeds)
        queries = self.norm3(queries + self.dropout(ts_out))
        
        self.last_ts_attn_weights = ts_attn.detach().cpu()
        
        # 4. FFN
        queries = self.norm4(queries + self.dropout(self.ffn(queries)))
        
        return queries


class QFormer(nn.Module):
    """
    多层迭代式 Q-Former (Iterative Q-Former)。
    通过多层交互，实现深度的文本引导和特征精炼。
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        
        # 1. 可学习的 Query Tokens (Base Queries)
        # 这些 Token 是特征提取的“种子”
        self.query_tokens = nn.Parameter(
            torch.randn(config.query_tokens, config.patch_embedding_dim)
        )
        # 使用正态分布初始化，这通常比全0初始化更容易训练
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # 2. 堆叠多层 QFormerLayer
        # 建议层数：4 到 6 层。如果 config 中没有定义，默认使用 4 层。
        # 你可以在 CROMEConfig 中添加 qformer_layers 属性来控制
        num_layers = getattr(config, "qformer_layers", 4) 
        
        self.layers = nn.ModuleList([
            QFormerLayer(config) for _ in range(num_layers)
        ])
        
        # 暴露最后一层的 attention map 供外部调用 (例如 plot.py)
        self.last_text_attn_weights = None

    def forward(self, patch_tokens: Tensor, instruction_embeds: Optional[Tensor] = None) -> Tensor:
        """
        patch_tokens: [Batch, Num_Patches, Dim] (来自 Encoder)
        instruction_embeds: [Batch, Text_Len, LLM_Dim] (来自 LLM)
        """
        b = patch_tokens.size(0)
        
        # 1. 扩展 Query Tokens: [1, N_q, D] -> [B, N_q, D]
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        
        # 2. 迭代式更新 Query
        # 每一层 Query 都会变得更加“聪明”，因为它反复看了文本和数据
        for layer in self.layers:
            queries = layer(queries, instruction_embeds, patch_tokens)
            
        # 3. 记录最后一层的 Attention Map (为了兼容旧的可视化代码)
        self.last_text_attn_weights = self.layers[-1].last_text_attn_weights
            
        return queries

class DetailProjection(nn.Module):
    def __init__(self, config: CROMEConfig, input_dim: int = None):
        super().__init__()
        
        dim = input_dim if input_dim is not None else config.patch_embedding_dim
        target_dim = config.patch_embedding_dim 
        
        hidden_dim = target_dim * 4  
        drop_rate = config.proj_dropout

        self.proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(hidden_dim, target_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.proj(patch_tokens)

class RobustFiLMGenerator(nn.Module):
    """
    Robust Log-Space FiLM Generator.
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        # input_dim = 3 (log_mu, log_sigma, sign_mu)
        input_dim = 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.adapter_hidden_dim),
            nn.LayerNorm(config.adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_hidden_dim, config.adapter_hidden_dim * 2) # gamma, beta
        )
        
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stats: Tensor) -> Tuple[Tensor, Tensor]:
        if stats.dim() == 2:
            mu = stats[..., 0:1]
            sigma = stats[..., 1:2]
        else:
            mu = stats[..., 0]
            sigma = stats[..., 1]
            
        log_mu = torch.log1p(mu.abs())
        log_sigma = torch.log1p(sigma)
        sign_mu = torch.sign(mu)
        
        features = torch.cat([log_mu, log_sigma, sign_mu], dim=-1)
        out = self.mlp(features)
        
        if out.dim() == 2:
            out = out.unsqueeze(1) 
            
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class CROMEAdapterBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None) -> Tensor:
        z = F.silu(self.down(x)) * self.gate(x)
        if gamma is not None and beta is not None:
            z = z * (1 + gamma) + beta
        return x + self.up(z)


# --- [修改] 3. 适配器 (丢弃 Detail Tokens 输入) ---
class CROMEAdapter(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        # 只需要一个 Query Adapter
        # 移除了原有的 patch_adapter
        self.query_adapter = CROMEAdapterBlock(
            config.patch_embedding_dim, config.adapter_hidden_dim
        )

    def forward(
        self, 
        query_tokens: Tensor, 
        # patch_tokens: Tensor,  <-- 移除此参数
        # sep_embed: Optional[Tensor] = None, <-- 移除此参数
        gamma: Optional[Tensor] = None, 
        beta: Optional[Tensor] = None
    ) -> Tensor:
        
        # [关键修改] 将 FiLM 统计量 (Gamma/Beta) 注入到 Query Tokens 中
        # 这确保了即使丢弃了 Detail Tokens，LLM 依然能感知到序列的幅度信息
        return self.query_adapter(query_tokens, gamma=gamma, beta=beta)


class InstructionTokenizer:
    def __init__(self, config: CROMEConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_path, use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __call__(self, texts: Sequence[str], device: torch.device) -> Dict[str, Tensor]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in encoded.items()}


class FrozenLLM(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        dtype = _resolve_dtype(config.llm_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_path,
            torch_dtype=dtype,
            device_map=config.llm_device_map,
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def embed_layer(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def embed(self, input_ids: Tensor) -> Tensor:
        embed_layer = self.embed_layer
        target_device = input_ids.device
        weight_device = embed_layer.weight.device
        if target_device != weight_device:
            input_ids = input_ids.to(weight_device)
        embeds = embed_layer(input_ids)
        if embeds.device != target_device:
            embeds = embeds.to(target_device)
        return embeds

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, **kwargs):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
    
class SeriesDecomp(nn.Module):
    """
    序列分解模块：将序列分解为 趋势项(Trend) 和 残差项(Residual)。
    X_residual = X_input - MovingAvg(X_input)
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # 使用平均池化实现移动平均
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        
        # Padding 以保持序列长度不变
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        # 计算 Trend
        # [B, L, C] -> [B, C, L] -> AvgPool -> [B, C, L] -> [B, L, C]
        x_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 计算 Residual (这就包含了 Noise 和 Anomaly)
        x_resid = x - x_trend
        
        return x_resid, x_trend

# ==========================================
# 1. 修改 CNNDetailEncoder: 纯微观 ResNet-1D
# ==========================================
class CNNDetailEncoder(nn.Module):
    """
    纯微观细节编码器 (Pure Micro Encoder)。
    特点：
    1. 无池化 (No Pooling)：绝对不丢失高频细节。
    2. 无膨胀 (No Dilation)：只关注局部 (Kernel=3)，不越界去管宏观。
    3. ResNet结构：深层特征提取，训练稳定。
    """
    def __init__(self, input_channels: int, patch_embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 隐藏层维度
        hidden_dim = 64 
        
        # 第一层投影: 将输入 (Resid+Diff) 映射到隐藏空间
        self.first_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        
        # 定义 3 个标准的残差块
        # 坚持 Kernel=3, Dilation=1 (不膨胀), Padding=1 (保持长度)
        # 这种结构只关注 "t" 时刻及其左右邻居，非常符合 "Micro" 的定义
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(3) # 堆叠 3 层，足以提取复杂的微观特征
        ])
        
        # 最终投影到 PatchTST 的维度，以便 Q-Former 处理
        self.final_proj = nn.Conv1d(hidden_dim, patch_embedding_dim, kernel_size=1)

    def forward(self, x):
        # x input shape: [Batch, Channels, Seq_Len]
        # Channels = 2 (Resid + Diff)
        
        x = self.first_conv(x)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual # ResNet 风格连接
            
        x = self.final_proj(x)
        
        # 输出形状: [Batch, Dim, Seq_Len]
        # 转置为 [Batch, Seq_Len, Dim] 给 Q-Former
        return x.transpose(1, 2)


# ==========================================
# 2. 修改 CROMETSModel: 初始化与双流逻辑
# ==========================================
class CROMETSModel(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        
        # 1. 预处理
        self.preprocessor = InputPreprocessor(config)
        
        # 2. Macro 流编码器
        pre_input_dim = (
            config.input_channels if config.fuse_mode == "add" else config.input_channels * 2
        )
        self.shape_encoder = PatchTSTEncoder(config, pre_input_dim)
        
        # 3. Micro 流编码器 (纯微观 ResNet)
        self.decomp = SeriesDecomp(kernel_size=65) # 保持 Kernel=65
        self.detail_encoder = CNNDetailEncoder(
            input_channels=2, # Resid + Diff
            patch_embedding_dim=config.patch_embedding_dim,
            dropout=config.proj_dropout
        )
        
        # ==========================================
        # [新增] 4. 指令蒸馏器
        # ==========================================
        self.instr_refiner = InstructionRefiner(config)
        
        # 5. Q-Former, FiLM, Adapter
        self.qformer = QFormer(config)
        self.film_generator = RobustFiLMGenerator(config)
        self.adapter = CROMEAdapter(config)
        
        # 6. 融合门控
        self.fusion_gate = FusionGate(config)
        
        # 7. LLM 投影
        self.llm_proj = nn.Sequential(
            nn.Linear(config.patch_embedding_dim, config.llm_embed_dim),
            nn.GELU(),
            nn.Dropout(config.proj_dropout),
            nn.Linear(config.llm_embed_dim, config.llm_embed_dim)
        )

    def _process_single_channel(
        self,
        channel_data: Tensor,
        instruction_embeds: Optional[Tensor] = None,
        sep_embed: Optional[Tensor] = None,
    ) -> Tensor:
        # =========================================================
        # [步骤 0] 指令蒸馏 (Instruction Refinement)
        # =========================================================
        # 将原始的含指代指令 (Raw) 转化为纯任务指令 (Refined)
        # task_embeds: [B, Num_Task_Queries, LLM_Dim]
        # 这里的 task_embeds 将替代 raw instruction_embeds 传给下游
        task_embeds = self.instr_refiner(instruction_embeds)
        
        # 如果蒸馏器返回 None (即没有输入指令)，则沿用 None，
        # 下游的 Gate 会处理 None (返回 0.5)，QFormer 也会处理 None (不进行 Text Attention)
        
        # 1. 预处理与分解
        x, stats = self.preprocessor(channel_data)
        gamma, beta = self.film_generator(stats)
        
        # Kernel=65 分解
        x_resid, x_trend = self.decomp(x)
        
        # 2. Macro 流：兜底看 Raw
        _, deep_feats = self.shape_encoder(x)
        
        # 3. Micro 流：Resid + Diff
        x_t = x.permute(0, 2, 1)        # [B, 1, L]
        x_resid_t = x_resid.permute(0, 2, 1) # [B, 1, L]
        x_diff_t = torch.diff(x_t, dim=2, prepend=x_t[:, :, :1])
        
        micro_input = torch.cat([x_resid_t, x_diff_t], dim=1) # [B, 2, L]
        detail_feats = self.detail_encoder(micro_input)
        
        # =========================================================
        # 4. 独立查询 (使用 task_embeds 替代 instruction_embeds)
        # =========================================================
        
        # (A) 查询宏观趋势
        # Q-Former 内部 Cross-Attn 现在看到的是纯粹的任务语义
        q_macro = self.qformer(deep_feats, task_embeds)
        
        # (B) 查询微观细节
        q_micro = self.qformer(detail_feats, task_embeds)
        
        # 5. 动态融合 (使用 task_embeds)
        # Gate 会对 task_embeds 进行 Pooling，然后决定权重
        # 由于 task_embeds 是“提纯”过的，Gate 判别会更准
        gate = self.fusion_gate(task_embeds) 
        
        q_fused = gate * q_macro + (1 - gate) * q_micro
        
        # 6. Adapter & Projection
        ts_tokens = self.adapter(q_fused, gamma=gamma, beta=beta)
        ts_tokens = self.llm_proj(ts_tokens)
        
        return ts_tokens
    # def _process_single_channel(
    #     self,
    #     channel_data: Tensor,
    #     instruction_embeds: Optional[Tensor] = None,
    #     sep_embed: Optional[Tensor] = None, # 
    # ) -> Tensor:
    #     x, stats = self.preprocessor(channel_data)
    #     gamma, beta = self.film_generator(stats)
        
    #     # 接收双流特征
    #     raw_embeds, deep_feats = self.shape_encoder(x)
        
    #     # 1. Q-Former (语义)
    #     query_tokens = self.qformer(deep_feats, instruction_embeds)
        
    #     # 2. Detail Projector (细节)
    #     detail_tokens = self.detail_proj(raw_embeds) 
        
    #     # 3. 融合 (传入 sep_embed)
    #     ts_tokens = self.adapter(
    #         query_tokens, detail_tokens, 
    #         sep_embed=sep_embed, 
    #         gamma=gamma, beta=beta
    #     )
    #     ts_tokens = self.llm_proj(ts_tokens)
        
    #     return ts_tokens

    # def forward(
    #     self,
    #     raw_series: Tensor,
    #     text_prefix: Tensor,
    #     text_suffix: Tensor,
    #     instruction_embeds: Optional[Tensor] = None,
    # ) -> Dict[str, Tensor]:
    #     target_dtype = text_prefix.dtype
    #     x, stats = self.preprocessor(raw_series)
    #     gamma, beta = self.film_generator(stats)
        
    #     raw_embeds, deep_feats = self.shape_encoder(x)
        
    #     query_tokens = self.qformer(deep_feats, instruction_embeds)
    #     detail_tokens = self.detail_proj(raw_embeds)
        
    #     # 通用 forward 暂时不处理 sep_embed，或者也可以加上
    #     ts_tokens = self.adapter(query_tokens, detail_tokens, gamma=gamma, beta=beta)
    #     ts_tokens = self.llm_proj(ts_tokens)
        
    #     if ts_tokens.dtype != target_dtype:
    #         ts_tokens = ts_tokens.to(dtype=target_dtype)
            
    #     assembled = torch.cat(
    #         [text_prefix, ts_tokens, text_suffix],
    #         dim=1,
    #     )
    #     return {
    #         "ts_tokens": ts_tokens,
    #         "assembled": assembled,
    #     }


class StatBypassCROMETS1(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.ts_model = CROMETSModel(config)
        self.llm = FrozenLLM(config)
        self.tokenizer = InstructionTokenizer(config)
        
        self.sep_token = nn.Parameter(
            torch.randn(1, config.llm_embed_dim) * 0.02
        )
        
        #  定义模态特殊标记 (随机初始化)
        # 使用 nn.Parameter 确保在 SFT/LoRA 时能被优化器捕获
        self.ts_start_token = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        self.ts_end_token   = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        # self.feat_sep_token = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        # self.feat_sep_token = nn.Parameter(torch.randn(1, 1, config.patch_embedding_dim) * 0.02)

    def _prepare_text(
        self,
        text_input: Union[Tensor, Sequence[str]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(text_input, Tensor):
            mask = torch.ones(
                text_input.size(0),
                text_input.size(1),
                dtype=torch.long,
                device=device,
            )
            return text_input, mask
        encoded = self.tokenizer(text_input, device)
        embeds = self.llm.embed(encoded["input_ids"])
        return embeds, encoded["attention_mask"]
    def prepare_multimodal_embeds(
        self,
        input_texts: Sequence[str],
        timeseries_lists: Sequence[Sequence[Tensor]],
        output_texts: Optional[Sequence[str]] = None,
        # 新增 Mask 参数用于 Ablation
        mask_query: bool = False,
        mask_detail: bool = False,
        mask_text_stats: bool = False
    ) -> Dict[str, Any]:
        """
        核心逻辑封装：统一构建训练和推理用的 Multimodal Embeddings。
        拼接顺序：[Start] -> [Stats] -> [TS Features] -> [End] 
        """
        device = next(self.parameters()).device
        batch_size = len(input_texts)
        
        assembled_embeds_list = []
        attention_masks_list = []
        prefix_mask_lengths = []
        suffix_mask_lengths = []
        
        target_dtype = next(self.llm.parameters()).dtype
        
        for i in range(batch_size):
            input_text = input_texts[i]
            timeseries_list = list(timeseries_lists[i]) 
            output_text = output_texts[i] if output_texts is not None else None
            
            ts_marker = "<ts><ts/>"
            text_parts = input_text.split(ts_marker)
            
            # 1. 提取全局指令 (Text-Guided)
            full_instruction_text = " ".join([p.strip() for p in text_parts if p.strip()])
            current_instruction_embeds = None
            drop_text = self.training and (torch.rand(1).item() < 0.15)
            
            if full_instruction_text and not drop_text:
                instr_encoded = self.tokenizer([full_instruction_text], device)
                input_ids = instr_encoded["input_ids"]
                if input_ids.shape[1] > 2048:
                     input_ids = input_ids[:, :2048]
                current_instruction_embeds = self.llm.embed(input_ids)
            
            num_markers = len(text_parts) - 1
            num_timeseries = len(timeseries_list)
            
            if num_timeseries < num_markers:
                for _ in range(num_markers - num_timeseries):
                    timeseries_list.append(
                        torch.zeros(self.config.input_channels, device=device)
                    )
            elif num_timeseries > num_markers:
                timeseries_list = timeseries_list[:num_markers]
            
            segment_embeds = []
            segment_masks = []
            
            # 2. Prefix Text
            if text_parts[0]:
                prefix_encoded = self.tokenizer([text_parts[0]], device)
                prefix_embed = self.llm.embed(prefix_encoded["input_ids"])
                prefix_mask = prefix_encoded["attention_mask"]
                segment_embeds.append(prefix_embed[0])
                segment_masks.append(prefix_mask[0])
                prefix_length = prefix_mask[0].sum().item()
            else:
                prefix_length = 0
            
            prefix_mask_lengths.append(prefix_length)
            
            # 3. Time Series Loop
            for ts_idx, ts_tensor in enumerate(timeseries_list):
                ts_tensor = ts_tensor.to(device)
                
                if ts_tensor.numel() > 0:
                    ts_mean = ts_tensor.mean().item()
                    ts_std = ts_tensor.std().item()
                else:
                    ts_mean = 0.0
                    ts_std = 1.0
                
                # Stats Dropout / Mask Logic
                if mask_text_stats:
                    stats_str = ""
                elif self.training and torch.rand(1).item() < 0.5:
                    stats_str = ""
                else:
                    stats_str = f" [Scale: {ts_std:.2f}, Offset: {ts_mean:.2f}] "
                
                # 获取时序特征 (Output: [Query, Sep, Detail])
                ts_batch = ts_tensor.unsqueeze(0)
                ts_tokens = self.ts_model._process_single_channel(
                    ts_batch, 
                    instruction_embeds=current_instruction_embeds
                )
                
                if mask_query:
                    ts_tokens.fill_(0.0)

                if ts_tokens.dtype != target_dtype:
                    ts_tokens = ts_tokens.to(dtype=target_dtype)
                ts_embed = ts_tokens[0]
                
                # (A) Start Token
                segment_embeds.append(self.ts_start_token[0]) 
                segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                # (B) Stats Text (前置!)
                if stats_str:
                    stats_encoded = self.tokenizer([stats_str], device)
                    stats_embed = self.llm.embed(stats_encoded["input_ids"])
                    stats_mask = stats_encoded["attention_mask"]
                    segment_embeds.append(stats_embed[0])
                    segment_masks.append(stats_mask[0])
                
                # (C) TS Features
                segment_embeds.append(ts_embed)
                segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
                
                # (D) End Token
                segment_embeds.append(self.ts_end_token[0])
                segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                # (E) Multi-TS Separator (可选)
                if ts_idx < len(timeseries_list) - 1:
                    sep_embed = self.sep_token
                    if sep_embed.dtype != target_dtype:
                        sep_embed = sep_embed.to(dtype=target_dtype)
                    segment_embeds.append(sep_embed)
                    segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                # (F) Next Text Part
                text_idx = ts_idx + 1
                if text_idx < len(text_parts) and text_parts[text_idx]:
                    text_encoded = self.tokenizer([text_parts[text_idx]], device)
                    text_embed = self.llm.embed(text_encoded["input_ids"])
                    text_mask = text_encoded["attention_mask"]
                    segment_embeds.append(text_embed[0])
                    segment_masks.append(text_mask[0])
            
            # 4. Suffix Text (Output)
            if output_text:
                suffix_encoded = self.tokenizer([output_text], device)
                suffix_embed = self.llm.embed(suffix_encoded["input_ids"])
                suffix_mask = suffix_encoded["attention_mask"]
                segment_embeds.append(suffix_embed[0])
                segment_masks.append(suffix_mask[0])
                suffix_length = suffix_mask[0].sum().item()
            else:
                suffix_length = 0
            
            suffix_mask_lengths.append(suffix_length)
            
            # Concat for this sample
            full_embed = torch.cat(segment_embeds, dim=0)
            full_mask = torch.cat(segment_masks, dim=0)
            
            assembled_embeds_list.append(full_embed)
            attention_masks_list.append(full_mask)
        
        # 5. Padding Batch
        max_len = max(emb.shape[0] for emb in assembled_embeds_list)
        embed_dim = assembled_embeds_list[0].shape[1]
        
        padded_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=assembled_embeds_list[0].dtype)
        padded_masks = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        
        for i, (emb, mask) in enumerate(zip(assembled_embeds_list, attention_masks_list)):
            seq_len = emb.shape[0]
            padded_embeds[i, :seq_len] = emb
            padded_masks[i, :seq_len] = mask
            
        padded_embeds = padded_embeds.to(self.llm.model.dtype)
        
        return {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_masks,
            "prefix_mask_lengths": prefix_mask_lengths,
            "suffix_mask_lengths": suffix_mask_lengths
        }

    def forward_chatts(
        self,
        input_texts: Sequence[str],
        timeseries_lists: Sequence[Sequence[Tensor]],
        output_texts: Sequence[str],
        llm_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """训练时调用，计算 Loss"""
        llm_kwargs = llm_kwargs or {}
        
        prepared = self.prepare_multimodal_embeds(input_texts, timeseries_lists, output_texts)
        
        outputs = self.llm(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            **llm_kwargs,
        )
        
        return {
            "llm_outputs": outputs,
            **prepared
        }

    @torch.no_grad()
    def generate(
        self,
        input_texts: Union[str, Sequence[str]],
        timeseries_lists: Union[Sequence[Tensor], Sequence[Sequence[Tensor]]],
        mask_query: bool = False,
        mask_detail: bool = False,
        mask_text_stats: bool = False,
        **gen_kwargs
    ):
        """
        推理时调用，支持 Ablation Masking。
        """
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        if len(timeseries_lists) > 0 and isinstance(timeseries_lists[0], Tensor):
            timeseries_lists = [timeseries_lists]
            
        prepared = self.prepare_multimodal_embeds(
            input_texts, 
            timeseries_lists, 
            output_texts=None,
            mask_query=mask_query,
            mask_detail=mask_detail,
            mask_text_stats=mask_text_stats
        )
        
        return self.llm.model.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            **gen_kwargs
        )