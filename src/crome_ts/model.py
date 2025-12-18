from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ==========================================
# 1. 修改后的 CNN 细节编码器 (Micro Stream)
# ==========================================
class CNNDetailEncoder(nn.Module):
    """
    特化细节编码器：
    1. 输入为 Residual + Diff 双通道时序信号 [B, 2, L]
    2. 使用 ResNet 提取高频特征
    3. 使用 Stride Conv 进行 8x 降采样，保留区间定位能力
    """
    def __init__(self, config: CROMEConfig, input_channels: int = 2):
        super().__init__()
        dim = config.patch_embedding_dim
        
        # 1. Stem: 将输入通道映射到特征维度
        # Kernel=3 保持局部性
        self.stem = nn.Conv1d(input_channels, dim, kernel_size=3, padding=1)
        
        # 2. 卷积残差块 (Conv1D + BN + GELU)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim) 
        self.act = nn.GELU()
        
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(dim)
        
        # 增加深度
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(dim)

        self.dropout = nn.Dropout(config.proj_dropout)
        
        # 3. [关键] 降采样投影层 (Direct Downsampling Projector)
        # Stride=8: 1024点 -> 128 Token
        # Kernel=8: 覆盖整个步长窗口
        self.final_proj = nn.Conv1d(dim, dim, kernel_size=8, stride=8)

    def forward(self, x):
        # x input shape: [Batch, Channels=2, Length]
        x = self.stem(x)
        
        # Residual Block 1
        res1 = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + res1)
        
        # Residual Block 2
        res2 = out
        out = self.bn3(self.conv3(out))
        out = self.act(out + res2)
        
        # [关键] 直连降采样
        # out: [B, Dim, L] -> [B, Dim, L/8]
        out = self.final_proj(out)
        
        # Transpose to [B, L/8, D] for LLM
        return out.transpose(1, 2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len + 128, device=x.device)
        return (
            self.cos_cached[..., :seq_len, :].to(dtype=x.dtype),
            self.sin_cached[..., :seq_len, :].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = rotary_emb(v, seq_len=L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
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
        x = x + self.attn(self.norm1(x), rotary_emb)
        x = x + self.mlp(self.norm2(x))
        return x
        
@dataclass
class CROMEConfig:
    input_channels: int
    llm_embed_dim: int
    patch_len: int = 16
    patch_stride: int = 8
    patch_embedding_dim: int = 512
    patch_num_heads: int = 8
    patch_num_layers: int = 4
    freeze_patch_encoder: bool = False
    query_tokens: int = 32
    adapter_hidden_dim: int = 256
    fuse_mode: str = "add"
    epsilon: float = 1e-4
    proj_dropout: float = 0.0
    llm_model_path: str = "/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"
    llm_dtype: str = "bfloat16"
    llm_device_map: str = "auto"

def _resolve_dtype(name: str) -> torch.dtype:
    name = name.lower()
    return getattr(torch, name)

def get_llm_embed_dim(llm_model_path: str) -> int:
    try:
        config = AutoConfig.from_pretrained(llm_model_path)
        if hasattr(config, 'hidden_size'): return config.hidden_size
        elif hasattr(config, 'd_model'): return config.d_model
        elif hasattr(config, 'n_embd'): return config.n_embd
        else: raise ValueError(f"无法获取embed_dim: {llm_model_path}")
    except Exception as e:
        raise RuntimeError(f"加载模型配置失败: {e}")

class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        mu = x_fp32.mean(dim=1, keepdim=True)
        sigma = x_fp32.std(dim=1, keepdim=True).clamp_min(self.eps)
        x_norm = (x_fp32 - mu) / sigma
        x_norm = x_norm.to(dtype=orig_dtype)
        stats = torch.stack((mu.squeeze(1), sigma.squeeze(1)), dim=-1).to(dtype=orig_dtype)
        return x_norm, stats

class InputPreprocessor(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(config.epsilon)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_norm, stats = self.revin(x)
        return x_norm, stats

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
    def __init__(self, config: CROMEConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.embedding = PatchEmbedding(config, input_dim)
        head_dim = config.patch_embedding_dim // config.patch_num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim)
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(
                dim=config.patch_embedding_dim,
                num_heads=config.patch_num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(config.patch_num_layers)
        ])
        self.norm = nn.LayerNorm(config.patch_embedding_dim)
        if config.freeze_patch_encoder:
            self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        emb = self.embedding(x) 
        x_out = emb
        for block in self.blocks:
            x_out = block(x_out, self.rotary_emb)
        x_out = self.norm(x_out)
        return emb, x_out

class QFormerLayer(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        dim = config.patch_embedding_dim
        num_heads = config.patch_num_heads
        dropout = getattr(config, "dropout", 0.1)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.text_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.text_proj = nn.Linear(config.llm_embed_dim, dim) 
        self.ts_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm4 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.last_text_attn_weights = None
        self.last_ts_attn_weights = None

    def forward(self, queries, text_embeds, ts_embeds):
        q_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.dropout(q_out))
        if text_embeds is not None:
            text_embeds_input = text_embeds.to(self.text_proj.weight.dtype)
            text_kv = self.text_proj(text_embeds_input).to(queries.dtype)
            text_out, text_attn = self.text_attn(queries, text_kv, text_kv)
            queries = self.norm2(queries + self.dropout(text_out))
            self.last_text_attn_weights = text_attn.detach().cpu()
        ts_out, ts_attn = self.ts_attn(queries, ts_embeds, ts_embeds)
        queries = self.norm3(queries + self.dropout(ts_out))
        self.last_ts_attn_weights = ts_attn.detach().cpu()
        queries = self.norm4(queries + self.dropout(self.ffn(queries)))
        return queries

class QFormer(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(torch.randn(config.query_tokens, config.patch_embedding_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        num_layers = getattr(config, "qformer_layers", 4) 
        self.layers = nn.ModuleList([QFormerLayer(config) for _ in range(num_layers)])
        self.last_text_attn_weights = None

    def forward(self, patch_tokens: Tensor, instruction_embeds: Optional[Tensor] = None) -> Tensor:
        b = patch_tokens.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            queries = layer(queries, instruction_embeds, patch_tokens)
        self.last_text_attn_weights = self.layers[-1].last_text_attn_weights
        return queries

class RobustFiLMGenerator(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, config.adapter_hidden_dim),
            nn.LayerNorm(config.adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_hidden_dim, config.adapter_hidden_dim * 2)
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
        if out.dim() == 2: out = out.unsqueeze(1) 
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

class CROMEAdapter(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.query_adapter = CROMEAdapterBlock(
            config.patch_embedding_dim, config.adapter_hidden_dim
        )

    def forward(self, tokens: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None) -> Tensor:
        return self.query_adapter(tokens, gamma=gamma, beta=beta)

class InstructionTokenizer:
    def __init__(self, config: CROMEConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def __call__(self, texts: Sequence[str], device: torch.device) -> Dict[str, Tensor]:
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(device) for k, v in encoded.items()}

class FrozenLLM(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        dtype = _resolve_dtype(config.llm_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_path,
            torch_dtype=dtype,
            device_map=config.llm_device_map,
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
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        x_resid = x - x_trend
        return x_resid, x_trend

# ==========================================
# 2. 修改后的 CROMETSModel (Direct Concatenation)
# ==========================================
class CROMETSModel(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        pre_input_dim = (
            config.input_channels if config.fuse_mode == "add" else config.input_channels * 2
        )
        self.preprocessor = InputPreprocessor(config)
        self.shape_encoder = PatchTSTEncoder(config, pre_input_dim)
        
        self.qformer = QFormer(config)
        self.film_generator = RobustFiLMGenerator(config)
        self.adapter = CROMEAdapter(config)
        self.decomp = SeriesDecomp(kernel_size=25)

        # [修改] 使用新的 CNN 细节编码器 (输入=2通道: Resid + Diff)
        # 不再需要 resid_embedding 和 PE，因为 CNN 直接吃时序
        self.detail_encoder = CNNDetailEncoder(config, input_channels=2)
        
        # [修改] 移除融合门控，改用直连
        # self.fusion_gate = FusionGate(config)
        
        self.llm_proj = nn.Sequential(
            nn.Linear(config.patch_embedding_dim, config.llm_embed_dim),
            nn.GELU(),
            nn.Dropout(config.proj_dropout),
            nn.Linear(config.llm_embed_dim, config.llm_embed_dim)
        )
        
        # 清理旧属性
        if hasattr(self, 'resid_embedding'): del self.resid_embedding
        if hasattr(self, 'resid_pos_encoding'): del self.resid_pos_encoding

    def _process_single_channel(
        self,
        channel_data: Tensor,
        instruction_embeds: Optional[Tensor] = None,
        sep_embed: Optional[Tensor] = None, 
    ) -> Tensor:
        # 1. 预处理与分解
        x, stats = self.preprocessor(channel_data)
        gamma, beta = self.film_generator(stats)
        x_resid, x_trend = self.decomp(x)
        
        # 2. 宏观流 (Macro Stream) -> Q-Former
        _, deep_feats = self.shape_encoder(x)
        q_macro = self.qformer(deep_feats, instruction_embeds) # [B, N_macro, D]
        
        # 3. 微观流 (Micro Stream) -> 直连 CNN (Residual + Diff)
        # 构造 Residual + Diff 输入
        x_t = x.permute(0, 2, 1)        # [B, 1, L]
        x_resid_t = x_resid.permute(0, 2, 1) # [B, 1, L]
        
        # 计算一阶差分 (Diff)
        # Prepend 第一个值以保持长度一致
        x_diff_t = torch.diff(x_t, dim=2, prepend=x_t[:, :, :1])
        
        # 拼接双通道: [B, 2, L]
        micro_input = torch.cat([x_resid_t, x_diff_t], dim=1)
        
        # CNN 编码 + 降采样 -> [B, L/8, D]
        micro_tokens = self.detail_encoder(micro_input)
        
        # 4. 特征融合 (Direct Concatenation)
        # 将 Q-Former 的宏观语义 Token 与 CNN 的微观序列 Token 拼接
        ts_tokens = torch.cat([q_macro, micro_tokens], dim=1)
        
        # 5. Adapter & Projection
        # 统一注入 FiLM 统计量，并映射到 LLM 空间
        ts_tokens = self.adapter(ts_tokens, gamma=gamma, beta=beta)
        ts_tokens = self.llm_proj(ts_tokens)
        
        return ts_tokens

class StatBypassCROMETS1(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.ts_model = CROMETSModel(config)
        self.llm = FrozenLLM(config)
        self.tokenizer = InstructionTokenizer(config)
        
        self.sep_token = nn.Parameter(torch.randn(1, config.llm_embed_dim) * 0.02)
        self.ts_start_token = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        self.ts_end_token   = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        self.feat_sep_token = nn.Parameter(torch.randn(1, 1, config.patch_embedding_dim) * 0.02)

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
        mask_query: bool = False,
        mask_detail: bool = False,
        mask_text_stats: bool = False
    ) -> Dict[str, Any]:
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
                    timeseries_list.append(torch.zeros(self.config.input_channels, device=device))
            elif num_timeseries > num_markers:
                timeseries_list = timeseries_list[:num_markers]
            
            segment_embeds = []
            segment_masks = []
            
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
            
            for ts_idx, ts_tensor in enumerate(timeseries_list):
                ts_tensor = ts_tensor.to(device)
                
                if ts_tensor.numel() > 0:
                    ts_mean = ts_tensor.mean().item()
                    ts_std = ts_tensor.std().item()
                else:
                    ts_mean = 0.0
                    ts_std = 1.0
                
                if mask_text_stats:
                    stats_str = ""
                elif self.training and torch.rand(1).item() < 0.5:
                    stats_str = ""
                else:
                    stats_str = f" [Scale: {ts_std:.2f}, Offset: {ts_mean:.2f}] "
                
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
                
                segment_embeds.append(self.ts_start_token[0]) 
                segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                if stats_str:
                    stats_encoded = self.tokenizer([stats_str], device)
                    stats_embed = self.llm.embed(stats_encoded["input_ids"])
                    stats_mask = stats_encoded["attention_mask"]
                    segment_embeds.append(stats_embed[0])
                    segment_masks.append(stats_mask[0])
                
                segment_embeds.append(ts_embed)
                segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
                
                segment_embeds.append(self.ts_end_token[0])
                segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                if ts_idx < len(timeseries_list) - 1:
                    sep_embed = self.sep_token
                    if sep_embed.dtype != target_dtype:
                        sep_embed = sep_embed.to(dtype=target_dtype)
                    segment_embeds.append(sep_embed)
                    segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                text_idx = ts_idx + 1
                if text_idx < len(text_parts) and text_parts[text_idx]:
                    text_encoded = self.tokenizer([text_parts[text_idx]], device)
                    text_embed = self.llm.embed(text_encoded["input_ids"])
                    text_mask = text_encoded["attention_mask"]
                    segment_embeds.append(text_embed[0])
                    segment_masks.append(text_mask[0])
            
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
            
            full_embed = torch.cat(segment_embeds, dim=0)
            full_mask = torch.cat(segment_masks, dim=0)
            
            assembled_embeds_list.append(full_embed)
            attention_masks_list.append(full_mask)
        
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
        timeseries_lists: Union[Sequence[Tensor], Sequence[Sequence[Tensor]]],\
        mask_query: bool = False,
        mask_detail: bool = False,
        mask_text_stats: bool = False,
        **gen_kwargs
    ):
        if isinstance(input_texts, str): input_texts = [input_texts]
        if len(timeseries_lists) > 0 and isinstance(timeseries_lists[0], Tensor): timeseries_lists = [timeseries_lists]
        prepared = self.prepare_multimodal_embeds(
            input_texts, timeseries_lists, output_texts=None,
            mask_query=mask_query, mask_detail=mask_detail, mask_text_stats=mask_text_stats
        )
        return self.llm.model.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            **gen_kwargs
        )