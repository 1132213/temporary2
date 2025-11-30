from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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
    query_tokens: int = 32
    adapter_hidden_dim: int = 256
    fuse_mode: str = "add"  # or "concat"
    epsilon: float = 1e-4
    # LLM 接口
    llm_model_path: str = "/root/emhua/btwu/Llama-2-7b-hf"
    llm_dtype: str = "bfloat16"
    llm_device_map: str = "auto"
    # 注意：原本的 stat_fourier_features 等参数已废弃，保留是为了兼容旧Config加载


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
    """Reversible Instance Normalization。"""
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mu = x.mean(dim=1, keepdim=True)
        sigma = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        x_norm = (x - mu) / sigma
        stats = torch.stack((mu.squeeze(1), sigma.squeeze(1)), dim=-1)
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
        self.pos_encoding = FixedSinePositionalEncoding(config.input_channels)
        self.fuse_mode = config.fuse_mode

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, l, c = x.shape
        x_norm, stats = self.revin(x)
        
        if c != self.config.input_channels:
            pos_encoding = FixedSinePositionalEncoding(c)
        else:
            pos_encoding = self.pos_encoding
        
        time_emb = pos_encoding(l, device=x.device, dtype=x.dtype)
        time_emb = time_emb.unsqueeze(0).expand(b, -1, -1)
        
        if self.fuse_mode == "add":
            fused = x_norm + time_emb
        elif self.fuse_mode == "concat":
            fused = torch.cat([x_norm, time_emb], dim=-1)
        else:
            raise ValueError(f"未知融合模式: {self.fuse_mode}")
            
        return fused, stats


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
    """模块 II：冻结 PatchTST 形态编码器。"""
    def __init__(self, config: CROMEConfig, input_dim: int):
        super().__init__()
        self.embedding = PatchEmbedding(config, input_dim)
        self.config=config
        layer = nn.TransformerEncoderLayer(
            d_model=config.patch_embedding_dim,
            nhead=config.patch_num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.patch_num_layers)
        if config.freeze_patch_encoder:
            self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        # 如果需要解冻，这里可以去掉 no_grad，但通常 encoder forward 逻辑简单
        # 为了兼容性保持原状，外层训练脚本控制 requires_grad 即可
        # 如果需要梯度回传，PatchTSTEncoder 内部不应该强制 no_grad
        # 修正：根据 freeze_patch_encoder 决定是否启用 grad
        if self.config.freeze_patch_encoder:
            with torch.no_grad():
                emb = self.embedding(x)
                return self.encoder(emb)
        else:
            emb = self.embedding(x)
            return self.encoder(emb)


class QFormer(nn.Module):
    """模块 III 分支 A：Query Former。"""
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.query_tokens = nn.Parameter(
            torch.randn(config.query_tokens, config.patch_embedding_dim)
        )
        self.attn = nn.MultiheadAttention(
            config.patch_embedding_dim, config.patch_num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(config.patch_embedding_dim)

    def forward(self, patch_tokens: Tensor, instruction_embeds: Optional[Tensor] = None) -> Tensor:
        b = patch_tokens.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        key = patch_tokens if instruction_embeds is None else torch.cat(
            [patch_tokens, instruction_embeds], dim=1
        )
        attn_out, _ = self.attn(queries, key, key)
        return self.ln(attn_out)


class DetailProjection(nn.Module):
    """模块 III 分支 B：局部细节投影。"""
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.proj = nn.Linear(config.patch_embedding_dim, config.patch_embedding_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.proj(patch_tokens)


class RobustFiLMGenerator(nn.Module):
    """
    Robust Log-Space FiLM Generator.
    解决梯度冲突与尺度爆炸问题：
    1. 独立于主干特征流，梯度不互通。
    2. 使用对数变换预处理，适应不同量级的时间序列。
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        # 输入特征：log_mu, log_sigma, sign_mu, cv
        input_dim = 4
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.adapter_hidden_dim),
            nn.LayerNorm(config.adapter_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_hidden_dim, config.adapter_hidden_dim * 2) # gamma, beta
        )
        
        # 零初始化最后一层，使初始状态为恒等变换
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, stats: Tensor) -> Tuple[Tensor, Tensor]:
        # stats: [B, 2] -> (mean, std)
        if stats.dim() == 2:
            mu = stats[..., 0:1]
            sigma = stats[..., 1:2]
        else:
            mu = stats[..., 0]
            sigma = stats[..., 1]
            
        # 数值稳定化预处理
        log_mu = torch.log1p(mu.abs())
        log_sigma = torch.log1p(sigma)
        sign_mu = torch.sign(mu)
        cv = mu / (sigma + 1e-5) # 变异系数特征
        
        features = torch.cat([log_mu, log_sigma, sign_mu, cv], dim=-1)
        out = self.mlp(features)
        
        if out.dim() == 2:
            out = out.unsqueeze(1) # [B, 1, 2*H]
            
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
        
        # FiLM 调制
        if gamma is not None and beta is not None:
            z = z * (1 + gamma) + beta

        return x + self.up(z)


class CROMEAdapter(nn.Module):
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.query_adapter = CROMEAdapterBlock(
            config.patch_embedding_dim, config.adapter_hidden_dim
        )
        self.patch_adapter = CROMEAdapterBlock(
            config.patch_embedding_dim, config.adapter_hidden_dim
        )

    def forward(
        self, 
        query_tokens: Tensor, 
        patch_tokens: Tensor, 
        gamma: Optional[Tensor] = None, 
        beta: Optional[Tensor] = None
    ) -> Tensor:
        query_out = self.query_adapter(query_tokens)
        
        # 仅对 Detail Path 应用 FiLM
        patch_out = self.patch_adapter(patch_tokens, gamma=gamma, beta=beta)
        
        return torch.cat([query_out, patch_out], dim=1)


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


class CROMETSModel(nn.Module):
    """
    模块 V：CROME 主体模型 (FiLM Enhanced, No Stat Token)
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        pre_input_dim = (
            config.input_channels if config.fuse_mode == "add" else config.input_channels * 2
        )
        self.preprocessor = InputPreprocessor(config)
        self.shape_encoder = PatchTSTEncoder(config, pre_input_dim)
        self.qformer = QFormer(config)
        self.detail_proj = DetailProjection(config)
        
        # FiLM Generator
        self.film_generator = RobustFiLMGenerator(config)
        
        self.adapter = CROMEAdapter(config)
        
        # 移除 StatProjector
        # self.stat_projector = StatProjector(config)
        
        self.llm_proj = nn.Linear(config.patch_embedding_dim, config.llm_embed_dim)

    def _process_single_channel(
        self,
        channel_data: Tensor,
        instruction_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """
        处理单通道数据，仅返回 ts_tokens (stat token 已移除)
        """
        x, stats = self.preprocessor(channel_data)
        
        # 生成 FiLM 参数
        gamma, beta = self.film_generator(stats)
        
        patch_tokens = self.shape_encoder(x)
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        detail_tokens = self.detail_proj(patch_tokens)
        
        # FiLM 调制与融合
        ts_tokens = self.adapter(query_tokens, detail_tokens, gamma=gamma, beta=beta)
        ts_tokens = self.llm_proj(ts_tokens)
        
        return ts_tokens

    def forward(
        self,
        raw_series: Tensor,
        text_prefix: Tensor,
        text_suffix: Tensor,
        instruction_embeds: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        target_dtype = text_prefix.dtype
        
        x, stats = self.preprocessor(raw_series)
        gamma, beta = self.film_generator(stats)
        
        patch_tokens = self.shape_encoder(x)
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        detail_tokens = self.detail_proj(patch_tokens)
        
        ts_tokens = self.adapter(query_tokens, detail_tokens, gamma=gamma, beta=beta)
        ts_tokens = self.llm_proj(ts_tokens)
        
        if ts_tokens.dtype != target_dtype:
            ts_tokens = ts_tokens.to(dtype=target_dtype)
            
        # 拼接：移除 stat_token
        assembled = torch.cat(
            [text_prefix, ts_tokens, text_suffix],
            dim=1,
        )
        return {
            "ts_tokens": ts_tokens,
            "assembled": assembled,
        }


class StatBypassCROMETS1(nn.Module):
    """
    端到端系统封装：支持 ChatTS 格式与 Text Stats Dropout
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.ts_model = CROMETSModel(config)
        self.llm = FrozenLLM(config)
        self.tokenizer = InstructionTokenizer(config)
        
        self.sep_token = nn.Parameter(
            torch.randn(1, config.llm_embed_dim) * 0.02
        )

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
    
    def forward_chatts(
        self,
        input_texts: Sequence[str],
        timeseries_lists: Sequence[Sequence[Tensor]],
        output_texts: Sequence[str],
        llm_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        llm_kwargs = llm_kwargs or {}
        batch_size = len(input_texts)
        
        assembled_embeds_list = []
        attention_masks_list = []
        prefix_mask_lengths = []
        suffix_mask_lengths = []
        
        for i in range(batch_size):
            input_text = input_texts[i]
            timeseries_list = timeseries_lists[i]
            output_text = output_texts[i]
            
            ts_marker = "<ts><ts/>"
            text_parts = input_text.split(ts_marker)
            
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
            
            target_dtype = next(self.llm.parameters()).dtype
            
            # Prefix
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
            
            # Process Time Series
            for ts_idx, ts_tensor in enumerate(timeseries_list):
                ts_tensor = ts_tensor.to(device)
                
                if ts_tensor.numel() > 0:
                    ts_mean = ts_tensor.mean().item()
                    ts_std = ts_tensor.std().item()
                    ts_min = ts_tensor.min().item()
                    ts_max = ts_tensor.max().item()
                else:
                    ts_mean = ts_std = ts_min = ts_max = 0.0
                
                # === Text Stats Dropout ===
                # 训练时 50% 概率丢弃统计量文本
                if self.training and torch.rand(1).item() < 0.5:
                    stats_str = ""
                else:
                    stats_str = f" [Stats: mean={ts_mean:.2f}, std={ts_std:.2f}, min={ts_min:.2f}, max={ts_max:.2f}] "
                
                stats_encoded = self.tokenizer([stats_str], device)
                stats_embed = self.llm.embed(stats_encoded["input_ids"])
                stats_mask = stats_encoded["attention_mask"]
                
                segment_embeds.append(stats_embed[0])
                segment_masks.append(stats_mask[0])
                
                ts_batch = ts_tensor.unsqueeze(0)
                
                # 获取 TS Tokens (无 stat token)
                ts_tokens = self.ts_model._process_single_channel(
                    ts_batch, instruction_embeds=None
                )
                
                if ts_tokens.dtype != target_dtype:
                    ts_tokens = ts_tokens.to(dtype=target_dtype)
                
                # 拼接：直接使用 ts_tokens
                ts_embed = ts_tokens[0]
                
                segment_embeds.append(ts_embed)
                segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
                
                if ts_idx < len(timeseries_list) - 1:
                    sep_embed = self.sep_token
                    if sep_embed.dtype != target_dtype:
                        sep_embed = sep_embed.to(dtype=target_dtype)
                    segment_embeds.append(sep_embed)
                    segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                # Middle Text
                text_idx = ts_idx + 1
                if text_idx < len(text_parts) and text_parts[text_idx]:
                    text_encoded = self.tokenizer([text_parts[text_idx]], device)
                    text_embed = self.llm.embed(text_encoded["input_ids"])
                    text_mask = text_encoded["attention_mask"]
                    segment_embeds.append(text_embed[0])
                    segment_masks.append(text_mask[0])
            
            # Suffix
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
        
        # Batch Padding
        max_len = max(emb.shape[0] for emb in assembled_embeds_list)
        embed_dim = assembled_embeds_list[0].shape[1]
        
        padded_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=assembled_embeds_list[0].dtype)
        padded_masks = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        
        for i, (emb, mask) in enumerate(zip(assembled_embeds_list, attention_masks_list)):
            seq_len = emb.shape[0]
            padded_embeds[i, :seq_len] = emb
            padded_masks[i, :seq_len] = mask
        
        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_masks,
            **llm_kwargs,
        )
        
        return {
            "assembled": padded_embeds,
            "attention_mask": padded_masks,
            "llm_outputs": outputs,
            "prefix_mask_lengths": prefix_mask_lengths,
            "suffix_mask_lengths": suffix_mask_lengths,
        }

    def forward(
        self,
        raw_series: Tensor,
        prefix: Union[Tensor, Sequence[str]],
        suffix: Union[Tensor, Sequence[str]],
        instruction_text: Optional[Sequence[str]] = None,
        llm_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        device = raw_series.device
        llm_kwargs = llm_kwargs or {}
        prefix_embeds, prefix_mask = self._prepare_text(prefix, device)
        suffix_embeds, suffix_mask = self._prepare_text(suffix, device)

        instruction_embeds = None
        if instruction_text is not None:
            instr_encoded = self.tokenizer(instruction_text, device)
            instruction_embeds = self.llm.embed(instr_encoded["input_ids"])
            
        ts_outputs = self.ts_model(
            raw_series,
            prefix_embeds,
            suffix_embeds,
            instruction_embeds=instruction_embeds,
        )
        batch = raw_series.size(0)
        
        ts_tokens = ts_outputs["ts_tokens"]
        
        ts_mask = torch.ones(
            batch,
            ts_tokens.size(1),
            device=device,
            dtype=prefix_mask.dtype,
        )
        # 移除 stat_token 对应的 mask (ones)
        attention_mask = torch.cat(
            [prefix_mask, ts_mask, suffix_mask],
            dim=1,
        )
        
        outputs = self.llm(
            inputs_embeds=ts_outputs["assembled"],
            attention_mask=attention_mask,
            **llm_kwargs,
        )
        return {
            **ts_outputs,
            "attention_mask": attention_mask,
            "llm_outputs": outputs,
            "prefix_mask": prefix_mask,
            "suffix_mask": suffix_mask,
        }