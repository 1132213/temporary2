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
    fuse_mode: str = "add"
    epsilon: float = 1e-4
    # LLM 接口
    llm_model_path: str = "/root/emhua/btwu/Llama-2-7b-hf"
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
        self.config = config 
        self.embedding = PatchEmbedding(config, input_dim)
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
        if self.config.freeze_patch_encoder:
            with torch.no_grad():
                emb = self.embedding(x)
                return self.encoder(emb)
        else:
            emb = self.embedding(x)
            return self.encoder(emb)


class QFormer(nn.Module):
    """
    模块 III 分支 A：Text-Guided Q-Former (方案 B+ 增强版).
    包含：
    1. Cross-Attention 文本引导
    2. Attention 权重暴露 (用于可视化)
    3. 全面 Dropout (防止过拟合)
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        
        # [新增] 获取 Dropout 率，优先从 config 读取，默认为 0.1
        dropout_rate = getattr(config, "dropout", 0.1)
        
        # 1. 可学习的 Query Tokens (Base Queries)
        self.query_tokens = nn.Parameter(
            torch.randn(config.query_tokens, config.patch_embedding_dim)
        )
        
        # 2. 文本交互层 (Text Interaction)
        # 将 LLM Embedding (如 4096) 投影到 Patch 维度 (如 512)
        self.text_proj = nn.Linear(config.llm_embed_dim, config.patch_embedding_dim)
        
        # Cross-Attention: Query 关注 Text
        # [修改] 启用 dropout
        self.text_attn = nn.MultiheadAttention(
            embed_dim=config.patch_embedding_dim,
            num_heads=config.patch_num_heads,
            dropout=dropout_rate, 
            batch_first=True
        )
        self.ln_text = nn.LayerNorm(config.patch_embedding_dim)

        # 3. 时序提取层 (Time-Series Extraction)
        # Cross-Attention: Query 关注 TS Patch
        # [修改] 启用 dropout
        self.ts_attn = nn.MultiheadAttention(
            embed_dim=config.patch_embedding_dim,
            num_heads=config.patch_num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.ln_ts = nn.LayerNorm(config.patch_embedding_dim)
        
        # [新增] 通用 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 初始化投影层
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

        # 用于存储可视化权重
        self.last_text_attn_weights = None

    def forward(self, patch_tokens: Tensor, instruction_embeds: Optional[Tensor] = None) -> Tensor:
        b = patch_tokens.size(0)
        
        # 1. 扩展 Base Query
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1) # [B, 32, D]
        
        # 重置可视化权重 (防止 Text Dropout 时残留旧权重)
        self.last_text_attn_weights = None
        
        # 2. [阶段一] 文本引导：Query 主动“读取”指令
        if instruction_embeds is not None:
            # instruction_embeds: [B, Text_Len, LLM_Dim]
            
            # (A) 类型转换 (解决 BF16 vs FP32 冲突)
            instruction_embeds = instruction_embeds.to(dtype=self.text_proj.weight.dtype)

            # (B) 投影文本特征
            text_kv = self.text_proj(instruction_embeds) # [B, Text_Len, D]
            
            # (C) Cross-Attention: Q=Queries, K=Text, V=Text
            # [修改] 获取 Attention 权重
            text_out, attn_weights = self.text_attn(
                query=queries,
                key=text_kv,
                value=text_kv
            )
            
            # [新增] 保存权重供可视化 (Detach + CPU 以节省显存)
            # 形状通常为 [Batch, Queries, Text_Len] (若 average_attn_weights=True)
            self.last_text_attn_weights = attn_weights.detach().cpu()
            
            # (D) 残差 + Norm + [新增] Dropout
            # 注意：这里采用了 Pre-Norm 或 Post-Norm 结构均可，这里维持原有的 Post-Norm 逻辑
            # Query = Norm(Query + Dropout(Attn(Q, Text)))
            queries = self.ln_text(queries + self.dropout(text_out))
        
        # 3. [阶段二] 时序提取：带着意图的 Query 提取时序特征
        # Q=Queries(Text-Aware), K=TS_Patch, V=TS_Patch
        ts_out, _ = self.ts_attn(
            query=queries, 
            key=patch_tokens, 
            value=patch_tokens
        )
        
        # [修改] 输出前应用 Dropout
        return self.ln_ts(self.dropout(ts_out))


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
    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        pre_input_dim = (
            config.input_channels if config.fuse_mode == "add" else config.input_channels * 2
        )
        self.preprocessor = InputPreprocessor(config)
        self.shape_encoder = PatchTSTEncoder(config, pre_input_dim)
        # 使用新的 Text-Guided QFormer
        self.qformer = QFormer(config)
        self.detail_proj = DetailProjection(config)
        
        self.film_generator = RobustFiLMGenerator(config)
        self.adapter = CROMEAdapter(config)
        
        self.llm_proj = nn.Linear(config.patch_embedding_dim, config.llm_embed_dim)

    def _process_single_channel(
        self,
        channel_data: Tensor,
        instruction_embeds: Optional[Tensor] = None, # 接收文本特征
    ) -> Tensor:
        x, stats = self.preprocessor(channel_data)
        gamma, beta = self.film_generator(stats)
        
        patch_tokens = self.shape_encoder(x)
        
        # 传入 instruction_embeds 给 QFormer
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        
        detail_tokens = self.detail_proj(patch_tokens)
        
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
        # 传入 instruction_embeds
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        detail_tokens = self.detail_proj(patch_tokens)
        
        ts_tokens = self.adapter(query_tokens, detail_tokens, gamma=gamma, beta=beta)
        ts_tokens = self.llm_proj(ts_tokens)
        
        if ts_tokens.dtype != target_dtype:
            ts_tokens = ts_tokens.to(dtype=target_dtype)
            
        assembled = torch.cat(
            [text_prefix, ts_tokens, text_suffix],
            dim=1,
        )
        return {
            "ts_tokens": ts_tokens,
            "assembled": assembled,
        }


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
            
            # =================================================================
            # [新增] 拼接所有文本片段作为全局指令 (Fix from discussion)
            # =================================================================
            full_instruction_text = " ".join([p.strip() for p in text_parts if p.strip()])
            
            current_instruction_embeds = None
            drop_text = self.training and (torch.rand(1).item() < 0.15)
            
            if full_instruction_text and not drop_text:
                # 正常提取文本特征
                instr_encoded = self.tokenizer([full_instruction_text], device)
                input_ids = instr_encoded["input_ids"]
                if input_ids.shape[1] > 512:
                     input_ids = input_ids[:, :512]
                current_instruction_embeds = self.llm.embed(input_ids)
            else:
                # 触发 Dropout，传入 None，强迫模型使用静态 Query
                current_instruction_embeds = None

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
            
            # 处理 Prefix Text (用于输入 LLM)
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
            
            # 处理 TimeSeries (插入)
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
                if self.training and torch.rand(1).item() < 0.5:
                    stats_str = ""
                else:
                    stats_str = f" [Stats: mean={ts_mean:.2f}, std={ts_std:.2f}, min={ts_min:.2f}, max={ts_max:.2f}] "
                
                if stats_str:
                    stats_encoded = self.tokenizer([stats_str], device)
                    stats_embed = self.llm.embed(stats_encoded["input_ids"])
                    stats_mask = stats_encoded["attention_mask"]
                    segment_embeds.append(stats_embed[0])
                    segment_masks.append(stats_mask[0])
                
                ts_batch = ts_tensor.unsqueeze(0)
                
                # [关键] 传入 instruction_embeds
                ts_tokens = self.ts_model._process_single_channel(
                    ts_batch, instruction_embeds=current_instruction_embeds
                )
                
                if ts_tokens.dtype != target_dtype:
                    ts_tokens = ts_tokens.to(dtype=target_dtype)
                
                ts_embed = ts_tokens[0]
                
                segment_embeds.append(ts_embed)
                segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
                
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

    # 用于 Stage 2 或其他非 ChatTS 格式的 Forward
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
            # 使用 no_grad 避免在这里计算 LLM 的梯度 (只训练 projector)
            # with torch.no_grad():
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