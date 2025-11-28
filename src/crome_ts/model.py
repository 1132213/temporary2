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
    # Stat bypass
    use_stats_projector: bool = True
    stat_fourier_features: int = 64
    stat_hidden_dim: int = 512
    stat_keep_per_channel: bool = False
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
    """
    从LLM模型路径获取embedding维度。
    
    Args:
        llm_model_path: LLM模型路径
        
    Returns:
        embed_dim: 模型的embedding维度
    """
    try:
        config = AutoConfig.from_pretrained(llm_model_path)
        # 不同模型架构可能使用不同的属性名
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
    """
    Reversible Instance Normalization。
    """

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
    """
    标准固定正弦位置编码。直接生成指定长度和维度的 PE。
    """

    def __init__(self, dim: int, scale: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, length: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # 先在 float32 中构建，最后再转换回目标 dtype，避免数值不稳定。
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
    """
    模块 I：输入预处理 + 去量纲 + 时间编码。
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.revin = RevIN(config.epsilon)
        
        self.pos_encoding = FixedSinePositionalEncoding(config.input_channels)
        self.fuse_mode = config.fuse_mode

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, l, c = x.shape
        x_norm, stats = self.revin(x)
        
        # 根据实际输入的通道数动态生成位置编码
        # 如果输入通道数与配置不同（多通道模式下的单通道输入），创建临时编码器
        if c != self.config.input_channels:
            pos_encoding = FixedSinePositionalEncoding(c)
        else:
            pos_encoding = self.pos_encoding
        
        # 生成固定正弦位置编码
        time_emb = pos_encoding(l, device=x.device, dtype=x.dtype)  # [L, c]
        
        # 扩展 Batch 维度
        time_emb = time_emb.unsqueeze(0).expand(b, -1, -1)  # [B, L, c]
        
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
    """
    模块 II：冻结 PatchTST 形态编码器。
    """

    def __init__(self, config: CROMEConfig, input_dim: int):
        super().__init__()
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
        with torch.no_grad():
            emb = self.embedding(x)
            return self.encoder(emb)


class QFormer(nn.Module):
    """
    模块 III 分支 A：Query Former。
    """

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

    def forward(
        self, patch_tokens: Tensor, instruction_embeds: Optional[Tensor] = None
    ) -> Tensor:
        b = patch_tokens.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        key = patch_tokens if instruction_embeds is None else torch.cat(
            [patch_tokens, instruction_embeds], dim=1
        )
        attn_out, _ = self.attn(queries, key, key)
        return self.ln(attn_out)


class DetailProjection(nn.Module):
    """
    模块 III 分支 B：局部细节投影。
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.proj = nn.Linear(config.patch_embedding_dim, config.patch_embedding_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        return self.proj(patch_tokens)


class CROMEAdapterBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        z = F.silu(self.down(x)) * self.gate(x)
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

    def forward(self, query_tokens: Tensor, patch_tokens: Tensor) -> Tensor:
        query_out = self.query_adapter(query_tokens)
        patch_out = self.patch_adapter(patch_tokens)
        return torch.cat([query_out, patch_out], dim=1)


class StatProjector(nn.Module):
    """
    模块 IV：可切换的统计量投影器。
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        if config.use_stats_projector:
            base_dim = 4  # mu, sigma, log1p|mu|, log1p sigma
            self.register_buffer(
                "fourier_matrix",
                torch.randn(config.stat_fourier_features, base_dim) * 0.5,
            )
            in_dim = config.stat_fourier_features * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, config.stat_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.stat_hidden_dim, config.llm_embed_dim),
            )
        else:
            self.linear = nn.Linear(2, config.llm_embed_dim)

    def forward(self, stats: Tensor) -> Tensor:
        if not self.config.use_stats_projector:
            tokens = self.linear(stats)
            return tokens.mean(dim=1, keepdim=True)

        mu = stats[..., 0]
        sigma = stats[..., 1]
        sigma = sigma.clamp_min(self.config.epsilon)
        aug = torch.stack(
            (mu, sigma, torch.log1p(mu.abs()), torch.log1p(sigma)), dim=-1
        )
        mapped = torch.matmul(aug, self.fourier_matrix.t())
        features = torch.cat([torch.sin(mapped), torch.cos(mapped)], dim=-1)
        tokens = self.mlp(features)
        if self.config.stat_keep_per_channel:
            return tokens
        return tokens.mean(dim=1, keepdim=True)


class InstructionTokenizer:
    """
    模块 III：指令流，冻结 Tokenizer。
    """

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
    """
    模块 V：冻结 LLM 推理引擎。
    """

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
    模块 V：拼接至 LLM 嵌入空间。
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
        self.adapter = CROMEAdapter(config)
        self.stat_projector = StatProjector(config)
        self.llm_proj = nn.Linear(config.patch_embedding_dim, config.llm_embed_dim)

    def _process_single_channel(
        self,
        channel_data: Tensor,  # [B, L, 1]
        instruction_embeds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        处理单个通道，返回 stat_token 和 ts_tokens。
        
        Args:
            channel_data: 单个通道的时间序列数据 [B, L, 1]
            instruction_embeds: 可选的指令嵌入
            
        Returns:
            stat_token: 统计量token [B, 1, D]
            ts_tokens: 时序tokens [B, N, D] (N = query_tokens + detail_tokens)
        """
        x, stats = self.preprocessor(channel_data)
        patch_tokens = self.shape_encoder(x)
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        detail_tokens = self.detail_proj(patch_tokens)
        ts_tokens = self.adapter(query_tokens, detail_tokens)
        ts_tokens = self.llm_proj(ts_tokens)
        stat_token = self.stat_projector(stats)
        return stat_token, ts_tokens

    def forward(
        self,
        raw_series: Tensor,  # [B, L, C] 其中C是通道数
        text_prefix: Tensor,
        text_suffix: Tensor,
        instruction_embeds: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        b, l, c = raw_series.shape
        
        # ✨【修复】混合精度训练的关键：强制数据类型对齐
        target_dtype = text_prefix.dtype
        
        x, stats = self.preprocessor(raw_series)
        patch_tokens = self.shape_encoder(x)
        query_tokens = self.qformer(patch_tokens, instruction_embeds)
        detail_tokens = self.detail_proj(patch_tokens)
        ts_tokens = self.adapter(query_tokens, detail_tokens)
        ts_tokens = self.llm_proj(ts_tokens)
        stat_token = self.stat_projector(stats)
        
        # 数据类型对齐
        if ts_tokens.dtype != target_dtype:
            ts_tokens = ts_tokens.to(dtype=target_dtype)
        if stat_token.dtype != target_dtype:
            stat_token = stat_token.to(dtype=target_dtype)
            
        assembled = torch.cat(
            [text_prefix, stat_token, ts_tokens, text_suffix],
            dim=1,
        )
        return {
            "ts_tokens": ts_tokens,
            "stat_token": stat_token,
            "assembled": assembled,
        }


class StatBypassCROMETS1(nn.Module):
    """
    端到端系统封装：形态流 + 量级流 + 指令流 + 冻结 LLM。
    支持 ChatTS 格式：将多个时间序列插入到文本的 <ts><ts/> 标记位置。
    """

    def __init__(self, config: CROMEConfig):
        super().__init__()
        self.config = config
        self.ts_model = CROMETSModel(config)
        self.llm = FrozenLLM(config)
        self.tokenizer = InstructionTokenizer(config)
        
        # SEP token（用于分隔多个时间序列）
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
        input_texts: Sequence[str],  # 包含 <ts><ts/> 标记的文本列表
        timeseries_lists: Sequence[Sequence[Tensor]],  # 每个样本的时间序列列表
        output_texts: Sequence[str],  # 输出文本列表
        llm_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        """
        处理 ChatTS 格式的输入。
        
        将 input_texts 中的 <ts><ts/> 标记替换为对应的时间序列 embeddings。
        每个时间序列被处理为：[Stat_i][Q_Tokens_i][Detail_Tokens_i][SEP]
        
        Args:
            input_texts: 包含 <ts><ts/> 标记的输入文本列表
            timeseries_lists: 每个样本对应的时间序列列表，每个时间序列为 [T, C] 的 Tensor
            output_texts: 输出文本列表
            llm_kwargs: 传递给 LLM 的额外参数
            
        Returns:
            包含模型输出的字典
        """
        device = next(self.parameters()).device
        llm_kwargs = llm_kwargs or {}
        batch_size = len(input_texts)
        
        # 用于存储每个样本的最终嵌入和mask
        assembled_embeds_list = []
        attention_masks_list = []
        prefix_mask_lengths = []
        suffix_mask_lengths = []
        
        # 处理每个样本
        for i in range(batch_size):
            input_text = input_texts[i]
            timeseries_list = timeseries_lists[i]
            output_text = output_texts[i]
            
            # 分割输入文本：找到所有 <ts><ts/> 标记并分割
            ts_marker = "<ts><ts/>"
            text_parts = input_text.split(ts_marker)
            
            # 确保时间序列数量与标记数量匹配
            num_markers = len(text_parts) - 1
            num_timeseries = len(timeseries_list)
            
            if num_timeseries < num_markers:
                # 如果时间序列不够，用零填充
                for _ in range(num_markers - num_timeseries):
                    timeseries_list.append(
                        torch.zeros(self.config.input_channels, device=device)
                    )
            elif num_timeseries > num_markers:
                # 如果时间序列太多，只使用前 num_markers 个
                timeseries_list = timeseries_list[:num_markers]
            
            # 收集所有片段的嵌入
            segment_embeds = []
            segment_masks = []
            
            # 确定目标 dtype（从 LLM 的配置获取）
            # 使用 bfloat16 或模型的实际 dtype
            target_dtype = next(self.llm.parameters()).dtype
            
            # 处理第一个文本片段（prefix）
            if text_parts[0]:
                prefix_encoded = self.tokenizer([text_parts[0]], device)
                prefix_embed = self.llm.embed(prefix_encoded["input_ids"])  # [1, L, D]
                prefix_mask = prefix_encoded["attention_mask"]  # [1, L]
                segment_embeds.append(prefix_embed[0])  # [L, D]
                segment_masks.append(prefix_mask[0])  # [L]
                prefix_length = prefix_mask[0].sum().item()
            else:
                prefix_length = 0
            
            prefix_mask_lengths.append(prefix_length)
            
            # 处理每个时间序列和后续文本片段
            for ts_idx, ts_tensor in enumerate(timeseries_list):
                # 确保时间序列在正确的设备上
                ts_tensor = ts_tensor.to(device)
                
                # 添加 batch 维度：[T, C] -> [1, T, C]
                ts_batch = ts_tensor.unsqueeze(0)
                
                # 生成时间序列的嵌入
                # 使用 _process_single_channel 方法
                stat_token, ts_tokens = self.ts_model._process_single_channel(
                    ts_batch, instruction_embeds=None
                )
                # stat_token: [1, 1, D], ts_tokens: [1, N, D]
                
                # 确保数据类型对齐
                if stat_token.dtype != target_dtype:
                    stat_token = stat_token.to(dtype=target_dtype)
                if ts_tokens.dtype != target_dtype:
                    ts_tokens = ts_tokens.to(dtype=target_dtype)
                
                # 组装：[Stat][Q_Tokens][Detail_Tokens]
                ts_embed = torch.cat([stat_token[0], ts_tokens[0]], dim=0)  # [1+N, D]
                
                # 添加到片段列表
                segment_embeds.append(ts_embed)
                segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
                
                # 添加 SEP token（如果不是最后一个时间序列）
                if ts_idx < len(timeseries_list) - 1:
                    sep_embed = self.sep_token  # [1, D]
                    # 确保 dtype 对齐
                    if sep_embed.dtype != target_dtype:
                        sep_embed = sep_embed.to(dtype=target_dtype)
                    # 保持 2D 形状 [1, D]，与其他 embeddings 的形状一致
                    segment_embeds.append(sep_embed)  # [1, D]
                    segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
                
                # 处理该时间序列后的文本片段
                text_idx = ts_idx + 1
                if text_idx < len(text_parts) and text_parts[text_idx]:
                    text_encoded = self.tokenizer([text_parts[text_idx]], device)
                    text_embed = self.llm.embed(text_encoded["input_ids"])  # [1, L, D]
                    text_mask = text_encoded["attention_mask"]  # [1, L]
                    segment_embeds.append(text_embed[0])  # [L, D]
                    segment_masks.append(text_mask[0])  # [L]
            
            # 处理输出文本（suffix）
            if output_text:
                suffix_encoded = self.tokenizer([output_text], device)
                suffix_embed = self.llm.embed(suffix_encoded["input_ids"])  # [1, L, D]
                suffix_mask = suffix_encoded["attention_mask"]  # [1, L]
                segment_embeds.append(suffix_embed[0])  # [L, D]
                segment_masks.append(suffix_mask[0])  # [L]
                suffix_length = suffix_mask[0].sum().item()
            else:
                suffix_length = 0
            
            suffix_mask_lengths.append(suffix_length)
            
            # 合并所有片段
            full_embed = torch.cat(segment_embeds, dim=0)  # [Total_L, D]
            full_mask = torch.cat(segment_masks, dim=0)  # [Total_L]
            
            assembled_embeds_list.append(full_embed)
            attention_masks_list.append(full_mask)
        
        # 对齐批次（padding）
        max_len = max(emb.shape[0] for emb in assembled_embeds_list)
        embed_dim = assembled_embeds_list[0].shape[1]
        
        padded_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=assembled_embeds_list[0].dtype)
        padded_masks = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        
        for i, (emb, mask) in enumerate(zip(assembled_embeds_list, attention_masks_list)):
            seq_len = emb.shape[0]
            padded_embeds[i, :seq_len] = emb
            padded_masks[i, :seq_len] = mask
        
        # 通过 LLM
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
        
        stat_tokens = ts_outputs["stat_token"]
        ts_tokens = ts_outputs["ts_tokens"]
        ones = torch.ones(
            batch,
            stat_tokens.size(1),
            device=device,
            dtype=prefix_mask.dtype,
        )
        ts_mask = torch.ones(
            batch,
            ts_tokens.size(1),
            device=device,
            dtype=prefix_mask.dtype,
        )
        attention_mask = torch.cat(
            [prefix_mask, ones, ts_mask, suffix_mask],
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
            # ✨【修复】返回 Prefix Mask 和 Suffix Mask
            # 供外部训练脚本计算 Loss 偏移量使用
            "prefix_mask": prefix_mask,
            "suffix_mask": suffix_mask,
        }