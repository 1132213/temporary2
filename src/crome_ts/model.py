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
        # self.pos_encoding = FixedSinePositionalEncoding(config.input_channels)
        # self.fuse_mode = config.fuse_mode

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, l, c = x.shape
        x_norm, stats = self.revin(x)
        return x_norm,stats
        # if c != self.config.input_channels:
        #     pos_encoding = FixedSinePositionalEncoding(c)
        # else:
        #     pos_encoding = self.pos_encoding
        
        # time_emb = pos_encoding(l, device=x.device, dtype=x.dtype)
        # time_emb = time_emb.unsqueeze(0).expand(b, -1, -1)
        
        # if self.fuse_mode == "add":
        #     fused = x_norm + time_emb
        # elif self.fuse_mode == "concat":
        #     fused = torch.cat([x_norm, time_emb], dim=-1)
        # else:
        #     raise ValueError(f"未知融合模式: {self.fuse_mode}")
            
        # return fused, stats


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
        self.pos_encoding = FixedSinePositionalEncoding(
            dim=config.patch_embedding_dim, 
            scale=10000.0
        )
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [Batch, Seq_Len, Channels]
        
        # 1. Patchify & Project
        # emb: [Batch, Num_Patches, Patch_Dim]
        emb = self.embedding(x) 
        
        # 2. [核心修改] 动态生成 Patch 级位置编码
        # 获取当前的 patch 数量
        b, num_patches, d = emb.shape
        
        # 生成对应的 PE: [Num_Patches, Patch_Dim]
        # 注意：这里传入的是 num_patches，代表“第几个Patch”，而不是“第几秒”
        pe = self.pos_encoding(num_patches, device=x.device, dtype=emb.dtype)
        
        # 广播并相加: [1, N, D] + [B, N, D]
        emb = emb + pe.unsqueeze(0)

        if self.config.freeze_patch_encoder:
            with torch.no_grad():
                enc_out = self.encoder(emb)
                return emb, enc_out 
        else:
            enc_out = self.encoder(emb)
            return emb, enc_out 


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
        
        dropout_rate = getattr(config, "dropout", 0.1)
        
        # 1. 可学习的 Query Tokens (Base Queries)
        self.query_tokens = nn.Parameter(
            torch.randn(config.query_tokens, config.patch_embedding_dim)
        )
        
        # 2. 文本交互层 (Text Interaction)
        self.layernorm_text_input = nn.LayerNorm(config.llm_embed_dim) # 注意维度是 llm_embed_dim
        self.text_proj = nn.Linear(config.llm_embed_dim, config.patch_embedding_dim)
        # Cross-Attention: Query 关注 Text
        self.text_attn = nn.MultiheadAttention(
            embed_dim=config.patch_embedding_dim,
            num_heads=config.patch_num_heads,
            dropout=dropout_rate, 
            batch_first=True
        )
        self.ln_text = nn.LayerNorm(config.patch_embedding_dim)

        # 3. 时序提取层 (Time-Series Extraction)
        # Cross-Attention: Query 关注 TS Patch
        self.ln_ts_input = nn.LayerNorm(config.patch_embedding_dim)
        self.ts_attn = nn.MultiheadAttention(
            embed_dim=config.patch_embedding_dim,
            num_heads=config.patch_num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.ln_ts = nn.LayerNorm(config.patch_embedding_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        # nn.init.zeros_(self.text_proj.weight)
        # nn.init.zeros_(self.text_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        if self.text_proj.bias is not None:
            nn.init.zeros_(self.text_proj.bias)

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
            # text_kv = self.ln_text_input(text_kv)
            
            # (C) Cross-Attention: Q=Queries, K=Text, V=Text
            text_out, attn_weights = self.text_attn(
                query=queries,
                key=text_kv,
                value=text_kv
            )
            
            # 保存权重供可视化 (Detach + CPU 以节省显存)
            # 形状通常为 [Batch, Queries, Text_Len] (若 average_attn_weights=True)
            self.last_text_attn_weights = attn_weights.detach().cpu()
            
            # (D) 残差 + Norm + Dropout
            # 注意：这里采用了 Pre-Norm 或 Post-Norm 结构均可，这里维持原有的 Post-Norm 逻辑
            # Query = Norm(Query + Dropout(Attn(Q, Text)))
            queries = self.ln_text(queries + self.dropout(text_out))
        
        # 3. [阶段二] 时序提取：带着意图的 Query 提取时序特征
        # Q=Queries(Text-Aware), K=TS_Patch, V=TS_Patch
        patch_tokens_norm = self.ln_ts_input(patch_tokens)
        ts_out, _ = self.ts_attn(
            query=queries, 
            key=patch_tokens_norm, 
            value=patch_tokens
        )
        
        # 输出前应用 Dropout
        return self.ln_ts(self.dropout(ts_out))


class DetailProjection(nn.Module):
    """
    模块 III 分支 B：局部细节投影 (3-Layer MLP)。
    用于将原始的 Shallow Patch 特征映射到 LLM 语义空间。
    结构：Dim -> 4*Dim -> 4*Dim -> Dim
    """
    def __init__(self, config: CROMEConfig):
        super().__init__()
        
        # 输入和输出维度都是 patch_embedding_dim
        dim = config.patch_embedding_dim
        hidden_dim = dim * 4  # 中间层扩维以增加表达能力
        drop_rate = config.proj_dropout
        # 3层 MLP 结构
        self.proj = nn.Sequential(
            # 第一层：升维 + 激活
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            
            # # 第二层：特征变换 + 激活 (深度的来源)
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.GELU(),
            
            # 第三层：降维回目标空间
            nn.Linear(hidden_dim, dim)
        )
        
        # 初始化策略 (可选，保持默认通常也可以，但 Xavier/Kaiming 有助于深层网络)
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
        sep_embed: Optional[Tensor] = None, # [新增]
        gamma: Optional[Tensor] = None, 
        beta: Optional[Tensor] = None
    ) -> Tensor:
        query_out = self.query_adapter(query_tokens)
        patch_out = self.patch_adapter(patch_tokens, gamma=gamma, beta=beta)
        
        # [修改] 插入分隔符: [Query, Sep, Detail]
        if sep_embed is not None:
            # sep_embed shape: [1, 1, Dim] -> 扩展到 [B, 1, Dim]
            b = query_tokens.size(0)
            sep = sep_embed.expand(b, -1, -1)
            return torch.cat([query_out, sep, patch_out], dim=1)
        else:
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

        self.decomp = SeriesDecomp(kernel_size=25)
        self.resid_patch_len = 4  # 使用更小的 Patch
        self.resid_stride = 4     # 配合 Stride=4 或 2
        
        # 独立的线性层，随机初始化
        self.resid_embedding = nn.Linear(
            self.resid_patch_len * config.input_channels, 
            config.patch_embedding_dim
        )
        nn.init.xavier_uniform_(self.resid_embedding.weight)
        
        # self.llm_proj = nn.Linear(config.patch_embedding_dim, config.llm_embed_dim)
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
        sep_embed: Optional[Tensor] = None, # [新增]
    ) -> Tensor:
        x, stats = self.preprocessor(channel_data)
        gamma, beta = self.film_generator(stats)
        # 1. 分解
        x_resid, x_trend = self.decomp(x)
        
        # 2. 主流 (Query): 看趋势 (Patch=16)
        raw_embeds_global, deep_feats = self.shape_encoder(x) # 还是喂原始x效果最好
        query_tokens = self.qformer(deep_feats, instruction_embeds)
        
        # 3. 辅流 (Detail): 看残差 (Patch=4)
        # 手动 Patching
        # [B, L, C] -> [B, N, P, C]
        resid_patches = x_resid.unfold(dimension=1, size=self.resid_patch_len, step=self.resid_stride)
        b, n, p, c = resid_patches.shape
        resid_patches = resid_patches.contiguous().view(b, n, -1)
        
        # 使用独立 Embedding 层
        resid_embeds = self.resid_embedding(resid_patches) # [B, N_resid, 512]
        
        # 送入 Detail Projector
        detail_tokens = self.detail_proj(resid_embeds)
        
        # 4. 融合
        # 注意: 此时 detail_tokens 的长度 (N_resid) 是 query_tokens (32) 或原 patch (N_main) 的 4 倍
        # Adapter 和 LLM 可以处理变长序列，直接拼接即可
        ts_tokens = self.adapter(
            query_tokens, detail_tokens, 
            sep_embed=sep_embed, 
            gamma=gamma, beta=beta
        )
        ts_tokens = self.llm_proj(ts_tokens)
        
        return ts_tokens
    # def _process_single_channel(
    #     self,
    #     channel_data: Tensor,
    #     instruction_embeds: Optional[Tensor] = None,
    #     sep_embed: Optional[Tensor] = None, # [新增]
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
        
        raw_embeds, deep_feats = self.shape_encoder(x)
        
        query_tokens = self.qformer(deep_feats, instruction_embeds)
        detail_tokens = self.detail_proj(raw_embeds)
        
        # 通用 forward 暂时不处理 sep_embed，或者也可以加上
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
        
        # [新增] 定义模态特殊标记 (随机初始化)
        # 使用 nn.Parameter 确保在 SFT/LoRA 时能被优化器捕获
        self.ts_start_token = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        self.ts_end_token   = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
        # self.feat_sep_token = nn.Parameter(torch.randn(1, 1, config.llm_embed_dim) * 0.02)
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
        # 新增 Mask 参数用于 Ablation
        mask_query: bool = False,
        mask_detail: bool = False,
        mask_text_stats: bool = False
    ) -> Dict[str, Any]:
        """
        核心逻辑封装：统一构建训练和推理用的 Multimodal Embeddings。
        拼接顺序：[Start] -> [Stats] -> [TS Features] -> [End] (方案 B)
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
                if input_ids.shape[1] > 512:
                     input_ids = input_ids[:, :512]
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
                    instruction_embeds=current_instruction_embeds,
                    sep_embed=self.feat_sep_token
                )
                
                # [新增] Ablation Masking Logic
                if mask_query or mask_detail:
                    num_q = self.config.query_tokens
                    # ts_tokens 结构: [Batch=1, Num_Tokens, Dim]
                    # 结构顺序: [Query(0:num_q), Sep(num_q), Detail(num_q+1:)]
                    
                    if mask_query:
                        ts_tokens[:, :num_q, :] = 0.0
                    
                    if mask_detail:
                        # 如果有 sep_token，则 Detail 从 num_q + 1 开始
                        # 当前实现中 self.feat_sep_token 始终初始化，故有 Sep
                        start_idx = num_q + 1
                        ts_tokens[:, start_idx:, :] = 0.0

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