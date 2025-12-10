import sys
import os
import json
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# --- 1. 环境设置 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim

def visualize_full_attention(
    jsonl_path, 
    checkpoint_path, 
    llm_model_path, 
    sample_idx=None,
    save_dir="vis_output"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    
    # --- 2. 加载模型 ---
    llm_embed_dim = get_llm_embed_dim(llm_model_path)
    config = CROMEConfig(
        input_channels=1,
        llm_embed_dim=llm_embed_dim,
        patch_len=16,
        patch_stride=8, # 确保与训练一致
        llm_model_path=llm_model_path,
        llm_device_map="cpu", 
        llm_dtype="bfloat16"
    )
    
    print(">>> Loading model...")
    model = StatBypassCROMETS1(config)
    
    if os.path.exists(checkpoint_path):
        # 修复权重键名 (如果需要)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(">>> Checkpoint loaded.")
    else:
        print(f"!!! Checkpoint {checkpoint_path} not found.")
        return
    
    model.ts_model.to(device) # 只把 TS 部分放 GPU 即可
    model.ts_model.eval()

    # --- 3. 读取数据 ---
    print(f">>> Loading data from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(lines) - 1)
    
    raw_item = json.loads(lines[sample_idx])
    input_text = raw_item.get("input", "")
    timeseries_list = raw_item.get("timeseries", [])
    
    print(f">>> Selected Sample ID: {sample_idx}")
    print(f">>> Number of Time Series: {len(timeseries_list)}")

    # --- 4. 准备输入数据 ---
    
    # A. 文本处理 (拼接)
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    valid_parts = [p.strip() for p in text_parts if p.strip()]
    full_instruction_text = " ".join(valid_parts)
    
    print(f">>> Instruction: {full_instruction_text[:100]}...")

    tokenizer = model.tokenizer.tokenizer
    tokenized = tokenizer(full_instruction_text, return_tensors="pt")
    input_ids = tokenized.input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    readable_tokens = [t.replace('Ġ', '').replace(' ', ' ').strip() for t in tokens]

    # B. 时序处理 (取第一条或全部)
    # 为了简化可视化，我们这里只取第一条时序进行分析
    # 如果想看多条，可以在这里循环
    target_ts_idx = 0 
    ts_data = timeseries_list[target_ts_idx]
    ts_array = np.array(ts_data, dtype=np.float32).flatten()
    
    # 转换为 Tensor [1, L, 1]
    ts_tensor = torch.tensor(ts_array).unsqueeze(0).unsqueeze(-1).to(device)

    # --- 5. 运行推理 (模拟 Forward 过程) ---
    with torch.no_grad():
        # A. 文本 Embedding
        # 注意：这里需要暂时把 LLM Embedding 层放到 device
        model.llm.model.to(device) 
        input_ids = input_ids.to(device)
        instr_embeds = model.llm.embed(input_ids) 
        
        # 类型转换 (BFloat16 -> Float32 适配 QFormer)
        instr_embeds = instr_embeds.to(dtype=torch.float32)

        # B. 时序编码 (Shape Encoder)
        # 1. 预处理
        x_norm, stats = model.ts_model.preprocessor(ts_tensor)
        # 2. Encoder 提取特征
        _, deep_feats = model.ts_model.shape_encoder(x_norm)
        
        # C. Q-Former 推理
        # 这会自动触发多层交互，并保存最后一层的权重
        model.ts_model.qformer(deep_feats, instruction_embeds=instr_embeds)
        
        # --- 6. 获取注意力权重 ---
        # Q-Former 最后一层 (layers[-1])
        last_layer = model.ts_model.qformer.layers[-1]
        
        # (1) Text Attention: [Batch, Num_Query, Text_Len]
        text_attn = last_layer.last_text_attn_weights
        # (2) TS Attention: [Batch, Num_Query, Num_Patch]
        ts_attn = last_layer.last_ts_attn_weights
        
        if text_attn is None or ts_attn is None:
            print("!!! Error: Attention weights not captured. Did you replace QFormer code correctly?")
            return

        text_attn_map = text_attn[0].float().cpu().numpy() # [32, Text_Len]
        ts_attn_map = ts_attn[0].float().cpu().numpy()     # [32, Num_Patch]

    # --- 7. 绘图 1: Text Attention (哪些词最重要) ---
    plt.figure(figsize=(14, 8))
    
    # 筛选 Top-20 词
    token_avg = np.mean(text_attn_map, axis=0)
    top_indices = np.argsort(token_avg)[-25:][::-1] # Top 25
    top_tokens = [readable_tokens[i] for i in top_indices]
    filtered_map = text_attn_map[:, top_indices]
    
    sns.heatmap(
        filtered_map,
        xticklabels=top_tokens,
        yticklabels=[f"Q{i}" for i in range(32)],
        cmap="Blues",
        annot=False
    )
    plt.title(f"Text Attention (Instruction Guidance) - Sample {sample_idx}")
    plt.xlabel("Top Attended Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{sample_idx}_text_attn.png", dpi=300)
    print(f">>> Saved Text Attention to {save_dir}/sample_{sample_idx}_text_attn.png")

    # --- 8. 绘图 2: TS Attention (关注哪些时间段) ---
    plt.figure(figsize=(14, 6))
    
    # 绘制原始波形作为背景参考 (归一化到 0-32 范围以便对比?) 
    # 不，直接画两个子图最好
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(ts_array, label="Original Time Series", color='black', alpha=0.6)
    ax1.set_xlim(0, len(ts_array))
    ax1.set_title("Original Time Series")
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # TS Attn Map: [32, Num_Patch] -> 需要扩展回原始长度
    # 简单插值
    import cv2
    resized_map = cv2.resize(ts_attn_map, (len(ts_array), 32), interpolation=cv2.INTER_NEAREST)
    
    sns.heatmap(
        resized_map,
        cmap="Reds",
        cbar_kws={'label': 'Attention'},
        ax=ax2
    )
    ax2.set_ylabel("Query Tokens")
    ax2.set_xlabel("Time Steps")
    ax2.set_title("Time Series Attention (Which part is focused?)")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_{sample_idx}_ts_attn.png", dpi=300)
    print(f">>> Saved TS Attention to {save_dir}/sample_{sample_idx}_ts_attn.png")

if __name__ == "__main__":
    # 配置
    JSONL_PATH = "/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.jsonl"
    # 请替换为您最新训练好的模型路径
    CHECKPOINT = "/mnt/shared-storage-user/huaermo/code/test_wbt2/temporary2/model/aligned_8b_tattn_16_1208_stride8_residual_data_new.pth" 
    LLM_PATH = "/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"
    
    visualize_full_attention(JSONL_PATH, CHECKPOINT, LLM_PATH)