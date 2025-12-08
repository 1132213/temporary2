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
sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim

def visualize_text_attention(
    jsonl_path, 
    checkpoint_path, 
    llm_model_path, 
    sample_idx=None,
    save_path="attn_top20_heatmap.png"
):
    # --- 2. 加载配置与模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")
    
    llm_embed_dim = get_llm_embed_dim(llm_model_path)
    
    config = CROMEConfig(
        input_channels=1,
        llm_embed_dim=llm_embed_dim,
        patch_len=16,
        patch_stride=8,
        llm_model_path=llm_model_path,
        llm_device_map="cpu", # 可视化通常不需要太快，放 CPU 省显存
        llm_dtype="bfloat16"
    )
    
    print(">>> Loading model...")
    model = StatBypassCROMETS1(config)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(">>> Checkpoint loaded.")
    else:
        print(f"!!! Checkpoint {checkpoint_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()

    # --- 3. 读取数据 ---
    print(f">>> Loading data from {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if sample_idx is None:
        sample_idx = random.randint(0, len(lines) - 1)
    
    raw_item = json.loads(lines[sample_idx])
    input_text = raw_item.get("input", "")
    
    print(f">>> Selected Sample ID: {sample_idx}")

    # --- 4. 构造完整文本 (核心步骤) ---
    # 逻辑必须与 model.py 中的 forward_chatts 完全一致
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    
    # 过滤空串并拼接
    valid_parts = [p.strip() for p in text_parts if p.strip()]
    full_instruction_text = " ".join(valid_parts)
    while "\n\n" in full_instruction_text:
        full_instruction_text = full_instruction_text.replace("\n\n", "\n")
    # [修改点] 打印完整拼接文本
    print("\n" + "="*60)
    print(f">>> ACTUAL MODEL INPUT TEXT (Spliced)")
    print("="*60)
    print(full_instruction_text)
    print("="*60 + "\n")
    
    if not full_instruction_text:
        print("!!! No text instruction found after splicing.")
        return

    # Tokenize
    tokenizer = model.tokenizer.tokenizer
    tokenized = tokenizer(full_instruction_text, return_tensors="pt")
    input_ids = tokenized.input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # 清洗 Token 显示
    readable_tokens = [t.replace('Ġ', '').replace(' ', ' ').replace(' ', '') for t in tokens]

    # --- 5. 运行推理 ---
    with torch.no_grad():
        input_ids = input_ids.to(device)
        instr_embeds = model.llm.embed(input_ids) 
        
        # 构造虚拟 Patch (时序部分不影响 Text Attention)
        dummy_patch_tokens = torch.randn(1, 64, 512).to(device)
        
        # 运行 QFormer
        model.ts_model.qformer(dummy_patch_tokens, instruction_embeds=instr_embeds)
        
        if not hasattr(model.ts_model.qformer, "last_text_attn_weights") or \
           model.ts_model.qformer.last_text_attn_weights is None:
            print("!!! Error: 'last_text_attn_weights' is None.")
            return
            
        # 获取权重: [1, 32, Text_Len]
        attn_weights = model.ts_model.qformer.last_text_attn_weights 
        attn_map = attn_weights[0].float().numpy() # [32, Text_Len]

    # --- 6. 计算 Top-20 Token ---
    token_avg_attn = np.mean(attn_map, axis=0) # [Text_Len]

    
    # # 排序
    
    token_max_attn = np.max(attn_map, axis=0) 

    df_tokens = pd.DataFrame({
        'token': readable_tokens,
        'avg_weight': token_avg_attn,
        'max_weight': token_max_attn, # 新增
        'original_idx': range(len(readable_tokens))
    })
    
    # 按 max_weight 排序看看 "3" 是否由特定的 Query 负责
    top_20_df = df_tokens.sort_values(by='max_weight', ascending=False).head(20)
    print(f"{'Rank':<5} | {'Token':<20} | {'Max Weight':<10}")
    print("-" * 45)
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        print(f"{i+1:<5} | {row['token']:<20} | {row['max_weight']:.4f}")
    print("-" * 45 + "\n")
    

    # --- 7. 绘图 ---
    print(f">>> Plotting Heatmap...")
    
    top_20_indices = top_20_df['original_idx'].values
    top_20_tokens = top_20_df['token'].values
    
    # 提取对应的列
    filtered_attn_map = attn_map[:, top_20_indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        filtered_attn_map, 
        xticklabels=top_20_tokens, 
        yticklabels=[f"Q{i}" for i in range(32)],
        cmap="viridis",
        annot=False,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(f"Q-Former Attention Top-20 Tokens\n(Sample {sample_idx})")
    plt.xlabel("Top Attended Tokens (Sorted by Avg Weight)")
    plt.ylabel("Learned Query Tokens")
    plt.xticks(rotation=45, ha='right', fontsize=11, weight='bold')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    save_full_path = str(project_root / save_path)
    plt.savefig(save_full_path, dpi=300)
    print(f">>> Saved Top-20 heatmap to {save_full_path}")



    top_20_df = df_tokens.sort_values(by='avg_weight', ascending=False).head(20)
    print(f"{'Rank':<5} | {'Token':<20} | {'Avg Weight':<10}")
    print("-" * 45)
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        print(f"{i+1:<5} | {row['token']:<20} | {row['avg_weight']:.4f}")
    print("-" * 45 + "\n")
    print(f">>> Plotting Heatmap...")
    
    top_20_indices = top_20_df['original_idx'].values
    top_20_tokens = top_20_df['token'].values
    
    # 提取对应的列
    filtered_attn_map = attn_map[:, top_20_indices]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        filtered_attn_map, 
        xticklabels=top_20_tokens, 
        yticklabels=[f"Q{i}" for i in range(32)],
        cmap="viridis",
        annot=False,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(f"Q-Former Attention Top-20 Tokens\n(Sample {sample_idx})")
    plt.xlabel("Top Attended Tokens (Sorted by Avg Weight)")
    plt.ylabel("Learned Query Tokens")
    plt.xticks(rotation=45, ha='right', fontsize=11, weight='bold')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    save_full_path = "/mnt/shared-storage-user/huaermo/code/test_wbt/temporary2/attn_avg_top20_heatmap.png"
    plt.savefig(save_full_path, dpi=300)
    print(f">>> Saved Top-20 heatmap to {save_full_path}")

if __name__ == "__main__":
    # 配置你的路径
    # JSONL_PATH = "/mnt/shared-storage-user/huaermo/code/test_wbt2/alignment.jsonl"
    JSONL_PATH="/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.jsonl"
    CHECKPOINT = "/mnt/shared-storage-user/huaermo/code/test_wbt2/temporary2/model/aligned_8b_tattn_16_skip_1207_stride8.pth" # 确保这是你 Text-Guided 训练后的模型
    # LLM_PATH = "/mnt/shared-storage-user/dllm-share/Models/Qwen2.5-3B-Instruct"
    LLM_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"

    
    visualize_text_attention(
        JSONL_PATH, 
        CHECKPOINT, 
        LLM_PATH,
        sample_idx=None
    )