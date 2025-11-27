import sys
from pathlib import Path

# 路径设置
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import argparse
import torch
from torch.utils.data import DataLoader
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn

def generate_response(model, series, prefix, device, max_new_tokens=256):
    """
    生成回复
    """
    # 1. 处理 Prefix
    # 很多 Llama 模型没有 pad token，设为 eos 或 unk
    tokenizer = model.tokenizer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    prefix_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    prefix_embeds = model.llm.embed(prefix_inputs.input_ids)
    
    # 2. 构造空的 Suffix (推理阶段没有 Suffix)
    empty_suffix = torch.empty(
        series.size(0), 0, model.config.llm_embed_dim, 
        device=device, dtype=prefix_embeds.dtype
    )
    
    with torch.no_grad():
        # 3. 获取时序特征 + Prefix 特征
        ts_out = model.ts_model(series, prefix_embeds, empty_suffix)
        inputs_embeds = ts_out["assembled"] # [B, Seq_Len, D]
        
        # 4. 构建 Attention Mask (全 1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], 
            dtype=torch.long, 
            device=device
        )
        
        # 5. 调用 LLM Generate
        output_ids = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False, # 确定性生成，方便调试
            temperature=0.1,
            repetition_penalty=1.1
        )
        
    # 6. 解码
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main(args):
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None and ":" not in args.device:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    
    print(f">>> Loading Model from {args.checkpoint}...")
    
    # 1. 先加载数据集以自动检测通道数
    print(f">>> Loading Dataset to auto-detect channels: {args.jsonl_path}...")
    # 先创建一个临时数据集来检测通道数
    temp_ds = JSONLInstructDataset(
        args.jsonl_path, args.seq_len, 
        input_channels=getattr(args, 'input_channels', None),  # None表示自动检测
        split="val"
    )
    detected_channels = temp_ds.input_channels
    print(f">>> Auto-detected input channels: {detected_channels}")
    
    # 使用检测到的通道数（如果用户指定了则使用指定的）
    input_channels = getattr(args, 'input_channels', None) or detected_channels
    
    # 2. 配置 (需与训练一致)
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    print(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16",
        use_stats_projector=True,
        epsilon=1e-5,
    )
    
    # 3. 初始化模型
    model = StatBypassCROMETS1(config).to(device)
    
    # 4. 加载权重 (Full Weights)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=True) # 全量加载，必须严格匹配
    model.eval()
    print(">>> Model Loaded Successfully.")
    
    # 5. 正式加载验证数据
    val_ds = JSONLInstructDataset(
        args.jsonl_path, args.seq_len, input_channels, 
        split="val"
    )
    # batch_size=1 方便观察
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=instruct_collate_fn)
    
    print(f"\n{'='*60}")
    print(f"Starting Inference on {args.num_samples} samples...")
    print(f"{'='*60}\n")
    
    for i, (series, prefixes, suffixes) in enumerate(val_loader):
        if i >= args.num_samples:
            break
            
        series = series.to(device)
        prefix = prefixes[0]
        ground_truth = suffixes[0]
        
        # 去掉 Prompt 中的 "Assistant: " 以便打印看起来整洁，
        # 但传给模型的 prefix 必须保留完整格式 (train_instruct.py 里已处理好)
        
        try:
            prediction = generate_response(model, series, prefix, device)
        except Exception as e:
            print(f"[Error] Sample {i}: {e}")
            continue
            
        print(f"Sample #{i+1}")
        print(f"--- Input Prompt ---\n{prefix.strip()}")
        print(f"--- Ground Truth ---\n{ground_truth.strip()}")
        print(f"--- Model Prediction ---\n{prediction.strip()}")
        print(f"{'-'*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to crome_instruct_best.pth")
    
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=None, help="输入通道数（多通道数据中的通道数）。如果未指定，将从数据中自动检测")
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None)
    
    args = parser.parse_args()
    main(args)