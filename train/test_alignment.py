import sys
from pathlib import Path

# 路径设置：确保能找到 src
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import argparse
import torch
from torch.utils.data import DataLoader
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn

def generate_response(model, series, prefix, device, max_new_tokens=128):
    """
    执行推理生成
    """
    tokenizer = model.tokenizer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 1. 准备 Prompt
    prefix_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    prefix_embeds = model.llm.embed(prefix_inputs.input_ids)
    
    # 2. 空 Suffix (让模型接着续写)
    empty_suffix = torch.empty(
        series.size(0), 0, model.config.llm_embed_dim, 
        device=device, dtype=prefix_embeds.dtype
    )
    
    with torch.no_grad():
        # 3. 获取对齐后的特征 (TS -> LLM Embeds)
        ts_out = model.ts_model(series, prefix_embeds, empty_suffix)
        inputs_embeds = ts_out["assembled"] 
        
        # 4. 构造 Attention Mask
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], 
            dtype=torch.long, 
            device=device
        )
        
        # 5. 生成
        output_ids = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,      # 采样模式，生成的描述更自然
            temperature=0.7,     # 控制多样性
            top_p=0.9,
            repetition_penalty=1.1
        )
        
    # 6. 解码
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

def main(args):
    # 设备处理
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    
    print(f">>> Loading Config & Model...")
    
    # 1. 配置 (需与 train_alignment.py 保持一致)
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    print(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16",
        use_stats_projector=True,
        epsilon=1e-5
    )
    
    # 2. 初始化模型 (此时加载了原始 LLM 权重)
    model = StatBypassCROMETS1(config).to(device)
    
    # 3. 加载 Stage 2 训练好的对齐权重
    print(f">>> Loading Stage 2 Checkpoint from {args.checkpoint}...")
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 关键：strict=False
        # 因为 checkpoint 里只有 Adapter/Projector/Encoder，没有 LLM 权重
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        
        # 验证是否加载成功 (忽略 LLM 相关的 missing keys)
        real_missing = [k for k in missing if "llm.model" not in k]
        if len(real_missing) > 0:
            print(f"!!! Warning: Potential missing keys: {real_missing[:5]} ...")
        else:
            print(">>> Stage 2 Weights merged successfully (LLM weights skipped as expected).")
    else:
        print(f"!!! Error: Checkpoint {args.checkpoint} not found!")
        return

    model.eval()
    
    # 4. 加载验证数据
    print(f">>> Loading Validation Dataset...")
    val_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=instruct_collate_fn)
    
    print(f"\n{'='*60}")
    print(f"Starting Stage 2 Alignment Test on {args.num_samples} samples")
    print(f"{'='*60}\n")
    
    for i, (series, prefixes, suffixes) in enumerate(val_loader):
        if i >= args.num_samples:
            break
            
        series = series.to(device)
        prefix = prefixes[0]
        ground_truth = suffixes[0]
        
        try:
            # 生成
            prediction = generate_response(model, series, prefix, device)
            
            print(f"Sample #{i+1}")
            print(f"--- Prompt ---\n{prefix.strip()}")
            print(f"--- Ground Truth ---\n{ground_truth.strip()}")
            print(f"--- Stage 2 Output ---\n{prediction.strip()}")
            print(f"{'-'*60}\n")
            
        except Exception as e:
            print(f"[Error] Sample {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    # 默认加载 Stage 2 的权重文件
    parser.add_argument("--checkpoint", type=str, default="crome_stage2_aligned.pth")
    
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    # 确保这里指向您的 LLM 路径
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None)
    
    args = parser.parse_args()
    main(args)