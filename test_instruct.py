import argparse
import torch
from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple

# 引入项目模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn
from test.common import set_seed
为

def _resolve_device(
    device_arg: str,
    gpu_id: Optional[int],
    cuda_visible_devices: Optional[str],
) -> Tuple[torch.device, str]:
    """
    根据命令行参数解析推理所使用的计算设备。
    返回 torch.device 与字符串形式（供配置使用）。
    """
    if cuda_visible_devices:
        # 允许通过命令行直接设置 CUDA_VISIBLE_DEVICES，例如 "0" 或 "0,1"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("检测到 --device=cuda 但当前环境没有可用的 CUDA 设备。")
        device_str = device_arg
        if gpu_id is not None and gpu_id >= 0:
            device_str = f"cuda:{gpu_id}"
        device = torch.device(device_str)
        if device.index is not None:
            torch.cuda.set_device(device.index)
    else:
        device = torch.device(device_arg)
        device_str = device_arg
    return device, device_str


def load_trained_model(args, device, llm_device_str):
    """加载模型配置并恢复权重"""
    print(f">>> Loading Model Config & Weights from {args.checkpoint}...")
    
    # 1. 初始化配置 (必须与训练时一致)
    # 从模型路径动态获取embed_dim
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    print(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=llm_device_str,
        llm_dtype="bfloat16"
    )
    
    # 2. 实例化模型
    model = StatBypassCROMETS1(config).to(device)
    
    # 3. 加载权重
    # 注意：如果训练时只保存了 state_dict，直接加载
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False) # strict=False 以防有一些非关键键不匹配
    model.eval()
    
    return model

def generate_report(model, series, prefix, max_new_tokens=128, temperature=0.7):
    """执行单次推理生成"""
    device = series.device
    tokenizer = model.tokenizer.tokenizer
    
    # ✨【修复 1】解决 Pad==EOS 问题
    # 很多 Llama 模型没有 pad token，默认设为 eos 会导致生成停止或混乱
    # 我们将其临时设为 unk_token (id=0) 或者其他非特殊 token
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    
    # 1. 准备 Prefix Embeddings
    prefix_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    prefix_embeds = model.llm.embed(prefix_inputs.input_ids)
    
    # 2. 构造空的 Suffix
    empty_suffix = torch.empty(
        series.size(0), 0, model.config.llm_embed_dim, 
        device=device, dtype=prefix_embeds.dtype
    )
    
    with torch.no_grad():
        # 3. 获取时序特征
        ts_out = model.ts_model(series, prefix_embeds, empty_suffix)
        inputs_embeds = ts_out["assembled"] # [B, Seq_Len, D]
        input_length = inputs_embeds.shape[1]  # 记录输入长度
        
        # ✨【修复 2】显式构建 Attention Mask
        # inputs_embeds 的形状是 [1, Seq_Len, D]
        # 因为我们是 Batch=1 推理，且没有 Padding，所以 Mask 全为 1
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], 
            dtype=torch.long, 
            device=device
        )
        
        # 4. 调用 LLM Generate
        output_ids = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,  # ✨ 传入 Mask
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
    # 5. 解码：优先尝试获取真正生成的部分，如果失败则回退到全量文本
    generated_text = ""
    if output_ids.shape[1] > input_length:
        generated_ids = output_ids[:, input_length:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    if not generated_text:
        full_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        prefix_stripped = prefix.strip()
        if prefix_stripped and full_text.startswith(prefix_stripped):
            generated_text = full_text[len(prefix_stripped):].strip()
        else:
            generated_text = full_text
    
    return generated_text

def main(args):
    set_seed(args.seed)
    device, llm_device_str = _resolve_device(
        args.device,
        args.gpu_id,
        args.cuda_visible_devices,
    )

    split = args.split
    if split != "val":
        print(">>> INFO: 测试脚本强制使用验证集，已自动将 split 设置为 'val'.")
        split = "val"
    
    # 加载模型
    model = load_trained_model(args, device, llm_device_str)
    
    # 加载验证数据
    print(f">>> Loading Dataset (split={split}) from {args.jsonl_path}...")
    val_ds = JSONLInstructDataset(
        args.jsonl_path,
        args.seq_len,
        args.input_channels,
        split=split,
        split_ratio=args.split_ratio,
    )
    # 这里 batch_size=1 方便观察每一个样本的生成结果
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=instruct_collate_fn)
    
    print(f">>> Starting Inference on {min(len(val_ds), args.num_samples)} samples...\n")
    print("="*60)
    
    for i, (series, prefixes, suffixes) in enumerate(val_loader):
        if i >= args.num_samples:
            break
            
        series = series.to(device)
        prompt = prefixes[0]
        ground_truth = suffixes[0]
        
        # 生成
        try:
            prediction = generate_report(model, series, prompt, max_new_tokens=200)
        except Exception as e:
            print(f"[Error] Generation failed for sample {i}: {e}")
            continue

        # 打印结果
        print(f"Sample #{i+1}")
        print(f"--- Prompt ---\n{prompt.strip()}")
        print(f"--- Ground Truth ---\n{ground_truth}")
        print(f"--- CROME Prediction ---\n{prediction.strip()}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    
    # 模型参数 (必须与训练一致)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    
    # 推理参数
    parser.add_argument("--checkpoint", type=str, default="crome_instruct_best.pth", help="训练好的模型路径")
    parser.add_argument("--num-samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=4, help="选择具体使用的 GPU，留空则使用默认设备")
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="直接设置 CUDA_VISIBLE_DEVICES，例如 '0' 或 '0,1'",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="选择数据拆分，默认使用验证集")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="train/val 划分比例 (train 占比)")
    
    args = parser.parse_args()
    main(args)