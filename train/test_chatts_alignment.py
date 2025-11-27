import sys
import os
from pathlib import Path

# 禁用 tokenizers 并行以避免多进程警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 路径设置：确保能找到 src
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import argparse
import torch
from torch.utils.data import DataLoader
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

def build_chatts_embeddings_for_inference(model, input_text, ts_list, device):
    """
    使用与训练相同的方式构建 ChatTS 格式的嵌入（用于推理）
    这个函数复用了模型的 forward_chatts 中的逻辑，但不包含 output_text 部分
    
    参数：
        model: 模型实例
        input_text: 包含 <ts><ts/> 标记的输入文本
        ts_list: 时间序列列表 [tensor1, tensor2, ...]
        device: 设备
    
    返回：
        assembled_embeds: 组装好的嵌入 [1, total_length, llm_embed_dim]
        attention_mask: 注意力掩码 [1, total_length]
    """
    # 分割输入文本：找到所有 <ts><ts/> 标记并分割
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    
    # 确保时间序列数量与标记数量匹配
    num_markers = len(text_parts) - 1
    num_timeseries = len(ts_list)
    
    timeseries_list = list(ts_list)  # 复制以避免修改原列表
    
    if num_timeseries < num_markers:
        # 如果时间序列不够，用零填充
        for _ in range(num_markers - num_timeseries):
            timeseries_list.append(
                torch.zeros(model.config.seq_len, model.config.input_channels, device=device)
            )
    elif num_timeseries > num_markers:
        # 如果时间序列太多，只使用前 num_markers 个
        timeseries_list = timeseries_list[:num_markers]
    
    # 收集所有片段的嵌入
    segment_embeds = []
    segment_masks = []
    
    # 确定目标 dtype
    target_dtype = next(model.llm.parameters()).dtype
    
    # 获取 tokenizer
    tokenizer = model.tokenizer.tokenizer
    
    # 处理第一个文本片段（prefix）
    if text_parts[0]:
        # 使用 tokenizer 直接编码，不通过 model.tokenizer() 以避免多次添加特殊token
        prefix_tokens = tokenizer(
            text_parts[0], 
            return_tensors="pt", 
            add_special_tokens=True  # 只在第一个片段添加 BOS
        ).to(device)
        prefix_embed = model.llm.embed(prefix_tokens.input_ids)  # [1, L, D]
        prefix_mask = prefix_tokens.attention_mask  # [1, L]
        segment_embeds.append(prefix_embed[0])  # [L, D]
        segment_masks.append(prefix_mask[0])  # [L]
    
    # 处理每个时间序列和后续文本片段
    for ts_idx, ts_tensor in enumerate(timeseries_list):
        # 确保时间序列在正确的设备上
        ts_tensor = ts_tensor.to(device)
        
        # 添加 batch 维度：[T, C] -> [1, T, C]
        ts_batch = ts_tensor.unsqueeze(0)
        
        # 生成时间序列的嵌入（使用与训练相同的方法）
        stat_token, ts_tokens = model.ts_model._process_single_channel(
            ts_batch, instruction_embeds=None
        )
        # stat_token: [1, 1, D], ts_tokens: [1, N, D]
        
        # 确保数据类型对齐
        if stat_token.dtype != target_dtype:
            stat_token = stat_token.to(dtype=target_dtype)
        if ts_tokens.dtype != target_dtype:
            ts_tokens = ts_tokens.to(dtype=target_dtype)
        
        # 组装：[Stat][TS_Tokens]
        ts_embed = torch.cat([stat_token[0], ts_tokens[0]], dim=0)  # [1+N, D]
        
        # 添加到片段列表
        segment_embeds.append(ts_embed)
        segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
        
        # 添加 SEP token（如果不是最后一个时间序列）
        if ts_idx < len(timeseries_list) - 1:
            sep_embed = model.sep_token  # [1, D]
            # 确保 dtype 对齐
            if sep_embed.dtype != target_dtype:
                sep_embed = sep_embed.to(dtype=target_dtype)
            segment_embeds.append(sep_embed)  # [1, D]
            segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
        
        # 处理该时间序列后的文本片段
        text_idx = ts_idx + 1
        if text_idx < len(text_parts) and text_parts[text_idx]:
            # 中间文本片段不添加特殊 token
            text_tokens = tokenizer(
                text_parts[text_idx], 
                return_tensors="pt", 
                add_special_tokens=False  # 中间片段不添加特殊token
            ).to(device)
            text_embed = model.llm.embed(text_tokens.input_ids)  # [1, L, D]
            text_mask = text_tokens.attention_mask  # [1, L]
            segment_embeds.append(text_embed[0])  # [L, D]
            segment_masks.append(text_mask[0])  # [L]
    
    # 合并所有片段
    if segment_embeds:
        full_embed = torch.cat(segment_embeds, dim=0)  # [Total_L, D]
        full_mask = torch.cat(segment_masks, dim=0)  # [Total_L]
        
        # ⭐ 添加生成触发 token（BOS）到末尾
        # 这告诉 LLM "输入结束，现在开始生成"
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embed = model.llm.embed(bos_tensor)  # [1, 1, D]
        bos_mask = torch.ones(1, device=device, dtype=torch.long)
        
        # 拼接 BOS token
        full_embed = torch.cat([full_embed, bos_embed[0]], dim=0)  # [Total_L+1, D]
        full_mask = torch.cat([full_mask, bos_mask], dim=0)  # [Total_L+1]
        
        # 添加 batch 维度
        assembled_embeds = full_embed.unsqueeze(0)  # [1, Total_L+1, D]
        attention_mask = full_mask.unsqueeze(0)  # [1, Total_L+1]
    else:
        # 如果没有任何内容，返回空嵌入
        assembled_embeds = torch.empty(1, 0, model.config.llm_embed_dim, device=device)
        attention_mask = torch.empty(1, 0, device=device, dtype=torch.long)
    
    return assembled_embeds, attention_mask

def generate_response(model, ts_list, input_text, device, max_new_tokens=128):
    """
    执行推理生成（针对 ChatTS 格式，使用与训练相同的嵌入构建方式）
    
    参数：
        model: 模型实例
        ts_list: 时间序列列表
        input_text: 包含 <ts><ts/> 标记的输入文本（不含 User:/Assistant: 前缀）
        device: 设备
        max_new_tokens: 最大生成 token 数
    """
    tokenizer = model.tokenizer.tokenizer
    
    # 确保 Pad Token ID 有效
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    
    # 构建完整的输入文本（包含 User: 和 Assistant: 提示）
    full_input = f"User: {input_text}\nAssistant: "
    
    with torch.no_grad():
        # 使用与训练相同的方式构建嵌入
        inputs_embeds, attention_mask = build_chatts_embeddings_for_inference(
            model, full_input, ts_list, device
        )
        
        # 检查输入长度
        input_length = inputs_embeds.shape[1]
        print(f"[Debug] Input embeddings length: {input_length}")
        
        # 如果输入太长，可能需要警告
        if input_length > 4000:
            print(f"[Warning] Input length ({input_length}) is very long, generation may be unstable")
        
        # 为 Stage 2 对齐模型使用更保守的生成参数
        output_ids = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=10,  # 至少生成10个token
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            # Stage 2 使用更保守的采样策略
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
            length_penalty=1.0,
            no_repeat_ngram_size=3,  # 避免重复3-gram
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 打印生成的 token IDs 用于调试
    print(f"[Debug] Generated {len(output_ids[0])} tokens")
    print(f"[Debug] First 20 token IDs: {output_ids[0][:20].tolist()}")
    
    return generated_text

def main(args):
    # 设备处理
    device_str = args.device
    if args.device.startswith("cuda") and args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    
    print(f">>> Loading Config & Model...")
    
    # 1. ChatTS 格式中每个时间序列都是单通道（与训练代码保持一致）
    # 多个时间序列通过 forward_chatts() 方法处理，而不是多通道模式
    input_channels = 1
    print(f">>> ChatTS format uses input_channels = {input_channels} (each TS is single-channel)")
    
    # 2. 配置
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
    
    # 4. 加载 Stage 2 训练好的对齐权重
    print(f">>> Loading Stage 2 Checkpoint from {args.checkpoint}...")
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 关键：strict=False
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        
        # 验证是否加载成功
        real_missing = [k for k in missing if "llm.model" not in k]
        if len(real_missing) > 0:
            print(f"!!! Warning: Potential missing keys: {real_missing[:5]} ...")
        else:
            print(">>> Stage 2 Weights merged successfully (LLM weights skipped as expected).")
    else:
        print(f"!!! Error: Checkpoint {args.checkpoint} not found!")
        return

    model.eval()
    
    # 4. 加载验证数据（ChatTS 格式）
    print(f">>> Loading ChatTS Validation Dataset...")
    val_ds = ChatTSDataset(args.jsonl_path, args.seq_len, input_channels, split="val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=chatts_collate_fn)
    
    print(f"\n{'='*60}")
    print(f"Starting Stage 2 Alignment Test (ChatTS Format) on {args.num_samples} samples")
    print(f"{'='*60}")
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"1. Stage 2 (Alignment) models may not generate coherent text yet")
    print(f"2. They are trained to align TS embeddings with LLM space")
    print(f"3. For better text generation, use Stage 3 (Instruct) models")
    print(f"4. Long inputs (many TS) may cause generation issues\n")
    print(f"{'='*60}\n")
    
    sample_count = 0
    for batch in val_loader:
        if sample_count >= args.num_samples:
            break
        
        input_texts = batch["input_texts"]
        timeseries_lists = batch["timeseries_lists"]
        output_texts = batch["output_texts"]
        
        # 处理第一个样本（batch_size=1）
        input_text = input_texts[0]
        ts_list = timeseries_lists[0]
        ground_truth = output_texts[0]
        
        try:
            # 计算输入统计
            input_text_len = len(input_text)
            num_ts = len(ts_list)
            ts_shapes = [ts.shape for ts in ts_list]
            
            # 生成（使用交替的文本和时间序列格式）
            prediction = generate_response(model, ts_list, input_text, device)
            
            print(f"\n{'='*60}")
            print(f"Sample #{sample_count+1}")
            print(f"{'='*60}")
            print(f"📊 Input Statistics:")
            print(f"  - Text length: {input_text_len} chars")
            print(f"  - Number of time series: {num_ts}")
            print(f"  - Time series shapes: {ts_shapes}")
            print(f"\n--- Input Text (with <ts><ts/> markers) ---")
            # 只显示前500个字符，避免过长
            display_text = input_text.strip()
            if len(display_text) > 500:
                display_text = display_text[:500] + "... (truncated)"
            print(f"{display_text}")
            print(f"\n--- Ground Truth ---")
            print(f"{ground_truth.strip()}")
            print(f"\n--- Stage 2 Output ---")
            print(f"{prediction.strip()}")
            print(f"{'-'*60}\n")
            
            sample_count += 1
            
        except Exception as e:
            print(f"[Error] Sample {sample_count}: {e}")
            import traceback
            traceback.print_exc()
            sample_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 2 Alignment with ChatTS Format")
    parser.add_argument("--jsonl-path", type=str, default="chatts_data.jsonl", help="ChatTS 格式的数据文件")
    parser.add_argument("--checkpoint", type=str, default="crome_stage2_aligned.pth", help="Stage 2 权重路径")
    
    parser.add_argument("--seq-len", type=int, default=1024)
    # 移除 input-channels 参数，改为自动检测
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    
    parser.add_argument("--num-samples", type=int, default=5, help="要测试的样本数量")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=None)
    
    args = parser.parse_args()
    main(args)

