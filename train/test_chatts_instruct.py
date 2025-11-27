import sys
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- 1. 环境与路径设置 ---
# 确保能找到 src 包
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# --- 2. 核心功能函数 ---

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

def compute_test_loss(model, dataloader, device):
    """
    计算测试集上的平均 Loss (Perplexity 指标)
    针对 ChatTS 格式：支持多个时间序列和 <ts><ts/> 标记
    """
    model.eval()
    total_loss = 0
    count = 0
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Test Loss"):
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            # 对于 ChatTS 格式，我们需要处理每个样本
            # 这里简化处理：取 batch 中的第一个样本
            for i in range(len(input_texts)):
                input_text = input_texts[i]
                ts_list = timeseries_lists[i]
                output_text = output_texts[i]
                
                # 将时间序列移到设备上
                ts_list_device = [ts.unsqueeze(0).to(device) for ts in ts_list]
                
                # 构建 prefix（将 <ts><ts/> 替换为实际时间序列的占位符）
                prefix = f"User: {input_text}\nAssistant: "
                
                # Forward pass - 这里需要根据实际模型接口调整
                # 对于 ChatTS，我们假设模型可以处理多个时间序列
                # 这里使用第一个时间序列作为示例（实际应用中需要处理所有序列）
                if len(ts_list_device) > 0:
                    series = ts_list_device[0]  # [1, T, C]
                else:
                    # 如果没有时间序列，创建一个零张量
                    series = torch.zeros(1, model.config.seq_len, 1, device=device)
                
                try:
                    model_out = model(series, [prefix], [output_text])
                    llm_out = model_out["llm_outputs"]
                    logits = llm_out.logits
                    
                    # 定位 Suffix (回答部分) 的起始位置
                    prefix_width = model_out["prefix_mask"].shape[1]
                    
                    # 使用stat_token + ts_tokens的宽度
                    ts_width = 1 + model_out["ts_tokens"].shape[1]
                    
                    start_idx = prefix_width + ts_width
                    
                    # 截取预测部分的 Logits
                    suffix_logits = logits[:, start_idx:, :]
                    
                    # 获取 Label
                    suffix_labels = model.tokenizer.tokenizer(
                        [output_text], return_tensors="pt", padding=True
                    ).input_ids.to(device)
                    
                    # 长度对齐
                    min_len = min(suffix_logits.shape[1], suffix_labels.shape[1])
                    suffix_logits = suffix_logits[:, :min_len, :]
                    suffix_labels = suffix_labels[:, :min_len]
                    
                    # Shift for Next Token Prediction
                    shift_logits = suffix_logits[..., :-1, :].contiguous()
                    shift_labels = suffix_labels[..., 1:].contiguous()
                    
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    
                    total_loss += loss.item()
                    count += 1
                except Exception as e:
                    print(f"[Warning] Error computing loss for sample {i}: {e}")
                    continue
            
    return total_loss / count if count > 0 else 0.0

def generate_predictions(model, dataloader, device, output_file, max_new_tokens=256, max_samples=None):
    """
    生成回复并保存到文件 (针对 ChatTS 格式)
    """
    model.eval()
    results = []
    
    print(f">>> Generating responses (Max Samples: {max_samples if max_samples else 'All'})...")
    
    tokenizer = model.tokenizer.tokenizer
    
    # 1. 确保 Pad Token 存在
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. 获取 BOS Token ID
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Inference"):
            if max_samples is not None and sample_count >= max_samples:
                break
            
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            # 处理 batch 中的每个样本（建议 batch_size=1）
            for i in range(len(input_texts)):
                if max_samples is not None and sample_count >= max_samples:
                    break
                
                input_text = input_texts[i]
                ts_list = timeseries_lists[i]
                ground_truth = output_texts[i]
                
                # 构建完整的输入文本（包含 User/Assistant 提示符和原始的 <ts><ts/> 标记）
                full_input = f"User: {input_text}\nAssistant: "
                
                try:
                    # 使用与训练相同的方式构建嵌入
                    # 格式：<text>User: <text>[Stat_1][TS_1][SEP]<text> 和 <text>[Stat_2][TS_2][SEP]<text>Assistant: <text>
                    inputs_embeds, attention_mask = build_chatts_embeddings_for_inference(
                        model, full_input, ts_list, device
                    )
                    
                    # 检查输入长度
                    input_length = inputs_embeds.shape[1]
                    if input_length > 4000:
                        print(f"[Warning] Sample {sample_count}: Input length ({input_length}) is very long")
                    
                    # 生成（Stage 3 使用贪婪解码以获得稳定结果）
                    output_ids = model.llm.model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        do_sample=False,  # 贪婪解码
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )
                    
                    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    # F. 收集结果
                    results.append({
                        "sample_id": sample_count,
                        "input_text": input_text,
                        "num_timeseries": len(ts_list),
                        "ground_truth": ground_truth,
                        "prediction": pred_text
                    })
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"[Warning] Error generating prediction for sample {sample_count}: {e}")
                    sample_count += 1
                    continue
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f">>> Predictions saved to {output_file}")

# --- 3. 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="Test Stage 3 (ChatTS Instruct) Model")
    parser.add_argument("--jsonl-path", type=str, default="chatts_data.jsonl", help="测试数据路径（ChatTS 格式）")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 3 权重路径")
    parser.add_argument("--output-file", type=str, default="chatts_stage3_test_results.jsonl", help="结果输出文件路径")
    
    # 模型参数
    parser.add_argument("--seq-len", type=int, default=256)
    # 移除 input-channels 参数，改为自动检测
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--calc-loss", action="store_true", help="是否计算测试集 Loss")
    parser.add_argument("--num-gen-samples", type=int, default=100, help="生成测试样本的数量，-1表示全部")
    
    args = parser.parse_args()
    
    # 1. 设备设置
    device_str = args.device
    if "cuda" in args.device and args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    print(f">>> Using Device: {device}")

    # 2. ChatTS 格式中每个时间序列都是单通道（与训练代码保持一致）
    # 多个时间序列通过 forward_chatts() 方法处理
    input_channels = 1
    print(f">>> ChatTS format uses input_channels = {input_channels} (each TS is single-channel)")
    print(f">>> Multiple TS handled by forward_chatts method")
    
    # 3. 加载配置
    print(">>> Loading Configuration...")
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16",
        use_stats_projector=True,
    )
    
    # 4. 加载模型
    print(">>> Loading Model structure...")
    model = StatBypassCROMETS1(config).to(device)
    
    print(f">>> Loading Checkpoint from {args.checkpoint}...")
    if Path(args.checkpoint).exists():
        state_dict = torch.load(args.checkpoint, map_location=device)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f">>> Weights Loaded. Msg: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.eval()

    # 5. 加载验证数据（使用检测到的通道数）
    print(f">>> Loading Validation Dataset from {args.jsonl_path}...")
    val_ds = ChatTSDataset(
        args.jsonl_path, args.seq_len, input_channels, split="val"
    )
    
    # 6. 执行测试
    
    # 6.1 计算 Loss (可选)
    if args.calc_loss:
        # 对于 ChatTS，建议使用较小的 batch size
        loss_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=chatts_collate_fn)
        avg_loss = compute_test_loss(model, loss_loader, device)
        print(f"\n{'='*40}")
        print(f"Test Set Loss (Perplexity Metric): {avg_loss:.4f}")
        print(f"{'='*40}\n")
    
    # 6.2 生成样本
    gen_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=chatts_collate_fn)
    
    max_samples = args.num_gen_samples if args.num_gen_samples > 0 else None
    generate_predictions(model, gen_loader, device, args.output_file, max_samples=max_samples)
    
    print("\n>>> Testing Completed.")
    print(f">>> Check {args.output_file} for detailed results.")

if __name__ == "__main__":
    main()

