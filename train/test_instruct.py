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
from src.crome_ts.data_instruct import JSONLInstructDataset, instruct_collate_fn

# --- 2. 核心功能函数 ---

def compute_test_loss(model, dataloader, device):
    """
    计算测试集上的平均 Loss (Perplexity 指标)
    """
    model.eval()
    total_loss = 0
    count = 0
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for series, prefixes, suffixes in tqdm(dataloader, desc="Computing Test Loss"):
            series = series.to(device)
            
            # Forward pass
            model_out = model(series, prefixes, suffixes)
            llm_out = model_out["llm_outputs"]
            logits = llm_out.logits
            
            # 定位 Suffix (回答部分) 的起始位置
            # 序列结构: [Prefix] + [Stat] + [TS] + [Suffix]
            prefix_width = model_out["prefix_mask"].shape[1]
            ts_width = 1 + model_out["ts_tokens"].shape[1] # 1 是 stat_token
            start_idx = prefix_width + ts_width
            
            # 截取预测部分的 Logits
            suffix_logits = logits[:, start_idx:, :]
            
            # 获取 Label
            suffix_labels = model.tokenizer.tokenizer(
                suffixes, return_tensors="pt", padding=True
            ).input_ids.to(device)
            
            # 长度对齐 (防止 Logits 和 Labels 长度不一致)
            min_len = min(suffix_logits.shape[1], suffix_labels.shape[1])
            suffix_logits = suffix_logits[:, :min_len, :]
            suffix_labels = suffix_labels[:, :min_len]
            
            # Shift for Next Token Prediction (错位预测)
            shift_logits = suffix_logits[..., :-1, :].contiguous()
            shift_labels = suffix_labels[..., 1:].contiguous()
            
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            total_loss += loss.item()
            count += 1
            
    return total_loss / count if count > 0 else 0.0

def generate_predictions(model, dataloader, device, output_file, max_new_tokens=256, max_samples=None):
    """
    生成回复并保存到文件 (包含 BOS Token 修复)
    """
    model.eval()
    results = []
    
    print(f">>> Generating responses (Max Samples: {max_samples if max_samples else 'All'})...")
    
    tokenizer = model.tokenizer.tokenizer
    
    # 1. 确保 Pad Token 存在
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. 获取 BOS Token ID (用于触发生成)
    # Llama 2/3 通常是 1 或 128000，这里自动获取
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    
    with torch.no_grad():
        for i, (series, prefixes, suffixes) in enumerate(tqdm(dataloader, desc="Inference")):
            if max_samples is not None and i >= max_samples:
                break
            
            # 建议 Batch Size = 1 进行生成，避免 Padding 干扰
            series = series.to(device)
            prefix = prefixes[0] 
            ground_truth = suffixes[0]
            
            # A. 准备 Prefix Embedding
            # add_special_tokens=False，因为我们会在最后手动加 BOS
            prefix_inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(device)
            prefix_embeds = model.llm.embed(prefix_inputs.input_ids)
            
            # B. 准备空的 Suffix (推理阶段还没有回答)
            empty_suffix = torch.empty(
                series.size(0), 0, model.config.llm_embed_dim, 
                device=device, dtype=prefix_embeds.dtype
            )
            
            # C. 获取时序特征 (Stat + TS)
            ts_out = model.ts_model(series, prefix_embeds, empty_suffix)
            inputs_embeds = ts_out["assembled"] # 目前是 [Prefix, Stat, TS]
            
            # ==================【关键修复】==================
            # 手动拼接 BOS Token 到序列末尾
            # 只有看到这个 Token，LLM 才知道“User 说完了，轮到 Assistant 开始了”
            bos_tensor = torch.tensor([[bos_token_id]], device=device) # [1, 1]
            bos_embeds = model.llm.embed(bos_tensor)                 # [1, 1, Dim]
            
            # 扩展 Batch 维度 (虽然这里 BS=1，但为了健壮性)
            if inputs_embeds.size(0) > 1:
                bos_embeds = bos_embeds.expand(inputs_embeds.size(0), -1, -1)
            
            # 最终输入: [Prefix, Stat, TS, BOS]
            inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)
            # ===============================================
            
            # D. 生成
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=device
            )
            
            output_ids = model.llm.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, # 贪婪搜索 (Greedy)，结果最稳定
                repetition_penalty=1.1 # 避免重复
            )
            
            pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # E. 收集结果
            results.append({
                "sample_id": i,
                "prompt": prefix,
                "ground_truth": ground_truth,
                "prediction": pred_text
            })
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f">>> Predictions saved to {output_file}")

# --- 3. 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="Test Stage 3 (Instruct) Model")
    parser.add_argument("--jsonl-path", type=str, default="sft_full_data_en.jsonl", help="测试数据路径")
    # 注意：这里默认加载 Stage 3 训练好的最佳权重
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 3 权重路径 (例如 train/crome_instruct_best.pth)")
    parser.add_argument("--output-file", type=str, default="stage3_test_results.jsonl", help="结果输出文件路径")
    
    # 模型参数 (需与训练时一致)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--input-channels", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=16)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--calc-loss", action="store_true", help="是否计算测试集 Loss (耗时较长)")
    parser.add_argument("--num-gen-samples", type=int, default=100, help="生成测试样本的数量，-1表示全部")
    
    args = parser.parse_args()
    
    # 1. 设备设置
    device_str = args.device
    if "cuda" in args.device and args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    print(f">>> Using Device: {device}")

    # 2. 加载配置
    print(">>> Loading Configuration...")
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    
    config = CROMEConfig(
        input_channels=args.input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=device_str,
        llm_dtype="bfloat16",
        use_stats_projector=True
    )
    
    # 3. 加载模型
    print(">>> Loading Model structure...")
    model = StatBypassCROMETS1(config).to(device)
    
    print(f">>> Loading Checkpoint from {args.checkpoint}...")
    if Path(args.checkpoint).exists():
        state_dict = torch.load(args.checkpoint, map_location=device)
        # strict=False 以防万一有些非关键键名不匹配，但在测试阶段通常应该尽量匹配
        msg = model.load_state_dict(state_dict, strict=False)
        print(f">>> Weights Loaded. Msg: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.eval()

    # 4. 加载数据
    print(f">>> Loading Validation Dataset from {args.jsonl_path}...")
    # 这里使用 'val' split，你也可以根据需要改为 'test'
    val_ds = JSONLInstructDataset(args.jsonl_path, args.seq_len, args.input_channels, split="val")
    
    # 5. 执行测试
    
    # 5.1 计算 Loss (可选)
    if args.calc_loss:
        # Batch size 可稍大
        loss_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=instruct_collate_fn)
        avg_loss = compute_test_loss(model, loss_loader, device)
        print(f"\n{'='*40}")
        print(f"Test Set Loss (Perplexity Metric): {avg_loss:.4f}")
        print(f"{'='*40}\n")
    
    # 5.2 生成样本
    # Batch size 必须为 1
    gen_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=instruct_collate_fn)
    
    max_samples = args.num_gen_samples if args.num_gen_samples > 0 else None
    generate_predictions(model, gen_loader, device, args.output_file, max_samples=max_samples)
    
    print("\n>>> Testing Completed.")
    print(f">>> Check {args.output_file} for detailed results.")

if __name__ == "__main__":
    main()