"""
测试 Stage 2 对齐效果的专用脚本

Stage 2 对齐模型的评估指标：
1. 对齐损失（Alignment Loss）
2. 下一个 token 预测准确率
3. 困惑度（Perplexity）
4. 嵌入空间质量

不应该评估：
- 生成文本质量（这是 Stage 3 的任务）
"""

import sys
from pathlib import Path

# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn


def compute_alignment_loss(model, dataloader, device):
    """
    计算对齐损失 - Stage 2 的核心指标
    这衡量模型能否正确预测下一个 token
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct_predictions = 0
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    
    print("\n" + "="*60)
    print("计算对齐损失（Alignment Loss）")
    print("="*60)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Loss"):
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            try:
                # 使用 forward_chatts 方法
                model_out = model.forward_chatts(
                    input_texts=input_texts,
                    timeseries_lists=timeseries_lists,
                    output_texts=output_texts,
                    llm_kwargs={}
                )
                
                llm_out = model_out["llm_outputs"]
                logits = llm_out.logits  # [B, Total_L, Vocab]
                
                # 获取输出文本的标签
                suffix_labels = model.tokenizer.tokenizer(
                    output_texts, return_tensors="pt", padding=True
                ).input_ids.to(device)
                
                batch_size = logits.shape[0]
                suffix_mask_lengths = model_out["suffix_mask_lengths"]
                
                # 计算每个样本的 loss 和准确率
                for i in range(batch_size):
                    valid_len = model_out["attention_mask"][i].sum().item()
                    suffix_len = suffix_mask_lengths[i]
                    
                    if suffix_len == 0:
                        continue
                    
                    suffix_start = int(valid_len - suffix_len)
                    
                    # 提取 logits 和 labels
                    sample_logits = logits[i, suffix_start:suffix_start+suffix_len, :]
                    sample_labels = suffix_labels[i, :suffix_len]
                    
                    if sample_logits.shape[0] > 1:
                        # Causal LM: 预测下一个 token
                        shift_logits = sample_logits[:-1, :]
                        shift_labels = sample_labels[1:]
                        
                        # 计算损失
                        loss = loss_fct(shift_logits, shift_labels)
                        total_loss += loss.item()
                        total_tokens += shift_labels.numel()
                        
                        # 计算准确率
                        predictions = shift_logits.argmax(dim=-1)
                        correct = (predictions == shift_labels).sum().item()
                        correct_predictions += correct
                        
            except Exception as e:
                print(f"[Warning] Error processing batch: {e}")
                continue
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "total_tokens": total_tokens
    }


def compute_embedding_quality(model, dataloader, device, num_samples=100):
    """
    评估嵌入质量
    - 分别评估 stat_token, query_tokens, detail_tokens 的范数
    - 嵌入的范数分布
    """
    model.eval()
    
    print("\n" + "="*60)
    print("评估嵌入空间质量（分别显示 Query 和 Detail）")
    print("="*60)
    
    stat_norms = []
    query_norms = []
    detail_norms = []
    combined_norms = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Analyzing Embeddings"):
            if sample_count >= num_samples:
                break
                
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            for i in range(len(input_texts)):
                if sample_count >= num_samples:
                    break
                    
                ts_list = timeseries_lists[i]
                
                # 处理每个时间序列
                for ts_tensor in ts_list[:3]:  # 只看前3个
                    ts_tensor = ts_tensor.to(device)
                    ts_batch = ts_tensor.unsqueeze(0)  # [1, T, C]
                    
                    try:
                        # ⭐ 分别提取 query 和 detail tokens
                        x, stats = model.ts_model.preprocessor(ts_batch)
                        
                        # 使用 encoder
                        patch_tokens = model.ts_model.shape_encoder(x)
                        
                        # 分别获取 query 和 detail
                        query_tokens = model.ts_model.qformer(patch_tokens, instruction_embeds=None)
                        detail_tokens = model.ts_model.detail_proj(patch_tokens)
                        
                        # 通过 adapter 处理
                        query_out = model.ts_model.adapter.query_adapter(query_tokens)
                        detail_out = model.ts_model.adapter.patch_adapter(detail_tokens)
                        
                        # 投影到 LLM 空间
                        query_projected = model.ts_model.llm_proj(query_out)
                        detail_projected = model.ts_model.llm_proj(detail_out)
                        combined_projected = torch.cat([query_projected, detail_projected], dim=1)
                        
                        # 获取 stat token
                        stat_token = model.ts_model.stat_projector(stats)
                        
                        # 计算范数
                        stat_norm = torch.norm(stat_token, dim=-1).mean().item()
                        query_norm = torch.norm(query_projected, dim=-1).mean().item()
                        detail_norm = torch.norm(detail_projected, dim=-1).mean().item()
                        combined_norm = torch.norm(combined_projected, dim=-1).mean().item()
                        
                        stat_norms.append(stat_norm)
                        query_norms.append(query_norm)
                        detail_norms.append(detail_norm)
                        combined_norms.append(combined_norm)
                        
                    except Exception as e:
                        print(f"[Warning] Error processing TS: {e}")
                        continue
                
                sample_count += 1
    
    return {
        "stat_norm_mean": np.mean(stat_norms),
        "stat_norm_std": np.std(stat_norms),
        "query_norm_mean": np.mean(query_norms),
        "query_norm_std": np.std(query_norms),
        "detail_norm_mean": np.mean(detail_norms),
        "detail_norm_std": np.std(detail_norms),
        "combined_norm_mean": np.mean(combined_norms),
        "combined_norm_std": np.std(combined_norms),
    }


def analyze_prediction_patterns(model, dataloader, device, num_samples=20):
    """
    分析预测模式
    - 看模型在不同位置的置信度
    - 看预测的 token 分布
    """
    model.eval()
    
    print("\n" + "="*60)
    print("分析预测模式")
    print("="*60)
    
    top1_confidences = []
    entropy_values = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Analyzing Predictions"):
            if sample_count >= num_samples:
                break
                
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            try:
                model_out = model.forward_chatts(
                    input_texts=input_texts,
                    timeseries_lists=timeseries_lists,
                    output_texts=output_texts,
                    llm_kwargs={}
                )
                
                llm_out = model_out["llm_outputs"]
                logits = llm_out.logits
                
                # 计算 softmax 概率
                probs = torch.softmax(logits, dim=-1)
                
                # Top-1 置信度
                top1_conf = probs.max(dim=-1).values.mean().item()
                top1_confidences.append(top1_conf)
                
                # 熵（衡量不确定性）
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                entropy_values.append(entropy)
                
                sample_count += len(input_texts)
                
            except Exception as e:
                print(f"[Warning] Error: {e}")
                continue
    
    return {
        "avg_top1_confidence": np.mean(top1_confidences),
        "avg_entropy": np.mean(entropy_values),
    }


def main():
    parser = argparse.ArgumentParser(description="Test Stage 2 Alignment Quality")
    parser.add_argument("--jsonl-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100, help="用于评估的样本数")
    
    args = parser.parse_args()
    
    # 设备设置
    device_str = args.device
    if "cuda" in args.device and args.gpu_id is not None:
        device_str = f"cuda:{args.gpu_id}"
    device = torch.device(device_str)
    
    print("\n" + "="*70)
    print(" "*20 + "Stage 2 对齐效果评估")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.jsonl_path}")
    print(f"Device: {device}")
    print("="*70)
    
    # ChatTS 格式配置
    input_channels = 1
    
    # 加载配置
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
        epsilon=1e-4,
    )
    
    # 加载模型
    print("\n>>> Loading Model...")
    model = StatBypassCROMETS1(config).to(device)
    
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f">>> Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.eval()
    
    # 加载数据
    print(f"\n>>> Loading Dataset...")
    val_ds = ChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, # 作为 max_len
        input_channels=input_channels, 
        split="val",
        patch_stride=args.patch_stride # <--- 新增：传入 stride 以启用动态对齐和 Edge Padding
    )
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(val_ds):
        val_ds.records = val_ds.records[:args.num_samples]
    
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=chatts_collate_fn)
    
    print(f">>> Evaluating on {len(val_ds)} samples...")
    
    # ==================== 核心评估 ====================
    
    # 1. 对齐损失（最重要）
    loss_metrics = compute_alignment_loss(model, val_loader, device)
    
    # 2. 嵌入质量
    embedding_metrics = compute_embedding_quality(model, val_loader, device, num_samples=50)
    
    # 3. 预测模式
    prediction_metrics = analyze_prediction_patterns(model, val_loader, device, num_samples=20)
    
    # ==================== 输出结果 ====================
    
    print("\n" + "="*70)
    print(" "*25 + "评估结果")
    print("="*70)
    
    print("\n📊 1. 对齐损失（Alignment Loss）")
    print("-" * 70)
    print(f"  Loss (per token):        {loss_metrics['loss']:.4f}")
    print(f"  Perplexity:              {loss_metrics['perplexity']:.4f}")
    print(f"  Next Token Accuracy:     {loss_metrics['accuracy']*100:.2f}%")
    print(f"  Total Tokens Evaluated:  {loss_metrics['total_tokens']}")
    
    # 解读
    print("\n  💡 解读:")
    if loss_metrics['loss'] < 0.5:
        print("     ✅ 优秀！对齐效果很好")
    elif loss_metrics['loss'] < 0.8:
        print("     ⚠️  一般，可以继续训练")
    elif loss_metrics['loss'] < 1.2:
        print("     ❌ 较差，建议继续训练")
    else:
        print("     ❌ 很差，检查训练过程")
    
    print("\n📐 2. 嵌入空间质量（分别显示 Query 和 Detail）")
    print("-" * 70)
    print(f"  Stat Token Norm:         {embedding_metrics['stat_norm_mean']:.4f} ± {embedding_metrics['stat_norm_std']:.4f}")
    print(f"  Query Token Norm:        {embedding_metrics['query_norm_mean']:.4f} ± {embedding_metrics['query_norm_std']:.4f}  (关键信息)")
    print(f"  Detail Token Norm:       {embedding_metrics['detail_norm_mean']:.4f} ± {embedding_metrics['detail_norm_std']:.4f}  (细节信息)")
    print(f"  Combined (Q+D) Norm:     {embedding_metrics['combined_norm_mean']:.4f} ± {embedding_metrics['combined_norm_std']:.4f}  (拼接后)")
    
    print("\n  💡 解读:")
    stat_norm = embedding_metrics['stat_norm_mean']
    query_norm = embedding_metrics['query_norm_mean']
    detail_norm = embedding_metrics['detail_norm_mean']
    combined_norm = embedding_metrics['combined_norm_mean']
    
    issues = []
    if stat_norm < 10 or stat_norm > 200:
        issues.append("Stat token 范数异常")
    if query_norm < 10 or query_norm > 200:
        issues.append("Query token 范数异常")
    if detail_norm < 10 or detail_norm > 200:
        issues.append("Detail token 范数异常")
    
    if not issues:
        print("     ✅ 所有嵌入范数正常，在合理范围内")
    else:
        print(f"     ⚠️  {', '.join(issues)}")
    
    # 检查稳定性
    query_std = embedding_metrics['query_norm_std']
    detail_std = embedding_metrics['detail_norm_std']
    if query_std < 5 and detail_std < 5:
        print("     ✅ Query 和 Detail 都非常稳定（标准差小）")
    elif query_std < 10 and detail_std < 10:
        print("     ✅ Query 和 Detail 稳定性良好")
    else:
        print("     ⚠️  嵌入变化较大，可能需要更多训练")
    
    print("\n🎯 3. 预测模式")
    print("-" * 70)
    print(f"  Avg Top-1 Confidence:    {prediction_metrics['avg_top1_confidence']:.4f}")
    print(f"  Avg Entropy:             {prediction_metrics['avg_entropy']:.4f}")
    
    print("\n  💡 解读:")
    conf = prediction_metrics['avg_top1_confidence']
    if conf > 0.5:
        print("     ✅ 模型预测较自信")
    elif conf > 0.3:
        print("     ⚠️  模型预测中等自信")
    else:
        print("     ❌ 模型预测不自信，可能未学好")
    
    # ==================== 总结与建议 ====================
    
    print("\n" + "="*70)
    print(" "*25 + "总结与建议")
    print("="*70)
    
    loss = loss_metrics['loss']
    acc = loss_metrics['accuracy']
    
    print("\n🎓 当前模型状态:")
    if loss < 0.5 and acc > 0.5:
        print("   ✅ 对齐效果优秀，可以进入 Stage 3 训练")
    elif loss < 0.8 and acc > 0.3:
        print("   ⚠️  对齐效果一般，建议继续训练 Stage 2")
        print("      目标: Loss < 0.5, Accuracy > 50%")
    else:
        print("   ❌ 对齐效果较差，需要继续训练")
        print("      建议: 检查学习率、训练轮数、数据质量")
    
    print("\n📝 下一步行动:")
    if loss < 0.5:
        print("   1. ✅ 当前 Stage 2 已训练好")
        print("   2. 🎯 可以开始训练 Stage 3")
        print("   3. 📊 使用 test_chatts_instruct.py 评估 Stage 3")
    else:
        print("   1. 🔄 继续训练 Stage 2")
        print(f"      当前: Loss = {loss:.4f}")
        print(f"      目标: Loss < 0.5")
        print("   2. ⚙️  可以尝试:")
        print("      - 增加训练轮数 (--epochs 20)")
        print("      - 调整学习率 (--lr 2e-4)")
        print("      - 使用更多数据")
    
    print("\n⚠️  重要提醒:")
    print("   - Stage 2 的目标是对齐，不是生成")
    print("   - 不要用生成质量评估 Stage 2")
    print("   - 只有 Stage 3 才能生成好的文本")
    
    print("\n" + "="*70)
    print(" "*20 + "评估完成")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

