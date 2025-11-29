"""
测试 Stage 2 对齐效果的专用脚本 (已适配动态长度 + 显式统计量)
"""

import sys
from pathlib import Path
import numpy as np

# 路径设置
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn


def compute_alignment_loss(model, dataloader, device):
    """
    计算对齐损失 (自动适配模型内部的 forward_chatts 逻辑)
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
                # model.forward_chatts 内部已经包含了 "显式统计量" 的插入逻辑
                model_out = model.forward_chatts(
                    input_texts=input_texts,
                    timeseries_lists=timeseries_lists,
                    output_texts=output_texts,
                    llm_kwargs={}
                )
                
                llm_out = model_out["llm_outputs"]
                logits = llm_out.logits
                
                suffix_labels = model.tokenizer.tokenizer(
                    output_texts, return_tensors="pt", padding=True
                ).input_ids.to(device)
                
                batch_size = logits.shape[0]
                suffix_mask_lengths = model_out["suffix_mask_lengths"]
                
                for i in range(batch_size):
                    valid_len = model_out["attention_mask"][i].sum().item()
                    suffix_len = suffix_mask_lengths[i]
                    
                    if suffix_len == 0: continue
                    
                    suffix_start = int(valid_len - suffix_len)
                    
                    sample_logits = logits[i, suffix_start:suffix_start+suffix_len, :]
                    sample_labels = suffix_labels[i, :suffix_len]
                    
                    # 长度对齐保护
                    min_len = min(sample_logits.shape[0], sample_labels.shape[0])
                    sample_logits = sample_logits[:min_len]
                    sample_labels = sample_labels[:min_len]
                    
                    if min_len > 1:
                        shift_logits = sample_logits[:-1, :]
                        shift_labels = sample_labels[1:]
                        
                        loss = loss_fct(shift_logits, shift_labels)
                        total_loss += loss.item()
                        total_tokens += shift_labels.numel()
                        
                        predictions = shift_logits.argmax(dim=-1)
                        correct = (predictions == shift_labels).sum().item()
                        correct_predictions += correct
                        
            except Exception as e:
                print(f"[Warning] Error processing batch: {e}")
                continue
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return {
        "loss": avg_loss,
        "perplexity": np.exp(avg_loss),
        "accuracy": correct_predictions / total_tokens if total_tokens > 0 else 0,
        "total_tokens": total_tokens
    }


def compute_embedding_quality(model, dataloader, device, num_samples=100):
    """
    评估嵌入质量 (已更新：包含显式文本统计量)
    """
    model.eval()
    tokenizer = model.tokenizer.tokenizer
    
    print("\n" + "="*60)
    print("评估嵌入空间质量")
    print("="*60)
    
    stat_norms = []       # 隐式 Stat Projector
    text_stat_norms = []  # ✨ 新增：显式文本统计量
    query_norms = []
    detail_norms = []
    combined_norms = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in tqdm(dataloader, desc="Analyzing Embeddings"):
            if sample_count >= num_samples: break
                
            timeseries_lists = batch["timeseries_lists"]
            input_texts = batch["input_texts"]
            
            for i in range(len(input_texts)):
                if sample_count >= num_samples: break
                ts_list = timeseries_lists[i]
                
                for ts_tensor in ts_list[:3]: # 只看前3个
                    ts_tensor = ts_tensor.to(device)
                    ts_batch = ts_tensor.unsqueeze(0)
                    
                    try:
                        # 1. 计算显式统计量文本的范数 (复用 model.py 的逻辑)
                        if ts_tensor.numel() > 0:
                            ts_mean = ts_tensor.mean().item()
                            ts_std = ts_tensor.std().item()
                            ts_min = ts_tensor.min().item()
                            ts_max = ts_tensor.max().item()
                        else:
                            ts_mean = ts_std = ts_min = ts_max = 0.0
                            
                        stats_str = f" [Stats: mean={ts_mean:.2f}, std={ts_std:.2f}, min={ts_min:.2f}, max={ts_max:.2f}] "
                        
                        stats_tokens = tokenizer(
                            stats_str, return_tensors="pt", add_special_tokens=False
                        ).to(device)
                        stats_embed = model.llm.embed(stats_tokens.input_ids) # [1, L, D]
                        
                        # ✨ 记录文本统计量范数
                        text_stat_norm = torch.norm(stats_embed, dim=-1).mean().item()
                        text_stat_norms.append(text_stat_norm)

                        # 2. 计算时序相关范数
                        x, stats = model.ts_model.preprocessor(ts_batch)
                        patch_tokens = model.ts_model.shape_encoder(x)
                        query_tokens = model.ts_model.qformer(patch_tokens, instruction_embeds=None)
                        detail_tokens = model.ts_model.detail_proj(patch_tokens)
                        
                        query_out = model.ts_model.adapter.query_adapter(query_tokens)
                        detail_out = model.ts_model.adapter.patch_adapter(detail_tokens)
                        
                        query_projected = model.ts_model.llm_proj(query_out)
                        detail_projected = model.ts_model.llm_proj(detail_out)
                        stat_token = model.ts_model.stat_projector(stats)
                        
                        combined_projected = torch.cat([query_projected, detail_projected], dim=1)
                        
                        stat_norms.append(torch.norm(stat_token, dim=-1).mean().item())
                        query_norms.append(torch.norm(query_projected, dim=-1).mean().item())
                        detail_norms.append(torch.norm(detail_projected, dim=-1).mean().item())
                        combined_norms.append(torch.norm(combined_projected, dim=-1).mean().item())
                        
                    except Exception as e:
                        print(f"[Warning] Error processing TS: {e}")
                        continue
                
                sample_count += 1
    
    return {
        "stat_norm": (np.mean(stat_norms), np.std(stat_norms)),
        "text_stat_norm": (np.mean(text_stat_norms), np.std(text_stat_norms)), # ✨ 新增
        "query_norm": (np.mean(query_norms), np.std(query_norms)),
        "detail_norm": (np.mean(detail_norms), np.std(detail_norms)),
        "combined_norm": (np.mean(combined_norms), np.std(combined_norms)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=2048, help="Max length")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else args.device)
    
    input_channels = 1
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=str(device),
        llm_dtype="bfloat16",
        use_stats_projector=True,
    )
    
    print(f">>> Loading Model from {args.checkpoint}...")
    model = StatBypassCROMETS1(config).to(device)
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    print(f">>> Loading Dataset...")
    # ✨ 修改点 1：传入 patch_stride
    val_ds = ChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, 
        input_channels=input_channels, 
        split="val",
        patch_stride=args.patch_stride 
    )
    
    if args.num_samples > 0:
        val_ds.records = val_ds.records[:args.num_samples]
        
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=chatts_collate_fn)
    
    # 1. Loss
    loss_metrics = compute_alignment_loss(model, val_loader, device)
    
    # 2. Embeddings
    emb_metrics = compute_embedding_quality(model, val_loader, device, num_samples=50)
    
    print("\n" + "="*70)
    print("评估结果")
    print("-" * 70)
    print(f"Loss: {loss_metrics['loss']:.4f} | Perplexity: {loss_metrics['perplexity']:.4f} | Acc: {loss_metrics['accuracy']*100:.2f}%")
    print("-" * 70)
    print(f"Implicit Stat Token Norm (Soft): {emb_metrics['stat_norm'][0]:.4f} ± {emb_metrics['stat_norm'][1]:.4f}")
    print(f"Explicit Text Stat Norm (Hard):  {emb_metrics['text_stat_norm'][0]:.4f} ± {emb_metrics['text_stat_norm'][1]:.4f}  <-- ✨ New!")
    print(f"Query Token Norm:                {emb_metrics['query_norm'][0]:.4f} ± {emb_metrics['query_norm'][1]:.4f}")
    print(f"Detail Token Norm:               {emb_metrics['detail_norm'][0]:.4f} ± {emb_metrics['detail_norm'][1]:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()