import sys
import json
import argparse
import torch
import os
import glob
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
# 新增分布式相关库
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# === 1. 新增：PEFT 库支持 ===
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA will not be available.")

# --- 环境与路径设置 ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn

# --- 分布式辅助 ---
def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        print(f"[Init] Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        return rank, world_size, local_rank
    else:
        print("[Init] Running in Single GPU Mode")
        return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- 2. 核心功能函数 ---

def build_chatts_embeddings_for_inference(model, input_text, ts_list, device):
    ts_marker = "<ts><ts/>"
    text_parts = input_text.split(ts_marker)
    num_markers = len(text_parts) - 1
    num_timeseries = len(ts_list)
    
    # 复制列表以防修改原数据
    timeseries_list = list(ts_list)
    
    # === 修改点 1: 处理缺失的时间序列 ===
    if num_timeseries < num_markers:
        # 既然是占位符，不需要填充到 seq_len (max_len)
        # 只需要填充一个最小的合法长度 (patch_len)，以节省推理计算量
        # 且因为是缺失数据，保持全 0 是正确的（没有"最后一个点"可供Padding）
        min_len = model.config.patch_len
        
        for _ in range(num_markers - num_timeseries):
            timeseries_list.append(
                torch.zeros(min_len, model.config.input_channels, device=device)
            )
            
    elif num_timeseries > num_markers:
        timeseries_list = timeseries_list[:num_markers]
        
    segment_embeds = []
    segment_masks = []
    
    # 获取目标 dtype (兼容 LoRA/bf16)
    target_dtype = next(model.llm.parameters()).dtype
    tokenizer = model.tokenizer.tokenizer
    
    # 1. 处理前缀文本
    if text_parts[0]:
        prefix_tokens = tokenizer(
            text_parts[0], 
            return_tensors="pt", 
            add_special_tokens=True # 添加 BOS
        ).to(device)
        prefix_embed = model.llm.embed(prefix_tokens.input_ids)
        prefix_mask = prefix_tokens.attention_mask
        segment_embeds.append(prefix_embed[0])
        segment_masks.append(prefix_mask[0])
    
    # 2. 循环处理 [TS] + [Text]
    for ts_idx, ts_tensor in enumerate(timeseries_list):
        ts_tensor = ts_tensor.to(device)
        
        ts_batch = ts_tensor.unsqueeze(0)
        
        # === 核心: 这里的 ts_tensor 长度是动态的 ===
        # 只要 Dataset 处理正确（已对齐 stride），这里就能直接通过
        stat_token, ts_tokens = model.ts_model._process_single_channel(
            ts_batch, instruction_embeds=None
        )
        
        # 类型对齐
        if stat_token.dtype != target_dtype: 
            stat_token = stat_token.to(dtype=target_dtype)
        if ts_tokens.dtype != target_dtype: 
            ts_tokens = ts_tokens.to(dtype=target_dtype)
            
        ts_embed = torch.cat([stat_token[0], ts_tokens[0]], dim=0)
        
        segment_embeds.append(ts_embed)
        # 生成对应的 Mask (全1)
        segment_masks.append(torch.ones(ts_embed.shape[0], device=device, dtype=torch.long))
        
        # 添加 SEP
        if ts_idx < len(timeseries_list) - 1:
            sep_embed = model.sep_token
            if sep_embed.dtype != target_dtype: 
                sep_embed = sep_embed.to(dtype=target_dtype)
            segment_embeds.append(sep_embed)
            segment_masks.append(torch.ones(1, device=device, dtype=torch.long))
        
        # 处理中间文本
        text_idx = ts_idx + 1
        if text_idx < len(text_parts) and text_parts[text_idx]:
            text_tokens = tokenizer(
                text_parts[text_idx], 
                return_tensors="pt", 
                add_special_tokens=False # 中间文本不加特殊 Token
            ).to(device)
            text_embed = model.llm.embed(text_tokens.input_ids)
            text_mask = text_tokens.attention_mask
            segment_embeds.append(text_embed[0])
            segment_masks.append(text_mask[0])
    
    # 3. 最终组装
    if segment_embeds:
        full_embed = torch.cat(segment_embeds, dim=0)
        full_mask = torch.cat(segment_masks, dim=0)
        
        # 添加生成触发符 (BOS/EOS 视模型而定，通常生成任务需要一个触发)
        # 这里逻辑保持原样，追加一个 token 提示 LLM 开始生成
        bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
        bos_tensor = torch.tensor([[bos_token_id]], device=device)
        bos_embed = model.llm.embed(bos_tensor)
        bos_mask = torch.ones(1, device=device, dtype=torch.long)
        
        full_embed = torch.cat([full_embed, bos_embed[0]], dim=0)
        full_mask = torch.cat([full_mask, bos_mask], dim=0)
        
        assembled_embeds = full_embed.unsqueeze(0)
        attention_mask = full_mask.unsqueeze(0)
    else:
        assembled_embeds = torch.empty(1, 0, model.config.llm_embed_dim, device=device)
        attention_mask = torch.empty(1, 0, device=device, dtype=torch.long)
        
    return assembled_embeds, attention_mask

def compute_test_loss(model, dataloader, device, rank):
    """计算测试集 Loss - 支持分布式汇总"""
    model.eval()
    local_loss_sum = 0
    local_count = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    
    iterator = tqdm(dataloader, desc="Computing Test Loss") if rank == 0 else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            for i in range(len(input_texts)):
                input_text = input_texts[i]
                ts_list = timeseries_lists[i]
                output_text = output_texts[i]
                ts_list_device = [ts.unsqueeze(0).to(device) for ts in ts_list]
                prefix = f"User: {input_text}\nAssistant: "
                if len(ts_list_device) > 0: series = ts_list_device[0]
                else: series = torch.zeros(1, model.config.seq_len, 1, device=device)
                
                try:
                    # 使用 forward_chatts (兼容 LoRA)
                    # model(series, ...) 这种调用如果是 DDP 会有问题，但在 test 脚本中 model 没包 DDP
                    # 但为了统一，如果 StatBypassCROMETS1 内部逻辑正确，直接调用实例即可
                    model_out = model.forward_chatts([input_text], [ts_list_device], [output_text])
                    
                    llm_out = model_out["llm_outputs"]
                    logits = llm_out.logits
                    
                    prefix_width = model_out["prefix_mask_lengths"][0]
                    suffix_len = model_out["suffix_mask_lengths"][0]
                    
                    # 重新定位 suffix (因为 forward_chatts 返回的是 padded batch)
                    # forward_chatts 已经处理好了 mask
                    
                    # 这里为了简化，直接用训练时的逻辑可能更稳
                    # 但 forward_chatts 内部可能有 padding，我们只取有效部分
                    valid_len = model_out["attention_mask"][0].sum().item()
                    start_idx = int(valid_len - suffix_len)
                    
                    suffix_logits = logits[0, start_idx:start_idx+suffix_len, :]
                    suffix_labels = model.tokenizer.tokenizer([output_text], return_tensors="pt", padding=True).input_ids.to(device)[0]
                    
                    # 确保长度一致
                    min_len = min(suffix_logits.shape[0], suffix_labels.shape[0])
                    suffix_logits = suffix_logits[:min_len, :]
                    suffix_labels = suffix_labels[:min_len]
                    
                    if min_len > 1:
                        shift_logits = suffix_logits[:-1, :].contiguous()
                        shift_labels = suffix_labels[1:].contiguous()
                        loss = loss_fct(shift_logits, shift_labels)
                        local_loss_sum += loss.item()
                        local_count += 1
                except Exception as e:
                    print(f"[Warning Rank {rank}] Error computing loss: {e}")
                    continue
    
    if dist.is_initialized():
        stats = torch.tensor([local_loss_sum, local_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss_sum = stats[0].item()
        total_count = stats[1].item()
        return total_loss_sum / total_count if total_count > 0 else 0.0
    else:
        return local_loss_sum / local_count if local_count > 0 else 0.0

def generate_predictions(model, dataloader, device, output_file, rank, max_new_tokens=256, max_samples=None):
    """生成回复并保存"""
    model.eval()
    results = []
    
    tokenizer = model.tokenizer.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    sample_count = 0
    iterator = tqdm(dataloader, desc=f"Rank {rank} Inference") if rank == 0 else dataloader

    with torch.no_grad():
        for batch in iterator:
            if max_samples is not None and sample_count >= max_samples: break
            
            input_texts = batch["input_texts"]
            timeseries_lists = batch["timeseries_lists"]
            output_texts = batch["output_texts"]
            
            for i in range(len(input_texts)):
                if max_samples is not None and sample_count >= max_samples: break
                
                input_text = input_texts[i]
                ts_list = timeseries_lists[i]
                ground_truth = output_texts[i]
                full_input = f"User: {input_text}\nAssistant: "
                
                try:
                    inputs_embeds, attention_mask = build_chatts_embeddings_for_inference(
                        model, full_input, ts_list, device
                    )
                    
                    # 兼容 LoRA: model.llm.model 可能是 PeftModel
                    # generate 方法通常是透传的，但最好显式调用
                    output_ids = model.llm.model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=10,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        do_sample=False,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )
                    
                    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    
                    results.append({
                        "id": f"r{rank}_s{sample_count}",
                        "input": input_text,
                        "ground_truth": ground_truth,
                        "prediction": pred_text
                    })
                    sample_count += 1
                except Exception as e:
                    print(f"[Warning Rank {rank}] Gen error: {e}")
                    continue
    
    temp_file = f"{output_file}.rank{rank}"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f">>> Rank {rank} saved {len(results)} samples to {temp_file}")

# --- 3. 主程序入口 ---

def main():
    parser = argparse.ArgumentParser(description="Test Stage 3 (ChatTS Instruct) Model - Multi GPU")
    parser.add_argument("--jsonl-path", type=str, default="chatts_data.jsonl")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="chatts_stage3_test_results.jsonl")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-3.2-3B")
    parser.add_argument("--calc-loss", action="store_true")
    parser.add_argument("--num-gen-samples", type=int, default=100)
    
    # === 2. 新增：LoRA 参数 ===
    parser.add_argument("--use-lora", action="store_true", help="是否加载 LoRA 模型进行推理")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    input_channels = 1
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    
    # 配置
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=f"cuda:{local_rank}",
        llm_dtype="bfloat16",
        use_stats_projector=True,
    )
    
    # 初始化基础模型
    model = StatBypassCROMETS1(config).to(device)
    
    # === 3. 关键：在加载权重之前应用 LoRA ===
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not installed but --use-lora requested.")
        
        if rank == 0:
            print(f">>> Applying LoRA config (r={args.lora_r}, alpha={args.lora_alpha}) to model...")
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True, # 推理模式
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # 注入 LoRA
        model.llm.model = get_peft_model(model.llm.model, peft_config)
    
    # === 4. 加载权重 ===
    if Path(args.checkpoint).exists():
        if rank == 0:
            print(f">>> Loading Checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # strict=False 允许加载 LoRA 权重 (它们现在在 model 结构里了)
        msg = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(f">>> Weights Loaded. Msg: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model.eval()

    # 数据集
    val_ds = ChatTSDataset(
        args.jsonl_path, 
        seq_len=args.seq_len, # 作为 max_len
        input_channels=input_channels, 
        split="val",
        patch_stride=args.patch_stride # <--- 新增：传入 stride
    )
    sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    
    if args.calc_loss:
        loss_loader = DataLoader(val_ds, batch_size=1, sampler=sampler, collate_fn=chatts_collate_fn)
        avg_loss = compute_test_loss(model, loss_loader, device, rank)
        if rank == 0:
            print(f"Test Set Loss: {avg_loss:.4f}")
    
    gen_loader = DataLoader(val_ds, batch_size=1, sampler=sampler, collate_fn=chatts_collate_fn)
    local_max_samples = None
    if args.num_gen_samples > 0:
        import math
        local_max_samples = math.ceil(args.num_gen_samples / world_size)
    
    generate_predictions(model, gen_loader, device, args.output_file, rank, max_samples=local_max_samples)
    
    if dist.is_initialized():
        dist.barrier()
        
    if rank == 0:
        print(">>> Merging output files...")
        merged_results = []
        for r in range(world_size):
            fname = f"{args.output_file}.rank{r}"
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as f:
                    for line in f:
                        merged_results.append(json.loads(line))
                os.remove(fname)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for res in merged_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f">>> Final results saved to {args.output_file} (Total: {len(merged_results)})")

    cleanup_distributed()

if __name__ == "__main__":
    main()