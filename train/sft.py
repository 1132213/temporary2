import sys
from pathlib import Path
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
from datetime import datetime
import math
import os

# --- 项目模块导入 ---
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn, WeightedMixingDataset
from test.common import set_seed

# === [修改点] 导入 autoeval 模块 ===
try:
    from train.autoeval import run_internal_eval
except ImportError:
    try:
        from autoeval import run_internal_eval
    except ImportError:
        print("Warning: Could not import 'run_internal_eval' from autoeval.py. Auto-eval will fail.")
        run_internal_eval = None

# === PEFT 库 ===
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到分布式环境变量，使用单卡模式")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_chatts_loss(model, batch, device):
    """
    计算 ChatTS 格式的 Causal LM Loss
    """
    input_texts = batch["input_texts"]
    timeseries_lists = batch["timeseries_lists"]
    output_texts = batch["output_texts"]
    
    if isinstance(model, DDP):
        tokenizer = model.module.tokenizer.tokenizer
    else:
        tokenizer = model.tokenizer.tokenizer
        
    eos_token = tokenizer.eos_token
    output_texts_with_eos = [str(text) + eos_token for text in output_texts]
    
    actual_model = model.module if isinstance(model, DDP) else model
    model_out = actual_model.forward_chatts(
        input_texts=input_texts,
        timeseries_lists=timeseries_lists,
        output_texts=output_texts_with_eos,
        llm_kwargs={}
    )
    
    llm_out = model_out["llm_outputs"]
    logits = llm_out.logits
    
    suffix_labels = tokenizer(
        output_texts_with_eos, 
        return_tensors="pt", 
        padding=True,
        add_special_tokens=True 
    ).input_ids.to(device)
    
    batch_size = logits.shape[0]
    suffix_mask_lengths = model_out["suffix_mask_lengths"]
    
    losses = []
    for i in range(batch_size):
        valid_len = model_out["attention_mask"][i].sum().item()
        suffix_len = suffix_mask_lengths[i]
        
        if suffix_len == 0:
            continue
        
        suffix_start = int(valid_len - suffix_len)
        
        sample_logits = logits[i, suffix_start:suffix_start+suffix_len, :] 
        sample_labels = suffix_labels[i, :suffix_len]
        
        min_len = min(sample_logits.shape[0], sample_labels.shape[0])
        sample_logits = sample_logits[:min_len]
        sample_labels = sample_labels[:min_len]
        
        if sample_logits.shape[0] > 1:
            shift_logits = sample_logits[:-1, :] 
            shift_labels = sample_labels[1:] 
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return torch.stack(losses).mean()

def evaluate(model, val_loader, device, rank):
    """验证函数"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss = compute_chatts_loss(model, batch, device)
            total_loss += loss.item()
            num_batches += 1
    
    if dist.is_initialized():
        loss_tensor = torch.tensor([total_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        num_batches = int(loss_tensor[1].item())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def train_chatts_instruct_ddp(args, rank, world_size, local_rank):
    """
    多卡分布式训练主函数 - Stage 3 指令微调
    """
    device = torch.device(f"cuda:{local_rank}")
    
    # --- 日志设置 ---
    logger = None
    log_dir = None
    model_dir = None
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if rank == 0:
        log_dir = project_root / "log"
        model_dir = project_root / "model"
        log_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)
        
        log_filename = log_dir / f"chatts_instruct_ddp_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"日志文件保存在: {log_filename}")
        logger.info(f"分布式训练: Rank {rank}/{world_size}, Local Rank: {local_rank}")
    
    set_seed(args.seed + rank)

    # 1. 配置与模型初始化
    llm_embed_dim = get_llm_embed_dim(args.llm_model_path)
    if rank == 0 and logger:
        logger.info(f">>> LLM Embedding Dimension: {llm_embed_dim} (from {args.llm_model_path})")
    
    input_channels = 1
    
    config = CROMEConfig(
        input_channels=input_channels,
        llm_embed_dim=llm_embed_dim, 
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        llm_model_path=args.llm_model_path,
        llm_device_map=f"cuda:{local_rank}",
        llm_dtype="bfloat16",
        freeze_patch_encoder=args.freeze_encoder,
        epsilon=1e-3, 
        proj_dropout=args.proj_dropout
    )
    
    if rank == 0 and logger:
        logger.info(">>> Initializing Model for ChatTS Instruction Tuning...")
    
    model = StatBypassCROMETS1(config).to(device)

    # 2. 加载 Stage 2 权重
    stage2_path = Path(args.stage2_checkpoint)
    if not stage2_path.is_absolute():
        if not stage2_path.exists():
            model_path = project_root / "model" / stage2_path.name
            if model_path.exists():
                stage2_path = model_path
            else:
                root_path = project_root / stage2_path
                if root_path.exists():
                    stage2_path = root_path
    
    if stage2_path.exists():
        if rank == 0 and logger:
            logger.info(f">>> Loading Stage 2 Aligned Weights from {stage2_path}...")
        
        state_dict = torch.load(str(stage2_path), map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if rank == 0 and logger:
            logger.info(f"Weights Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        if rank == 0 and logger:
            logger.warning(f"!!! Stage 2 checkpoint not found at {stage2_path}! Training from scratch (NOT RECOMMENDED). !!!")

    # 3. LoRA 配置
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT library is not installed. Please install it to use LoRA.")
        
        if rank == 0 and logger:
            logger.info(f">>> Applying LoRA to LLM (r={args.lora_r}, alpha={args.lora_alpha})...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model.llm.model = get_peft_model(model.llm.model, peft_config)
        
        if rank == 0 and logger:
            model.llm.model.print_trainable_parameters()
            logger.info(">>> LoRA applied successfully.")
    else:
        if rank == 0 and logger:
            logger.info(">>> Full Fine-tuning Mode Detected (No LoRA). Unfreezing LLM...")
        for param in model.llm.model.parameters():
            param.requires_grad = True

    # ================= [关键修改] 开启梯度检查点 (解决 OOM) =================
    if hasattr(model.llm.model, "gradient_checkpointing_enable"):
        model.llm.model.gradient_checkpointing_enable()
        if rank == 0 and logger:
            logger.info(">>> Gradient Checkpointing ENABLED. (Memory Efficient Mode)")
    # ======================================================================

    # 4. 冻结策略
    if args.freeze_encoder:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = True
    
    # 5. DDP 包装
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # 分组参数设置学习率
    # 分组参数设置学习率
    ts_params = []  # 整个时序模型 (Low LR)
    llm_params = [] # LLM & LoRA (High LR)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # [修改] 只要是 ts_model 下的参数，全部归为低学习率
        # 这包括了 shape_encoder, detail_encoder(CNN), qformer, adapter, llm_proj, fusion_gate
        if "ts_model" in name:
            ts_params.append(param) # LR = 1e-5 (args.lr * 0.1)
        else:
            # 剩下的就是 llm.model (LoRA)
            llm_params.append(param)   # LR = 1e-4 (args.lr)

    optimizer = AdamW([
        {'params': llm_params, 'lr': args.lr},
        {'params': ts_params, 'lr': args.lr * 0.1}
    ], weight_decay=args.weight_decay)

    if rank == 0 and logger:
        logger.info(f"-" * 40)
        logger.info(f">>> [Differential LR Strategy Applied]")
        logger.info(f">>> Encoder Params (LR={args.lr * 0.1:.2e}): {len(ts_params)} tensors")
        logger.info(f">>> Other Params   (LR={args.lr:.2e})    : {len(llm_params)} tensors")
        logger.info(f"-" * 40)

    optimizer = AdamW([
        {'params': llm_params, 'lr': args.lr},
        {'params': ts_params, 'lr': args.lr * 0.1}
    ], weight_decay=args.weight_decay)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 6. 数据加载
    paths = args.mix_jsonl_paths.split(',')
    probs = [float(p) for p in args.mix_probs.split(',')]
    eval_path = args.eval_jsonl_path.strip() 
    
    if rank == 0 and logger:
        logger.info(f">>> [Dynamic Mixing Mode] Loading {len(paths)} datasets...")
        
    sub_datasets = []
    for p in paths:
        p = p.strip()
        current_split_ratio = 0.9 if p == eval_path else 1.0
        ds = ChatTSDataset(
            p, 
            args.seq_len, 
            input_channels, 
            split="train", 
            split_ratio=current_split_ratio, 
            patch_stride=args.patch_stride
        )
        sub_datasets.append(ds)
        
    train_ds = WeightedMixingDataset(
        sub_datasets, 
        probs, 
        epoch_len=None, 
        seed=args.seed
    )
    
    val_ds = ChatTSDataset(
        eval_path, 
        args.seq_len, 
        input_channels, 
        split="val", 
        split_ratio=0.9, 
        patch_stride=args.patch_stride
    )
    if rank == 0:
        logger.info(">>> [Smart Batching] Sorting dataset by length to boost speed...")
    
    # 获取所有样本的长度（这里我们以 text 长度近似，或者您如果有 ts 长度更好）
    # 对于 WeightedMixingDataset，我们需要处理一下子数据集
    
    # 简单的策略：不做复杂的 Sampler，直接在 Dataset 层面把数据排好序
    # 注意：这会轻微破坏随机性（Randomness），但在 SFT 阶段通常是可以接受的，
    # 或者可以只在每个 Epoch 开始时做 "Chunked Shuffle" (局部乱序)
    
    # 这里提供一个最简单有效的补丁：直接对 sub_datasets 内部进行排序
    if isinstance(train_ds, WeightedMixingDataset):
        for sub_ds in train_ds.datasets:
            # 假设 sub_ds.records 是列表
            if hasattr(sub_ds, 'records'):
                # 按照 input 文本长度 + 时序长度 进行排序
                # 这样长数据会聚在一起，极大减少 Padding
                sub_ds.records.sort(key=lambda x: len(x.get('input', '')) + len(x.get('context', '')))
    
    # =========================================================

    # 修改 Sampler：使用 SequentialSampler 或者保持 DistributedSampler 但关闭 shuffle
    # 为了保持一定的随机性，建议使用 DistributedSampler(shuffle=False) 配合我们刚才的手动排序
    # 这样每个 GPU 分到的是一段连续的（长度相近的）数据
    
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, # 注意这里 shuffle=False
        collate_fn=chatts_collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    # train_sampler = DistributedSampler(
    #     train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    # ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    
    # train_loader = DataLoader(
    #     train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
    #     collate_fn=chatts_collate_fn, num_workers=args.num_workers, pin_memory=True
    # )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        collate_fn=chatts_collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    # Scheduler 配置
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * 0.05)
    
    if rank == 0:
        logger.info(f">>> Total Optimization Steps: {max_train_steps}")
        logger.info(f">>> Warmup Steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # ==============================================================================
    # 辅助函数：执行评测和保存 (In-Process Eval + 分布式同步)
    # ==============================================================================
    best_val_loss = float("inf")
    def log_exam_stats(results_list, logger):
        if not results_list: return
        
        from collections import defaultdict
        total_stats = {"correct": 0, "total": 0}
        cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for item in results_list:
            cat = item.get("category", "Uncategorized")
            score = item.get("judge_score", 0)
            total_stats["total"] += 1
            cat_stats[cat]["total"] += 1
            if score == 1:
                total_stats["correct"] += 1
                cat_stats[cat]["correct"] += 1
        
        logger.info(f"\n{'='*25} Regex Judge Report (Epoch {epoch+1} Step {step}) {'='*25}")
        logger.info(f"{'Category':<35} | {'Acc':<8} | {'Correct':<8} | {'Total':<8}")
        logger.info("-" * 75)
        
        forced_order = ["Pattern Recognition", "Noise Understanding", "Anomaly Detection", "Similarity Analysis", "Causality Analysis"]
        for cat in forced_order:
            # 兼容拼写错误
            target_key = cat if cat in cat_stats else ("Anolmaly Detection" if "Anolmaly Detection" in cat_stats and cat == "Anomaly Detection" else None)
            if target_key and target_key in cat_stats:
                s = cat_stats[target_key]
                acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
                logger.info(f"{target_key:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

        remaining = sorted([c for c in cat_stats.keys() if c not in forced_order and c != "Anolmaly Detection"])
        for cat in remaining:
            s = cat_stats[cat]
            acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
            logger.info(f"{cat:<35} | {acc:<7.2f}% | {s['correct']:<8} | {s['total']:<8}")

        logger.info("-" * 75)
        total_acc = (total_stats['correct'] / total_stats['total']) * 100 if total_stats['total'] > 0 else 0.0
        logger.info(f"{'OVERALL':<35} | {total_acc:<7.2f}% | {total_stats['correct']:<8} | {total_stats['total']:<8}")
        logger.info("=" * 75)
    def run_evaluation_pipeline(current_epoch, current_step, current_best_loss, current_train_loss):
        if rank == 0:
            logger.info(f"Step {current_step} [Eval] Computing Validation Loss...")
        
        # 1. 计算 Val Loss
        avg_val_loss = evaluate(model, val_loader, device, rank)
        
        # 2. 决策：是否是最佳模型 (由 Rank 0 决定)
        is_best = torch.tensor([0], device=device)
        new_best_loss = current_best_loss
        
        if rank == 0:
            logger.info(f"Epoch {current_epoch+1} (Step {current_step}) | Train Loss: {current_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < current_best_loss:
                is_best[0] = 1
                new_best_loss = avg_val_loss
                logger.info(f">>> Val Loss Improved ({current_best_loss:.4f} -> {avg_val_loss:.4f}).")
        
        # 3. 广播决策 (防止死锁)
        if world_size > 1:
            dist.broadcast(is_best, src=0)
            
        if is_best.item() == 1:
            # 3.1 保存模型 (仅 Rank 0)
            if rank == 0:
                if args.model_suffix:
                    model_filename = f"sft_{args.model_suffix}.pth"
                else:
                    model_filename = "sft.pth"
                save_path = model_dir / model_filename
                
                model_to_save = model.module if isinstance(model, DDP) else model
                
                if args.save_only_trainable:
                    full_state_dict = model_to_save.state_dict()
                    dict_to_save = {}
                    for name, param in model_to_save.named_parameters():
                        if param.requires_grad or "ts_model" in name:
                            dict_to_save[name] = full_state_dict[name]
                    for name, buf in model_to_save.named_buffers():
                        if "ts_model" in name:
                            dict_to_save[name] = buf
                    torch.save(dict_to_save, save_path)
                else:
                    torch.save(model_to_save.state_dict(), save_path)
                
                logger.info(f">>> Best ChatTS Instruct Model saved to: {save_path}")

            # 3.2 自动评测 (所有 Rank 共同参与)
            if args.test_exam_path and run_internal_eval is not None:
                if rank == 0: logger.info(f">>> [Auto-Eval] Triggering in-process evaluation...")
                
                model.eval()
                
                # 获取 unwrapped model
                if isinstance(model, DDP):
                    tokenizer = model.module.tokenizer.tokenizer
                    inference_model = model
                else:
                    tokenizer = model.tokenizer.tokenizer
                    inference_model = model
                
                class EvalArgs:
                    pass
                eval_args = EvalArgs()
                eval_args.jsonl_path = args.test_exam_path
                eval_args.output_file = str(project_root / "log" / f"sft_eval_result_e{current_epoch+1}_s{current_step}_{args.model_suffix}.jsonl")
                eval_args.seq_len = args.seq_len
                eval_args.patch_len = args.patch_len
                eval_args.patch_stride = args.patch_stride
                eval_args.enable_cot = False
                eval_args.mask_query = False
                eval_args.mask_detail = False
                eval_args.mask_text_stats = False
                
                try:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        # 获取评测结果列表
                        final_results = run_internal_eval(
                            model=inference_model, tokenizer=tokenizer, args=eval_args,
                            device=device, rank=rank, world_size=world_size
                        )
                    
                    # [关键修改] Rank 0 负责将统计结果写入 Logger
                    if rank == 0:
                        log_exam_stats(final_results, logger)
                        logger.info(f">>> [Auto-Eval] Finished.")
                        
                except Exception as e:
                    if rank == 0:
                        logger.error(f">>> [Auto-Eval] Exception: {e}")
                        import traceback
                        traceback.print_exc()
                
                model.train()
                torch.cuda.empty_cache()
        
        # 同步 Best Loss
        if world_size > 1:
            loss_tensor = torch.tensor([new_best_loss], device=device)
            dist.broadcast(loss_tensor, src=0)
            new_best_loss = loss_tensor.item()
            
        return new_best_loss

    # 7. 训练循环
    accumulation_steps = args.gradient_accumulation_steps
    eval_interval_steps = max(1, int(len(train_loader) // accumulation_steps * args.eval_interval)) * accumulation_steps

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)
        
        model.train()
        train_loss_sum = 0
        
        if rank == 0:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            progress = train_loader
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            loss = compute_chatts_loss(model, batch, device)
            
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.warning(f"!!! NaN/Inf Loss at Step {step}.")
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            train_loss_sum += loss.item() * accumulation_steps
            
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.2e}")
            
            # --- 步骤内评测 ---
            if (step + 1) % accumulation_steps == 0:
                if ((step + 1) % eval_interval_steps == 0) and ((step + 1) != len(train_loader)):
                    avg_train_loss = train_loss_sum / (step + 1)
                    best_val_loss = run_evaluation_pipeline(epoch, step + 1, best_val_loss, avg_train_loss)
                    model.train()
        
        # End of Epoch Loop
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if dist.is_initialized():
            dist.barrier()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        best_val_loss = run_evaluation_pipeline(epoch, len(train_loader), best_val_loss, avg_train_loss)
        
        model.train() 

    if rank == 0 and logger:
        logger.info(">>> ChatTS Instruction Tuning Completed!")
def main():
    parser = argparse.ArgumentParser(description="ChatTS 格式数据 - Stage 3 多卡指令微调 (支持 LoRA)")
    
    parser.add_argument("--mix-jsonl-paths", type=str, required=True, help="逗号分隔的多个训练数据路径")
    parser.add_argument("--mix-probs", type=str, required=True, help="逗号分隔的混合概率")
    parser.add_argument("--eval-jsonl-path", type=str, required=True, help="用于验证 Loss 的数据集路径 (切分10%)")
    parser.add_argument("--test-exam-path", type=str, default="/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.json", help="[可选] 自动评测脚本用的测试集路径")
    
    parser.add_argument("--eval-num-samples", type=int, default=746)
    parser.add_argument("--eval-interval", type=float, default=1.0)
    parser.add_argument("--stage2-checkpoint", type=str, default="model/chatts_stage2_aligned.pth")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--llm-model-path", type=str, default="/root/emhua/btwu/Llama-2-7b-hf")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5) 
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-suffix", type=str, default="")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-only-trainable", action="store_true")
    parser.add_argument("--proj-dropout", type=float, default=0.05, 
                        help="Dropout rate for MLP projectors.")
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        train_chatts_instruct_ddp(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()