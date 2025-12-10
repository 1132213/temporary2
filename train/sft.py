import sys
from pathlib import Path

# 路径设置：确保能找到 src 和 test 包
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 引入项目模块
from src.crome_ts.model import CROMEConfig, StatBypassCROMETS1, get_llm_embed_dim
from src.crome_ts.data_instruct import ChatTSDataset, chatts_collate_fn, WeightedMixingDataset
from test.common import set_seed
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import logging
from datetime import datetime
import math
import os
import subprocess 
from transformers import get_cosine_schedule_with_warmup

# === 新增：PEFT 库 ===
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA will not be available.")

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

    # 1. 配置
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
        epsilon=1e-3, # 推荐: 1e-3 防止梯度爆炸
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

    # 3. LoRA
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
        
        # 1. 强制解冻 LLM 所有参数
        # 注意：model.llm 是 FrozenLLM 类，model.llm.model 才是 HF 的 AutoModel
        for param in model.llm.model.parameters():
            param.requires_grad = True
    # 4. 冻结策略
    if args.freeze_encoder:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.ts_model.shape_encoder.parameters():
            p.requires_grad = True
    
    # 5. DDP
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    encoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "ts_model.shape_encoder" in name or "ts_model.preprocessor" in name or "ts_model.qformer" in name:
            encoder_params.append(param) # LR = 1e-5
        else:
            # 此时这里只剩下 llm.model (LoRA), detail_proj, film_generator
            other_params.append(param)   # LR = 1e-4

    if rank == 0 and logger:
        logger.info(f"-" * 40)
        logger.info(f">>> [Differential LR Strategy Applied]")
        logger.info(f">>> Encoder Params (LR={args.lr * 0.1:.2e}): {len(encoder_params)} tensors")
        logger.info(f">>> Other Params   (LR={args.lr:.2e})    : {len(other_params)} tensors")
        logger.info(f"-" * 40)

    optimizer = AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': encoder_params, 'lr': args.lr * 0.1}
    ], weight_decay=args.weight_decay)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 6. 数据加载
    paths = args.mix_jsonl_paths.split(',')
    probs = [float(p) for p in args.mix_probs.split(',')]
    eval_path = args.eval_jsonl_path.strip() # 主要关注的数据集
    
    if rank == 0 and logger:
        logger.info(f">>> [Dynamic Mixing Mode] Loading {len(paths)} datasets...")
        logger.info(f"    Train Mix Paths: {paths}")
        logger.info(f"    Eval Split Path: {eval_path}")
        
    sub_datasets = []
    for p in paths:
        p = p.strip()
        # === 修改点 2: 智能切分逻辑 ===
        # 只有当路径等于 eval_path 时，才进行 90% 切分，否则 100% 全用
        current_split_ratio = 0.9 if p == eval_path else 1.0
        
        ds = ChatTSDataset(
            p, 
            args.seq_len, 
            input_channels, 
            split="train", 
            split_ratio=current_split_ratio, # 动态传入
            patch_stride=args.patch_stride
        )
        sub_datasets.append(ds)
        
    train_ds = WeightedMixingDataset(
        sub_datasets, 
        probs, 
        epoch_len=None, 
        seed=args.seed
    )
    
    # === 修改点 3: 验证集加载 ===
    # 强制加载 eval_path 的后 10%
    if rank == 0 and logger:
        logger.info(f">>> Loading Validation Set (last 10%) from {eval_path}...")

    val_ds = ChatTSDataset(
        eval_path, 
        args.seq_len, 
        input_channels, 
        split="val", 
        split_ratio=0.9, # 切出后10%
        patch_stride=args.patch_stride
    )

    
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=chatts_collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        collate_fn=chatts_collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    
    # 1. 计算总优化步数 (Total Training Steps)
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    # 2. 计算预热步数
    warmup_steps = int(max_train_steps * 0.02)
    
    if rank == 0:
        logger.info(f">>> Scheduler: Cosine with Warmup")
        logger.info(f">>> Total Optimization Steps: {max_train_steps}")
        logger.info(f">>> Warmup Steps: {warmup_steps} (Ratio: 0.02)")
        eval_steps_interval = int(len(train_loader) * args.eval_interval)
        logger.info(f">>> Eval Interval: Every {args.eval_interval} epoch (~{eval_steps_interval} steps)")

    # 3. 初始化调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # --- 辅助函数：执行评测和保存 ---
    # [修改点1]: 增加 current_train_loss 参数
    best_val_loss = float("inf")
    
    def run_evaluation_pipeline(current_epoch, current_step, current_best_loss, current_train_loss):
        if rank == 0:
            logger.info(f"Step {current_step} [Eval] Computing Validation Loss...")
        
        # 1. 计算 Val Loss
        avg_val_loss = evaluate(model, val_loader, device, rank)
        
        if rank == 0:
            # [修改点2]: 日志输出包含 Train Loss
            logger.info(f"Epoch {current_epoch+1} (Step {current_step}) | Train Loss: {current_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # 2. 保存 Best Model
            if avg_val_loss < current_best_loss:
                logger.info(f">>> Val Loss Improved ({current_best_loss:.4f} -> {avg_val_loss:.4f}). Saving Model...")
                new_best_loss = avg_val_loss
                
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

                # 3. 自动评测 (test_exam.py)
                if args.test_exam_path:
                    logger.info(f">>> [Auto-Eval] Triggering evaluation...")
                    eval_log_name = f"sft_eval_result_e{current_epoch+1}_s{current_step}_{args.model_suffix}.jsonl"
                    eval_output_path = project_root / "log" / eval_log_name
                    
                    cmd = [
                        sys.executable, "train/test_exam.py",
                        "--jsonl-path", args.test_exam_path,
                        "--checkpoint", str(save_path),
                        "--output-file", str(eval_output_path),
                        "--llm-model-path", args.llm_model_path,
                        "--seq-len", str(args.seq_len),
                        "--patch-len", str(args.patch_len),
                        "--patch-stride", str(args.patch_stride),
                        "--num-gen-samples", str(args.eval_num_samples),
                    ]
                    
                    if args.use_lora:
                        cmd.extend(["--use-lora", "--lora-r", str(args.lora_r), "--lora-alpha", str(args.lora_alpha)])
                    
                    try:
                        clean_env = os.environ.copy()
                        for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                            clean_env.pop(key, None)
                        
                        clean_env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

                        result = subprocess.run(cmd, env=clean_env, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            logger.info("\n" + "="*30 + f" EVAL REPORT (Step {current_step}) " + "="*30)
                            logger.info(result.stdout)
                            logger.info("="*80)
                        else:
                            logger.error(f">>> [Auto-Eval] Failed. Stderr: {result.stderr}")
                    except Exception as e:
                        logger.error(f">>> [Auto-Eval] Exception: {e}")
                
                return new_best_loss
            
        return current_best_loss

    # 7. 训练循环
    accumulation_steps = args.gradient_accumulation_steps
    current_grad_norm = 0.0
    
    # eval_steps_interval = int(len(train_loader) * args.eval_interval)
    # if eval_steps_interval <= 0: eval_steps_interval = len(train_loader)
    num_updates_per_epoch = len(train_loader) // accumulation_steps
    
    # 2. 计算每多少次更新评测一次
    eval_updates_interval = int(num_updates_per_epoch * args.eval_interval)
    if eval_updates_interval < 1: 
        eval_updates_interval = 1
        
    # 3. 换算回 Dataloader 的 Step 数
    eval_steps_interval = eval_updates_interval * accumulation_steps

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)
        
        model.train()
        train_loss_sum = 0
        
        if rank == 0:
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Instruct-LoRA]")
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
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if isinstance(grad_norm, torch.Tensor):
                    current_grad_norm = grad_norm.item()
                else:
                    current_grad_norm = grad_norm
                
                if math.isnan(current_grad_norm) or math.isinf(current_grad_norm):
                    if rank == 0:
                        logger.warning(f"!!! NaN/Inf Gradient (norm={current_grad_norm}) at Step {step}.")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                progress.set_postfix(loss=loss.item() * accumulation_steps, lr=f"{current_lr:.2e}", gn=f"{current_grad_norm:.2f}")
            
            # --- 评测逻辑修正 ---
            if (step + 1) % accumulation_steps == 0:
                if ((step + 1) % eval_steps_interval == 0) and ((step + 1) != len(train_loader)):
                    # [修改点3]: 计算 avg_train_loss 并传入
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
            
        # [修改点3]: 计算全 Epoch 的 avg_train_loss 并传入
        avg_train_loss = train_loss_sum / len(train_loader)
        best_val_loss = run_evaluation_pipeline(epoch, len(train_loader), best_val_loss, avg_train_loss)
        
        model.train() 

    if rank == 0 and logger:
        logger.info(">>> ChatTS Instruction Tuning Completed!")

def main():
    parser = argparse.ArgumentParser(description="ChatTS 格式数据 - Stage 3 多卡指令微调 (支持 LoRA)")
    
    # === 修改点 1: 参数调整 ===
    # 移除了 jsonl-path, 增加了 test-exam-path
    parser.add_argument("--mix-jsonl-paths", type=str, required=True, help="逗号分隔的多个训练数据路径")
    parser.add_argument("--mix-probs", type=str, required=True, help="逗号分隔的混合概率")
    parser.add_argument("--eval-jsonl-path", type=str, required=True, help="用于验证 Loss 的数据集路径 (切分10%)")
    
    # 重命名原有的自动评测路径参数，避免混淆
    parser.add_argument("--test-exam-path", type=str, default="/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.json", help="[可选] 自动评测脚本(test_exam.py)用的测试集路径")
    
    # 其他参数...
    parser.add_argument("--eval-num-samples", type=int, default=746)
    parser.add_argument("--eval-interval", type=float, default=1.0)
    parser.add_argument("--stage2-checkpoint", type=str, default="model/chatts_stage2_aligned.pth")
    # ... (其余参数保持不变) ...
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
                        help="Dropout rate for MLP projectors (DetailProj & LLMProj).")
    
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    try:
        train_chatts_instruct_ddp(args, rank, world_size, local_rank)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()