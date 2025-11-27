#!/bin/bash

# ChatTS 指令微调 - 多卡分布式训练启动脚本
# 使用方法: 
#   bash run_ddp_instruct.sh [GPU数量]
#   或者指定GPU: CUDA_VISIBLE_DEVICES=4,5 bash run_ddp_instruct.sh 2

# 默认使用的GPU数量
NUM_GPUS=${1:-4}

# 如果没有设置 CUDA_VISIBLE_DEVICES，使用默认的前 NUM_GPUS 张卡
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "未指定 CUDA_VISIBLE_DEVICES，将使用前 ${NUM_GPUS} 张GPU"
else
    echo "使用指定的GPU: $CUDA_VISIBLE_DEVICES"
fi

# 训练参数配置
JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/train_cleaned.jsonl"
STAGE2_CHECKPOINT="model/chatts_stage2_aligned_ddp.pth"
LLM_PATH="/root/emhua/btwu/Llama-3.2-3B"

# 训练超参数
BATCH_SIZE=4          # 每个GPU的批次大小
GRADIENT_ACCUM=8      # 梯度累积步数
EPOCHS=20
LR=5e-5               # 微调阶段使用较小的学习率
SEQ_LEN=512
PATCH_LEN=16
PATCH_STRIDE=8

# 计算有效批次大小
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUM))

echo "=========================================="
echo "ChatTS 指令微调 - 多卡分布式训练"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "每卡批次大小: $BATCH_SIZE"
echo "梯度累积步数: $GRADIENT_ACCUM"
echo "有效批次大小: $EFFECTIVE_BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "Stage 2 权重: $STAGE2_CHECKPOINT"
echo "=========================================="
echo ""

# 使用 torchrun 启动分布式训练
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train/train_chatts_instruct_ddp.py \
    --jsonl-path "$JSONL_PATH" \
    --stage2-checkpoint "$STAGE2_CHECKPOINT" \
    --llm-model-path "$LLM_PATH" \
    --seq-len $SEQ_LEN \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --lr $LR \
    --epochs $EPOCHS \
    --num-workers 4 \
    --seed 42 \
    --freeze-encoder

echo ""
echo "训练完成！"

