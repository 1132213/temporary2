#!/bin/bash

# ChatTS 格式 Stage 2 对齐训练脚本
# 使用方法: bash train_chatts_alignment.sh

# 设置参数
# JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/align_256/train_cleaned.jsonl"
JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/align_256/train_cleaned_2w.jsonl"

PRETRAINED_ENCODER="model/patchtst_pretrained_full.pth"
SEQ_LEN=256
BATCH_SIZE=4
GRADIENT_ACCUM=8  # 实际批次大小 = 4 × 8 = 32
EPOCHS=10
LR=1e-3
DEVICE="cuda:0"
GPU_ID=0

# LLM 模型路径
LLM_MODEL_PATH="/root/emhua/btwu/Llama-3.2-3B"

# 模型架构参数
PATCH_LEN=16
PATCH_STRIDE=8

# 训练参数
WEIGHT_DECAY=0.01
SEED=42

# 检查数据文件是否存在
if [ ! -f "$JSONL_PATH" ]; then
    echo "错误: 数据文件不存在: $JSONL_PATH"
    exit 1
fi

# 检查 LLM 模型是否存在
if [ ! -d "$LLM_MODEL_PATH" ]; then
    echo "错误: LLM 模型不存在: $LLM_MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "ChatTS Stage 2: 对齐训练"
echo "=========================================="
echo "数据文件: $JSONL_PATH"
echo "预训练编码器: $PRETRAINED_ENCODER"
echo "序列长度: $SEQ_LEN"
echo "批次大小: $BATCH_SIZE × $GRADIENT_ACCUM = $((BATCH_SIZE * GRADIENT_ACCUM))"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "设备: $DEVICE"
echo "=========================================="

# 构建命令
CMD="python train/train_chatts_alignment.py \
    --jsonl-path $JSONL_PATH \
    --pretrained-encoder-path $PRETRAINED_ENCODER \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --device $DEVICE \
    --gpu-id $GPU_ID \
    --llm-model-path $LLM_MODEL_PATH \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --weight-decay $WEIGHT_DECAY \
    --seed $SEED"

echo ""
echo "开始训练..."
echo ""

# 执行训练
eval $CMD

echo ""
echo "=========================================="
echo "对齐训练完成"
echo "模型保存在: model/chatts_stage2_aligned.pth"
echo "=========================================="

