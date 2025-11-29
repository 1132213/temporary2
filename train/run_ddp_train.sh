#!/bin/bash

# ChatTS 对齐训练 - 多卡分布式训练启动脚本
# 使用方法: 
#   bash run_ddp_train.sh [GPU数量]
#   或者指定GPU: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_ddp_train.sh 4

# 禁用 tokenizers 的并行处理以避免 fork 警告
export TOKENIZERS_PARALLELISM=false

# 默认使用的GPU数量
NUM_GPUS=${1:-4}

# 如果没有设置 CUDA_VISIBLE_DEVICES，使用默认的前 NUM_GPUS 张卡
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "未指定 CUDA_VISIBLE_DEVICES，将使用前 ${NUM_GPUS} 张GPU"
else
    echo "使用指定的GPU: $CUDA_VISIBLE_DEVICES"
fi

# 训练参数配置
LAST_MODEL="nonoverlap"
# MODEL_SUFFIX=$LAST_MODEL
MODEL_SUFFIX="stat"

# JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/align_256/train_cleaned.jsonl"
JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/my_data/alignment.jsonl"
PRETRAINED_PATH="model/patchtst_pretrained_full_$LAST_MODEL.pth"
LLM_PATH="/root/emhua/btwu/Qwen2.5-3B-Instruct"

# 训练超参数
BATCH_SIZE=2          # 每个GPU的批次大小
GRAD_ACCUM=16          # 梯度累积步数
EPOCHS=5
LR=1e-3
SEQ_LEN=1024
PATCH_LEN=16
PATCH_STRIDE=16
# PATCH_STRIDE=8

# 计算有效批次大小
EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

echo "=========================================="
echo "ChatTS 对齐训练 - 多卡分布式训练"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "每卡批次大小: $BATCH_SIZE"
echo "梯度累积步数: $GRAD_ACCUM"
echo "有效批次大小: $EFFECTIVE_BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "=========================================="
echo ""

# 构建训练命令
CMD="torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train/train_chatts_alignment_ddp.py \
    --jsonl-path \"$JSONL_PATH\" \
    --pretrained-encoder-path \"$PRETRAINED_PATH\" \
    --llm-model-path \"$LLM_PATH\" \
    --seq-len $SEQ_LEN \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --lr $LR \
    --epochs $EPOCHS \
    --num-workers 4 \
    --seed 42"

# 如果设置了模型后缀，添加参数
if [ -n "$MODEL_SUFFIX" ]; then
    CMD="$CMD --model-suffix \"$MODEL_SUFFIX\""
    echo "模型后缀: $MODEL_SUFFIX"
fi

# 使用 torchrun 启动分布式训练
eval $CMD

echo ""
echo "训练完成！"

