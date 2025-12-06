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
LAST_MODEL="8b_tattn_16_skip_new"
# MODEL_SUFFIX=$LAST_MODEL
MODEL_SUFFIX="8b_tattn_16_skip_new"

# JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/new_merged.jsonl"
JSONL_PATH="/mnt/shared-storage-user/huaermo/code/test_wbt2/sft.jsonl"
STAGE2_CHECKPOINT="model/aligned_$LAST_MODEL.pth"
LLM_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"
# LLM_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen2.5-3B-Instruct"
 # 可选：模型名称后缀，例如设置为 "1st" 则保存为 chatts_instruct_best_ddp_1st.pth

# 训练超参数
BATCH_SIZE=2         # 每个GPU的批次大小
GRADIENT_ACCUM=16      # 梯度累积步数
EPOCHS=4
LR=1e-4               # 微调阶段使用较小的学习率
SEQ_LEN=1024
PATCH_LEN=16
PATCH_STRIDE=16
# PATCH_STRIDE=8

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

# CMD="torchrun --nproc_per_node=$NUM_GPUS \
#     --master_port=29502 \
#     train/sft.py \
#     --jsonl-path \"$JSONL_PATH\" \
#     --stage2-checkpoint \"$STAGE2_CHECKPOINT\" \
#     --llm-model-path \"$LLM_PATH\" \
#     --seq-len $SEQ_LEN \
#     --patch-len $PATCH_LEN \
#     --patch-stride $PATCH_STRIDE \
#     --batch-size $BATCH_SIZE \
#     --gradient-accumulation-steps $GRADIENT_ACCUM \
#     --lr $LR \
#     --epochs $EPOCHS \
#     --num-workers 16 \
#     --save-only-trainable \
#     --weight-decay 0.05 \
#     --seed 42 "

CMD="torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    train/sft.py \
    --jsonl-path \"$JSONL_PATH\" \
    --stage2-checkpoint \"$STAGE2_CHECKPOINT\" \
    --llm-model-path \"$LLM_PATH\" \
    --seq-len $SEQ_LEN \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --lr $LR \
    --epochs $EPOCHS \
    --num-workers 16 \
    --save-only-trainable \
    --seed 42 \
    --use-lora \
    --lora-r 64 \
    --lora-alpha 128 "

# 构建训练命令
# CMD="torchrun --nproc_per_node=$NUM_GPUS \
#     --master_port=29502 \
#     train/train_chatts_instruct_ddp.py \
#     --jsonl-path \"$JSONL_PATH\" \
#     --stage2-checkpoint \"$STAGE2_CHECKPOINT\" \
#     --llm-model-path \"$LLM_PATH\" \
#     --seq-len $SEQ_LEN \
#     --patch-len $PATCH_LEN \
#     --patch-stride $PATCH_STRIDE \
#     --batch-size $BATCH_SIZE \
#     --gradient-accumulation-steps $GRADIENT_ACCUM \
#     --lr $LR \
#     --epochs $EPOCHS \
#     --num-workers 4 \
#     --seed 42 \
#     --freeze-encoder \
#     --use-lora \
#     --lora-r 16 \
#     --lora-alpha 32 "

# 如果设置了模型后缀，添加参数
if [ -n "$MODEL_SUFFIX" ]; then
    CMD="$CMD --model-suffix \"$MODEL_SUFFIX\""
    echo "模型后缀: $MODEL_SUFFIX"
fi

# 使用 torchrun 启动分布式训练
eval $CMD

echo ""
echo "训练完成！"

