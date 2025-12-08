#!/bin/bash

# ChatTS 对齐训练 - 多卡分布式训练启动脚本
export TOKENIZERS_PARALLELISM=false
NUM_GPUS=${1:-4}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "未指定 CUDA_VISIBLE_DEVICES，将使用前 ${NUM_GPUS} 张GPU"
else
    echo "使用指定的GPU: $CUDA_VISIBLE_DEVICES"
fi

# 训练参数配置
LAST_MODEL="16"
MODEL_SUFFIX="8b_tattn_16_1208_stride8_residual"

# 数据路径定义
PRETRAINED_PATH="model/encoder_$LAST_MODEL.pth"
LLM_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"

# 数据集
IFT_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/ChatTS/ift/train.jsonl"
ALIGN_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/ChatTS/align_256/train.jsonl"

# 混合配置
MIX_PATHS="$ALIGN_DATA,$IFT_DATA"
MIX_PROBS="0.9,0.1"

# === 关键修改：指定验证集来源 ===
# 这将导致代码只从 ALIGN_DATA 切分 10% 做验证，IFT_DATA 将 100% 用于训练
EVAL_DATA="$ALIGN_DATA"

# 训练超参数
BATCH_SIZE=2
GRAD_ACCUM=16
EPOCHS=3
LR=1e-3
SEQ_LEN=1024
PATCH_LEN=16
PATCH_STRIDE=8
WEIGHT_DECAY=0.01

echo "=========================================="
echo "ChatTS 对齐训练 (Mix: Align+IFT, Val: Align 10%)"
echo "=========================================="

CMD="torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train/alignment.py \
    --mix-jsonl-paths \"$MIX_PATHS\" \
    --mix-probs \"$MIX_PROBS\" \
    --eval-jsonl-path \"$EVAL_DATA\" \
    --pretrained-encoder-path \"$PRETRAINED_PATH\" \
    --llm-model-path \"$LLM_PATH\" \
    --seq-len $SEQ_LEN \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --weight-decay $WEIGHT_DECAY \
    --lr $LR \
    --epochs $EPOCHS \
    --num-workers 4 \
    --seed 42"

if [ -n "$MODEL_SUFFIX" ]; then
    CMD="$CMD --model-suffix \"$MODEL_SUFFIX\""
fi

eval $CMD
echo ""
echo "训练完成！"