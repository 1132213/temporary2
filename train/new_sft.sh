#!/bin/bash

# ChatTS 指令微调 - 多卡分布式训练启动脚本
NUM_GPUS=${1:-4}

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "未指定 CUDA_VISIBLE_DEVICES，将使用前 ${NUM_GPUS} 张GPU"
else
    echo "使用指定的GPU: $CUDA_VISIBLE_DEVICES"
fi

# 训练参数配置
LAST_MODEL="8b_tattn_16_1208_stride8_residual"
MODEL_SUFFIX="8b_tattn_16_1208_stride8_residual"

STAGE2_CHECKPOINT="model/aligned_$LAST_MODEL.pth"
LLM_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"

# 数据集
SFT_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/ChatTS/sft/train.jsonl"
IFT_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/ChatTS/ift/train.jsonl"
ALIGN_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/ChatTS/align_random/train.jsonl"

# 自动评测脚本用的测试集 (如果需要)
TEST_EXAM_DATA="/mnt/shared-storage-user/huaermo/code/test_wbt2/convert.jsonl"

# 混合配置
MIX_PATHS="$SFT_DATA,$IFT_DATA,$ALIGN_DATA"
MIX_PROBS="0.6,0.1,0.3"

# 代码将从 SFT_DATA 切出 10% 做 Loss 验证，IFT 和 ALIGN 将 100% 参与训练
EVAL_DATA="$SFT_DATA"

# 训练超参数
BATCH_SIZE=1
GRADIENT_ACCUM=16
EPOCHS=2
LR=1e-4
SEQ_LEN=1024
PATCH_LEN=16
PATCH_STRIDE=8
INTERVAL=0.25

echo "=========================================="
echo " ChatTS sft "
echo "=========================================="

CMD="torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    train/sft.py \
    --mix-jsonl-paths \"$MIX_PATHS\" \
    --mix-probs \"$MIX_PROBS\" \
    --eval-jsonl-path \"$EVAL_DATA\" \
    --test-exam-path \"$TEST_EXAM_DATA\" \
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
    --lora-alpha 128 \
    --eval-interval $INTERVAL "
    

if [ -n "$MODEL_SUFFIX" ]; then
    CMD="$CMD --model-suffix \"$MODEL_SUFFIX\""
fi

eval $CMD
echo ""
echo "训练完成！"