#!/bin/bash
# CROME 三阶段训练脚本
# 使用方法: bash train_all_stages.sh

# ==================== 配置区域 ====================
# 请根据实际情况修改以下配置

# 数据路径
DATA_PATH="data/train.jsonl"  # 训练数据路径（JSONL格式）

# LLM模型路径
LLM_PATH="/root/models/Llama-3.2-3B"  # 根据实际情况修改

# GPU设置
GPU_ID=0  # GPU编号，如果使用CPU则设置为-1

# 训练参数
SEQ_LEN=1024          # 时序长度
PATCH_LEN=32          # Patch长度
PATCH_STRIDE=16       # Patch步长

# Stage 1 参数
STAGE1_EPOCHS=10
STAGE1_BATCH_SIZE=128
STAGE1_LR=1e-3

# Stage 2 参数
STAGE2_EPOCHS=10
STAGE2_BATCH_SIZE=32
STAGE2_LR=1e-3
STAGE2_GRAD_ACCUM=1

# Stage 3 参数
STAGE3_EPOCHS=50
STAGE3_BATCH_SIZE=16
STAGE3_LR=5e-5
FREEZE_ENCODER=true  # 是否冻结Encoder，true/false

# 检查点路径
STAGE1_CHECKPOINT="model/patchtst_pretrained_full_3b.pth"
STAGE2_CHECKPOINT="model/crome_stage2_aligned.pth"
STAGE3_CHECKPOINT="model/crome_instruct_best.pth"

# ==================== 训练脚本 ====================

set -e  # 遇到错误立即退出

echo "========================================="
echo "CROME 三阶段训练脚本"
echo "========================================="
echo "数据路径: ${DATA_PATH}"
echo "LLM路径: ${LLM_PATH}"
echo "GPU ID: ${GPU_ID}"
echo "时序长度: ${SEQ_LEN}"
echo "========================================="
echo ""

# 检查数据文件是否存在
if [ ! -f "${DATA_PATH}" ]; then
    echo "错误: 数据文件不存在: ${DATA_PATH}"
    exit 1
fi

# 检查LLM模型路径是否存在
if [ ! -d "${LLM_PATH}" ]; then
    echo "警告: LLM模型路径不存在: ${LLM_PATH}"
    echo "请确认路径是否正确，或修改脚本中的LLM_PATH变量"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 设置设备
if [ "${GPU_ID}" -ge 0 ]; then
    DEVICE="cuda:${GPU_ID}"
else
    DEVICE="cpu"
fi

# ==================== Stage 1: Encoder 预训练 ====================
echo ""
echo "========================================="
echo "Stage 1: Encoder 预训练"
echo "========================================="
echo ""

python train/train_encoder.py \
    --jsonl-path "${DATA_PATH}" \
    --seq-len ${SEQ_LEN} \
    --patch-len ${PATCH_LEN} \
    --patch-stride ${PATCH_STRIDE} \
    --llm-model-path "${LLM_PATH}" \
    --epochs ${STAGE1_EPOCHS} \
    --batch-size ${STAGE1_BATCH_SIZE} \
    --lr ${STAGE1_LR} \
    --device cuda \
    --gpu-id ${GPU_ID}

if [ ! -f "${STAGE1_CHECKPOINT}" ]; then
    echo "错误: Stage 1 检查点未生成: ${STAGE1_CHECKPOINT}"
    exit 1
fi

echo ""
echo "✓ Stage 1 完成! 检查点保存在: ${STAGE1_CHECKPOINT}"
echo ""

# ==================== Stage 2: Alignment 对齐 ====================
echo ""
echo "========================================="
echo "Stage 2: Alignment 对齐训练"
echo "========================================="
echo ""

python train/train_alignment.py \
    --jsonl-path "${DATA_PATH}" \
    --pretrained-encoder-path "${STAGE1_CHECKPOINT}" \
    --seq-len ${SEQ_LEN} \
    --patch-len ${PATCH_LEN} \
    --patch-stride ${PATCH_STRIDE} \
    --llm-model-path "${LLM_PATH}" \
    --batch-size ${STAGE2_BATCH_SIZE} \
    --gradient-accumulation-steps ${STAGE2_GRAD_ACCUM} \
    --lr ${STAGE2_LR} \
    --weight-decay 0.01 \
    --epochs ${STAGE2_EPOCHS} \
    --seed 42 \
    --device cuda \
    --gpu-id ${GPU_ID}

if [ ! -f "${STAGE2_CHECKPOINT}" ]; then
    echo "错误: Stage 2 检查点未生成: ${STAGE2_CHECKPOINT}"
    exit 1
fi

echo ""
echo "✓ Stage 2 完成! 检查点保存在: ${STAGE2_CHECKPOINT}"
echo ""

# ==================== Stage 3: Instruction Tuning ====================
echo ""
echo "========================================="
echo "Stage 3: Instruction Tuning"
echo "========================================="
echo ""

FREEZE_FLAG=""
if [ "${FREEZE_ENCODER}" = "true" ]; then
    FREEZE_FLAG="--freeze-encoder"
    echo "冻结 Encoder: 是"
else
    echo "冻结 Encoder: 否"
fi

python train/train_instruct.py \
    --jsonl-path "${DATA_PATH}" \
    --stage2-checkpoint "${STAGE2_CHECKPOINT}" \
    --seq-len ${SEQ_LEN} \
    --patch-len ${PATCH_LEN} \
    --patch-stride ${PATCH_STRIDE} \
    --llm-model-path "${LLM_PATH}" \
    --batch-size ${STAGE3_BATCH_SIZE} \
    --lr ${STAGE3_LR} \
    --weight-decay 0.01 \
    --epochs ${STAGE3_EPOCHS} \
    --seed 42 \
    --device cuda \
    --gpu-id ${GPU_ID} \
    ${FREEZE_FLAG}

if [ ! -f "${STAGE3_CHECKPOINT}" ]; then
    echo "错误: Stage 3 检查点未生成: ${STAGE3_CHECKPOINT}"
    exit 1
fi

echo ""
echo "✓ Stage 3 完成! 检查点保存在: ${STAGE3_CHECKPOINT}"
echo ""

# ==================== 训练完成 ====================
echo ""
echo "========================================="
echo "训练完成!"
echo "========================================="
echo ""
echo "所有检查点文件:"
echo "  - Stage 1: ${STAGE1_CHECKPOINT}"
echo "  - Stage 2: ${STAGE2_CHECKPOINT}"
echo "  - Stage 3: ${STAGE3_CHECKPOINT}"
echo ""
echo "可以使用以下命令进行测试:"
echo "  python train/test_instruct.py \\"
echo "      --jsonl-path ${DATA_PATH} \\"
echo "      --checkpoint ${STAGE3_CHECKPOINT} \\"
echo "      --llm-model-path ${LLM_PATH}"
echo ""
echo "或进行推理:"
echo "  python inference.py \\"
echo "      --jsonl-path ${DATA_PATH} \\"
echo "      --checkpoint ${STAGE3_CHECKPOINT} \\"
echo "      --llm-model-path ${LLM_PATH}"
echo ""

