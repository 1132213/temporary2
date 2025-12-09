#!/bin/bash

# PatchTST Encoder 预训练脚本
# Stage 1: Masked Autoencoding Pretraining
# 使用方法: bash train/run_train_encoder.sh

# ==================== 配置参数 ====================

# 数据文件路径
JSONL_PATH="/mnt/shared-storage-user/huaermo/code/test_wbt2/gifteval_windows1.jsonl"

# LLM 模型路径（用于获取 embed_dim）
LLM_MODEL_PATH="/mnt/shared-storage-user/dllm-share/Models/Qwen3/Qwen3-8B"

# 序列长度
SEQ_LEN=1024

# 训练参数
BATCH_SIZE=128
EPOCHS=20
LR=1e-3

# 模型架构参数
# PATCH_LEN=8
# PATCH_STRIDE=8

PATCH_LEN=16
PATCH_STRIDE=16

# PATCH_STRIDE=8
INPUT_CHANNELS=1

# GPU 设置
DEVICE="cuda"
GPU_ID=1

# 模型名称后缀（可选）
MODEL_SUFFIX="new16"  # 可选：模型名称后缀，例如设置为 "1st" 则保存为 patchtst_pretrained_full_1st.pth

# ==================== 检查配置 ====================

echo "=========================================="
echo "PatchTST Encoder 预训练 (Stage 1)"
echo "=========================================="
echo ""

# 检查数据文件是否存在
if [ ! -f "$JSONL_PATH" ]; then
    echo "❌ 错误: 数据文件不存在: $JSONL_PATH"
    echo ""
    echo "请检查文件路径或修改脚本中的 JSONL_PATH 变量"
    exit 1
fi
echo "✓ 数据文件: $JSONL_PATH"

# 检查 LLM 模型是否存在
if [ ! -d "$LLM_MODEL_PATH" ]; then
    echo "❌ 错误: LLM 模型路径不存在: $LLM_MODEL_PATH"
    echo ""
    echo "请检查路径或修改脚本中的 LLM_MODEL_PATH 变量"
    exit 1
fi
echo "✓ LLM 模型: $LLM_MODEL_PATH"

echo ""
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "序列长度: $SEQ_LEN"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "Patch 长度: $PATCH_LEN"
echo "Patch 步长: $PATCH_STRIDE"
echo "输入通道数: $INPUT_CHANNELS"
echo "设备: $DEVICE"
if [ -n "$MODEL_SUFFIX" ]; then
    echo "模型后缀: $MODEL_SUFFIX"
fi
echo "=========================================="
echo ""

# ==================== 构建训练命令 ====================

CMD="python train/train_encoder.py \
    --jsonl-path \"$JSONL_PATH\" \
    --llm-model-path \"$LLM_MODEL_PATH\" \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --input-channels $INPUT_CHANNELS \
    --device $DEVICE \
    --gpu-id $GPU_ID"

# 如果设置了模型后缀，添加参数
if [ -n "$MODEL_SUFFIX" ]; then
    CMD="$CMD --model-suffix \"$MODEL_SUFFIX\""
fi

echo "开始训练..."
echo ""
echo "执行命令:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# ==================== 执行训练 ====================

eval $CMD

# ==================== 训练完成 ====================

echo ""
echo "=========================================="
echo "✓ PatchTST Encoder 预训练完成！"
echo "=========================================="
echo ""

if [ -n "$MODEL_SUFFIX" ]; then
    echo "模型检查点已保存: model/patchtst_pretrained_full_${MODEL_SUFFIX}.pth"
else
    echo "模型检查点已保存: model/patchtst_pretrained_full.pth"
fi

echo ""
echo "日志文件保存在: log/encoder_log_*.log"
echo ""
echo "=========================================="
echo "后续步骤"
echo "=========================================="
echo ""
echo "1. 运行 Stage 2 对齐训练："
echo "   bash train/run_ddp_train.sh 4"
echo ""
echo "2. 运行 Stage 3 指令微调："
echo "   bash train/run_ddp_instruct.sh 4"
echo ""

