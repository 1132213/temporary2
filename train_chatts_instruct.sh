#!/bin/bash

# ChatTS 格式 Stage 3 指令微调训练脚本
# 使用方法: bash train_chatts_instruct.sh

# ==================== 配置参数 ====================

# 数据文件路径
JSONL_PATH="/root/emhua/btwu/timedataset/ChatTS-Training-Dataset/sft/train_cleaned.jsonl"

# Stage 2 对齐阶段的检查点路径
STAGE2_CHECKPOINT="model/chatts_stage2_aligned.pth"

# LLM 模型路径
LLM_MODEL_PATH="/root/emhua/btwu/Llama-3.2-3B"

# 序列长度
SEQ_LEN=512

# 训练参数
BATCH_SIZE=4
EPOCHS=20
LR=5e-5  # Stage 3 学习率通常比 Stage 2 小

# 模型架构参数
PATCH_LEN=16
PATCH_STRIDE=8

# 其他训练参数
WEIGHT_DECAY=0.01
SEED=42

# GPU 设置
DEVICE="cuda:0"
GPU_ID=0

# 是否冻结编码器（通常在 Stage 3 不冻结以获得更好的微调效果）
FREEZE_ENCODER=false

# ==================== 检查配置 ====================

echo "=========================================="
echo "ChatTS Stage 3: 指令微调训练"
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

# 检查 Stage 2 检查点是否存在
if [ ! -f "$STAGE2_CHECKPOINT" ]; then
    echo "❌ 警告: Stage 2 检查点不存在: $STAGE2_CHECKPOINT"
    echo ""
    echo "建议先运行 Stage 2 对齐训练："
    echo "  bash train_chatts_alignment.sh"
    echo ""
    read -p "是否继续从头训练? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Stage 2 检查点: $STAGE2_CHECKPOINT"
fi

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
echo "冻结编码器: $FREEZE_ENCODER"
echo "设备: $DEVICE"
echo "随机种子: $SEED"
echo "=========================================="
echo ""

# ==================== 构建训练命令 ====================

CMD="python train/train_chatts_instruct.py \
    --jsonl-path $JSONL_PATH \
    --stage2-checkpoint $STAGE2_CHECKPOINT \
    --seq-len $SEQ_LEN \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --device $DEVICE \
    --gpu-id $GPU_ID \
    --llm-model-path $LLM_MODEL_PATH \
    --patch-len $PATCH_LEN \
    --patch-stride $PATCH_STRIDE \
    --weight-decay $WEIGHT_DECAY \
    --seed $SEED"

# 如果冻结编码器，添加 --freeze-encoder 标志
if [ "$FREEZE_ENCODER" = "true" ]; then
    CMD="$CMD --freeze-encoder"
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
echo "✓ ChatTS Stage 3 指令微调训练完成！"
echo "=========================================="
echo ""
echo "模型检查点已保存："
echo "  - 最佳模型: model/chatts_instruct_best.pth"
echo ""
echo "日志文件保存在: log/chatts_instruct_log_*.log"
echo ""
echo "=========================================="
echo "后续步骤"
echo "=========================================="
echo ""
echo "1. 测试模型性能："
echo "   python train/test_instruct.py \\"
echo "       --jsonl-path $JSONL_PATH \\"
echo "       --checkpoint model/chatts_instruct_best.pth \\"
echo "       --llm-model-path $LLM_MODEL_PATH"
echo ""
echo "2. 进行推理："
echo "   python inference.py \\"
echo "       --jsonl-path $JSONL_PATH \\"
echo "       --checkpoint model/chatts_instruct_best.pth \\"
echo "       --llm-model-path $LLM_MODEL_PATH"
echo ""

