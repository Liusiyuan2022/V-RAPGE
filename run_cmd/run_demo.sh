#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31

export PYTHONPATH=/home/bingxing2/home/scx7655/workspace/VisRAG:$PYTHONPATH

# 设置CUDA相关环境变量
export CUDA_LAUNCH_BLOCKING=1  # 调试时启用以获取更清晰的错误信息

# 要使用的GPU数量
# 从Slurm环境变量中获取分配的GPU数量
if [ -n "$SLURM_GPUS" ]; then
    NUM_GPUS=$SLURM_GPUS
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_JOB_GPUS" ]; then
    NUM_GPUS=$SLURM_JOB_GPUS
else
    # 默认值（如果无法从环境变量获取）
    NUM_GPUS=1
fi

# 设置环境变量以支持模型分片
export ACCELERATE_USE_DEVICE_MAP=True
# 根据num_gpus设置visible_devices
# 动态生成CUDA_VISIBLE_DEVICES值
# 初始化为空字符串
CUDA_DEVICES=""
for ((i=0; i<$NUM_GPUS; i++))
do
    # 如果不是第一个设备，添加逗号
    if [ "$i" -gt 0 ]; then
        CUDA_DEVICES="$CUDA_DEVICES,"
    fi
    CUDA_DEVICES="$CUDA_DEVICES$i"
done

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# 解析外部传入的参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_field)
            TEST_FIELD=$2
            shift 2
            ;;
        --model_type)
            MODEL_TYPE=$2
            shift 2
            ;;
        --rag_en)
            RAG_EN=$2
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置默认值（如果未传入参数）
TEST_FIELD=${TEST_FIELD:-EE}
MODEL_TYPE=${MODEL_TYPE:-Qwen-VL-2B}
RAG_EN=${RAG_EN:-True}

echo "启动VisRAG演示，使用 $NUM_GPUS 个GPU..."

/home/bingxing2/home/scx7655/.conda/envs/VisRAG/bin/python \
    /home/bingxing2/home/scx7655/workspace/VisRAG/scripts/demo/visrag_pipeline/answer.py \
    --test_field $TEST_FIELD \
    --model_type $MODEL_TYPE \
    --rag_en $RAG_EN

echo "VisRAG演示运行完成"