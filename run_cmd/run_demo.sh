#!/bin/bash

# 设置默认值（如果未传入参数）
TEST_FIELD=${TEST_FIELD:-EE}
MODEL_TYPE=${MODEL_TYPE:-Qwen-VL-3B}
RAG_EN=${RAG_EN:-True}

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

echo "启动VisRAG演示"

/home/liusiyuan/.conda/envs/VisRAG/bin/python \
    /datacenter/liusiyuan/VisRAG_test/scripts/demo/visrag_pipeline/answer.py \
    --test_field $TEST_FIELD \
    --model_type $MODEL_TYPE \
    --rag_en $RAG_EN

echo "VisRAG演示运行完成"