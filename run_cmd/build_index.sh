#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31

echo "开始构建图像index，由于RET的第三方模型实现，只能使用单个GPU..."

/home/liusiyuan/.conda/envs/VisRAG/bin/python /datacenter/liusiyuan/VisRAG_test/scripts/demo/visrag_pipeline/build_index.py

echo "图像index构建完成"