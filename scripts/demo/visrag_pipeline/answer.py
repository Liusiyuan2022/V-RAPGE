from PIL import Image
import torch
import os
from transformers import AutoModel
from transformers import AutoTokenizer
from PIL import Image
import torch
import os
import numpy as np
import json
import datetime
from utils import encode
import conf
# from ds_gen import deepseek_answer_question
from transformers import AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_gen import qwen_answer_question
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from memlog import log_memory
from retrieve import *
import json
from tqdm import tqdm 
import argparse


def load_qa_pairs(jsonl_file_path):
    qa_pairs = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行 JSON 数据
            data = json.loads(line.strip())
            # 将 question 和 answer 添加到 qa_pairs 列表
            qa_pairs.append(data)
    return qa_pairs

def export_result(answer_path, qa_info, images_path_topk, response):
   
    with open(os.path.join(answer_path, f"answer.jsonl"), 'a') as f:
        f.write(json.dumps({
            'task': qa_info["task"],
            'sub_type': qa_info["sub_type"],
            'question': qa_info["question"],
            'retrieved_images': images_path_topk,
            'answer': response,
            'reference': qa_info["answer"],
        },ensure_ascii=False) + '\n')
    
def main():
    
    qa_pairs = load_qa_pairs(os.path.join(conf.TEST_DIR, f'test_QA_{conf.TEST_FIELD}.jsonl'))
    
    # 加载Ret模型
    log_memory("before load VisRAG-Ret model")
    model_ret, tokenizer, device = load_ret_models()
    log_memory("after load VisRAG-Ret model")
    
    # 加载Gen模型
    log_memory(f"before load {conf.MODEL_TYPE} model")
    
    
    
    if conf.MODEL_TYPE == "Qwen-VL-3B":
        # Qwen2-VL-2B-Instruct
        model_path = "/datacenter/models/Qwen/Qwen2.5-VL-3B-Instruct"
        model_gen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{conf.GPU_ID}",cache_dir=conf.CACHE_DIR, attn_implementation="flash_attention_2"
        )
        # default processer
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=conf.CACHE_DIR)
        
    elif conf.MODEL_TYPE == "Qwen-VL-7B":
        #Qwen2.5-VL-7B-Instruct
        model_path = "/datacenter/models/Qwen/Qwen2.5-VL-7B-Instruct"
        model_gen =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,torch_dtype=torch.bfloat16, device_map=f"cuda:{conf.GPU_ID}",
            cache_dir=conf.CACHE_DIR, attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=conf.CACHE_DIR)
    elif conf.MODEL_TYPE == "Qwen-VL-32B":
        model_path = "/datacenter/models/Qwen/Qwen2.5-VL-32B-Instruct"
        model_gen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{conf.GPU_ID}",
            cache_dir=conf.CACHE_DIR, attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=conf.CACHE_DIR)
    else:
        raise ValueError(f"Unsupported model type: {conf.MODEL_TYPE}")
        
    log_memory(f"after load {conf.MODEL_TYPE} model")

    
    
    # answer_log目录下的result.jsonl 作为存储question， answer， reference的jsonl文件
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = os.path.join(conf.RESULT_DIR + f'/{conf.MODEL_TYPE}_{"RAG" if conf.RAG_EN else "NOR"}_{conf.TEST_FIELD}_{timestamp}')
    os.makedirs(answer_path, exist_ok=True)
    
    
    for qa_info in tqdm(qa_pairs, desc="Processing QA Pairs"):
        # 调用检索函数
        query = qa_info["question"]
        
        images_path_topk = retrieve(query, model_ret, tokenizer, device)
        
        
        # print the size
        # img_0 = images_topk[0]
        # print(f"Image size: {img_0.size}")

        # 生成答案
        if conf.MODEL_TYPE == "Qwen-VL-3B":
            response = qwen_answer_question(images_path_topk, query, model_gen, processor)
        elif conf.MODEL_TYPE == "Qwen-VL-7B":
            response = qwen_answer_question(images_path_topk, query, model_gen, processor)
        elif conf.MODEL_TYPE == "Qwen-VL-32B":
            response = qwen_answer_question(images_path_topk, query, model_gen, processor)
        # print(answer)
        export_result(answer_path, qa_info, images_path_topk, response)
        
    print(f"Answer saved at {answer_path}/{timestamp}/result.json")

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Modify conf.py settings dynamically.")
    parser.add_argument("--test_field", type=str, choices=["BI", "EE"], help="Set TEST_FIELD value.")
    parser.add_argument("--model_type", type=str, choices=["Qwen-VL-3B", "Qwen-VL-7B", "Qwen-VL-32B"], help="Set MODEL_TYPE value.")
    # 当使用 type=bool 时，argparse 会尝试将字符串转换为布尔值，但在 Python 中，除了空字符串，几乎所有字符串转换为布尔值都是 True，包括字符串 "False"！
    # 因此这里使用lambda 函数来处理字符串转换为布尔值
    parser.add_argument("--rag_en", type=lambda x: x.lower() == "true", default=True, help="Set RAG_EN value.")

    args = parser.parse_args()

    # 动态修改 conf.py 中的值
    if args.test_field:
        conf.TEST_FIELD = args.test_field
    if args.model_type:
        conf.MODEL_TYPE = args.model_type

    conf.RAG_EN = args.rag_en
    # 打印修改后的配置（可选）
    print(f"Using{conf.MODEL_TYPE} model, RAG settings: {conf.RAG_EN}, test field: {conf.TEST_FIELD}")
    
    main()