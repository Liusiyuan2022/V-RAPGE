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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_gen import qwen_answer_question
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from memlog import log_memory
from retrieve import *
import json
from tqdm import tqdm 


model_path = 'openbmb/VisRAG-Ret'
# gen_model_path = 'openbmb/MiniCPM-V-2_6'
# gen_model_path = 'Qwen/Qwen2-VL-2B-Instruct'

def load_qa_pairs(jsonl_file_path):
    qa_pairs = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行 JSON 数据
            data = json.loads(line.strip())
            # 将 question 和 answer 添加到 qa_pairs 列表
            qa_pairs.append((
                data.get("question", "N/A"),
                data.get("answer", "N/A")
            ))
    return qa_pairs

def export_result(answer_path, query, images_path_topk, answer, reference_answer):
   
    with open(os.path.join(answer_path, f"answer.jsonl"), 'a') as f:
        f.write(json.dumps({
            'query': query, 
            'retrieved_images': images_path_topk,
            'answer': answer,
            'reference': reference_answer
        }, indent=4, ensure_ascii=False))
    # save images
    # images_topk = [Image.open(i) for i in images_path_topk]
    # for idx, image in enumerate(images_topk):
    #     image.save(os.path.join(answer_path, os.path.basename(images_path_topk[idx])))
    
    
    
def main():
    
    # query = "弃耕农田上面会不会发生群落演替，演替类型是什么，请说一下这个例子中演替的几个阶段"
    
    qa_pairs = load_qa_pairs(os.path.join(conf.TEST_DIR, 'test_QA.jsonl'))
    
    # 加载Ret模型
    log_memory("before load VisRAG-Ret model")
    model_ret, tokenizer, device = load_ret_models()
    log_memory("after load VisRAG-Ret model")
    
    
    # 加载Gen模型
    log_memory("before load Qwen model")
    model_gen = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",cache_dir=conf.CACHE_DIR,
        attn_implementation="flash_attention_2",
    )
    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", cache_dir=conf.CACHE_DIR)
    log_memory("after load Qwen model")
    
    
    # answer_log目录下的result.jsonl 作为存储question， answer， reference的jsonl文件
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = os.path.join(conf.RESULT_DIR + f'/{timestamp}')
    os.makedirs(answer_path, exist_ok=True)
    
    
    for query, reference_answer in tqdm(qa_pairs, desc="Processing QA Pairs"):
        # 调用检索函数
        images_path_topk = retrieve(query, model_ret, tokenizer, device)
        
        
        # print the size
        # img_0 = images_topk[0]
        # print(f"Image size: {img_0.size}")

        # 生成答案
        answer = qwen_answer_question(images_path_topk, query, model_gen, processor)
        # print(answer)
        export_result(answer_path, query, images_path_topk, answer, reference_answer)
        
    print(f"Answer saved at {answer_path}/{timestamp}/result.json")

if __name__ == "__main__":
    main()