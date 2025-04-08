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


def retrieve(query, model, tokenizer, device):
    model.eval()

    knowledge_base_path = conf.DATASTORE
    
    if not os.path.exists(knowledge_base_path):
        return None
    
    # 读取索引到图像文件名的映射
    with open(os.path.join(knowledge_base_path, 'index2img_filename.txt'), 'r') as f:
        index2img_filename = f.read().split('\n')
    
    doc_reps = np.load(os.path.join(knowledge_base_path, 'reps.npy'))
    doc_reps = torch.from_numpy(doc_reps).to(device)

    query_with_instruction = "Represent this query for retrieving relevant document: " + query
    with torch.no_grad():
        query_rep = torch.Tensor(encode(model, tokenizer, [query_with_instruction])).to(device)

    similarities = torch.matmul(query_rep, doc_reps.T)

    topk_values, topk_doc_ids = torch.topk(similarities, k=conf.TOP_K)

    topk_values_np = topk_values.squeeze(0).cpu().numpy()
    topk_doc_ids_np = topk_doc_ids.squeeze(0).cpu().numpy()

    images_path_topk = [os.path.join(knowledge_base_path, index2img_filename[idx]) for idx in topk_doc_ids_np]

    return images_path_topk

# def answer_question(images, question):
#     global gen_model, gen_tokenizer
#     msgs = [{'role': 'user', 'content': [question, *images]}]
#     answer = gen_model.chat(
#         image=None,
#         msgs=msgs,
#         tokenizer=gen_tokenizer
#     )
#     return answer


def load_ret_models():
    model_path = '/datacenter/models/openbmb/VisRAG-Ret'  # 使用绝对路径指向本地模型
    # device = f'cuda:1' # use the 1th GPU for retrieval

    print(f"VisRAG-Ret load begin...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=conf.CACHE_DIR)
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto",
        attn_implementation='sdpa', torch_dtype=torch.bfloat16, cache_dir=conf.CACHE_DIR)
    
    # model.to(device)
    model.eval()
    print(f"VisRAG-Ret load success!")
    
    return model, tokenizer, device

