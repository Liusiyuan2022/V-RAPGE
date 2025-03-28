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
from ds_gen import deepseek_answer_question
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from qwen_gen import qwen_answer_question
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from memlog import log_memory


def retrieve(knowledge_base_path, query, topk, model, tokenizer, device):
    model.eval()

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

    topk_values, topk_doc_ids = torch.topk(similarities, k=topk)

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


def load_models():
    model_path = 'openbmb/VisRAG-Ret'
    device = f'cuda:0' # use the first GPU for retrieval

    print(f"VisRAG-Ret load begin...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=conf.CACHE_DIR)
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16, cache_dir=conf.CACHE_DIR)
    
    model.to(device)
    model.eval()
    print(f"VisRAG-Ret load success!")
    
    return model, tokenizer, device

model_path = 'openbmb/VisRAG-Ret'
# gen_model_path = 'openbmb/MiniCPM-V-2_6'
# gen_model_path = 'Qwen/Qwen2-VL-2B-Instruct'

knowledge_base_path = conf.DATASTORE


def main():
    
    # 加载模型
    log_memory("before load VisRAG-Ret model")
    model, tokenizer, device = load_models()
    log_memory("after load VisRAG-Ret model")
    
    knowledge_base_path = conf.DATASTORE
    
    query = "弃耕农田上面会不会发生群落演替，演替类型是什么，请说一下这个例子中演替的几个阶段"
    topk = conf.TOP_K
    
    # 调用检索函数
    images_path_topk = retrieve(knowledge_base_path, query, topk, model, tokenizer, device)
    images_topk = [Image.open(i) for i in images_path_topk]
    
    # print the size
    img_0 = images_topk[0]
    print(f"Image size: {img_0.size}")
    
    # 生成答案
    answer = qwen_answer_question(images_path_topk, query)
    print(answer)

    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = os.path.join(conf.RESULT_DIR + f'/{timestamp}')
    os.makedirs(answer_path, exist_ok=True)
    with open(os.path.join(answer_path, f"answer.json"), 'w') as f:
        f.write(json.dumps({
            'query': query, 
            'retrieved_images': images_path_topk,
            'answer': answer
        }, indent=4, ensure_ascii=False))
    # save images
    for idx, image in enumerate(images_topk):
        image.save(os.path.join(answer_path, os.path.basename(images_path_topk[idx])))
    print(f"Answer saved at {answer_path}/{timestamp}.json")
    

if __name__ == "__main__":
    main()