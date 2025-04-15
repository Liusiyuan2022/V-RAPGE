from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import conf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch
from memlog import log_memory
from utils import *


def qwen_answer_question(images_path_topk, query, model, processor):
    # default: Load the model on the available device(s)
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-2B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
 
    
    

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    PROMPT = """
    你是一个有用的智能助手，{}，回答一道考试题
    根据不同题型，写出对应格式的答案
    例如：
    - 选择题，答案只有你的选项：A
    - 填空题，答案只有填在空中的内容：有氧呼吸、厌氧呼吸
    - 简答题，答案只有简答题的内容：HRA的压头类型是120°金刚石锥形压头，总负荷是588.4N，测量范围是60-85HRA。
    - 判断题，包括判断题的内容和解释：错误，纱布的作用是吸收伤口渗出液，保持伤口干燥，而不是促进细胞呼吸。
    - 计算题，包括推理过程和计算结果：根据海弗利克极限理论，细胞的最大分裂次数决定了其寿命。题目中给出的最大分裂次数为50次，每次分裂周期为2年。因此，细胞从新生到衰老所需的时间可以通过以下计算得出：50次 × 2年/次 = 100年。这表明该细胞在达到其最大分裂次数后，将进入衰老状态，整个过程大约需要100年。
    题目如下：
    {}
    请做答,注意回答尽量简洁： 
    """


    combined_img_path_tmp = all_path_to_one_create(images_path_topk)
    
 
    if conf.RAG_EN:
        knowledge_instr = "请根据所给图片信息和你的知识"
        content = [
                {"type": "image", "image": combined_img_path_tmp,},
                {"type": "text" , "text" : PROMPT.format(knowledge_instr, query)},
            ]
    else:
        knowledge_instr = "请根据你的知识"
        content = [
                {"type": "text" , "text" : PROMPT.format(knowledge_instr, query)}
            ]
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if conf.RAG_EN:
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs, video_inputs = (None, None)
        
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    device = f"cuda:{conf.GPU_ID}"
    
    model.eval()
    
    # 对于分片模型,让模型处理输入设备分配
    model_inputs = inputs.to(device)

    # 使用torch.no_grad()以节省内存
    # log_memory("before Qwen generate")
    with torch.no_grad():
        # Inference: Generation of the output
        generated_ids = model.generate(**model_inputs, max_new_tokens=conf.MAX_TOKENS)
        # log_memory("after Qwen generate call")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    # log_memory("after Qwen generate")
    
    all_path_to_one_remove()
    
    
    return output_text[0]
