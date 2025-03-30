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
    
    #emm flash_attention_2 will cause torch reinstall. 
    

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    combined_img_path_tmp = all_path_to_one_create(images_path_topk)
    
    # prompt限制一下输入长度，精简
    RESTRICT = "回答要精简，控制在50字以内"
    content = [
                {"type": "image", "image": combined_img_path_tmp,},
                {"type": "text" , "text" : query + RESTRICT},
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
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    model.eval()

    # 使用torch.no_grad()以节省内存
    log_memory("before Qwen generate")
    with torch.no_grad():
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=conf.MAX_TOKENS)
        log_memory("after Qwen generate call")
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    log_memory("after Qwen generate")
    
    all_path_to_one_remove()
    
    
    return output_text[0]
