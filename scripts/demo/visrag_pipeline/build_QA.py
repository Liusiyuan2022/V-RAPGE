from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
import time
# 老师的
# ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"
# 我的
ZHIPU_API_KEY="28eb0cace1314b2bbc1824aa98b823a0.eOwjqu86fz9DtMRJ"

PROMPT = """Based on the document images provided, complete the following task:
[Task]: Generate question-reasoning-answer pairs from different perspectives.
[Constraints]: 
1. Questions must be based on the document images' content. Avoid directly referencing specific elements (e.g., "this image", "given img2-7" or "this table"). Ensure the answers can be confidently derived from the images.
2. Reasoning should explain why the question is meaningful and how to find the answer step by step.
3. Answers must be based on the document images and follow the specified format.
4. Generate 1 to 3 question-reasoning-answer pairs, depending on the meaningfulness of the questions.
[Response Language]: Chinese
[Response Format]: Reply in the following JSON format,here's an example:
```json
{
    "result": [
        {
            "question": "克里克的实验说明了什么关于遗传密码的发现？",
            "reasoning": "从所给文档可见，克里克的实验通过使用蛋白质体外合成技术，揭示了遗传密码的三个碱基编码一个氨基酸的规律。这一发现对理解遗传信息如何被翻译成蛋白质至关重要。",
            "answer": "克里克的实验说明了遗传密码中三个碱基编码一个氨基酸的规律",
        },
        {
            "question": " 短期记忆和长期记忆有什么区别？", 
            "reasoning": "图2-7展示了不同形式记忆的关系。短期记忆通常指的是信息的短暂存储，其特征是容易遗忘。而长期记忆则涉及信息的持久保留，需要神经元之间的复杂联系来维持。", 
            "answer": "短期记忆和长期记忆的区别在于它们的持续时间和神经机制。"
        },
    ]
}
```"""


def dump_jsonl(req_imgs, data_path ,file_path):
    with open(file_path, 'w') as f:
        for i, batch in enumerate(req_imgs):
            content = []
            content.append({
                "type": "text",
                "text": PROMPT
            })
            for img_name in batch:
                img_path = os.path.join(data_path, img_name)
                # 由于图片是png的，需要jpeg的，所以需要临时转换一下
                with open(img_path, 'rb') as img_f:
                    # img_base64 = base64.b64encode(img_f.read()).decode('utf-8')
                    img_jpeg = Image.open(img_f).convert("RGB")
                    img_jpeg_bytes = img_jpeg.tobytes("jpeg", "RGB")
                    img_jpeg_base64 = base64.b64encode(img_jpeg_bytes).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_jpeg_base64}
                })
            
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in document question-answering dialogue analysis."
                },
                {
                    "role": "user",
                    "content": content
                }
            ]
            # # test
            # client = ZhipuAI(api_key=ZHIPU_API_KEY)
            # response = client.chat.completions.create(
            #     model="glm-4v",  # 填写需要调用的模型名称
            #     messages=messages,
            #     response_format = {'type': 'json_object'},
            # )
            # print(response)          
            # # test
            
            f.write(json.dumps({
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": "glm-4v-plus",
                    "messages": messages,
                    # "response_format":{'type': 'json_object'},
                    "max_tokens": 1000
                }
            }) + '\n')

def create_batch_jsonl(data_path):
    # 找到index2img_filename.txt,每行提取出来,作为图片文件名的list
    namefile = os.path.join(data_path, 'index2img_filename.txt')
    if not os.path.exists(namefile):
        print(f"index2img_filename.txt not found in {data_path}")
        return None
    with open(namefile, 'r') as f:
        img_names = f.read().split('\n')
    # 每k个作为一组，组成一个请求
    k = conf.QA_IMG_NUM
    req_imgs = []
    line = 0
    j = 0
    for i in range(0, len(img_names), k):
        req_imgs.append(img_names[i:i + k])
        line += 1
        if line == conf.QA_BATCH_SIZE:
            # 生成jsonl文件
            dump_jsonl(req_imgs, data_path, os.path.join(conf.TEST_DIR, f'batch_{j}.jsonl'))
            print(f"Batch {j} created")
            req_imgs = []
            line = 0
            j += 1
    if len(req_imgs) > 0:
        dump_jsonl(req_imgs, data_path, os.path.join(conf.TEST_DIR, f'batch_{j}.jsonl'))
        print(f"Batch {j} created")
    
    file_paths = [os.path.join(conf.TEST_DIR, f'batch_{i}.jsonl') for i in range(j+1)]
            
    return file_paths




def upload_batchfile(file_paths):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    ids = []
    for file_path in file_paths:
        print(f"Uploading {file_path}")
        upload_file = open(file_path, "rb")
        
        result = client.files.create(
            file=upload_file,
            purpose="batch"
        )
        print(f"Upload success! File ID: {result.id}")
        ids.append(result.id)
    return ids

def submit_batch_task(file_ids):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    batch_ids = []
    for i , file_id in enumerate(file_ids):
        print(f"Submitting task for file {file_id}")
        create = client.batches.create(
            input_file_id=file_id,
            endpoint="/v4/chat/completions", 
            auto_delete_input_file=True,
            metadata={
                "description": f"QA Batch {i}"
            }
        )
        batch_ids.append(create.id)
    # # 记录file_ids为一个json文件，用于后续查询以及下载
    with open(os.path.join(conf.TEST_DIR, 'batch_ids.json'), 'w') as f:
        json.dump(batch_ids, f)
    print(f"Submit Success! batch_ids were saved to {os.path.join(conf.TEST_DIR, 'batch_ids.json')}")
        
def check_jobs():
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    batch_job = client.batches.retrieve("batch_id")
    print(batch_job)

# 
def download_output(batch_ids):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    for batch_id in batch_ids:
        # client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
        content = client.files.content(batch_id)
        # 使用write_to_file方法把返回结果写入文件
        # content.write_to_file("generated_answer.jsonl")
        content.write_to_file(os.path.join(conf.TEST_DIR, f"generated_QA_{batch_id}.jsonl"))

# file_paths = create_batch_jsonl(conf.DATASTORE)


# file_ids = upload_batchfile(file_paths)
# batch_ids = submit_batch_task(file_ids)




# check_jobs()
# batch_ids = json.load(open(os.path.join(conf.TEST_DIR, 'batch_ids.json')))
# print(f"batch_ids: {batch_ids}")
# download_output(batch_ids)
