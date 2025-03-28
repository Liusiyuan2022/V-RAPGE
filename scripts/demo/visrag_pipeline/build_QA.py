from zhipuai import ZhipuAI
import json
import os
import conf
import base64
# 老师的
# ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"
# 我的
ZHIPU_API_KEY="28eb0cace1314b2bbc1824aa98b823a0.eOwjqu86fz9DtMRJ"

# PROMPT = """你是一个在文档问答对话领域的分析专家。请根据我给你的若干文档图像，完成下列形式的任务：
# [任务描述]：你的任务是找从不同角度去找问题-推理-答案对。
# [限制]：1.问题必须基于文档图像的内容，不要直接指着某个图片的元素(如这张图片，这张表)提问，而是让人们回答时去找到对应的图像，能够自信地根据对应文档内容回答。，2.推理是一步步地指解释为什么这个问题是有意义的，以及如何从文档图像中找到答案。3.答案是指对问题的回答，必须基于文档图像的内容。且按照格式。4.至多5个问题-推理-答案对。最少1个问题-推理-答案对，如果你觉得有意义的问题比较多，就多生成几条，否则就少生成。
# [回复格式]：请按照以下json文件格式回复：
# ```json
# {
#     "结果":[
#         {
#             "问题": "",
#             "推理": "",
#             "答案": ""
#         },
#         ...
# }
# ```
# """

PROMPT = """You are an expert in document question-answering dialogue analysis. Based on the document images provided, complete the following task:
[Task]: Generate question-reasoning-answer pairs from different perspectives.
[Constraints]: 
1. Questions must be based on the document images' content. Avoid directly referencing specific elements (e.g., "this image" or "this table"). Ensure the answers can be confidently derived from the images.
2. Reasoning should explain why the question is meaningful and how to find the answer step by step.
3. Answers must be based on the document images and follow the specified format.
4. Generate 1 to 5 question-reasoning-answer pairs, depending on the meaningfulness of the questions.
[Response Format]: Reply in the following JSON format:
```json
{
    "result": [
        {
            "question": "",
            "reasoning": "",
            "answer": ""
        },
        ...
    ]
}
```"""


def dump_jsonl(req_imgs, data_path ,file_path):
    with open(file_path, 'w') as f:
        for i, batch in enumerate(req_imgs):
            content = []
            for img_name in batch:
                img_path = os.path.join(data_path, img_name)
                with open(img_path, 'rb') as img_f:
                    img_base64 = base64.b64encode(img_f.read()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_base64}
                })
            content.append({
                "type": "text",
                "text": PROMPT
            })
            
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            # # test
            # client = ZhipuAI(api_key=ZHIPU_API_KEY)
            # response = client.chat.completions.create(
            #     model="glm-4v-plus",  # 填写需要调用的模型名称
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
                    "response_format" : {'type': 'json_object'},
                    "max_tokens": 1000
                }
            }, ensure_ascii=False) + '\n')

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
        print(create)
        
def check_jobs():
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    batch_job = client.batches.retrieve("batch_id")
    print(batch_job)

# 
def download_output(file_ids):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    for file_id in file_ids:
        # client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
        content = client.files.content(file_id)
        # 使用write_to_file方法把返回结果写入文件
        content.write_to_file("write_to_file_batchoutput.jsonl")

file_paths = create_batch_jsonl(conf.DATASTORE)
file_ids = upload_batchfile(file_paths)
submit_batch_task(file_ids)
# 记录file_ids为一个json文件，用于后续查询以及下载
with open(os.path.join(conf.TEST_DIR, 'file_ids.json'), 'w') as f:
    json.dump(file_ids, f)
# check_jobs()
# file_ids = json.load(open(os.path.join(conf.TEST_DIR, 'file_ids.json')))
# download_output(file_ids)
