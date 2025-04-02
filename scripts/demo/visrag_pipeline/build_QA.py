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
1. Questions must be based on the document images' content. Avoid directly referencing specific elements (e.g., "this image", "given img2-7" or "this table"). Ensure the answers can be confidently derived from the images. And cross-page and numeric questions(if any) are encouraged.
2. Reasoning should explain why the question is meaningful and how to find the answer step by step.
3. Answers must be based on the document images and follow the specified format.
4. Generate 1 to 3 question-reasoning-answer pairs, depending on the meaningfulness of the questions.
[Response Language]: Chinese
[Response Format]: Reply in the following JSON format,here's an example:
```json
{
    "result": [
        {
            "question": "洛氏硬度HRA的压头类型，总负荷是什么，测量范围是多少?", #numreic question
            "reasoning": "找到所给文档表1-1“常用洛氏硬度标尺的试验条件与应用范围”的第一行可以看见，洛氏硬度HRA的压头是120°金刚石锥形压头，负荷是588.4N，测量范围是60-85HRA，根据表格的内容可以得出这个结论。",
            "answer": "洛氏硬度HRA的压头类型是120°金刚石锥形压头，总负荷是588.4N，测量范围是60-85HRA。",
        },
        {
            "question": " 低碳钢比例极限和弹性极限的区别是什么?", # cross-page question
            "reasoning": "第一页图片中图1-6展示了低碳钢的拉伸应力‐应变曲线,第二张图片中的文本解释了该曲线的特征中比例极限点A和弹性极限点A'。根据这几页中图1-6和相关文本,可以共同结合得出低碳钢比例极限和弹性极限的区别：OA 段为弹性阶段。这种变形称为弹性变形 ，A 点的应力 σe 称为弹性极限 ，为材料不产生永久变形可承受的最大应力值。OA 线中 OA′段为一斜直线，在OA′段应变与应力始终成比例，所以 A′点的应力 σp 称为比例极限，即应变量与应力成比例所对应的最大应力值。", 
            "answer": " 弹性极限：材料不产生永久变形的最大应力值；比例极限： 即应变量与应力成比例所对应的最大应力值 。 由于 这两个点很接近 ，工程上一般不作区分 。" ,  # don't forget comma, and other json format
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
                "text": "给定以下几张图片，生成问题"
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
                    "content": PROMPT
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
            }, ensure_ascii=False) + '\n')

def create_batch_jsonl(data_path):
    # 找到index2img_filename.txt,每行提取出来,作为图片文件名的list
 
    namefile = os.path.join(data_path, 'index2img_filename.txt')
    if not os.path.exists(namefile):
        print(f"index2img_filename.txt not found in {data_path}")
        return None
    with open(namefile, 'r') as f:
        img_names = f.read().split('\n')
        
    # 如果只提取特定的书EEmaterials和EEdesign
    img_names = [img_name for img_name in img_names if "EEmaterials" in img_name or "EEdesign" in img_name]
    
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

file_paths = create_batch_jsonl(conf.DATASTORE)


file_ids = upload_batchfile(file_paths)
batch_ids = submit_batch_task(file_ids)




# check_jobs()
# batch_ids = json.load(open(os.path.join(conf.TEST_DIR, 'batch_ids.json')))
# print(f"batch_ids: {batch_ids}")
# download_output(batch_ids)
