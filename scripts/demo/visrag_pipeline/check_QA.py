from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
from tqdm import tqdm
import time
import re
# 老师的
ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"
# 我的
# ZHIPU_API_KEY="28eb0cace1314b2bbc1824aa98b823a0.eOwjqu86fz9DtMRJ"

PROMPT = """你是一个资深的考试出题人,你需要对已经出好的题目进行审核。
所传入的内容是一个出好的题目，包含四个字段“type”、“question”、“answer”、“analysis”，分别表示题目类型、题目内容、答案和解析。
你的任务是判断题目的质量，给出一个分数，范围是0-10分，0分表示题目质量极差，10分表示题目质量极好。
以下是判断标准：
1. 问题虽然基于教材所出，但考生考场上不知道教材内容，也不知道是哪本书，所以，问题必须围绕知识点，而不能是对于某张图片，某张表格或者某本书信息的提问。如，"表3提里面可以看到什么"、"图2中可以看到什么"、"本书的题目是什么"等问题都是不合格的。
2. 问题格式需要完整，如果是选择题，却没有选项，或者是填空题却没有填空内容这种情况,都是不合格的。
3. 问题需要是有价值的。知识领域的知识点，而不能是某本书的编者，出版社等信息的问题。
以下是相关问题：{}
请作出你的判断。
请返回json格式,包括你的理由和最终给分，例如:
```json
{{
    "reason": "该问题和答案聚焦于知识点，且问题格式完整，解析也比较完善，符合出题标准。",
    "score": 9 
    
}}
```
或者
```json
{{
    "reason": "该问题问的是编者和出版社等信息，这并不适合作为考试问题。是不合格的。",
    "score": 0
}}
```"""

def quality_sort(test_path, quality_path, qualified_path, unqualified_path):
    # test_path是原始的jsonl文件，quality_path是质量评分的jsonl文件
    # quality的id是testpath的行号-1
    quality_data = []
    with open(quality_path, 'r') as f :
        for line in f:
            data = json.loads(line)
            # 提取 "id" 和 "score" 字段
            id = data.get("id", "N/A")
            score = data.get("score", "N/A")
            reason = data.get("reason", "N/A")
            quality_data.append({
                "id": id,
                "score": score,
            })
    # 根据id排序,将排好的score拿出来
    quality_data.sort(key=lambda x: int(x["id"]))
    scores = [data["score"] for data in quality_data]
    qnum = 0
    unum = 0
    # 读取test_path的内容,根据对应的得分分类到两个文件
    with open(test_path, 'r') as f, open(qualified_path, 'w') as qf, open(unqualified_path, 'w') as uf:
        for i, line in enumerate(f):
            # score = scores[i]
            score = scores[i]
            if score >= conf.QA_QUALITY_THRESHOLD:
                qf.write(line)
                qnum += 1
            else:
                uf.write(line)
                unum += 1
    print(f"Sorted and filtered QA data based on quality scores.")
    print(f"Qualified QAs was saved to {qualified_path}, total: {qnum}")
    print(f"Unqualified QAs was saved to {unqualified_path}, total: {unum}")
            

def dump_jsonl(qas, file_path):
    with open(file_path, 'w') as f:
        for i, qa in tqdm(enumerate(qas), desc="Creating batch requests", total=len(qas)):
            content = PROMPT.format(json.dumps(qa, ensure_ascii=False))
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # 序号即为编号
            f.write(json.dumps({
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": "glm-4-plus",
                    "messages": messages,
                    "response_format":{
                        'type': 'json_object'
                    },
                    "max_tokens": 2048,
                }
            }, ensure_ascii=False) + '\n')
                

def create_batch_jsonl(test_path):
    # 找到index2img_filename.txt,每行提取出来,作为图片文件名的list
    # test是jsonl文件,每行是一个json对象
    # 这个文件是文本处理，在本实验条件中不需要考虑batch100M大小(已经足够)
    # 读取文件内容
    qas = []
    with open(test_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 提取 "knowledge" 字段
            type = data.get("sub_type", "N/A")
            question = data.get("question", "N/A")
            answer = data.get("answer", "N/A")
            analysis = data.get("analysis", "N/A")
            qas.append({
                "type": type,
                "question": question,
                "answer": answer,
                "analysis": analysis
            })
    
    file_path = os.path.join(conf.TEST_DIR, f'QAcheck_batch_{conf.TEST_FIELD}.jsonl')
    dump_jsonl(qas, file_path)
            
    return file_path




def upload_batchfile(file_path):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)


    print(f"Uploading {file_path}")
    upload_file = open(file_path, "rb")
    
    result = client.files.create(
        file=upload_file,
        purpose="batch"
    )
    print(f"Upload success! File ID: {result.id}")

    return result.id

def submit_batch_task(file_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    print(f"Submitting QAcheck task for file {file_id}")
    create = client.batches.create(
        input_file_id=file_id,
        endpoint="/v4/chat/completions", 
        auto_delete_input_file=True,
        metadata={
            "description": f"QA gen Batch"
        }
    )
    batch_id = create.id
    # # 记录file_ids为一个json文件,用于后续查询以及下载
    batch_id_path = os.path.join(conf.TEST_DIR, f'QAcheck_batch_ids_{conf.TEST_FIELD}.json')
    with open(batch_id_path, 'w') as f:
        json.dump(batch_id, f)
    print(f"Submit Success! batch_ids were saved to {batch_id_path}")
        
def check_jobs(batch_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    batch_job = client.batches.retrieve(batch_id)
    output_file_id = batch_job.output_file_id
    if batch_job.status != "completed":
        print(f"Batch job {batch_id} is not completed yet. Status: {batch_job.status}")
        print(f"Download canceled, please check the status later.")
        return None
    # print(batch_job)
    return output_file_id

# 目前实际上只支持单文件
def download_output(output_file_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    # client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
    print(f"downloading output file {output_file_id}")
    content = client.files.content(output_file_id)
    # 使用write_to_file方法把返回结果写入文件
    output_file = os.path.join(conf.TEST_DIR, f"raw_QA_quality_{conf.TEST_FIELD}.jsonl")
    content.write_to_file(output_file)
    print(f"Download success! Content was saved to {output_file}")
    parse_filter_jsonl(output_file)




def parse_filter_jsonl(input_path):
    # 先清空输出文件
    output_path = os.path.join(conf.TEST_DIR, f'QA_quality_{conf.TEST_FIELD}.jsonl')
    with open(output_path, 'w') as out_f:
        out_f.write('')
    i = 1    
    tot = 0
    err_num = 0
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # 检查是否有 "response" 和 "body" 字段
            if "response" in data and "body" in data["response"]:
                body = data["response"]["body"]
                if "request_id" in body:
                    request_id = body["request_id"]
                    # 提取出对应的行号
                    pattern = r"request-(\d+)"
                    match = re.search(pattern, request_id)
                    if match:
                        id = match.group(1)
                    else:
                        print(f"Failed to parse request_id at line: {i}, request_id: {request_id}")
                        continue
                
                # 检查是否有 "choices" 字段
                if "choices" in body and len(body["choices"]) > 0:
                    content = body["choices"][0]["message"]["content"]
                    # 提取 JSON 格式的 "result"
                try:
                    stripped_content = content.strip("`json").strip()
                    result_data = json.loads(stripped_content)
                    
                    
                    reason = result_data.get("reason", "")
                    score = result_data.get("score", "")
                    
                    with open(output_path, 'a') as out_f:
                        out_f.write(json.dumps({
                            "id": id,
                            "reason": reason,
                            "score": score
                        }, ensure_ascii=False) + '\n')
                    tot += 1
                            
                except json.JSONDecodeError:
                    print(f"Failed to parse result content as JSON at line: {i}, stripped_content:\n{stripped_content}")
                    err_num += 1
                    # set score为0
                    with open(output_path, 'a') as out_f:
                        out_f.write(json.dumps({
                            "id": id,
                            "reason": "parse error",
                            "score": 0
                        }, ensure_ascii=False) + '\n')
            i += 1
    print(f"Parsed JSONL file and extracted text content to {output_path}")
    print(f"Total : {tot - 1} QAs to Parse, Success : {tot-1-err_num} lines, Error : {err_num} lines")
    
    



def upload_task():
    # 创建批量 JSONL 文件
    test_path = os.path.join(conf.TEST_DIR, f'test_QA_{conf.TEST_FIELD}.jsonl')
    file_path = create_batch_jsonl(test_path)
    if not os.path.exists(file_path):
        print("No files to upload.")
        return

    # 上传文件并获取文件 ID
    file_id = upload_batchfile(file_path)

    # 提交任务并获取批次 ID
    submit_batch_task(file_id)



def download_result():
    # 从保存的 JSON 文件中加载批次 ID
    batch_id_path = os.path.join(conf.TEST_DIR, f'QAcheck_batch_ids_{conf.TEST_FIELD}.json')
    if not os.path.exists(batch_id_path):
        print(f"Batch IDs file not found: {batch_id_path}")
        return

    batch_id = json.load(open(batch_id_path, 'r'))
    
    # 检查任务状态并获取输出文件 ID
    output_file_id = check_jobs(batch_id)
    
    if output_file_id is None:
        print("No completed jobs found.")
        return
    # print(f"batch_ids: {batch_ids}, output_file_ids: {output_file_ids}")
    # 下载输出文件
    download_output(output_file_id)
     # 接下创建两个文件，qualified_QA.jsonl和unqualified_QA.jsonl
    qualified_path = os.path.join(conf.TEST_DIR, f'qualified_QA_{conf.TEST_FIELD}.jsonl')
    unqualified_path = os.path.join(conf.TEST_DIR, f'unqualified_QA_{conf.TEST_FIELD}.jsonl')
    test_path = os.path.join(conf.TEST_DIR, f'test_QA_{conf.TEST_FIELD}.jsonl')
    score_path = os.path.join(conf.TEST_DIR, f'QA_quality_{conf.TEST_FIELD}.jsonl')
    quality_sort(test_path, score_path, qualified_path, unqualified_path)


if __name__ == "__main__":
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    parser.add_argument("--test_field", type=str, choices=["BI", "EE"], help="Set TEST_FIELD value.")

    args = parser.parse_args()
    conf.TEST_FIELD = args.test_field if args.test_field else conf.TEST_FIELD
    # 根据参数调用对应的函数
    if args.action == "upload":
        upload_task()
    elif args.action == "download":
        download_result()