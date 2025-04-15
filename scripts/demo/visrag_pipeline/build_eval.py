from zhipuai import ZhipuAI
import json
import os
import conf
import base64
import time
import re
import numpy
# 老师的
ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"
# 我的
# ZHIPU_API_KEY="28eb0cace1314b2bbc1824aa98b823a0.eOwjqu86fz9DtMRJ"


ANSWER_ID = "Qwen-VL-3B_RAG_EE_20250402152431"

PROMPT= """你是一个资深公正的试卷评审员。
[任务]: 根据参考答案和你的知识，评估AI助手对考题的回答质量，考虑以下几个方面：有用性、相关性、准确性和细节。
[说明]:
给定的json格式中，你只需要关注三个字段"question"是考题，"answer"是AI助手的回答，"reference"是参考答案。
[评估] 开始评估时，请提供简短的解释。尽量客观。
[评分规则]：
- 对于选择题，如果选对就是10分，选错就是0分。
- 对于判断题，如果选对就是10分，选错就是0分。
- 对于填空题，根据答案和参考答案的相似性评分，给出0-10分。
- 对于简答题，根据参考答案，考察AI助手的回答是否全面、准确、相关，给出0-10分。
- 对于计算题，根据参考答案，考察AI助手的推理过程和计算结果是否正确，给出0-10分。

以下是需要评分的json格式数据：
{}

[格式要求]：
请返回json格式如下面的例子
```json
{{

    "Explaination": "回答中，胞吞作用识别，囊泡形成，囊泡移动，囊泡融合的每个阶段都被提及，且描述清晰。AI助手的回答与参考答案一致，准确性高，相关性强，细节丰富。",
    "Rating": <<10>> #打分必须遵循格式<<rating>>

}}
```

"""


def dump_jsonl(judges,file_path):
    with open(file_path, 'w') as f:
        for i, data in enumerate(judges):
            judge_data = data
            
            judge_tuple = {
                "question": judge_data["question"],
                "answer": judge_data["answer"],
                "reference": judge_data["reference"]
            }
            judge_text = json.dumps(judge_tuple, 
                                    ensure_ascii=False)
            messages = [
                {
                    "role": "user",
                    "content": PROMPT.format(judge_text),
                }
            ]
            
            # 构造对应的请求id，包括序号和任务类型，以便在回复的时候进行区分
            req_id = f"request-{i}-<<{judge_data['task']}>>-<<{judge_data['sub_type']}>>"
            
            f.write(json.dumps({
                "custom_id": req_id,
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": "glm-4-plus",
                    "messages": messages,
                    "response_format":{'type': 'json_object'},
                    "max_tokens": 2048
                }
            }, ensure_ascii=False ) + '\n')

def create_batch_jsonl(answer_dir):
    # 读取jsonl文件
    answer_path = os.path.join(answer_dir, "answer.jsonl")
    
    judges = []
    with open(answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行 JSON 数据
            data = json.loads(line.strip())
            # 去掉"retrieved_images"这个key
            data.pop("retrieved_images")
            judges.append(data)
            
    file_path = os.path.join(answer_dir, "batch.jsonl")
    dump_jsonl(judges, file_path)
    
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
    id = result.id
    return id

def submit_batch_task(file_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    print(f"Submitting Eval task for file {file_id}")
    create = client.batches.create(
        input_file_id=file_id,
        endpoint="/v4/chat/completions", 
        auto_delete_input_file=True,
        metadata={
            "description": f"Evaluate Batch"
        }
    )
    batch_id= create.id
    # # 记录file_ids为一个json文件，用于后续查询以及下载
    with open(os.path.join(conf.RESULT_DIR, ANSWER_ID, 'batch_id.json'), 'w') as f:
        json.dump(batch_id, f)
    print(f"Submit Success! batch_id were saved to {os.path.join(conf.RESULT_DIR, ANSWER_ID, 'batch_id.json')}")
    
def check_jobs(batch_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    # 检查任务状态
    batch_job = client.batches.retrieve(batch_id)
    output_file_id = batch_job.output_file_id
    if batch_job.status != "completed":
        print(f"Batch job {batch_id} is still in progress. Status: {batch_job.status}")
        print(f"Download canceled, please check the status later.")
        return None
    return output_file_id

def download_output(output_file_id):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    # 下载输出文件
    print(f"downloading output file {output_file_id}")
    content = client.files.content(output_file_id)
    # 使用write_to_file方法把返回结果写入文件
    # content.write_to_file("generated_answer.jsonl")
    eval_path = os.path.join(conf.RESULT_DIR, ANSWER_ID, 'eval.jsonl')
    
    content.write_to_file(eval_path)
    print(f"Download success! Content was saved to {os.path.join(conf.RESULT_DIR, ANSWER_ID, f'eval.jsonl')}")
    # 解析 JSONL 文件并计算平均分数
    calculate_score(eval_path)
    
    
    

def calculate_score(file_path):
    """
    Parse the input JSONL file and extract the text content.
    """
    i = 1 
    scores = {}
    with open(file_path, 'r') as f:
        for line in f:
            task_type = None
            sub_type = None
            data = json.loads(line)

            # 检查是否有 "response" 和 "body" 字段
            if "response" in data and "body" in data["response"]:
                body = data["response"]["body"]
                # 从request_id字段中提取任务类型
                if "request_id" in body:
                    request_id = body["request_id"]
                    pattern = r"request-(\d+)-<<([^>]*)>>-<<([^>]*)>>"
                    match = re.search(pattern, request_id)
                    if match:
                        task_type = match.group(2)
                        sub_type = match.group(3).lower()
                    else:
                        print(f"Failed to extract task type from request_id at line {i}")
                    # 检查任务类型是否合规
                    if task_type == "Understanding" :
                        if sub_type not in ["fill_in_the_blank", "multiple_choice", "short_answer"]:
                            print(f"Invalid at line {i}: task_type {task_type}, sub_type {sub_type}")
                            continue
                    elif task_type == "Reasoning":
                        if sub_type not in ["statement_judgment", "calculation", "short_answer"]:
                            print(f"Invalid at line {i}: task_type {task_type}, sub_type {sub_type}")
                            continue
                    else :
                        print(f"Invalid at line {i}: task_type {task_type}, sub_type {sub_type}")
                        continue
                    
                    
                # 检查是否有 "choices" 字段
                if "choices" in body and len(body["choices"]) > 0:
                    content = body["choices"][0]["message"]["content"]
                    # 提取 JSON 格式的 "result"
                try:
                    match = re.search(r'<<(\d+)>>', content)
                    if match:
                        rating = int(match.group(1))  # 提取数字并转换为整数
                        # 在对应类别中添加评分
                        if task_type not in scores:
                            scores[task_type] = {}
                        if sub_type not in scores[task_type]:
                            scores[task_type][sub_type] = []
                        scores[task_type][sub_type].append(rating)
                    else:
                        print(f"re search failed at line {i}, content:\n{content}")
                        
                except json.JSONDecodeError:
                    print(f"Failed to parse result content as JSON at line: {i}, content:\n content")
            i += 1
    # 输出每个任务类别的问题数量，平均分数，以json格式输出
    results = {}
    tot_questions = 0
    tot_score = 0
    for task_type, sub_types in scores.items():
        results[task_type] = {}
        type_tot_questions = 0
        type_tot_score = 0
        for sub_type, ratings in sub_types.items():
            type_tot_questions += len(ratings)
            type_tot_score += sum(ratings)
            avg_score = numpy.mean(ratings)
            results[task_type][sub_type] = {
                "count": len(ratings),
                "avg_score": avg_score
            }
        results[task_type]["total"] = {
            "count": type_tot_questions,
            "avg_score": type_tot_score / type_tot_questions if type_tot_questions > 0 else 0
        }
        tot_questions += type_tot_questions
        tot_score += type_tot_score
        # 在results[task_type]中添加总的评分和数量
    results["total"] = {
        "count": tot_questions,
        "avg_score": tot_score / tot_questions if tot_questions > 0 else 0
    }
    output_path = os.path.join(conf.RESULT_DIR, ANSWER_ID, f'score.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Parse Sucess! Score results saved to {output_path}")
    
    
    
    
    


def upload_task():
        
    file_path = create_batch_jsonl(os.path.join(conf.RESULT_DIR, ANSWER_ID))

    file_id = upload_batchfile(file_path)
    submit_batch_task(file_id)

def download_result():
    batch_id_path = os.path.join(conf.RESULT_DIR, ANSWER_ID, 'batch_id.json')
    if not os.path.exists(batch_id_path):
        print(f"Batch ID file not found: {batch_id_path}")
        return
    
    batch_id = json.load(open(batch_id_path, 'r'))
    
    output_file_id = check_jobs(batch_id)
    
    if output_file_id is None:
        print("No completed jobs found.")
        return

    download_output(output_file_id)



if __name__ == "__main__":
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    parser.add_argument("--answer_id", type=str, help="Answer ID for the task.")
    args = parser.parse_args()
    ANSWER_ID = args.answer_id if args.answer_id else ANSWER_ID
    # 根据参数调用对应的函数
    if args.action == "upload":
        upload_task()
    elif args.action == "download":
        download_result()