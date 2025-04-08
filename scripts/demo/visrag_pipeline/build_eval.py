from zhipuai import ZhipuAI
import json
import os
import conf
import base64
import time
# 老师的
ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"
# 我的
# ZHIPU_API_KEY="28eb0cace1314b2bbc1824aa98b823a0.eOwjqu86fz9DtMRJ"


ANSWER_ID = "Qwen-VL-3B_RAG_EE_20250402152431"

PROMPT = """You are an impartial judge. 
[Task]: Evaluate the quality of the AI assistant's response to the user's question based on helpfulness, relevance, accuracy, and detail with the reference answer and your knowledge.
[Instructions]: 
1. In the given format, "query“ means the user question, "answer" means the AI assistant's response, and "reference" means the reference answer.
2. Begin your evaluation by providing a short explanation. Be as objective as possible.
3. After providing your explanation, you must rate the response on a scale of <<1>> to <<10>>.
4. Your output should be strictly follow the given format: "Explaination:xxxx" "Rating" : <<score>>
[Response Language]: Chinese
[Response Format]: Reply in the following JSON format,here's an example:
```json
{
    "result": [
        {
            "Explaination": "AI的回答提到了生产者、消费者和分解者，这些都是生态系统的重要组成部分。然而，与参考答案相比，AI回答缺少了“非生物的物质和能量”这一重要部分，这使得回答不够全面。尽管如此，AI助手的回答在提到的部分上是准确和相关的。",
            "Rating": <<7>> #score must follow the format <<rating>>
        },
    ]
}
```"""



def dump_jsonl(judges,file_path):
    with open(file_path, 'w') as f:
        for i, judge_text in enumerate(judges):
            content = judge_text
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
            
            f.write(json.dumps({
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v4/chat/completions",
                "body": {
                    "model": "glm-4-plus",
                    "messages": messages,
                    # "response_format":{'type': 'json_object'},
                    "max_tokens": 1000
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
            judges.append(json.dumps(data, ensure_ascii=False))
            
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
    content.write_to_file(os.path.join(conf.RESULT_DIR, ANSWER_ID, f"eval.jsonl"))
    print(f"Download success! Content was saved to {os.path.join(conf.RESULT_DIR, ANSWER_ID, f'eval.jsonl')}")

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