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

PROMPT_R = """你是一个资深的考试出题人,面对以下的若干考点知识,请对其巧妙组织出若干是非判断和简答题,和计算题(如果存在相关知识点),并给出答案和解析。请注意以下几点：
难度要求：难度较高的Reasoning类型,虽然答案在考点知识中,但学生不能直接看出答案,而是需要进一步的推理才能做答。
答案解析：给出答案并且给出解析过程,说明怎么推理做出的答案。
问题个数：1-5个问题,数量取决于考点知识可细分的程度。
限制条件：问题内容需要聚焦于知识本身而非其参考的表格，图片等。不要问出"表xx中可以看出什么?"，"相关信息可以在表几找到?"这种具体指代某个表格或图片的问题。
考点知识：{}
输出格式：
请返回json格式,例如:
{{
    "questions":[
        {{
            "task": "Reasoning",
            "sub_type": "statement_judgment", 
            "question": "使用透气的消毒纱包扎伤口可以防止厌氧菌的生长,这是因为透气的纱布能够阻止细胞呼吸。",
            "answer": "错误",
            "analysis": "虽然透气的纱布确实可以防止厌氧菌的生长,但这是因为它允许氧气进入伤口,而不是因为透气的纱布本身能够阻止细胞呼吸。细胞呼吸是细胞内的代谢过程,透气的纱布并不直接影响细胞呼吸。"
        }}
    ]
}}
```"""

PROMPT_U =  """你是一个资深的考试出题人,面对一下的若干考点知识,请对其组织出若干选择,填空和简答题,并给出答案和解析。请注意以下几点：
任务类型：选择题multiple_choice、填空题fill_in_the_blank、简答题short_Answer
难度要求：难度较低的基础题Understanding,学生可以自信地直接从考点知识中找到答案。
答案解析：给出答案并且给出简单的解析过程。
问题个数：1-5个问题,数量取决于考点知识可细分的程度。
限制条件：问题内容需要聚焦于知识本身而非其参考的表格，图片等。不要问出"表xx中可以看出什么?"，"相关信息可以在表几找到?"这种具体指代某个表格或图片的问题。
考点知识：{}
格式要求：
请返回json格式,例如:
{{
    "questions":[
        {{
            "task": "Understanding",
            "sub_type" : "multiple_choice",
            "question" : "下列哪一项不是细胞呼吸原理在日常生活中的应用？ A. 使用透气性良好的敷料包扎伤口 B. 给盆栽植物松土促进根部呼吸 C. 在通风环境中使用活性炭吸附异味 D. 控制通气条件以制造葡萄酒",
            "answer" : "C",
            "analysis": "活性炭吸附异味的原理与细胞呼吸无关,而是通过物理吸附和化学吸附来去除异味分子。其他选项都与细胞呼吸原理有关。"
        }},
        {{  
            "task": "Understanding",
            "sub_type": "fill_in_the_blank",
            "question": "细胞呼吸是细胞内的代谢过程,主要分为__________和__________两个种类",
            "answer": "有氧呼吸、厌氧呼吸",
            "analysis": "细胞呼吸是细胞内的代谢过程,主要分为有氧呼吸和厌氧呼吸两个种类。有氧呼吸需要氧气参与,而厌氧呼吸则不需要氧气。"
        }}
    ]
}}
```"""


def dump_jsonl(fact_srcs, file_path):
    with open(file_path, 'w') as f:
        for i, fact_srcs in tqdm(enumerate(fact_srcs), desc="Creating batch requests", total=len(fact_srcs)):
            fact_chunk = fact_srcs["facts"]
            source = fact_srcs["source"]
            for prompt in [PROMPT_R, PROMPT_U]:
                content = prompt.format(fact_chunk)
                messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]
                
                task_class = "R" if prompt == PROMPT_R else "U"
                
                f.write(json.dumps({
                    "custom_id": f"request-{i}-<{task_class}>-<{source}>",
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
                

def create_batch_jsonl(fact_path):
    # 找到index2img_filename.txt,每行提取出来,作为图片文件名的list
    # facts 将是一个list,每个元素是一个fact字符串
    # 以页面对应知识点为单位
    # 这个文件是文本处理，在本实验条件中不需要考虑batch100M大小(已经足够)
    fact_srcs = json.loads(open(fact_path, 'r').read())
    
    file_path = os.path.join(conf.TEST_DIR, f'QAgen_batch_{conf.TEST_FIELD}.jsonl')
    dump_jsonl(fact_srcs, file_path)
            
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
    print(f"Submitting QAgen task for file {file_id}")
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
    batch_id_path = os.path.join(conf.TEST_DIR, f'QAgen_batch_ids_{conf.TEST_FIELD}.json')
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
    output_file = os.path.join(conf.TEST_DIR, f"generated_QA_{conf.TEST_FIELD}.jsonl")
    content.write_to_file(output_file)
    print(f"Download success! Content was saved to {output_file}")
    parse_filter_jsonl(output_file)




def parse_filter_jsonl(input_path):
    # 先清空输出文件
    output_path = os.path.join(conf.TEST_DIR, f'test_QA_{conf.TEST_FIELD}.jsonl')
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
                    # 形如ssource<EEdesign.pdf_0.png>,提取
                    pattern = r"request-(\d+)-<([^>]*)>-<([^>]*)>"
                    match = re.search(pattern, request_id)
                    if match:
                        source = match.group(3)
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
                    
                    if "questions" in result_data:
                        for item in result_data["questions"]:
                            task = item.get("task", "N/A")
                            sub_type = item.get("sub_type", "N/A")
                            question = item.get("question", "N/A")
                            answer = item.get("answer", "N/A")
                            analysis = item.get("analysis", "N/A")
                            
                            with open(output_path, 'a') as out_f:
                                out_f.write(json.dumps({
                                    "source": source,
                                    "task": task,
                                    "sub_type": sub_type,
                                    "question": question,
                                    "answer": answer,
                                    "analysis": analysis
                                }, ensure_ascii=False) + '\n')
                            tot += 1
                            
                except json.JSONDecodeError:
                    print(f"Failed to parse result content as JSON at line: {i}, stripped_content:\n{stripped_content}")
                    err_num += 1
            i += 1
    print(f"Parsed JSONL file and extracted text content to {output_path}")
    print(f"Total : {tot - 1} QAs to Parse, Success : {tot-1-err_num} lines, Error : {err_num} lines")



def upload_task():
    # 创建批量 JSONL 文件
    fact_path = os.path.join(conf.TEST_DIR, f'filtered_facts_{conf.TEST_FIELD}.jsonl')
    file_path = create_batch_jsonl(fact_path)
    if not os.path.exists(file_path):
        print("No files to upload.")
        return

    # 上传文件并获取文件 ID
    file_id = upload_batchfile(file_path)

    # 提交任务并获取批次 ID
    submit_batch_task(file_id)



def download_result():
    # 从保存的 JSON 文件中加载批次 ID
    batch_id_path = os.path.join(conf.TEST_DIR, f'QAgen_batch_ids_{conf.TEST_FIELD}.json')
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