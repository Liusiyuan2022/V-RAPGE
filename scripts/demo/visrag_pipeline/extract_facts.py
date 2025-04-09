
from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
from tqdm import tqdm
import time
# 老师的
ZHIPU_API_KEY="46ed99244d8f49b5b2eb18ed9292d4df.mgThjPTMvrV7XQbh"


PROMPT = """
你是一个资深的出题人，即将出一份试卷。
任务描述: 请所给文档中选出你认为可以作为出题点的事实，作为考试范围以便后续出题。
限制条件: 从图片或者有关文字中选择你认为可以作为出题点的事实论点+相关数据，例子(如有)，从文档中#提取#出1-5条事实性信息，并且对该页面的信息价值进行评估
信息要求：1.不要自己生成额外信息，严格遵照文档抽取,如果文档中对事实论点有相关例子或数据，请把例子和数据也提取出来。2.每个fact必须信息完整
信息条数：1-5条，取决于文档有用信息的多少。
打分格式：给出一个1-10的分数，1表示这些信息价值很低，10表示信息价值很高。
回复格式json格式严格遵循以下字段，例如：
{
   "facts":[
     "基因通过控制蛋白质的合成来控制生物体的性状。例如，人的白化症是由于控制酪氨酸酶的基因突变引起的。酪氨酸酶存在于正常人的皮肤、毛发等处，它能够将酪氨酸转化为黑色素。如果一个人由于基因不正常而缺少酪氨酸酶，那么这个人就不能合成黑色素，而表现出白化症。",
     "生态系统中的能量流动具有单向流动和逐级递减的特点。能量在相邻两个营养级间的传递效率大约是10%到20%。能量在相邻两个营养级间的传递效率大约是10%到20%。",
     "浆的化学组成包括水、蛋白质、无机盐、葡萄糖、氨基酸、激素、尿素、肌酐、乳酸、二氧化碳、尿酸等。血浆中约90%为水，其余10%分别是：无机盐（约1%）、蛋白质（7%-9%）、以及血液运输的物质、包括各种营养物质（如葡萄糖）、各种代谢废物、气体、激素等"
  ]
  "confidence": 8
}
"""


def dump_jsonl(req_imgs, data_path ,file_path):
    with open(file_path, 'w') as f:
        # for i, batch in enumerate(req_imgs):
        for i, batch in tqdm(enumerate(req_imgs), desc="Creating batch requests", total=len(req_imgs)):
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
                    "role": "user",
                    "content": content
                }
            ]
            
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
    
    print(f"Flitering on {conf.TEST_FIELD} field materials")
    # 如果只提取特定的书EEmaterials和EEdesign
    if conf.TEST_FIELD == "EE":
        img_names = [img_name for img_name in img_names if "EEmaterials" in img_name or "EEdesign" in img_name]
    elif conf.TEST_FIELD == "BI":
        img_names = [img_name for img_name in img_names if "biology_cell" in img_name or "biology_enviroment" in img_name or "biology_evolution" in img_name]
        
    
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
            dump_jsonl(req_imgs, data_path, os.path.join(conf.TEST_DIR, f'extract_batch_{j}_{conf.TEST_FIELD}.jsonl'))
            print(f"Batch {j} created")
            req_imgs = []
            line = 0
            j += 1
    if len(req_imgs) > 0:
        dump_jsonl(req_imgs, data_path, os.path.join(conf.TEST_DIR, f'extract_batch_{j}_{conf.TEST_FIELD}.jsonl'))
        print(f"Batch {j} created")
    
    file_paths = [os.path.join(conf.TEST_DIR, f'extract_batch_{i}_{conf.TEST_FIELD}.jsonl') for i in range(j+1)]
            
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
        print(f"Submitting QAgen task for file {file_id}")
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
    with open(os.path.join(conf.TEST_DIR, f'extract_batch_ids_{conf.TEST_FIELD}.json'), 'w') as f:
        json.dump(batch_ids, f)
    print(f"Submit Success! batch_ids were saved to {os.path.join(conf.TEST_DIR, f'extract_batch_ids_{conf.TEST_FIELD}.json')}")
        
def check_jobs(batch_ids):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    output_file_ids = []
    for batch_id in batch_ids:
        batch_job = client.batches.retrieve(batch_id)
        output_file_ids.append(batch_job.output_file_id)
        if batch_job.status != "completed":
            print(f"Batch job {batch_id} is not completed yet. Status: {batch_job.status}")
            print(f"Download canceled, please check the status later.")
            return None
        # print(batch_job)
    return output_file_ids

# 目前实际上只支持单文件
def download_output(output_file_ids):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    for output_file_id in output_file_ids:
        # client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
        print(f"downloading output file {output_file_id}")
        content = client.files.content(output_file_id)
        # 使用write_to_file方法把返回结果写入文件
        file_path = os.path.join(conf.TEST_DIR, f"extracted_facts_{conf.TEST_FIELD}.jsonl")
        content.write_to_file(file_path)
        print(f"Download success! Content was saved to {file_path}")
            
    parse_filter_jsonl(file_path)


def parse_filter_jsonl(input_path):
    i = 1
    tot = 0
    low_conf = 0
    err_num = 0
    facts = []
    print(f"Parsing and Filtering JSONL file: {input_path}")  
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # print(data)
            # 检查是否有 "response" 和 "body" 字段
            if "response" in data and "body" in data["response"]:
                body = data["response"]["body"]
                
                # 检查是否有 "choices" 字段
                if "choices" in body and len(body["choices"]) > 0:
                    content = body["choices"][0]["message"]["content"]
                    # 提取 JSON 格式的 "result"
                try:
                    stripped_content = content.strip("`json").strip()
                    result_data = json.loads(stripped_content)
                    
                    tot += 1
                    if "confidence" in result_data:
                      confidence = result_data["confidence"]
                      if confidence < conf.CONFIDENCE_THRESHOLD:
                          low_conf += 1
                          continue
                    if "facts" in result_data:
                      facts.extend(result_data["facts"])
                    
                except json.JSONDecodeError:
                    # print(f"Failed to parse result content as JSON at line: {i}, stripped_content:\n {stripped_content}")
                    err_num += 1
            i += 1
    print(f"Parsed JSONL file and extracted text content to {input_path}")
    print(f"Total : {tot} facts to Parse, Success : {tot-1-err_num} lines, Error : {err_num} lines, Low confidence : {low_conf} lines")
    # 将结果写入jsonl文件
    output_path = os.path.join(conf.TEST_DIR, f"filtered_facts_{conf.TEST_FIELD}.jsonl")
    with open(output_path, 'w') as out_f:
        out_f.write(json.dumps(facts, ensure_ascii=False) + '\n')
    print(f"Filtered facts were saved to {output_path}")

def upload_task(data_path):
    # 创建批量 JSONL 文件
    file_paths = create_batch_jsonl(data_path)
    if not file_paths:
        print("No files to upload.")
        return

    # 上传文件并获取文件 ID
    file_ids = upload_batchfile(file_paths)

    # 提交任务并获取批次 ID
    submit_batch_task(file_ids)



def download_result():
    # 从保存的 JSON 文件中加载批次 ID
    batch_ids_path = os.path.join(conf.TEST_DIR, f'extract_batch_ids_{conf.TEST_FIELD}.json')
    if not os.path.exists(batch_ids_path):
        print(f"Batch IDs file not found: {batch_ids_path}")
        return

    batch_ids = json.load(open(batch_ids_path))
    
    # 检查任务状态并获取输出文件 ID
    output_file_ids = check_jobs(batch_ids)
    
    if output_file_ids is None:
        print("No completed jobs found.")
        return
    # print(f"batch_ids: {batch_ids}, output_file_ids: {output_file_ids}")
    # 下载输出文件
    download_output(output_file_ids)



if __name__ == "__main__":
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    parser.add_argument("--test_field", type=str, choices=["BI", "EE"], help="Set TEST_FIELD value.")
    parser.add_argument("--data_path", type=str, default=conf.DATASTORE,
                        help="Path to the data directory (required for upload).")

    args = parser.parse_args()
    conf.TEST_FIELD = args.test_field if args.test_field else conf.TEST_FIELD
    # 根据参数调用对应的函数
    if args.action == "upload":
        upload_task(args.data_path)
    elif args.action == "download":
        download_result()