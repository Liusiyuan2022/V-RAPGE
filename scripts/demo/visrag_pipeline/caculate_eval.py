import json
import os
import conf
import re
import numpy
ANSWER_ID = "20250401124728"

def parse_jsonl(input_dir):
    """
    Parse the input JSONL file and extract the text content.
    """
    i = 1 
    scores = []  
    with open(input_dir, 'r') as f:
        for line in f:
            data = json.loads(line)

            # 检查是否有 "response" 和 "body" 字段
            if "response" in data and "body" in data["response"]:
                body = data["response"]["body"]
                
                # 检查是否有 "choices" 字段
                if "choices" in body and len(body["choices"]) > 0:
                    content = body["choices"][0]["message"]["content"]
                    # 提取 JSON 格式的 "result"
                try:
                    match = re.search(r'<<(\d+)>>', content)
                    if match:
                        rating = int(match.group(1))  # 提取数字并转换为整数
                        scores.append(rating)
                    else:
                        print(f"re search failed at line {i}, content:\n{content}")
                    
                    
            
                            
                except json.JSONDecodeError:
                    print(f"Failed to parse result content as JSON at line: {i}, content:\n content")
            i += 1
    avg_score = numpy.mean(scores)
    print(f"The average rating is : {avg_score}")

parse_jsonl(os.path.join(conf.RESULT_DIR, ANSWER_ID, 'eval.jsonl'))