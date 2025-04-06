import json
import os
import conf

def parse_jsonl(input_dir, output_dir):
    """
    Parse the input JSONL file and extract the text content.
    """
    # 先清空输出文件
    with open(output_dir, 'w') as out_f:
        out_f.write('')
    i = 1    
    tot = 0
    err_num = 0
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
                    stripped_content = content.strip("`json").strip()
                    result_data = json.loads(stripped_content)
                    # print(f"Parsed JSON content at line: {i}, stripped_content:\n {stripped_content}")
                    if "result" in result_data:
                        for item in result_data["result"]:
                            question = item.get("question", "N/A")
                            reasoning = item.get("reasoning", "N/A")
                            answer = item.get("answer", "N/A")
                            # 将结果写入jsonl文件
                            with open(output_dir, 'a') as out_f:
                                out_f.write(json.dumps({
                                    "question": question,
                                    # "reasoning": reasoning,
                                    "answer": answer
                                }, ensure_ascii=False) + '\n')
                            tot += 1
                            
                except json.JSONDecodeError:
                    # print(f"Failed to parse result content as JSON at line: {i}, stripped_content:\n {stripped_content}")
                    err_num += 1
            i += 1
    print(f"Parsed JSONL file and extracted text content to {output_dir}")
    print(f"Total : {tot - 1} QAs to Parse, Success : {tot-1-err_num} lines, Error : {err_num} lines")


if __name__ == "__main__":
    # 遍历 TEST_DIR 中所有以 generated_ 开头的 .jsonl 文件
    for filename in os.listdir(conf.TEST_DIR):
        if filename.startswith("generated_") and filename.endswith(".jsonl"):
            input_file = os.path.join(conf.TEST_DIR, filename)
            output_file = os.path.join(conf.TEST_DIR, filename.replace("generated_", "test_"))
            
            # 调用 parse_jsonl 解析文件
            parse_jsonl(input_file, output_file)