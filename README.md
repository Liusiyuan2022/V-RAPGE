# V-RAPGE:一个带有数据生成和评测的视觉检索增强框架
V-RAPGE是一个视觉检索增强生成模型的框架，支持多种模型和数据集，具有数据生成和评测功能。
本项目是基于[VisRAG](https://github.com/OpenBMB/VisRAG.git)进行开发实践的。
# ⚙️ 设置
所需的环境配置
```bash
git clone https://github.com/Liusiyuan2022/V-RAPGE.git
conda create --name VisRAG python==3.10.8
conda activate VisRAG
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
cd VisRAG
pip install -r requirements.txt
pip install -e .
cd timm_modified
pip install -e .
cd ..
```
# 项目介绍


# 如何使用
Makefile中有一些命令可以直接使用，或者直接使用python脚本

测试cuda环境
```bash
make test
```
构建知识库：
将需要的pdf文档放在pdf_materials文件夹中，运行以下命令
```bash
make build_index
```

就会在datastore位置生成图片以及索引信息


## 构建问题库：
问题库依赖于build_index的图片结果，需要先运行上面的步骤

**需要指定TEST_FIELD 变量**，以按照知识领域构建多个问题库
使用智谱AI的batch命令，故接下来的步骤需要成对运行上传和下载，当然如果自己有智谱api_key也可以登陆网页版进行上传和下载。

### 抽取信息
首先我们会从pdf图片中批量的抽取事实信息，送入智谱AI的API进行处理。
运行以下命令
```bash
make upload_extract TEST_FIELD=<your_test_field>
```
这就会构建出一个batch请求，上传到智谱AI的API
如果运行完毕可以下载输出文件
```bash
make download_extract TEST_FIELD=<your_test_field>
```
就可以在test文件夹中看到抽取的事实信息，最终结果是保存在scripts/demo/test目录下的filtered_facts_***.jsonl文件，

### 生成问题
我们会根据上一部分抽取的事实信息，生成问题和答案对
同样使用智谱AI的batch命令。
运行以下命令
```bash
make upload_QAgen
```
以构建生成问题的batch请求到智谱AI的API
如果运行完毕可以下载输出文件
```bash
make download_QAgen
```
就可以在test文件夹中看到生成的test_QA_**.jsonl文件了

### 问题质量过滤
我们会根据生成的问题和答案对，进行质量过滤，最终获得高质量的问题和答案对
同样使用智谱AI的batch命令。
运行以下命令
```bash
make upload_QAcheck
```
以构建batch请求到智谱AI的API
如果运行完毕可以下载输出文件
```bash
make download_QAcheck
```
就可以在test文件夹中看到QA_quality_**.jsonl文件了这对应着对应qa的质量

还可以看到生成的qualified_QA_**.jsonl以及unqualified文件，这便是经过质量分选后问题和答案对

### 进行推理
推理的时候
**需要指定TEST_FIELD 变量**，以按照知识领域推理
**需要指定MODEL 变量**，以指定使用的模型
**需要指定RAG_EN变量**，以指定是否使用RAG
例如
```bash
make answer TEST_FIELD=EE MODEL=Qwen-VL-3B RAG_EN=True
```

推理的结果将被存在answer_log里面，会有一个子文件夹记录有关信息，例如answer_log/Qwen-VL-3B_RAG_BI_20250408072103
推理结果是answer.jsonl文件

评测采用智谱平台的模型，同样是创建batch去处理
包括三个命令
根据answer.jsonl文件生成评测的batch请求，为batch.jsonl文件
同时上传到智谱AI的API，留下对应的batch_id.json文件
```bash
make upload_eval ANSWER_ID=<answer_dir_name>
```

下载评测结果，评测结果会存在对应的文件夹下面，例如answer_log/Qwen-VL-3B_RAG_BI_20250408072103/eval.jsonl
如果这个batch还没有执行玩，会有相关信息，需要在等待一段时间之后再执行
```bash
make download_eval ANSWER_ID=<answer_dir_name>
```

通过读取和解析评测结果，得到最终的均分分数
```bash
make score ANSWER_ID=<answer_dir_name>
```
