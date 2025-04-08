Colpali和VisRAG都配好环境，部署并跑了retriever部分，都是模型下到本地去跑的，还能撑得住。

VisRAG包含生成部分，项目中使用的模型是Vis-Gen(MiniCPM-V-2_6)，可以接受图片和文本输入，输出为文本。
问题是，Vis-Ret的大小还可以本地部署一下，但是Vis-Gen的模型太大了，本地不太好部署。
解决方法包括：
1. 使用线上算力平台看看能不能跑起来
2. 使用一些api接口，试了deepseek的，但是貌似不支持
```python
messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", 'content': [question,*images]},
        ],
```
的输入格式，只支持文本输入?试试其他的，或者再看看api文档？


在超算平台部署了这堆东西，至少把模型放上去了，跑answer.py显示torch版本不对，numpy也要求是1.x才行，需要改
此外超算平台提交作业看看怎么搞

配了半天只剩decord没配了，因为需要yum安装一堆东西，而且还要管理员权限，不太好搞(除非那些东西一个一个源码安装),试试硬跑一下

找到了pytorch和decord，在/home/bingxing2/apps/package下面有现成的whl，直接pipinstall就好。

但是报错找不到依赖库libcupti.so.11.8,看了一下自己的是12.0，为什么呢？

发现是因为安装toolkit11.8的时候，如果有默认的channel，conda本身都是12.4的，于是toolkit本身是11.8，但是其他东西都是12.4哪怕是nvcc -V 都显示12.4，于是就安装了但是却报错。

正确做法是去 -c nvidia/label/cuda-11.8.0 优先级必须高于默认的channel，这样安装的时候就不会出现问题了。

然后又发现有一个叫nsight-compute的东西，在conda安装过程中死活装不上。

遇到这样的情况，可以去anaconda里面找tar.bz2，本地下载，传到服务器，然后conda install xxx.tar.bz2


终于把answer.py跑通了，服务器环境不支持input现读，因此要改成预定的列表。

此外，多个GPU的时候不能指定叫cuda，这时候有cuda:0 cuda:1等等，具体的分配要研究一下。

用了ddp，现在的问题是，服务器有时候会分配12.x的版本，会与当前11.8冲突报错，除非进行一次手动的升到11.2，另外大多数生成VLM只接受一张图片，要利用拼接函数拼前k个

解决了多设备分配，无需指定某个模型to(device),只需要device_map="auto"，就会自动将模型不同的层分片在多个GPU
注意的是这里是推理，目前的需求是单机多卡推理，因此也无需写ddp(用于训练)

实现了拼接图片，在构建index的时候发现，由于VisRAG-Ret的三方实现，暂不支持分片操作：模型中的 get_vllm_embedding 函数在执行 scatter_ 操作时，发现操作的源张量和目标张量位于不同的设备上（cuda:0 和 cuda:3）而PyTorch 要求 scatter_ 操作的所有张量必须在同一设备上，因此无法分片

在执行build_index，Ret只能用一张GPU，而demo的Ret模型，必须指定在特定的设备上面(如cuda:0)才可以

将三个生物课本装了进去，检索目标比较好，但是生成模型毕竟只是个2B的qwen-VL，处理效果不太好，之后试一下cogvlm2-llama3-chinese-chat-19B 的效果

cogvlm2是需要xformer等库，还可能会用到remote code，这会导致服务器出现些问题，试试要不要先换点别的（minicpm也可以）或者是先做前后端交互。


弃用了cogvlm2

对给定数据集生成QA对的任务，直接给VLM一张图片和prompt，还是说利用OCR将文字信息扫描，布局信息记录下来，表格处理成md格式，同时图片换成一个文字总结，来让VLM生成QA对？如果是后者，需要先走一个OCR的流程，看看pymupdf和docmind是什么。

尝试了一下，阿里的文档智能，大概是可以获得解析后json格式的文件，这个太长，元素相关的信息需要处理，而且也调模型一个个需要生成对应小图片的总结。这个需要4o做高级LVLM而且还要用文档api
而直接用prompt去喂图片给大语言模型，也能生成QA，而且理解能力也比较好，所以可能会更不错？需要一个gpt-4o的模型。

部署deepseek-VL2 (small) 16B 和 MiniCPM-V2.6 8B 和 Qwen2.5-VL-7B-instruct
https://huggingface.co/openbmb/MiniCPM-V-2_6
https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

获得了GLM的api，老师的key在README里面，我的在网站上可以去找
https://www.bigmodel.cn
搭了一个生成QA的pipeline，目前可以提交上任务，但是还没跑成功(模型只能用4v-plus，不能用最新版，此外五张图一个请求的话报告prompt超长)，需要再找找看
只传一张图片，结果还行，不会爆超长，但是会有的成功有的失败。看官方文档中，batch目前还是支持glm4v,没有plus版本，改一下模型之后按格式上传看看。
改完之后还是不行，但是发现官方只能用jpeg格式的图片，png不行，改成jpeg之后试试
问题解决了，是png格式的问题
搞到test_QA了300多条数据之后可以部署推理测试了,这个parse里面还是会有"图5-35中"，需要手动将这种正则表达式替换掉


模型都传到服务器了，但是还没有对接开跑，符合缓存格式的是deepseekVL2，其他的直接平铺放在了目录里面之后调一下。先试试deepseekVL2
这个存在版本不对齐的情况，可能要为了这个重新配环境。

这几百条数据可以对接开跑了，用的暂且是原来那个Qwen2B，流水线搭好，最后还剩一步评估的步骤，也需要用GLM的batch
搭好了batch以及parse自动计算评分
Qwen-vl2B The average rating is : 7.110778443113772
但是可能是精简限制了输出，细节作为评分标准的话，评分不太高，可能需要重新生成一下。

模型方面：试试新的
试试minicpm和Qwen7B





对比无RAG的生成情况（不同领域的知识）
分析讨论部分可以强调一下来说明RAG适用的domain

航空数据集？因为普通的LLM在生物课本这种级别的数据早就经过训练了，需要之后替换。
换出了航空数据集，也是一些课本，但是领域更垂直，有相关QA对了

问题生成的分类也许可以丰富一下
目前不是特别需要，再搞个更加私密的数据集，丰富一下领域就好了

第二次过程记录
已经提交

本地batch和老师账号的batch
以及QAgen的参数化，同时在QAgen的步骤中打通了下载流程，接下来需要在eval部分进行参数化和下载流程。一旦实现，就可以将我的api换成老师的api了。

可以根据领域，RAG与否，不同模型进行配置运行了，以获得最后的实验数据。
实现了对于answer的参数化：领域，RAG与否，不同模型，接下来部署一下7B的Qwen和MiniCPM，看看效果如何。


有一些论文要读，了解数据生成？
读一下论文，但是不一定用得上

对比普通的强大LLM
之后可以试一下

-----------
接下来的任务

打通eval部分的batch

跑通Qwen7B和MiniCPM

找一个更加私域的数据集：比如复旦大学教务处的信息，因为这是内部信息，在Qwen2B上回答未知，是很好的材料。

原来的服务器没有办法用了，要在新的服务器上面配置有关环境
在新的服务器上面要下载模型需要使用
```bash
hf-download model <repo_id>
```
操作要在datacenter/models下面进行，防止太多模型副本占用空间

模型的话本地已经有Qwen3B/7B/32B的模型，用这三个应该差不多了