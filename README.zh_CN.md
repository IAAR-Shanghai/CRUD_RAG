[English](./README.md) | [中文简体](./README.zh_CN.md)

<h1 align="center">
    📖 CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models
</h1>
<p align="center">
<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-green.svg">
</a>
<a href="https://github.com/IAAR-Shanghai/CRUD_RAG/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/IAAR-Shanghai/CRUD_RAG?color=red">
</a>
<a href="https://arxiv.org/abs/2401.17043">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-blue.svg">
</a></p>


# 要点
- 全面支持中文RAG Benchmark评测，包括原生的中文数据集、评测任务、主流基座测试；
- 覆盖CRUD(增删改查)，即大模型信息新增能力、信息缩减能力、信息校正能力、信息查询问答能力全方位评测；
- 总测试数据量达到36166个，为中文RAG测试最多；
- 多个指标类型覆盖，包括 ROUGE, BLEU, bertScore, RAGQuestEval，一键评估；


# 介绍
此仓库包含 CRUD-RAG 的官方代码，这是评估 RAG 系统的一个新颖基准。 它包括我们为评估 RAG 系统而创建的数据集，以及有关如何在我们的基准测试上运行实验的教程。

# 项目结构
```bash
├── data  #  用于评测的数据集
│   ├── crud 
│   │   └── merged.json  # 完整的数据集
│   ├── crud_split
│   │   └── split_merged.json # 在论文中我们用于实验的数据集
│   └── 80000_docs
│       └── documents_dup_part... # 超过80,000条新闻文档, 用作 RAG 系统的检索文档库
│ 
├── src 
│   ├── configs  # 包含大模型相关设置的文件
│   │   
│   ├── datasets # 用于加载数据集的脚本
│   │
│   ├── embeddings  # 构建向量数据库的embedding
│   │       
│   ├── llms # 加载大模型的脚本
│   │   │
│   │   ├── api_model.py  # 调用GPT系列模型
│   │   │
│   │   ├── local_model.py # 调用本地部署的模型
│   │   │
│   │   └── remote_model.py # 调用部署在远程，封装成api的模型
│   │
│   ├── metric # 论文中使用的评估指标
│   │   │
│   │   ├── common.py  # bleu, rouge, bertScore.
│   │   │
│   │   └── quest_eval.py # RAGQuestEval. 请注意，使用此类指标需要调用 GPT 等大型语言模型来回答问题，或者自行修改代码并部署问答模型。
│   │
│   ├── prompts # 实验中用到的prompt
│   │ 
│   ├── quest_eval # 问答指标RAGQuestEval使用的数据集
│   │ 
│   ├── retrievers # 调用检索器的脚本
│   │ 
│   │
│   └── tasks # 评估任务
│       ├── base.py
│       ├── continue_writing.py
│       ├── hallucinated_modified.py
│       ├── quest_answer.py
│       └── summary.py
```

# 快速运行
- 安装依赖项
```bash
pip install -r requirements.txt
```

- 开启milvus-lite服务
```bash
milvus-server
```

- 下载bge-base-zh-v1.5 模型到 sentence-transformers/bge-base-zh-v1.5/ 路径下

- 根据需求修改 config.py

- 启动 quick_start.py

```bash
python quick_start.py \
  --model_name 'gpt-3.5-turbo' \
  --temperature 0.1 \
  --max_new_tokens 1280 \
  --data_path 'path/to/dataset' \
  --shuffle True \
  --docs_path 'path/to/retrieval_database' \
  --docs_type 'txt' \
  --chunk_size 128 \
  --chunk_overlap 0 \
  --retriever_name 'base' \
  --collection_name 'name/of/retrieval_database' \ 
  --retrieve_top_k 8 \
  --task 'all' \
  --num_threads 20 \
  --show_progress_bar True \
  --construct_index \ # 第一次运行代码时，需要为文本构建向量索引
```

# 重要事项
- RAGQuestEval 指标的使用依赖于 GPT，我们使用 GPT 作为问题回答和生成器。你也可以自行修改代码，更换问题回答和生成模型。
- 第一次运行代码时，需要为文本构建向量索引。 这是一次性过程，因此您以后无需重复。 当您再次使用该代码时，请确保省略了construct-index参数。

# 引用
```
@article{CRUDRAG,
    title={CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models},
    author={Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, Enhong Chen},
    journal={arXiv preprint arXiv:2401.17043},
    year={2024},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=IAAR-Shanghai/CRUD_RAG&type=Date)](https://star-history.com/#IAAR-Shanghai/CRUD_RAG&Date)
