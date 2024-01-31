# Highlights

- 全面支持中文RAG Benchmark评测，包括原生的中文数据集、评测任务、主流基座测试；
- 覆盖CRUD(增删改查)，即大模型信息新增能力、信息缩减能力、信息校正能力、信息查询问答能力 全方位的评测；
- 总测试数据量达到36166个，为中文RAG测试最多；
- 多个指标类型覆盖，包括 ROUGE, BLEU, bertScore, RAGQuestEval，一键评估；
- TODO：增加更多英文RAG评测，双语支持；欢迎 Star 持续关注！

# CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models

# Introduction
This repository contains the official code of CRUD-RAG, a novel benchmark for evaluting the RAG systems. It includes the datasets we created for evaluating RAG systems, and a tutorial on how to run the experiments on our benchmark.

# Project Structure
```bash
├── data  #  This folder comprises the datasets used for evaluation.
│   ├── crud 
│   │   └── merged.json  # The complete datasets.
│   ├── crud_split
│   │   └── split_merged.json # The dataset we used for experiments in the paper.
│   └── 80000_docs
│       └── documents_dup_part... # More than 80,000 news documents, which are used to build the retrieval database of the RAG system.
│ 
│ 
├── src 
│   ├── configs  # This folder comprises scripts used to initialize the loading parameters of the LLMs in RAG systems. 
│   │   
│   ├── datasets # This folder contains scripts used to load the dataset.
│   │
│   ├── embeddings  # The embedding model used to build vector databases.
│   │       
│   ├── llms # This folder contains scripts used to load the large language models. 
│   │   │
│   │   ├── api.py  # Call GPT-series models.
│   │   │
│   │   ├── local_model.py # Call a locally deployed model.
│   │   │
│   │   └── remote_model.py # Call the model deployed remotely and encapsulated into an API.
│   │
│   ├── metric # The evaluation metric we used in the experiments.
│   │   │
│   │   ├── common.py  # bleu, rouge, bertScore.
│   │   │
│   │   └── quest_eval.py # RAGQuestEval. Note that using such metric requires calling a large language model such as GPT to answer questions, or modifying the code and deploying the question answering model yourself.
│   │
│   ├── prompts # The prompts we used in the experiments.
│   │ 
│   ├── quest_eval # Question answering dataset for RAGQuestEval metric.
│   │ 
│   ├── retrievers # The retriever used in RAG system.
│   │ 
│   │
│   └── tasks # The evaluation tasks.
│       ├── base.py
│       ├── continue_writing.py
│       ├── hallucinated_modified.py
│       ├── quest_answer.py
│       └── summary.py
```

# Quick Start
- Install dependency packages
```bash
pip install -r requirements.txt
```

- Start the milvus-lite service(vector database)
```bash
milvus-server
```

- Download the bge-base-zh-v1.5 model to the sentence-transformers/bge-base-zh-v1.5/ directory

- Modify config.py according to your need.

- Run quick_start.py

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
  --construct_index True \ # you need to build a database when you use it first time
```

# CITATION
```
@article{CRUDRAG,
    title={CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models},
    author={Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, Enhong Chen},
    journal={arXiv preprint arXiv:2401.17043},
    year={2024},
}
```
