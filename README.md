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


# Highlights
- This project fully supports the Chinese RAG system evaluation, which includes native Chinese datasets, evaluation tasks, and baseline models;
- It covers CRUD (Create, Read, Update, Delete) operations, which are used to evaluate the RAG system's ability to add, reduce, correct information, as well as to answer questions based on the retrieve information;
- It contains 36166 test samples, which is the largest number of Chinese RAG tests available;
- It supports multiple evaluation metrics, such as ROUGE, BLEU, bertScore, RAGQuestEval, and provides a one-click evaluation function;

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
│   │   ├── api_model.py  # Call GPT-series models.
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
  --construct_index \ # you need to build a vector index when you use it first time
```

# Important Notes
- The use of RAGQuestEval metric relies on GPT, we use GPT as question answer and generator.**
- The first time you run the code, you need to build a vector index for the text. This is a one-time process, so you don't need to repeat it later. Please make sure to omit the construct-index parameter when you use the code again.

# CITATION
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
