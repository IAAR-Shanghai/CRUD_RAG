from abc import ABC
from typing import List
from operator import itemgetter

from llama_index.schema import TextNode
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType
from langchain.schema.embeddings import Embeddings

from src.retrievers import BaseRetriever, CustomBM25Retriever
from FlagEmbedding import FlagReranker


def bge_rerank_result(query_text: str, docs: List[str], top_n):
    reranker = FlagReranker('sentence-transformers/bge-rerank-base')

    scores = reranker.compute_score([[query_text, passage] for passage in docs])
    
    score_doc_pairs = zip(scores, docs)
    sorted_pairs = sorted(score_doc_pairs, key=lambda x: x[0], reverse=True)
    sorted_docs = [doc for _, doc in sorted_pairs]

    return sorted_docs[:top_n]

class EnsembleRerankRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
        ):
        super().__init__()
        self.weights = [0.5, 0.5]
        self.c: int = 60
        self.top_k = similarity_top_k
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k

        self.embedding_retriever = BaseRetriever(
            docs_directory, embed_model=embed_model, embed_dim=embed_dim,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            construct_index=construct_index, add_index=add_index,
            collection_name=collection_name, similarity_top_k=similarity_top_k,
        )
        self.bm25_retriever = CustomBM25Retriever(
            docs_directory, embed_model=embed_model,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            similarity_top_k=similarity_top_k,
        )

    def search_docs(self, query_text: str):
        bm25_search_docs = self.bm25_retriever.search_docs(query_text)
        bm25_search_docs = bm25_search_docs.split("\n")
        embedding_search_docs = self.embedding_retriever.search_docs(query_text)
        embedding_search_docs = embedding_search_docs.split("\n\n")

        doc_lists = [bm25_search_docs, embedding_search_docs]

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        doc_lists = list(all_documents)
        rerank_doc_lists = bge_rerank_result(query_text, doc_lists, top_n=self.top_k)

        return "\n\n".join(rerank_doc_lists)
