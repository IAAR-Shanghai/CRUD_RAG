from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import ElasticsearchStore

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index import QueryBundle
from elasticsearch import Elasticsearch


class CustomBM25Retriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs_80k",
            construct_index: bool = False,
            similarity_top_k: int=2,
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.collection_name = collection_name

        if construct_index:
            self.construct_index()

        self.es_client = Elasticsearch([{'host': 'localhost', 'port': 9221, "scheme":"http"}])
        print("Elasticsearch connected!")

    
    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = ElasticsearchStore(
            index_name=self.collection_name, es_url="http://localhost:9221"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

        print("Indexing finished!")

    def search_docs(self, query_text: str):
        query = QueryBundle(query_text)

        result = []
        dsl = {
            'query': {
                'match': {
                    'content': query.query_str
                }
            },
            "size": self.similarity_top_k
        }
        search_result = self.es_client.search(index=self.collection_name, body=dsl)
        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['content']
                result.append(text)
        
        result_str = '\n'.join(result)

        return result_str

