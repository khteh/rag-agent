import atexit, bs4, hashlib, logging
from uuid_extensions import uuid7, uuid7str
from sqlalchemy.exc import ProgrammingError
from typing_extensions import List, TypedDict, Optional, Any
from langchain_postgres import PGEngine
from langchain_core.tools.retriever import create_retriever_tool
#from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config
#https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
#https://realpython.com/python-class-constructor/
#https://stackoverflow.com/questions/79506120/python-subclass-constructor-calls-metaclass-with-args-and-kwargs
class VectorStoreSingleton(type): # Inherit from "type" in order to gain access to method __call__
    __registry = {}
    def __call__(cls, *args, **kwargs):
        registry = type(cls).__registry
        if cls not in registry:
              registry[cls] = (super().__call__(*args, **kwargs), args, kwargs)
        elif registry[cls][1] != args or registry[cls][2] != kwargs:
              raise TypeError(f"Class already initialized with different arguments!")
        return registry[cls][0]
class VectorStore(): #metaclass=VectorStoreSingleton):
    """
    Class constructor
    https://python.langchain.com/api_reference/_modules/langchain_core/vectorstores/in_memory.html
    https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
    https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
    """
    _embeddings: OllamaEmbeddings = None
    _chunk_size: int = None
    _chunk_overlap: int = None
    #vector_store: PGVector = None
    _pg_engine = None
    retriever_tool = None
    _collection: str = None
    _tenant: str = None
    _database: str = None
    _pg_engine = None
    _vectorStore: PGVectorStore = None
    _docs = set()
    #def __new__(cls, *args, **kwargs):
    #    return super().__new__(cls)
    def __init__(self, vectorStore:PGVectorStore, chunk_size, chunk_overlap, tenant="khteh", database="LLM-RAG-Agent", collection="LLM-RAG-Agent"):
        logging.info(f"\n=== {self.__class__.__name__}.{self.__init__.__name__} ===")
        self._vectorStore = vectorStore
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._collection = collection
        self._tenant = tenant
        self._database = database
        #https://python.langchain.com/api_reference/_modules/langchain_ollama/embeddings.html#OllamaEmbeddings
        #https://huggingface.co/blog/matryoshka
        #https://ollama.com/library/nomic-embed-text
        self._embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_LOCAL_URI, num_ctx=config.OLLAMA_CONTEXT_LENGTH, num_gpu=1, temperature=0)
        self._pg_engine = PGEngine.from_connection_string(url=config.SQLALCHEMY_DATABASE_URI)
        #self.vector_store = PGVector( # This will execute "CREATE EXTENSION vector;" in the database. So, it needs to have the right permission.
        #    embeddings = self._embeddings,
        #    collection_name = self._collection,
        #    connection = config.SQLALCHEMY_DATABASE_URI,
        #    use_jsonb = True,
        #    async_mode = True
        #)
        # https://api.python.langchain.com/en/latest/tools/langchain.tools.retriever.create_retriever_tool.html
        if self._vectorStore is not None:
            self.retriever_tool = create_retriever_tool(
                #self.vector_store.as_retriever(),
                self._vectorStore.as_retriever(),
                "retrieve_blog_posts",
                "Search and return information about the query from the documents available in the store",
            )
        atexit.register(self.Cleanup)

    def Cleanup(self):
        logging.info(f"\n=== {self.__class__.__name__}.{self.Cleanup.__name__} ===")
        # self._embeddings.close() 'OllamaEmbeddings' object has no attribute 'close'
        # self._client.close() 'Client' object has no attribute 'close'
        # self.vector_store.close()
        self.retriever_tool = None

    async def LoadDocuments(self, urls: List[dict]) -> int:
        """
        Load and chunk contents of the blog
        https://docs.langchain.com/oss/python/langchain/rag
        https://docs.langchain.com/oss/python/integrations/document_loaders
        """
        if self._vectorStore is None:
            try:
                await self._pg_engine.ainit_vectorstore_table(
                    table_name=config.VECTORSTORE_TABLE,
                    vector_size=config.EMBEDDING_DIMENSIONS
                )
            except ProgrammingError:
                logging.warning(f"{config.VECTORSTORE_TABLE} already exist!")
            self._vectorStore = await PGVectorStore.create(
                engine=self._pg_engine,
                table_name=config.VECTORSTORE_TABLE,
                # schema_name=SCHEMA_NAME,  # Default: "public"
                embedding_service=OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_LOCAL_URI, num_ctx=config.OLLAMA_CONTEXT_LENGTH, num_gpu=1, temperature=0),
            )        
        count: int = 0
        for url in urls:
            if url["url"] not in self._docs:
                logging.info(f"\n=== {self.LoadDocuments.__name__} loading {url["url"]}... ===")
                if url["type"] == "class":
                    loader = WebBaseLoader(
                        web_paths=(url["url"],),
                        bs_kwargs=dict(
                            parse_only=bs4.SoupStrainer(
                                #class_=("post-content", "post-title", "post-header")
                                class_ = url["filter"]
                            )
                        ),
                    )
                elif url["type"] == "article":
                    loader = WebBaseLoader(
                        web_paths=(url["url"],),
                        bs_kwargs=dict(
                            parse_only=bs4.SoupStrainer("article")))
                if loader:
                    docs = loader.load()
                    assert len(docs) == 1
                    logging.debug(f"Total characters: {len(docs[0].page_content)}")
                    if len(docs[0].page_content):
                        subdocs = self._SplitDocuments(docs)
                        count += await self._IndexChunks(subdocs)
                        self._docs.add(url["url"])
        return count

    def _SplitDocuments(self, docs):
        """
        Embedding models have a fixed-size context window, and as the size of the text grows, an embedding’s ability to accurately represent the text decreases.
        """
        logging.info(f"\n=== {self._SplitDocuments.__name__} ===")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        subdocs = text_splitter.split_documents(docs)
        logging.debug(f"Split blog post into {len(subdocs)} sub-documents.")
        return subdocs

    async def _IndexChunks(self, subdocs: List[Document]) -> int:
        # Index chunks
        logging.info(f"\n=== {self._IndexChunks.__name__} ===")
        # Create a list of unique ids for each document based on the content
        ids: List[str] = []
        for doc in subdocs:
            hash = hashlib.sha3_512()
            hash.update(doc.page_content.encode('utf8'))
            doc.id = uuid7str()
            ids.append(hash.hexdigest())
        unique_ids = list(set(ids))
        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [doc for doc, id in zip(subdocs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]
        ids = []
        if len(unique_docs) and len(unique_ids) and len(unique_docs) == len(unique_ids):
            #ids = await self._vectorStore.aadd_documents(unique_docs, ids = unique_ids)
            ids = await self._vectorStore.aadd_documents(unique_docs)
            logging.debug(f"{len(ids)} documents added successfully!")
        return len(ids)
    
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        logging.info(f"\n=== {self.asimilarity_search.__name__} ===")
        return await self._vectorStore.asimilarity_search(query=query, k=k, **kwargs)
