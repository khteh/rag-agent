import asyncio, atexit, bs4, hashlib, logging
#from chromadb.config import Settings
from psycopg_pool import AsyncConnectionPool
from typing_extensions import List, TypedDict, Optional, Any
from langchain.tools.retriever import create_retriever_tool
#from langchain_google_vertexai import VertexAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
#from langchain_chroma import Chroma
from langchain_core.documents import Document
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
    https://python.langchain.com/api_reference/_modules/langchain_core/vectorstores/in_memory.html#InMemoryVectorStore
    https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
    """
    _model: str = None
    _embeddings: OllamaEmbeddings = None
    _chunk_size: int = None
    _chunk_overlap: int = None
    #vector_store: Chroma = None
    vector_store: PGVector = None
    retriever_tool = None
    #_client: chromadb.HttpClient = None
    _collection: str = None
    _tenant: str = None
    _database: str = None
    _docs = set()
    #def __new__(cls, *args, **kwargs):
    #    return super().__new__(cls)
    def __init__(self, model, chunk_size, chunk_overlap, tenant="khteh", database="LLM-RAG-Agent", collection="LLM-RAG-Agent"):
        self._model = model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._collection = collection
        self._tenant = tenant
        self._database = database
        #https://python.langchain.com/api_reference/_modules/langchain_ollama/embeddings.html#OllamaEmbeddings
        #https://huggingface.co/blog/matryoshka
        #https://ollama.com/library/nomic-embed-text
        self._embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_URI, num_ctx=8192, num_gpu=1, temperature=0)
        #settings=Settings(chroma_client_auth_provider = "chromadb.auth.token_authn.TokenAuthenticationServerProvider",
        #                            chroma_client_auth_credentials = config.CHROMA_TOKEN,
        #                            #chroma_client_auth_token_transport_header = "X-Chroma-Token"
        #))
        #https://github.com/chroma-core/chroma/issues/1474
        #https://github.com/chroma-core/chroma/blob/main/chromadb/test/client/test_database_tenant.py
        # Create two databases with same name in different tenants
        #admin_client = client_factories.create_admin_client_from_system()
        #admin_client.create_tenant("test_tenant1")
        #admin_client.create_tenant("test_tenant2")
        #admin_client.create_database("test_db", tenant="test_tenant1")
        #admin_client.create_database("test_db", tenant="test_tenant2")

        # Create collections in each database with same name
        #client.set_tenant(tenant="test_tenant1", database="test_db")
        #coll_tenant1 = client.create_collection("collection")
        #client.set_tenant(tenant="test_tenant2", database="test_db")
        #coll_tenant2 = client.create_collection("collection")
        #self.CreateTenantDatabase()
        #self._client = chromadb.HttpClient(host=config.CHROMA_URI, port=80, headers={"X-Chroma-Token": config.CHROMA_TOKEN}, tenant=self._tenant, database=self._database)
        #self._client.reset()  # resets the database - delete all data. Must be enabled with ALLOW_RESET env in chroma server
        #self._vector_store = InMemoryVectorStore(self._embeddings)
        #self.vector_store = Chroma(client = self._client, collection_name = self._collection, embedding_function = self._embeddings)
        self.vector_store = PGVector( # This will execute "CREATE EXTENSION vector;" in the database. So, it needs to have the right permission.
            embeddings = self._embeddings,
            collection_name = self._collection,
            connection = config.SQLALCHEMY_DATABASE_URI,
            use_jsonb = True,
            async_mode = True
        )
        # https://api.python.langchain.com/en/latest/tools/langchain.tools.retriever.create_retriever_tool.html
        self.retriever_tool = create_retriever_tool(
            self.vector_store.as_retriever(),
            "retrieve_blog_posts",
            "Search and return information about the query from the documents available in the store",
        )
        atexit.register(self.Cleanup)

    def Cleanup(self):
        logging.info(f"\n=== {self.Cleanup.__name__} ===")
        # self._embeddings.close() 'OllamaEmbeddings' object has no attribute 'close'
        # self._client.close() 'Client' object has no attribute 'close'
        # self.vector_store.close()
        self.retriever_tool = None

    #def CreateTenantDatabase(self):
        #tenant_id = f"tenant_user:{user_id}"
        #For Local Chroma server:
        #adminClient = chromadb.AsyncAdminClient(Settings(
        #    chroma_api_impl="chromadb.api.segment.SegmentAPI",
        #    is_persistent=True,
        #    persist_directory="multitenant",
        #))
    #    logging.info(f"{self.CreateTenantDatabase.__name__} tenant: {self._tenant}, database: {self._database}")
        # For Remote Chroma server:
    #    adminClient = chromadb.AdminClient(Settings( # Does NOT support context manager protocol
    #       chroma_api_impl="chromadb.api.fastapi.FastAPI",
    #       chroma_server_host=config.CHROMA_URI,
    #       chroma_server_http_port=80,
    #    ))
    #    try:
    #        adminClient.get_tenant(name=self._tenant)
    #    except Exception:
    #        adminClient.create_tenant(name=self._tenant)
    #    try:
    #        adminClient.get_database(name=self._database, tenant=self._tenant)
    #    except Exception:
    #        adminClient.create_database(name=self._database, tenant=self._tenant)

    async def LoadDocuments(self, urls: List[str]) -> int:
        # Load and chunk contents of the blog
        count: int = 0
        for url in urls:
            if url not in self._docs:
                logging.info(f"\n=== {self.LoadDocuments.__name__} loading {url}... ===")
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(
                            class_=("post-content", "post-title", "post-header")
                        )
                    ),
                )
                docs = loader.load()
                assert len(docs) == 1
                logging.debug(f"Total characters: {len(docs[0].page_content)}")
                subdocs = self._SplitDocuments(docs)
                count += await self._IndexChunks(subdocs)
                self._docs.add(url)
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

    async def _IndexChunks(self, subdocs) -> int:
        # Index chunks
        logging.info(f"\n=== {self._IndexChunks.__name__} ===")
        # Create a list of unique ids for each document based on the content
        ids: List[str] = []
        for doc in subdocs:
            hash = hashlib.sha3_512()
            hash.update(doc.page_content.encode('utf8'))
            ids.append(hash.hexdigest())
        unique_ids = list(set(ids))
        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [doc for doc, id in zip(subdocs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]
        ids = []
        if len(unique_docs) and len(unique_ids) and len(unique_docs) == len(unique_ids):
            ids = await self.vector_store.aadd_documents(documents = unique_docs, ids = unique_ids)
            logging.debug(f"{len(ids)} documents added successfully!")
        return len(ids)
    
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        logging.info(f"\n=== {self.asimilarity_search.__name__} ===")
        return await self.vector_store.asimilarity_search(query=query, k=k, **kwargs)
