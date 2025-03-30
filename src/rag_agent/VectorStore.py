import bs4, logging
from dotenv import load_dotenv
from typing_extensions import List, TypedDict, Optional, Any
from langchain.tools.retriever import create_retriever_tool
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
"""
https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
https://realpython.com/python-class-constructor/
https://stackoverflow.com/questions/79506120/python-subclass-constructor-calls-metaclass-with-args-and-kwargs
"""
class VectorStoreSingleton(type): # Inherit from "type" in order to gain access to method __call__
    __registry = {}
    def __call__(cls, *args, **kwargs):
        registry = type(cls).__registry
        if cls not in registry:
              registry[cls] = (super().__call__(*args, **kwargs), args, kwargs)
        elif registry[cls][1] != args or registry(cls)[2] != kwargs:
              raise TypeError(f"Class already initialized with different arguments!")
        return registry[cls][0]
class VectorStore(metaclass=VectorStoreSingleton):
    """
    Class constructor
    https://python.langchain.com/api_reference/_modules/langchain_core/vectorstores/in_memory.html#InMemoryVectorStore
    https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
    """
    _model: str = None
    _embeddings: VertexAIEmbeddings = None
    _chunk_size = None
    _chunk_overlap = None
    _vector_store: InMemoryVectorStore = None
    retriever_tool: None
    _docs = set()
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    def __init__(self, model, chunk_size, chunk_overlap):
        self._model = model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        #vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location=os.environ.get("GOOGLE_CLOUD_LOCATION"))
        self._embeddings = VertexAIEmbeddings(model=self._model)
        self._vector_store = InMemoryVectorStore(self._embeddings)
        # https://api.python.langchain.com/en/latest/tools/langchain.tools.retriever.create_retriever_tool.html
        self.retriever_tool = create_retriever_tool(
            self._vector_store.as_retriever(),
            "Retrieve information related to a query",
            "Search and return information about the query from the documents available in the store",
        )   

    async def LoadDocuments(self, urls: List[str]):
        # Load and chunk contents of the blog
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
                await self._IndexChunks(subdocs)
                self._docs.add(url)

    def _SplitDocuments(self, docs):
        """
        Embedding models have a fixed-size context window, and as the size of the text grows, an embedding’s ability to accurately represent the text decreases.
        """
        logging.info(f"\n=== {self._SplitDocuments.__name__} ===")
        text_splitter = RecursiveCharacterTextSplitter(self._chunk_size, self._chunk_overlap)
        subdocs = text_splitter.split_documents(docs)
        logging.debug(f"Split blog post into {len(subdocs)} sub-documents.")
        return subdocs

    async def _IndexChunks(self, subdocs):
        # Index chunks
        logging.info(f"\n=== {self._IndexChunks.__name__} ===")
        ids = await self._vector_store.aadd_documents(documents=subdocs)
        logging.debug(f"{len(ids)} documents added successfully!")
    
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        logging.info(f"\n=== {self.asimilarity_search.__name__} ===")
        return await self._vector_store.asimilarity_search(query=query, k=k, **kwargs)

#vector_store = VectorStore(model="text-embedding-005", chunk_size=1000, chunk_overlap=100)