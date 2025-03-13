from typing_extensions import List, TypedDict, Optional, Any
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
#global vector_store
# https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
class VectorStoreSingleton(type): # Inherit from "type" in order to gain access to method __call__
    def __init__(self, *args, **kwargs):
        self.__instance = None # Create a variable to store the object reference
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            # if the object has not already been created
            self.__instance = super().__call__(*args, **kwargs) # Call the __init__ method of the subclass (Spam) and save the reference
            return self.__instance
        else:
            # if object (Spam) reference already exists; return it
            return self.__instance    
class VectorStore(metaclass=VectorStoreSingleton):
    """
    Class constructor
    https://python.langchain.com/api_reference/_modules/langchain_core/vectorstores/in_memory.html#InMemoryVectorStore
    """
    _model: str = None
    _embeddings: VertexAIEmbeddings = None
    _vector_store: InMemoryVectorStore = None
    def __init__(self, model="text-embedding-005"):
        self._model = model
        self._embeddings = VertexAIEmbeddings(model=self._model) # "text-embedding-005"
        self._vector_store = InMemoryVectorStore(self._embeddings)
        print(f"VectorStore::__init__ {self._model}")

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        return self._vector_store.add_documents(documents=documents, ids=ids, **kwargs)
    
    async def aadd_documents(
        self, documents: list[Document], ids: Optional[list[str]] = None, **kwargs: Any
    ) -> list[str]:
        print(f"\n=== {self.aadd_documents.__name__} ===")
        return await self._vector_store.aadd_documents(documents=documents, ids=ids, **kwargs)
    
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return self._vector_store.similarity_search(query-query, k=k, **kwargs)
    
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        print(f"\n=== {self.asimilarity_search.__name__} ===")
        return await self._vector_store.asimilarity_search(query=query, k=k, **kwargs)