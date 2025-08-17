import os
from core.vectorstore import VectorStore

def get_vectorstore() -> VectorStore:
    provider = os.getenv("VECTORSTORE_PROVIDER").lower()
    if provider == "qdrant":
        from infra.vectorstores.qdrant import QdrantStore
        collection = os.getenv("COLLECTION")
        return QdrantStore(collection_name=collection)
    raise ValueError(f"VectorStore provider not supported: {provider}") 