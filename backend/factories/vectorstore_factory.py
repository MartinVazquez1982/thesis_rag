import os
from core.vectorstore import VectorStore

def get_vectorstore() -> VectorStore:
    provider = os.getenv("VECTORSTORE_PROVIDER").lower()
    if provider == "chromadb":
        from infra.vectorstores.chroma import ChromaStore
        persist_dir = os.getenv("CHROMA_DB_DIR")
        collection = os.getenv("CHROMA_COLLECTION")
        return ChromaStore(persist_dir=persist_dir, collection_name=collection)
    raise ValueError(f"VectorStore provider not supported: {provider}") 