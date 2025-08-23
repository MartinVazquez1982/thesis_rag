import os
from core.vectorstore import VectorStore

def get_vectorstore() -> VectorStore:
    """Factory method to create a VectorStore instance.

    The provider is selected using the environment variable `VECTORSTORE_PROVIDER`.
    Currently supported:
        - "qdrant": Uses `QdrantStore`.

    Environment Variables:
        VECTORSTORE_PROVIDER (str): Vector store backend (e.g., "qdrant").
        COLLECTION (str): Name of the vector collection in the store.

    Raises:
        ValueError: If the provider is not supported.

    Returns:
        VectorStore: An instance of the selected vector store provider.
    """
    provider = os.getenv("VECTORSTORE_PROVIDER").lower()
    if provider == "qdrant":
        from infra.vectorstores.qdrant import QdrantStore
        collection = os.getenv("COLLECTION")
        return QdrantStore(collection_name=collection)
    raise ValueError(f"VectorStore provider not supported: {provider}") 