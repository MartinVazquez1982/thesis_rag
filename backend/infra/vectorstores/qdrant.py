import uuid, os
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from core.vectorstore import VectorStore # Asumo que esta es tu clase base

class QdrantStore(VectorStore):
    def __init__(self, collection_name: str = "thesis", vector_size: int = 768):
        """Qdrant-based implementation of a VectorStore.

        This class wraps the Qdrant client to provide storage, search, and reset 
        functionality for embeddings and their associated metadata.

        Args:
            collection_name (str, optional): Name of the Qdrant collection. Defaults to "thesis".
            vector_size (int, optional): Dimension of the embedding vectors. Defaults to 768.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size

        url = os.getenv("QDRANT_URL")
        self.client = QdrantClient(url=url)

        existing_collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Add a batch of texts, metadata, and embeddings to the collection.

        Args:
            texts (List[str]): List of documents to store.
            metadatas (List[Dict[str, Any]]): Metadata dictionaries corresponding to each text.
            embeddings (List[List[float]]): Embedding vectors corresponding to each text.

        Returns:
            List[str]: A list of unique IDs assigned to the stored documents.
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=[{**meta, "text": txt} for meta, txt in zip(metadatas, texts)]
            ),
            wait=False
        )
        return ids
    
    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        """Search for the most similar documents in the collection.

        Args:
            query_embedding (List[float]): Embedding vector of the query.
            k (int, optional): Number of top results to return. Defaults to 4.

        Returns:
            Dict[str, Any]: Dictionary with search results containing:
                - "documents" (List[str]): Retrieved texts.
                - "metadatas" (List[Dict[str, Any]]): Associated metadata for each result.
                - "distances" (List[float]): Similarity scores (higher is more similar).
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True,
            with_vectors=False
        )
        
        documents, metadatas, distances = [], [], []
        for hit in search_result:
            payload = hit.payload or {}
            documents.append(payload.pop("text", ""))
            metadatas.append(payload)
            distances.append(hit.score)
            
        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
    
    def reset(self) -> None:
        """Delete and recreate the collection.

        This removes all stored data and reinitializes the collection 
        with the configured vector size and cosine similarity.
        """
        self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                on_disk=True
            )
        )