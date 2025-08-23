from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class VectorStore(ABC):
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[str]:
        """Add texts and their embeddings to the vector store.

        Args:
            texts (List[str]): List of documents to store.
            metadatas (List[Dict[str, Any]]): Metadata associated with each document.
            embeddings (List[List[float]]): Embedding vectors corresponding to each text.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            List[str]: A list of IDs assigned to the stored documents.
        """
        raise NotImplementedError
    
    @abstractmethod
    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        """Query the vector store for the most relevant documents.

        Args:
            query_embedding (List[float]): Embedding vector of the query.
            k (int, optional): Number of top results to retrieve. Defaults to 4.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            Dict[str, Any]: A dictionary containing retrieved documents, 
                their metadata, and similarity scores/distances.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        """Clear the collection to allow reindexing.

        Raises:
            NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError