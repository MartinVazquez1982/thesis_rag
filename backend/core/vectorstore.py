from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class VectorStore(ABC):
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], emmbeddings: List[List[float]]) -> List[str]:
        """_summary_

        Args:
            texts (List[str]): _description_
            metadatas (List[Dict[str, Any]]): _description_
            emmbeddings (List[List[float]]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[str]: _description_
        """
        raise NotImplementedError
    
    @abstractmethod
    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        """_summary_

        Args:
            query_embedding (List[float]): _description_
            k (int, optional): _description_. Defaults to 4.

        Raises:
            NotImplementedError: _description_

        Returns:
            Dict[str, Any]: _description_
        """
        raise NotImplementedError
        