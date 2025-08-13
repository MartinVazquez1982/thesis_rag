from abc import ABC, abstractmethod
from typing import List

class Embeddings(ABC):
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """_summary_

        Args:
            texts (List[str]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            List[List[float]]: _description_
        """
        raise NotImplementedError