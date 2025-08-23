from abc import ABC, abstractmethod
from typing import List

class Embeddings(ABC):
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of texts.

        Args:
            texts (List[str]): List of input strings to embed.

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            List[List[float]]: A list of embedding vectors, one per input text.
        """
        raise NotImplementedError