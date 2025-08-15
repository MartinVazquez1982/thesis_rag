from abc import ABC, abstractmethod

class LLM(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """_summary_

        Args:
            promt (str): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            str: _description_
        """
        raise NotImplementedError