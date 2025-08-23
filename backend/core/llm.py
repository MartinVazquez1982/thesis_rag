from abc import ABC, abstractmethod

class LLM(ABC):
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the language model given a prompt.

        Args:
            prompt (str): Input text prompt to guide the model output.
            **kwargs: Additional parameters to configure generation (e.g., temperature, max tokens).

        Raises:
            NotImplementedError: Must be implemented in subclasses.

        Returns:
            str: Generated response from the language model.
        """
        raise NotImplementedError