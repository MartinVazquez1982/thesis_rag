import os
import google.generativeai as genai
from core.llm import LLM

class GeminiLLM(LLM):
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """Gemini implementation of the LLM interface.

        This class wraps the Google Gemini Generative AI model to provide
        text generation capabilities.

        Args:
            model_name (str, optional): Name of the Gemini model to use. 
                Defaults to "gemini-1.5-pro".

        Raises:
            RuntimeError: If the environment variable `GEMINI_API_KEY` is missing.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt, **kwargs):
        """Generate text from the Gemini model given a prompt.

        Args:
            prompt (str): Input prompt to guide the model's output.
            **kwargs: Additional optional parameters to configure generation.

        Returns:
            str: Generated text response.
        """
        return self.model.generate_content(prompt).text