import os
from typing import List
import google.generativeai as genai
from core.embeddings import Embeddings

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        """Gemini implementation of the Embeddings interface.

        This class wraps the Google Gemini embedding model to provide
        vector representations of text.

        Args:
            model_name (str, optional): Name of the Gemini embedding model.
                Defaults to "text-embedding-004".

        Raises:
            RuntimeError: If the environment variable `GEMINI_API_KEY` is missing.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model_name = model_name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: Embedding vectors for each input text.
        """
        out = []
        for t in texts:
            e = genai.embed_content(model=self.model_name, content=t)
            out.append(e["embedding"])
        return out