import os
from typing import List
import google.generativeai as genai
from core.embeddings import Embeddings

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model_name = model_name
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            e = genai.embed_content(model=self.model_name, content=t)
            out.append(e["embedding"])
        return out