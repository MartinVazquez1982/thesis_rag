import os
from typing import List
import google.generativeai as genai
from core.llm import LLM

class GeminiLLM(LLM):
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt, **kwargs):
        return self.model.generate_content(prompt)