import os
from core.llm import LLM

def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER").lower()
    if provider == "gemini":
        from infra.llm.gemini import GeminiLLM
        model = os.getenv("GEMINI_LLM_MODEL")
        return GeminiLLM(model_name=model)
    raise ValueError(f"LLM provider not supported: {provider}") 