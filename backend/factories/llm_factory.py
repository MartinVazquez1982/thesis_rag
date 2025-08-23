import os
from core.llm import LLM

def get_llm() -> LLM:
    """Factory method to create a Language Model (LLM) instance.

    The provider is selected using the environment variable `LLM_PROVIDER`.
    Currently supported:
        - "gemini": Uses `GeminiLLM`.

    Environment Variables:
        LLM_PROVIDER (str): LLM backend (e.g., "gemini").
        GEMINI_LLM_MODEL (str): Model name to use for Gemini LLM.

    Raises:
        ValueError: If the provider is not supported.

    Returns:
        LLM: An instance of the selected language model provider.
    """
    provider = os.getenv("LLM_PROVIDER").lower()
    if provider == "gemini":
        from infra.llm.gemini import GeminiLLM
        model = os.getenv("GEMINI_LLM_MODEL")
        return GeminiLLM(model_name=model)
    raise ValueError(f"LLM provider not supported: {provider}") 