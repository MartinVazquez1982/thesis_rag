import os
from core.embeddings import Embeddings

def get_embeddings() -> Embeddings:
    provider = os.getenv("EMBEDDINGS_PROVIDER").lower()
    if provider == "gemini":
        from infra.embeddings.gemini import GeminiEmbeddings
        model = os.getenv("GEMINI_EMBED_MODEL")
        return GeminiEmbeddings(model_name=model)
    raise ValueError(f"Embeddings provider not supported: {provider}")