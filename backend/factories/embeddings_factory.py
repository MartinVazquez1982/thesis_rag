import os
from core.embeddings import Embeddings

def get_embeddings() -> Embeddings:
    """Factory method to create an Embeddings provider instance.

    The provider is selected using the environment variable `EMBEDDINGS_PROVIDER`.
    Currently supported:
        - "gemini": Uses `GeminiEmbeddings`.

    Environment Variables:
        EMBEDDINGS_PROVIDER (str): Embeddings backend (e.g., "gemini").
        GEMINI_EMBED_MODEL (str): Model name to use for Gemini embeddings.

    Raises:
        ValueError: If the provider is not supported.

    Returns:
        Embeddings: An instance of the selected embeddings provider.
    """
    provider = os.getenv("EMBEDDINGS_PROVIDER").lower()
    if provider == "gemini":
        from infra.embeddings.gemini import GeminiEmbeddings
        model = os.getenv("GEMINI_EMBED_MODEL")
        return GeminiEmbeddings(model_name=model)
    raise ValueError(f"Embeddings provider not supported: {provider}")