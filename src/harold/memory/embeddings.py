"""Embedding generation via Pydantic AI's Embedder.

Provides thin wrappers around the Pydantic AI ``Embedder`` class to
generate vector embeddings for text. The embedder instance is cached
at module level to avoid recreating it on every call.
"""

from __future__ import annotations

from pydantic_ai import Embedder

from harold.config import HaroldSettings

_embedder_cache: dict[str, Embedder] = {}


def _get_embedder(settings: HaroldSettings) -> Embedder:
    """Return a cached Embedder instance for the configured model.

    Creates the embedder on first call and reuses it for subsequent
    calls with the same model identifier.

    Args:
        settings: Application configuration containing the embedding
            model identifier.

    Returns:
        A Pydantic AI Embedder instance for the configured model.
    """
    model = settings.embedding_model
    if model not in _embedder_cache:
        _embedder_cache[model] = Embedder(model)
    return _embedder_cache[model]


async def embed_text(text: str, settings: HaroldSettings) -> list[float]:
    """Generate an embedding vector for a single text string.

    Args:
        text: The text to embed.
        settings: Application configuration containing the embedding
            model identifier.

    Returns:
        A list of floats representing the text in vector space.
    """
    embedder = _get_embedder(settings)
    result = await embedder.embed_query(text)
    return list(result.embeddings[0])


async def embed_texts(
    texts: list[str], settings: HaroldSettings
) -> list[list[float]]:
    """Generate embedding vectors for multiple texts in a single batch.

    Args:
        texts: The texts to embed.
        settings: Application configuration containing the embedding
            model identifier.

    Returns:
        A list of embedding vectors, one per input text, each a list
        of floats.
    """
    embedder = _get_embedder(settings)
    result = await embedder.embed_documents(texts)
    return [list(embedding) for embedding in result.embeddings]
