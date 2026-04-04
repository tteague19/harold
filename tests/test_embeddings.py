"""Tests for the embedding generation module.

Verifies that ``embed_text`` and ``embed_texts`` correctly invoke
the Pydantic AI Embedder and return vectors of the expected shape.
Uses mocking to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harold.config import DEFAULT_EMBEDDING_DIMENSIONS, HaroldSettings
from harold.memory.embeddings import _embedder_cache, embed_text, embed_texts

MOCK_DIMENSIONS = DEFAULT_EMBEDDING_DIMENSIONS
MOCK_VECTOR_VALUE = 0.1
SINGLE_RESULT_COUNT = 1
SAMPLE_TEXTS = ["first text", "second text", "third text"]


@pytest.fixture(autouse=True)
def clear_embedder_cache() -> None:
    """Clear the module-level embedder cache before each test.

    Ensures test isolation by preventing cached embedder instances
    from leaking between tests.
    """
    _embedder_cache.clear()


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide a list of sample text inputs for batch embedding tests.

    Returns:
        A list of short text strings.
    """
    return list(SAMPLE_TEXTS)


def _make_mock_result(
    count: int,
    dimensions: int = MOCK_DIMENSIONS,
) -> MagicMock:
    """Create a mock EmbeddingResult with the given number of vectors.

    Args:
        count: Number of embedding vectors to include.
        dimensions: Dimensionality of each vector.

    Returns:
        A MagicMock with an ``embeddings`` attribute containing
        lists of floats.
    """
    result = MagicMock()
    result.embeddings = [
        [MOCK_VECTOR_VALUE] * dimensions for _ in range(count)
    ]
    return result


@patch("harold.memory.embeddings.Embedder")
async def test_embed_text_returns_correct_dimensions(
    mock_embedder_class: MagicMock,
) -> None:
    """Verify that embed_text returns a vector of the configured dimensions.

    Args:
        mock_embedder_class: Mocked Embedder class.
    """
    mock_instance = MagicMock()
    mock_instance.embed_query = AsyncMock(
        return_value=_make_mock_result(SINGLE_RESULT_COUNT)
    )
    mock_embedder_class.return_value = mock_instance

    settings = HaroldSettings()
    result = await embed_text("test input", settings)

    assert isinstance(result, list)
    assert len(result) == MOCK_DIMENSIONS
    mock_instance.embed_query.assert_awaited_once_with("test input")


@patch("harold.memory.embeddings.Embedder")
async def test_embed_texts_returns_batch(
    mock_embedder_class: MagicMock,
    sample_texts: list[str],
) -> None:
    """Verify that embed_texts returns one vector per input text.

    Args:
        mock_embedder_class: Mocked Embedder class.
        sample_texts: List of sample text inputs.
    """
    mock_instance = MagicMock()
    mock_instance.embed_documents = AsyncMock(
        return_value=_make_mock_result(len(sample_texts))
    )
    mock_embedder_class.return_value = mock_instance

    settings = HaroldSettings()
    result = await embed_texts(sample_texts, settings)

    assert len(result) == len(sample_texts)
    assert all(len(vec) == MOCK_DIMENSIONS for vec in result)
    mock_instance.embed_documents.assert_awaited_once_with(sample_texts)


@patch("harold.memory.embeddings.Embedder")
async def test_embedder_is_cached(
    mock_embedder_class: MagicMock,
) -> None:
    """Verify that repeated calls reuse the same Embedder instance.

    Args:
        mock_embedder_class: Mocked Embedder class.
    """
    mock_instance = MagicMock()
    mock_instance.embed_query = AsyncMock(
        return_value=_make_mock_result(SINGLE_RESULT_COUNT)
    )
    mock_embedder_class.return_value = mock_instance

    settings = HaroldSettings()
    await embed_text("first", settings)
    await embed_text("second", settings)

    mock_embedder_class.assert_called_once()
