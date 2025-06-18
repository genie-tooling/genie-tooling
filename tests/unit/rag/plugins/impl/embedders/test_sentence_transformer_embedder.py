### tests/unit/rag/plugins/impl/embedders/test_sentence_transformer_embedder.py
"""Unit tests for SentenceTransformerEmbedder."""
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector

# Updated import path for SentenceTransformerEmbedder
from genie_tooling.embedding_generators.impl.sentence_transformer import (
    SentenceTransformerEmbedder,
)

# Mock SentenceTransformer class if the library isn't installed for testing purposes
try:
    import numpy as ActualNumpy
    from sentence_transformers import SentenceTransformer as ActualSentenceTransformer
except ImportError:
    ActualSentenceTransformer = None # type: ignore
    ActualNumpy = None # type: ignore


class MockSentenceTransformerModel:
    """A mock for the SentenceTransformer model."""
    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        self.model_name = model_name_or_path
        self.device = device
        self.encode_should_raise: Optional[Exception] = None
        self.fixed_embedding_dim: int = 3 # Default dim for mock

    def encode(self, sentences: List[str], show_progress_bar: bool = False, batch_size: int = 32) -> List[List[float]]:
        if self.encode_should_raise:
            raise self.encode_should_raise
        # Return dummy embeddings based on index WITHIN THE CURRENT BATCH
        return [[0.1 * (i + 1)] * self.fixed_embedding_dim for i in range(len(sentences))]

    def set_encode_should_raise(self, error: Exception):
        self.encode_should_raise = error

    def set_fixed_embedding_dim(self, dim: int):
        self.fixed_embedding_dim = dim


@pytest.fixture()
def mock_sentence_transformer_instance() -> MockSentenceTransformerModel:
    return MockSentenceTransformerModel("mock-model")

@pytest.fixture()
def st_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()

# Mock Chunk for testing
class MockChunkForST(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id: Optional[str] = id
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata or {}

async def make_chunk_stream_st(chunks_data: List[Tuple[Optional[str], str]]) -> AsyncIterable[Chunk]:
    for chunk_id, content_text in chunks_data:
        yield MockChunkForST(id=chunk_id, content=content_text)

async def collect_embeddings(embedder: SentenceTransformerEmbedder, chunks_data: List[Tuple[Optional[str], str]], config: Optional[Dict[str, Any]] = None) -> List[Tuple[Chunk, EmbeddingVector]]:
    results: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in embedder.embed(make_chunk_stream_st(chunks_data), config=config):
        results.append((chunk_obj, vector))
    return results

@pytest.mark.asyncio()
async def test_st_setup_success(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test successful setup with default model."""
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance) as mock_st_constructor_patch, \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup(config={"model_name": "test-model", "device": "cpu"})

        mock_st_constructor_patch.assert_called_once_with("test-model", "cpu")
        assert st_embedder._model is mock_sentence_transformer_instance
        assert st_embedder._model_name == "test-model"
        assert st_embedder._device == "cpu"

@pytest.mark.asyncio()
async def test_st_setup_library_not_installed(st_embedder: SentenceTransformerEmbedder, caplog: pytest.LogCaptureFixture):
    """Test setup when sentence-transformers library is not installed."""
    caplog.set_level(logging.ERROR)
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", None), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup()
        assert st_embedder._model is None
        assert "'sentence-transformers' library not installed" in caplog.text

@pytest.mark.asyncio()
async def test_st_setup_numpy_not_installed(st_embedder: SentenceTransformerEmbedder, caplog: pytest.LogCaptureFixture):
    """Test setup when numpy library is not installed."""
    caplog.set_level(logging.ERROR)
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", ActualSentenceTransformer or MagicMock()), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", None):
        await st_embedder.setup()
        assert st_embedder._model is None
        assert "'numpy' library not installed" in caplog.text


@pytest.mark.asyncio()
async def test_st_setup_model_load_failure(st_embedder: SentenceTransformerEmbedder, caplog: pytest.LogCaptureFixture):
    """Test setup when SentenceTransformer model loading fails."""
    caplog.set_level(logging.ERROR)
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", side_effect=RuntimeError("Model load failed")), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup(config={"model_name": "failing-model"})
        assert st_embedder._model is None
        assert "Failed to load model 'failing-model': Model load failed" in caplog.text

@pytest.mark.asyncio()
async def test_st_embed_success_single_batch(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test successful embedding of a single batch of chunks."""
    mock_sentence_transformer_instance.set_fixed_embedding_dim(5)
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup(config={"model_name": "test-model"})

    chunks_data = [("c1", "text one"), ("c2", "text two")]
    results = await collect_embeddings(st_embedder, chunks_data, config={"batch_size": 5})

    assert len(results) == 2
    assert results[0][0].id == "c1"
    assert results[0][1] == [0.1] * 5
    assert results[1][0].id == "c2"
    assert results[1][1] == [0.2] * 5

@pytest.mark.asyncio()
async def test_st_embed_success_multiple_batches(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test successful embedding requiring multiple batches."""
    mock_sentence_transformer_instance.set_fixed_embedding_dim(2)
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup(config={"model_name": "test-model"})

    chunks_data = [("c1", "t1"), ("c2", "t2"), ("c3", "t3"), ("c4", "t4"), ("c5", "t5")]
    mock_encode_method = MagicMock(wraps=mock_sentence_transformer_instance.encode)
    mock_sentence_transformer_instance.encode = mock_encode_method # type: ignore

    results = await collect_embeddings(st_embedder, chunks_data, config={"batch_size": 2})

    assert len(results) == 5
    assert mock_encode_method.call_count == 3

    # Batch 1 ("t1", "t2") -> i=0, i=1 -> [0.1]*2, [0.2]*2
    # Batch 2 ("t3", "t4") -> i=0, i=1 -> [0.1]*2, [0.2]*2
    # Batch 3 ("t5")       -> i=0       -> [0.1]*2
    assert results[0][1] == [0.1] * 2 # for "t1"
    assert results[1][1] == [0.2] * 2 # for "t2"
    assert results[2][1] == [0.1] * 2 # for "t3"
    assert results[3][1] == [0.2] * 2 # for "t4"
    assert results[4][1] == [0.1] * 2 # for "t5" (Corrected expectation)


    # Check how the wrapped encode method was called by the partial
    assert mock_encode_method.call_args_list[0].kwargs["sentences"] == ["t1", "t2"]
    assert mock_encode_method.call_args_list[1].kwargs["sentences"] == ["t3", "t4"]
    assert mock_encode_method.call_args_list[2].kwargs["sentences"] == ["t5"]


@pytest.mark.asyncio()
async def test_st_embed_model_not_loaded(st_embedder: SentenceTransformerEmbedder, caplog: pytest.LogCaptureFixture):
    """Test embedding when the model failed to load during setup."""
    caplog.set_level(logging.ERROR)
    st_embedder._model = None
    # Updated patch path for numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        chunks_data = [("c1", "text one")]
        results = await collect_embeddings(st_embedder, chunks_data)

    assert len(results) == 0
    assert "Model not loaded. Cannot generate embeddings." in caplog.text

@pytest.mark.asyncio()
async def test_st_embed_empty_chunk_stream(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test embedding with an empty stream of chunks."""
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup()

    results = await collect_embeddings(st_embedder, [])
    assert len(results) == 0

@pytest.mark.asyncio()
async def test_st_embed_encode_failure(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel, caplog: pytest.LogCaptureFixture):
    """Test embedding when the model's encode method raises an exception."""
    caplog.set_level(logging.ERROR)
    mock_sentence_transformer_instance.set_encode_should_raise(ValueError("Encoding failed deliberately"))
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup()

    chunks_data = [("c1", "text one"), ("c2", "another text")]
    results = await collect_embeddings(st_embedder, chunks_data, config={"batch_size": 1})

    assert len(results) == 2
    assert results[0][1] == []
    assert results[1][1] == []
    assert caplog.text.count("Error during batch embedding: Encoding failed deliberately") == 2
    assert "Error during final batch embedding" not in caplog.text # because batch size is 1


@pytest.mark.asyncio()
async def test_st_teardown(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test that teardown releases the model."""
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup()
    assert st_embedder._model is not None
    await st_embedder.teardown()
    assert st_embedder._model is None

@pytest.mark.asyncio()
async def test_st_embed_show_progress_bar_config(st_embedder: SentenceTransformerEmbedder, mock_sentence_transformer_instance: MockSentenceTransformerModel):
    """Test that show_progress_bar config is passed to model.encode."""
    # Updated patch paths for SentenceTransformer and numpy
    with patch("genie_tooling.embedding_generators.impl.sentence_transformer.SentenceTransformer", return_value=mock_sentence_transformer_instance), \
         patch("genie_tooling.embedding_generators.impl.sentence_transformer.numpy", ActualNumpy or MagicMock()):
        await st_embedder.setup()

    mock_encode_method = MagicMock(wraps=mock_sentence_transformer_instance.encode)
    mock_sentence_transformer_instance.encode = mock_encode_method # type: ignore

    chunks_data = [("c1", "text")]
    await collect_embeddings(st_embedder, chunks_data, config={"show_progress_bar": True})

    mock_encode_method.assert_called_once()
    assert mock_encode_method.call_args.kwargs.get("show_progress_bar") is True

    mock_encode_method.reset_mock()
    await collect_embeddings(st_embedder, chunks_data, config={"show_progress_bar": False})
    mock_encode_method.assert_called_once()
    assert mock_encode_method.call_args.kwargs.get("show_progress_bar") is False

    mock_encode_method.reset_mock()
    await collect_embeddings(st_embedder, chunks_data) # Default
    mock_encode_method.assert_called_once()
    assert mock_encode_method.call_args.kwargs.get("show_progress_bar") is False
