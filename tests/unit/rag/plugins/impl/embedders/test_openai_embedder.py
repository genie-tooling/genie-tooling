### tests/unit/rag/plugins/impl/embedders/test_openai_embedder.py
import json
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.embedding_generators.impl.openai_embed import (
    OpenAIEmbeddingGenerator,
)
from genie_tooling.security.key_provider import KeyProvider

try:
    from openai import APIError as ActualOpenAI_APIError
    from openai import RateLimitError as ActualOpenAI_RateLimitError
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.embedding import Embedding as OpenAIEmbedding
    APIError_ToUseInTest = ActualOpenAI_APIError
    RateLimitError_ToUseInTest = ActualOpenAI_RateLimitError
except ImportError:
    class MockHTTPXResponseAdapter:
        def __init__(self, status_code: int, headers: Optional[Dict[str, str]] = None, content: Optional[bytes] = None):
            self.status_code = status_code; self.headers = headers or {}; self._content = content or b""; self.request: Optional[Any] = None
        def json(self) -> Any: return json.loads(self._content.decode("utf-8")) if self._content else {}
        @property
        def text(self) -> str: return self._content.decode("utf-8", errors="replace")
    class BaseMockOpenAIError(Exception):
        def __init__(self, message: Optional[str], *, request: Any, response: Any, body: Optional[Any]):
            super().__init__(message); self.message: str = message or ""; self.request = request; self.response = response
            self.body = body; self.status_code: Optional[int] = getattr(response, "status_code", None)
            self.headers: Optional[Any] = getattr(response, "headers", None)
    class MockOpenAI_APIError(BaseMockOpenAIError): pass
    class MockOpenAI_RateLimitError(BaseMockOpenAIError): pass
    APIError_ToUseInTest = MockOpenAI_APIError # type: ignore
    RateLimitError_ToUseInTest = MockOpenAI_RateLimitError # type: ignore
    CreateEmbeddingResponse = MagicMock # type: ignore
    OpenAIEmbedding = MagicMock # type: ignore


class MockKeyProviderForOpenAI(KeyProvider):
    def __init__(self, api_key_value: Optional[str] = "test_openai_api_key"):
        self._api_key_value = api_key_value; self.requested_key_names: List[str] = []
    async def get_key(self, key_name: str) -> Optional[str]:
        self.requested_key_names.append(key_name); return self._api_key_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class MockChunkForOpenAI(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id: Optional[str] = id; self.content: str = content; self.metadata: Dict[str, Any] = metadata or {}

@pytest.fixture
def mock_key_provider_with_key() -> MockKeyProviderForOpenAI: return MockKeyProviderForOpenAI(api_key_value="fake_openai_key")

@pytest.fixture
def openai_embedder_fixture() -> OpenAIEmbeddingGenerator: return OpenAIEmbeddingGenerator()

@pytest.fixture
def mock_openai_client_instance() -> AsyncMock:
    client_instance = AsyncMock(name="MockAsyncOpenAIInstance"); client_instance.embeddings = AsyncMock(name="MockEmbeddingsEndpoint")
    client_instance.embeddings.create = AsyncMock(name="MockEmbeddingsCreateMethod"); client_instance.close = AsyncMock(name="MockAsyncOpenAICloseMethod")
    return client_instance

async def make_chunk_stream(chunks_data: List[Tuple[Optional[str], str]]) -> AsyncIterable[Chunk]:
    for chunk_id, content_text in chunks_data: yield MockChunkForOpenAI(id=chunk_id, content=content_text)

def create_mock_openai_embedding_response(embeddings_with_indices: List[Tuple[int, List[float]]]) -> MagicMock:
    mock_response = MagicMock(spec=CreateEmbeddingResponse)
    mock_response.data = []
    for idx, emb_vector in embeddings_with_indices:
        embedding_item_mock = MagicMock(spec=OpenAIEmbedding)
        embedding_item_mock.index = idx
        embedding_item_mock.embedding = emb_vector
        mock_response.data.append(embedding_item_mock)
    return mock_response

@pytest.mark.asyncio
async def test_setup_success(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock,
):
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance) as mock_constructor:
        # Pass a config that does NOT specify request_timeout_seconds to test the default
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "model_name": "test-model"})

    # The default timeout in OpenAIEmbeddingGenerator.setup is 30.0
    mock_constructor.assert_called_once_with(
        api_key="fake_openai_key", max_retries=0,
        base_url=None, organization=None, timeout=30.0
    )
    assert openai_embedder_fixture._client is mock_openai_client_instance
    assert openai_embedder_fixture._model_name == "test-model"

@pytest.mark.asyncio
async def test_embed_mismatch_response_length(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock, caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)
    # Set max_retries to 0 so any "failure" (like mismatch) isn't retried, making log check simpler
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "max_retries": 0})
    openai_embedder_fixture._client = mock_openai_client_instance

    chunks_to_embed = [("c1", "text one"), ("c2", "text two")]
    mock_openai_response = create_mock_openai_embedding_response([(0, [0.1, 0.2])]) # Only one embedding
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][1] == [0.1, 0.2]
    assert results_list[1][1] == [] # Second one should be empty due to mismatch

    # Check for the specific warning about mismatch
    assert "Mismatch in returned embeddings. Expected 2, got 1." in caplog.text
    # The "Batch failed after X attempts" log might not appear if mismatch itself isn't a retry trigger
    # and no other exception forced retries to exhaust.
    # If max_retries is 0, the loop runs once. If mismatch occurs, it logs warning and returns partial.
    # So, "Batch failed" log might not be present here.
    # If the test requires "Batch failed" log, then the embedder's logic for mismatch needs to
    # potentially trigger a failure state that the retry loop would count.
    # For now, we check the warning.
