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
    from openai import APIStatusError as ActualAPIStatusError
    from openai import RateLimitError as ActualOpenAI_RateLimitError
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.embedding import Embedding as OpenAIEmbedding
    APIError_ToUseInTest = ActualAPIStatusError
    RateLimitError_ToUseInTest = ActualOpenAI_RateLimitError
except ImportError:
    class MockHTTPXResponseAdapter:
        def __init__(self, status_code: int, headers: Optional[Dict[str, str]] = None, content: Optional[bytes] = None):
            self.status_code = status_code
            self.headers = headers or {}
            self._content = content or b""
            self.request: Optional[Any] = None
        def json(self) -> Any: return json.loads(self._content.decode("utf-8")) if self._content else {}
        @property
        def text(self) -> str: return self._content.decode("utf-8", errors="replace")


    class BaseMockOpenAIError(Exception):
        def __init__(self, message: str, *, response: Any, body: Optional[Any] = None):
            super().__init__(f"{message} (status_code={response.status_code if response else 'N/A'})")
            self.message = message
            self.response = response
            self.body = body
            self.status_code = getattr(response, "status_code", None)
            self.headers = getattr(response, "headers", None)
            self.request = getattr(response, "request", None)

    class MockOpenAI_APIStatusError(BaseMockOpenAIError):
        pass
    class MockOpenAI_RateLimitError(BaseMockOpenAIError):
        pass
    APIError_ToUseInTest = MockOpenAI_APIStatusError # type: ignore
    RateLimitError_ToUseInTest = MockOpenAI_RateLimitError # type: ignore
    CreateEmbeddingResponse = MagicMock # type: ignore
    OpenAIEmbedding = MagicMock # type: ignore


class MockKeyProviderForOpenAI(KeyProvider):
    def __init__(self, api_key_value: Optional[str] = "test_openai_api_key"):
        self._api_key_value = api_key_value
        self.requested_key_names: List[str] = []
    async def get_key(self, key_name: str) -> Optional[str]:
        self.requested_key_names.append(key_name)
        return self._api_key_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    async def teardown(self) -> None:
        pass

class MockChunkForOpenAI(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id: Optional[str] = id
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata or {}

@pytest.fixture()
def mock_key_provider_with_key() -> MockKeyProviderForOpenAI: return MockKeyProviderForOpenAI(api_key_value="fake_openai_key")

@pytest.fixture()
def openai_embedder_fixture() -> OpenAIEmbeddingGenerator: return OpenAIEmbeddingGenerator()

@pytest.fixture()
def mock_openai_client_instance() -> AsyncMock:
    client_instance = AsyncMock(name="MockAsyncOpenAIInstance")
    client_instance.embeddings = AsyncMock(name="MockEmbeddingsEndpoint")
    client_instance.embeddings.create = AsyncMock(name="MockEmbeddingsCreateMethod")
    client_instance.close = AsyncMock(name="MockAsyncOpenAICloseMethod")
    return client_instance

async def make_chunk_stream(chunks_data: List[Tuple[Optional[str], str]]) -> AsyncIterable[Chunk]:
    for chunk_id, content_text in chunks_data:
        yield MockChunkForOpenAI(id=chunk_id, content=content_text)

def create_mock_openai_embedding_response(embeddings_with_indices: List[Tuple[int, List[float]]]) -> MagicMock:
    mock_response = MagicMock(spec=CreateEmbeddingResponse)
    mock_response.data = []
    for idx, emb_vector in embeddings_with_indices:
        embedding_item_mock = MagicMock(spec=OpenAIEmbedding)
        embedding_item_mock.index = idx
        embedding_item_mock.embedding = emb_vector
        mock_response.data.append(embedding_item_mock)
    return mock_response

@pytest.mark.asyncio()
async def test_setup_success(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock,
):
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance) as mock_constructor:
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "model_name": "test-model"})

    mock_constructor.assert_called_once_with(
        api_key="fake_openai_key", max_retries=0,
        base_url=None, organization=None, timeout=30.0
    )
    assert openai_embedder_fixture._client is mock_openai_client_instance
    assert openai_embedder_fixture._model_name == "test-model"

@pytest.mark.asyncio()
async def test_setup_no_keyprovider(openai_embedder_fixture: OpenAIEmbeddingGenerator, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR)
    await openai_embedder_fixture.setup(config={})
    assert "KeyProvider instance not provided" in caplog.text
    assert openai_embedder_fixture._client is None

@pytest.mark.asyncio()
async def test_embed_successful_batch(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock
):
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})
    openai_embedder_fixture._client = mock_openai_client_instance
    chunks_to_embed = [("c1", "text one"), ("c2", "text two")]
    mock_openai_response = create_mock_openai_embedding_response([(0, [0.1, 0.2]), (1, [0.3, 0.4])])
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][1] == [0.1, 0.2]
    assert results_list[1][1] == [0.3, 0.4]
    mock_openai_client_instance.embeddings.create.assert_awaited_once()

@pytest.mark.asyncio()
async def test_embed_handles_multiple_batches(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock
):
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})
    openai_embedder_fixture._client = mock_openai_client_instance
    chunks_to_embed = [("c1", "t1"), ("c2", "t2"), ("c3", "t3")]
    mock_resp1 = create_mock_openai_embedding_response([(0, [1.0]), (1, [2.0])])
    mock_resp2 = create_mock_openai_embedding_response([(0, [3.0])])
    mock_openai_client_instance.embeddings.create.side_effect = [mock_resp1, mock_resp2]

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed), config={"batch_size": 2}):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 3
    assert mock_openai_client_instance.embeddings.create.call_count == 2

@pytest.mark.asyncio()
async def test_embed_with_retries_on_ratelimiterror(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "max_retries": 1, "initial_retry_delay": 0.01})
    openai_embedder_fixture._client = mock_openai_client_instance
    chunks_to_embed = [("c1", "text one")]
    mock_success_resp = create_mock_openai_embedding_response([(0, [1.0, 2.0])])


    mock_error_response = MagicMock()
    mock_error_response.headers = {"retry-after": "0.02"}
    mock_error_response.request = MagicMock() # The real exception needs a request on its response.
    rate_limit_exception = RateLimitError_ToUseInTest(message="rate limited", response=mock_error_response, body=None)


    mock_openai_client_instance.embeddings.create.side_effect = [
        rate_limit_exception,
        mock_success_resp
    ]

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert "Rate Limit Error (attempt 1)" in caplog.text
    assert mock_openai_client_instance.embeddings.create.call_count == 2
    assert len(results_list) == 1
    assert results_list[0][1] == [1.0, 2.0]

@pytest.mark.asyncio()
async def test_embed_fails_after_all_retries(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "max_retries": 1, "initial_retry_delay": 0.01})
    openai_embedder_fixture._client = mock_openai_client_instance
    chunks_to_embed = [("c1", "text one")]


    mock_error_response = MagicMock()
    # The APIStatusError (base for APIError) needs the status_code on the response.
    mock_error_response.status_code = 500
    mock_error_response.request = MagicMock()
    api_error_exception = APIError_ToUseInTest(message="server error", response=mock_error_response, body=None)


    mock_openai_client_instance.embeddings.create.side_effect = api_error_exception

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert "Batch failed after 2 attempts" in caplog.text
    assert mock_openai_client_instance.embeddings.create.call_count == 2
    assert len(results_list) == 1
    assert results_list[0][1] == []

@pytest.mark.asyncio()
async def test_embed_mismatch_response_length(
    openai_embedder_fixture: OpenAIEmbeddingGenerator, mock_key_provider_with_key: MockKeyProviderForOpenAI, mock_openai_client_instance: AsyncMock, caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "max_retries": 0})
    openai_embedder_fixture._client = mock_openai_client_instance

    chunks_to_embed = [("c1", "text one"), ("c2", "text two")]
    mock_openai_response = create_mock_openai_embedding_response([(0, [0.1, 0.2])])
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][1] == [0.1, 0.2]
    assert results_list[1][1] == []

    assert "Mismatch in returned embeddings. Expected 2, got 1." in caplog.text
