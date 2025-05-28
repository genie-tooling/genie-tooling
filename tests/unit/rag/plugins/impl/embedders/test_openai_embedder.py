### tests/unit/rag/plugins/impl/embedders/test_openai_embedder.py
import json  # For mock_httpx_response_factory
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import httpx  # For mocking request/response objects
import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector

# Imports from the module under test
# Updated import path for OpenAIEmbeddingGenerator
from genie_tooling.embedding_generators.impl.openai_embed import (
    OpenAIEmbeddingGenerator,
)
from genie_tooling.security.key_provider import KeyProvider

# --- Start: Revised Mock Exception Handling ---
try:
    from openai import APIError as ActualOpenAI_APIError
    from openai import APIStatusError as ActualOpenAI_APIStatusError  # type: ignore
    from openai import RateLimitError as ActualOpenAI_RateLimitError

    APIError_ToUseInTest = ActualOpenAI_APIError
    RateLimitError_ToUseInTest = ActualOpenAI_RateLimitError

except ImportError:
    # Fallbacks for when openai library is not installed.
    # These mocks will be used if the real ones can't be imported.
    # Their __init__ should accept the arguments as they are called in the tests.

    class MockHTTPXResponseAdapter: # Adapter to provide attributes expected by error objects
        def __init__(self, status_code: int, headers: Optional[Dict[str, str]] = None, content: Optional[bytes] = None):
            self.status_code = status_code
            self.headers = headers or {}
            self._content = content or b""
            self.request: Optional[MockHTTPXRequestAdapter] = None # Can be set if needed

        def json(self) -> Any:
            if self._content:
                try:
                    return json.loads(self._content.decode("utf-8"))
                except json.JSONDecodeError:
                    raise ValueError("Failed to decode JSON from mock response content.")
            return {}

        @property
        def text(self) -> str:
            return self._content.decode("utf-8", errors="replace")

    class MockHTTPXRequestAdapter:
        def __init__(self, method: str, url: str):
            self.method = method
            self.url = url # Simplified URL handling for mock

    class BaseMockOpenAIError(Exception):
        def __init__(self, message: Optional[str], *, request: Any, response: Any, body: Optional[Any]):
            super().__init__(message)
            self.message: str = message or ""
            self.request = request
            self.response = response # Should be a MockHTTPXResponseAdapter instance
            self.body = body
            self.status_code: Optional[int] = getattr(response, "status_code", None)
            self.headers: Optional[Any] = getattr(response, "headers", None)

    class MockOpenAI_APIError(BaseMockOpenAIError):
        pass

    class MockOpenAI_RateLimitError(BaseMockOpenAIError): # For simplicity, inherit base
        pass

    APIError_ToUseInTest = MockOpenAI_APIError # type: ignore
    RateLimitError_ToUseInTest = MockOpenAI_RateLimitError # type: ignore

# --- End: Revised Mock Exception Handling ---


# Mock KeyProvider
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


# Mock Chunk
class MockChunkForOpenAI(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id: Optional[str] = id
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata or {}


# Fixtures
@pytest.fixture
def mock_key_provider_with_key() -> MockKeyProviderForOpenAI:
    return MockKeyProviderForOpenAI(api_key_value="fake_openai_key")


@pytest.fixture
def mock_key_provider_no_key() -> MockKeyProviderForOpenAI:
    return MockKeyProviderForOpenAI(api_key_value=None)


@pytest.fixture
def openai_embedder_fixture() -> OpenAIEmbeddingGenerator:
    return OpenAIEmbeddingGenerator()


@pytest.fixture
def mock_openai_client_instance() -> AsyncMock:
    client_instance = AsyncMock(name="MockAsyncOpenAIInstance")
    client_instance.embeddings = AsyncMock(name="MockEmbeddingsEndpoint")
    client_instance.embeddings.create = AsyncMock(name="MockEmbeddingsCreateMethod")
    client_instance.close = AsyncMock(name="MockAsyncOpenAICloseMethod")
    return client_instance

@pytest.fixture
def mock_httpx_request_object() -> Any: # Use Any for flexibility between real and mock
    try:
        return httpx.Request(method="POST", url="https://api.openai.com/v1/embeddings")
    except NameError: # httpx not available
        return MockHTTPXRequestAdapter(method="POST", url="https://api.openai.com/v1/embeddings")

@pytest.fixture
def mock_httpx_response_object_factory(): # Renamed for clarity
    def _factory(status_code: int, json_body: Optional[Dict] = None, text_body: Optional[str] = None, headers: Optional[Dict] = None) -> Any:
        try:
            content_bytes: Optional[bytes] = None
            final_headers = headers or {}

            if json_body is not None:
                content_bytes = json.dumps(json_body).encode("utf-8")
                if "content-type" not in final_headers:
                    final_headers["content-type"] = "application/json"
            elif text_body is not None:
                content_bytes = text_body.encode("utf-8")
                if "content-type" not in final_headers:
                    final_headers["content-type"] = "text/plain"

            return httpx.Response(
                status_code=status_code,
                content=content_bytes,
                headers=final_headers
            )
        except NameError: # httpx not available
            return MockHTTPXResponseAdapter(
                status_code=status_code,
                headers=headers,
                content=json.dumps(json_body).encode("utf-8") if json_body else (text_body.encode("utf-8") if text_body else None) # type: ignore
            )
    return _factory


# Helper to create AsyncIterable of Chunks
async def make_chunk_stream(chunks_data: List[Tuple[Optional[str], str]]) -> AsyncIterable[Chunk]:
    for chunk_id, content_text in chunks_data:
        yield MockChunkForOpenAI(id=chunk_id, content=content_text)


# --- Test Cases (Selected failing tests corrected) ---

@pytest.mark.asyncio
async def test_embed_success_multiple_batches(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    # This test doesn't involve error instantiation, so it should be fine.
    # Re-included for completeness of the file.
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    chunks_to_embed = [("c1", "text1"), ("c2", "text2"), ("c3", "text3")]

    mock_emb1 = MagicMock(); mock_emb1.embedding = [1.0]
    mock_emb2 = MagicMock(); mock_emb2.embedding = [2.0]
    mock_emb3 = MagicMock(); mock_emb3.embedding = [3.0]

    mock_response_batch1 = MagicMock(); mock_response_batch1.data = [mock_emb1, mock_emb2]
    mock_response_batch2 = MagicMock(); mock_response_batch2.data = [mock_emb3]

    mock_openai_client_instance.embeddings.create.side_effect = [
        mock_response_batch1,
        mock_response_batch2,
    ]

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed), config={"batch_size": 2}):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 3
    assert results_list[0][1] == [1.0]
    assert results_list[1][1] == [2.0]
    assert results_list[2][1] == [3.0]
    assert mock_openai_client_instance.embeddings.create.call_count == 2
    mock_openai_client_instance.embeddings.create.assert_any_call(input=["text1", "text2"], model=openai_embedder_fixture._model_name)
    mock_openai_client_instance.embeddings.create.assert_any_call(input=["text3"], model=openai_embedder_fixture._model_name)


@pytest.mark.asyncio
async def test_setup_success(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance) as mock_constructor:
        await openai_embedder_fixture.setup(
            config={"key_provider": mock_key_provider_with_key, "model_name": "test-model"}
        )

    mock_constructor.assert_called_once_with(
        api_key="fake_openai_key",
        max_retries=0,
        base_url=None,
        organization=None
    )
    assert openai_embedder_fixture._client is mock_openai_client_instance
    assert openai_embedder_fixture._model_name == "test-model"
    assert mock_key_provider_with_key.requested_key_names == [openai_embedder_fixture._api_key_name]


@pytest.mark.asyncio
async def test_setup_custom_config(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    custom_config = {
        "key_provider": mock_key_provider_with_key,
        "model_name": "custom-model-003",
        "api_key_name": "CUSTOM_OPENAI_KEY",
        "max_retries": 5,
        "initial_retry_delay": 0.5,
        "openai_api_base": "https://custom.openai.azure.com/",
        "openai_organization": "org-custom123"
    }
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance) as mock_constructor:
        await openai_embedder_fixture.setup(config=custom_config)

    mock_constructor.assert_called_once_with(
        api_key="fake_openai_key",
        max_retries=0,
        base_url="https://custom.openai.azure.com/",
        organization="org-custom123"
    )
    assert openai_embedder_fixture._model_name == "custom-model-003"
    assert openai_embedder_fixture._api_key_name == "CUSTOM_OPENAI_KEY"
    assert openai_embedder_fixture._max_retries == 5
    assert openai_embedder_fixture._initial_retry_delay == 0.5
    assert mock_key_provider_with_key.requested_key_names == ["CUSTOM_OPENAI_KEY"]


@pytest.mark.asyncio
async def test_setup_no_openai_library(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", None):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})
    assert openai_embedder_fixture._client is None
    assert "'openai' library not installed" in caplog.text


@pytest.mark.asyncio
async def test_setup_no_key_provider(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    await openai_embedder_fixture.setup(config={})
    assert openai_embedder_fixture._client is None
    assert "KeyProvider instance not provided" in caplog.text


@pytest.mark.asyncio
async def test_setup_api_key_not_found(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_no_key: MockKeyProviderForOpenAI,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_no_key})
    assert openai_embedder_fixture._client is None
    assert f"API key '{openai_embedder_fixture._api_key_name}' not found" in caplog.text


@pytest.mark.asyncio
async def test_setup_openai_client_init_fails(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", side_effect=Exception("Init failed")):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})
    assert openai_embedder_fixture._client is None
    assert "Failed to initialize AsyncOpenAI client: Init failed" in caplog.text


@pytest.mark.asyncio
async def test_embed_success_single_batch(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    chunks_to_embed = [("c1", "text one"), ("c2", "text two")]
    mock_embedding1 = MagicMock()
    mock_embedding1.embedding = [0.1, 0.2]
    mock_embedding2 = MagicMock()
    mock_embedding2.embedding = [0.3, 0.4]
    mock_openai_response = MagicMock()
    mock_openai_response.data = [mock_embedding1, mock_embedding2]
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][0].id == "c1"
    assert results_list[0][1] == [0.1, 0.2]
    assert results_list[1][0].id == "c2"
    assert results_list[1][1] == [0.3, 0.4]
    mock_openai_client_instance.embeddings.create.assert_awaited_once_with(
        input=["text one", "text two"], model=openai_embedder_fixture._model_name
    )

@pytest.mark.asyncio
async def test_embed_empty_chunks_list(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    with patch("genie_tooling.rag.plugins.impl.embedders.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream([])):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 0
    mock_openai_client_instance.embeddings.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_chunks_with_empty_content(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG)
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    chunks_to_embed = [("c1", "  "), ("c2", "text two")]
    mock_embedding2 = MagicMock(); mock_embedding2.embedding = [0.3, 0.4]
    mock_openai_response = MagicMock(); mock_openai_response.data = [mock_embedding2]
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][0].id == "c1"
    assert results_list[0][1] == []
    assert results_list[1][0].id == "c2"
    assert results_list[1][1] == [0.3, 0.4]
    assert "Skipping empty chunk content for ID: c1" in caplog.text
    mock_openai_client_instance.embeddings.create.assert_awaited_once_with(
        input=["text two"], model=openai_embedder_fixture._model_name
    )


@pytest.mark.asyncio
async def test_embed_client_not_initialized(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream([("c1", "text")])): # type: ignore[arg-type]
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 0
    assert "Client not initialized. Cannot generate embeddings." in caplog.text


@pytest.mark.asyncio
async def test_embed_unexpected_error(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key, "max_retries": 0})
    openai_embedder_fixture._client = mock_openai_client_instance

    mock_openai_client_instance.embeddings.create.side_effect = Exception("Network hiccup")

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream([("c1", "text")])):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 1
    assert results_list[0][1] == []
    assert "Unexpected error during OpenAI embedding on attempt 1: Network hiccup" in caplog.text
    assert "Unexpected error: Max retries reached. Batch failed." in caplog.text


@pytest.mark.asyncio
async def test_embed_mismatch_response_length(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    chunks_to_embed = [("c1", "text one"), ("c2", "text two")]
    mock_embedding1 = MagicMock(); mock_embedding1.embedding = [0.1, 0.2]
    mock_openai_response = MagicMock(); mock_openai_response.data = [mock_embedding1]
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    results_list: List[Tuple[Chunk, EmbeddingVector]] = []
    async for chunk_obj, vector in openai_embedder_fixture.embed(make_chunk_stream(chunks_to_embed)):
        results_list.append((chunk_obj, vector))

    assert len(results_list) == 2
    assert results_list[0][1] == []
    assert results_list[1][1] == []
    assert "Mismatch in returned embeddings count. Expected 2, got 1." in caplog.text


@pytest.mark.asyncio
async def test_embed_with_output_dimensions(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    chunks_to_embed = [("c1", "text one")]
    mock_embedding1 = MagicMock(); mock_embedding1.embedding = [0.1, 0.2, 0.3]
    mock_openai_response = MagicMock(); mock_openai_response.data = [mock_embedding1]
    mock_openai_client_instance.embeddings.create.return_value = mock_openai_response

    custom_dimensions = 3
    async for _chunk_obj, _vector in openai_embedder_fixture.embed(
        make_chunk_stream(chunks_to_embed), config={"dimensions": custom_dimensions}
    ):
        pass

    mock_openai_client_instance.embeddings.create.assert_awaited_once_with(
        input=["text one"], model=openai_embedder_fixture._model_name, dimensions=custom_dimensions
    )


@pytest.mark.asyncio
async def test_teardown_closes_client(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
):
    with patch("genie_tooling.rag.plugins.impl.embedders.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    assert openai_embedder_fixture._client is not None
    await openai_embedder_fixture.teardown()
    mock_openai_client_instance.close.assert_awaited_once()
    assert openai_embedder_fixture._client is None


@pytest.mark.asyncio
async def test_teardown_client_already_none(openai_embedder_fixture: OpenAIEmbeddingGenerator):
    openai_embedder_fixture._client = None
    await openai_embedder_fixture.teardown()
    assert openai_embedder_fixture._client is None


@pytest.mark.asyncio
async def test_teardown_client_close_fails(
    openai_embedder_fixture: OpenAIEmbeddingGenerator,
    mock_key_provider_with_key: MockKeyProviderForOpenAI,
    mock_openai_client_instance: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR)
    # Updated patch path for AsyncOpenAI
    with patch("genie_tooling.embedding_generators.impl.openai_embed.AsyncOpenAI", return_value=mock_openai_client_instance):
        await openai_embedder_fixture.setup(config={"key_provider": mock_key_provider_with_key})

    mock_openai_client_instance.close.side_effect = Exception("Failed to close client")
    await openai_embedder_fixture.teardown()
    assert "Error closing OpenAI client: Failed to close client" in caplog.text
    assert openai_embedder_fixture._client is None
