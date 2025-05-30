import asyncio
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

try:
    from openai import APIError, AsyncOpenAI, RateLimitError
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from openai.types.embedding import Embedding as OpenAIEmbedding
except ImportError:
    AsyncOpenAI = None
    APIError = Exception
    RateLimitError = Exception
    CreateEmbeddingResponse = Any
    OpenAIEmbedding = Any


class OpenAIEmbeddingGenerator(EmbeddingGeneratorPlugin):
    plugin_id: str = "openai_embedding_generator_v1"
    description: str = "Generates text embeddings using OpenAI's API (e.g., text-embedding-ada-002)."

    _client: Optional[AsyncOpenAI] = None
    _model_name: str
    _api_key_name: str = "OPENAI_API_KEY"
    _max_retries: int = 3
    _initial_retry_delay: float = 1.0

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not AsyncOpenAI:
            logger.error(f"{self.plugin_id} Error: 'openai' library not installed. ")
            return

        cfg = config or {}
        self._model_name = cfg.get("model_name", "text-embedding-ada-002")
        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._max_retries = int(cfg.get("max_retries", self._max_retries))
        self._initial_retry_delay = float(cfg.get("initial_retry_delay", self._initial_retry_delay))

        key_provider = cfg.get("key_provider")
        if not key_provider or not isinstance(key_provider, KeyProvider):
            logger.error(f"{self.plugin_id} Error: KeyProvider instance not provided.")
            return

        api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"{self.plugin_id} Error: API key '{self._api_key_name}' not found.")
            return

        try:
            client_timeout = cfg.get("request_timeout_seconds", 30.0)
            self._client = AsyncOpenAI(
                api_key=api_key, max_retries=0,
                base_url=cfg.get("openai_api_base"),
                organization=cfg.get("openai_organization"),
                timeout=client_timeout
            )
            logger.info(f"{self.plugin_id}: Client initialized for model '{self._model_name}' with timeout {client_timeout}s.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize client: {e}", exc_info=True)
            self._client = None

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        if not self._client:
            logger.error(f"{self.plugin_id} Error: Client not initialized. Cannot generate embeddings.")
            async for chunk_item_err in chunks:
                yield chunk_item_err, []
            return

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 2048))
        output_dimensions = cfg.get("dimensions")

        current_batch_chunks: List[Chunk] = []
        current_batch_texts: List[str] = []

        async for chunk in chunks:
            processed_text = chunk.content.replace("\n", " ").strip()
            if not processed_text:
                logger.debug(f"Skipping empty chunk content for ID: {chunk.id or 'N/A'}")
                yield chunk, []
                continue

            current_batch_chunks.append(chunk)
            current_batch_texts.append(processed_text)

            if len(current_batch_texts) >= batch_size:
                # P1 VERIFICATION POINT: Test batch alignment robustness here.
                # Ensure that if OpenAI returns fewer embeddings than texts sent,
                # the successful ones are correctly mapped and missing ones are handled (e.g., empty list).
                batch_embeddings = await self._process_batch_with_retries(current_batch_texts, output_dimensions)
                for i, chunk_in_batch in enumerate(current_batch_chunks):
                    yield chunk_in_batch, batch_embeddings[i] if i < len(batch_embeddings) else []
                current_batch_chunks, current_batch_texts = [], []

        if current_batch_texts:
            batch_embeddings = await self._process_batch_with_retries(current_batch_texts, output_dimensions)
            for i, chunk_in_batch in enumerate(current_batch_chunks):
                yield chunk_in_batch, batch_embeddings[i] if i < len(batch_embeddings) else []

        logger.info(f"{self.plugin_id}: Finished embedding process.")


    async def _process_batch_with_retries(
        self, batch_texts: List[str], output_dimensions: Optional[int]
    ) -> List[EmbeddingVector]:
        if not self._client or not batch_texts: return [[] for _ in batch_texts]

        embeddings_params: Dict[str, Any] = {"input": batch_texts, "model": self._model_name}
        if output_dimensions: embeddings_params["dimensions"] = output_dimensions

        current_retry_delay = self._initial_retry_delay
        batch_results: List[EmbeddingVector] = [[] for _ in range(len(batch_texts))]

        for attempt in range(self._max_retries + 1):
            try:
                response: CreateEmbeddingResponse = await self._client.embeddings.create(**embeddings_params)

                for embedding_data_item in response.data:
                    original_index = embedding_data_item.index
                    if 0 <= original_index < len(batch_results):
                        batch_results[original_index] = embedding_data_item.embedding
                    else:
                        logger.error(f"{self.plugin_id}: Embedding index {original_index} out of bounds for batch size {len(batch_texts)}.")

                if len(response.data) != len(batch_texts):
                    logger.warning(f"{self.plugin_id}: Mismatch in returned embeddings. Expected {len(batch_texts)}, got {len(response.data)}. Some embeddings might be missing (empty list).")

                return batch_results

            except RateLimitError as e:
                logger.warning(f"{self.plugin_id} Rate Limit Error (attempt {attempt + 1}): {getattr(e, 'message', str(e))}")
                if attempt >= self._max_retries: break
                # P1 VERIFICATION POINT: Test _get_retry_after with actual/mocked OpenAI headers.
                retry_after = self._get_retry_after(e) or current_retry_delay
                await asyncio.sleep(retry_after); current_retry_delay = max(current_retry_delay * 1.5, retry_after)
            except APIError as e:
                status = getattr(e, "status_code", "N/A")
                logger.error(f"{self.plugin_id} API Error (attempt {attempt + 1}): {status} - {getattr(e, 'message', str(e))}")
                if status == 401 or status == 400: break
                if attempt >= self._max_retries: break
                await asyncio.sleep(current_retry_delay); current_retry_delay *= 2
            except Exception as e:
                logger.error(f"{self.plugin_id} Unexpected error (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt >= self._max_retries: break
                await asyncio.sleep(current_retry_delay); current_retry_delay *= 2

        logger.error(f"{self.plugin_id}: Batch failed after {self._max_retries + 1} attempts. Returning list of empty embeddings for this batch.")
        return batch_results

    def _get_retry_after(self, error: RateLimitError) -> Optional[float]:
        # This logic seems plausible. Needs testing with real headers.
        if hasattr(error, "response") and error.response and hasattr(error.response, "headers"):
            headers = error.response.headers
            if "retry-after" in headers:
                try: return float(headers["retry-after"])
                except (ValueError, TypeError): pass
            reset_val = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
            if reset_val and isinstance(reset_val, str):
                try:
                    if reset_val.endswith("ms"): return float(reset_val[:-2]) / 1000.0
                    if reset_val.endswith("s"): return float(reset_val[:-1])
                except ValueError: pass
        return None

    async def teardown(self) -> None:
        if self._client:
            try: await self._client.close()
            except Exception as e: logger.error(f"Error closing OpenAI client: {e}", exc_info=True)
            finally: self._client = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
