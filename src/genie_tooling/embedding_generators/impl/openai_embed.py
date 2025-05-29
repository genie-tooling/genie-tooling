### src/genie_tooling/rag/plugins/impl/embedders/openai_embed.py
import asyncio  # For potential batch delays or retries
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from genie_tooling.core.types import Chunk, EmbeddingVector

# Updated import path for EmbeddingGeneratorPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.security.key_provider import KeyProvider  # Crucial for API keys

logger = logging.getLogger(__name__)

# Attempt to import openai, make it optional
try:
    from openai import APIError, AsyncOpenAI, RateLimitError  # type: ignore
except ImportError:
    AsyncOpenAI = None # type: ignore
    APIError = Exception # type: ignore # Fallback
    RateLimitError = Exception # type: ignore # Fallback

class OpenAIEmbeddingGenerator(EmbeddingGeneratorPlugin):
    plugin_id: str = "openai_embedding_generator_v1"
    description: str = "Generates text embeddings using OpenAI's API (e.g., text-embedding-ada-002)."

    _client: Optional[Any] = None # AsyncOpenAI client instance
    _model_name: str
    _api_key_name: str = "OPENAI_API_KEY" # Standard name for the key in KeyProvider
    _max_retries: int = 3
    _initial_retry_delay: float = 1.0 # seconds

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not AsyncOpenAI:
            logger.error("OpenAIEmbeddingGenerator Error: 'openai' library not installed. "
                         "Please install it with: poetry install --extras openai_services")
            return

        cfg = config or {}
        self._model_name = cfg.get("model_name", "text-embedding-ada-002")
        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._max_retries = int(cfg.get("max_retries", self._max_retries))
        self._initial_retry_delay = float(cfg.get("initial_retry_delay", self._initial_retry_delay))

        key_provider = cfg.get("key_provider")
        if not key_provider or not isinstance(key_provider, KeyProvider):
            logger.error("OpenAIEmbeddingGenerator Error: KeyProvider instance not provided in 'key_provider' config field.")
            return

        api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"OpenAIEmbeddingGenerator Error: API key '{self._api_key_name}' not found via KeyProvider.")
            return

        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                max_retries=0,
                base_url=cfg.get("openai_api_base"),
                organization=cfg.get("openai_organization")
            )
            logger.info(f"OpenAIEmbeddingGenerator: AsyncOpenAI client initialized for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"OpenAIEmbeddingGenerator: Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            self._client = None

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        if not self._client:
            logger.error("OpenAIEmbeddingGenerator Error: Client not initialized. Cannot generate embeddings.")
            if False: yield # type: ignore
            return

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 50))
        output_dimensions = cfg.get("dimensions")

        current_batch_chunks: List[Chunk] = []
        current_batch_texts: List[str] = []

        logger.debug(f"OpenAIEmbeddingGenerator: Starting embedding process with batch_size={batch_size}, model='{self._model_name}'.")
        processed_chunk_count = 0

        async for chunk in chunks:
            processed_text = chunk.content.replace("\n", " ").strip()
            if not processed_text:
                logger.debug(f"Skipping empty chunk content for ID: {chunk.id or 'N/A'}")
                processed_chunk_count +=1
                yield chunk, []
                continue

            current_batch_chunks.append(chunk)
            current_batch_texts.append(processed_text)

            if len(current_batch_texts) >= batch_size:
                # Corrected: Only one call to _process_batch_with_retries here
                batch_embeddings = await self._process_batch_with_retries(current_batch_chunks, current_batch_texts, output_dimensions)
                for i, chunk_in_batch in enumerate(current_batch_chunks):
                    processed_chunk_count += 1
                    yield chunk_in_batch, batch_embeddings[i] if i < len(batch_embeddings) else []
                current_batch_chunks = []
                current_batch_texts = []

        if current_batch_texts:
            batch_embeddings = await self._process_batch_with_retries(current_batch_chunks, current_batch_texts, output_dimensions)
            for i, chunk_in_batch in enumerate(current_batch_chunks):
                processed_chunk_count += 1
                yield chunk_in_batch, batch_embeddings[i] if i < len(batch_embeddings) else []

        logger.info(f"OpenAIEmbeddingGenerator: Finished embedding {processed_chunk_count} chunks.")

    async def _process_batch_with_retries(
        self,
        batch_chunks: List[Chunk],
        batch_texts: List[str],
        output_dimensions: Optional[int]
    ) -> List[EmbeddingVector]:
        if not self._client: return [[] for _ in batch_texts]

        embeddings_params: Dict[str, Any] = {"input": batch_texts, "model": self._model_name}
        if output_dimensions:
            embeddings_params["dimensions"] = output_dimensions

        current_retry_delay = self._initial_retry_delay
        batch_results: List[EmbeddingVector] = [[] for _ in batch_texts]

        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(f"OpenAIEmbeddingGenerator: Attempt {attempt + 1}/{self._max_retries + 1} for batch of {len(batch_texts)} texts.")
                response = await self._client.embeddings.create(**embeddings_params)

                embeddings_data = response.data
                if len(embeddings_data) == len(batch_texts):
                    for i in range(len(batch_texts)):
                        batch_results[i] = embeddings_data[i].embedding
                    logger.debug(f"OpenAIEmbeddingGenerator: Successfully embedded batch of {len(batch_texts)} texts.")
                    return batch_results
                else:
                    logger.error(f"OpenAIEmbeddingGenerator: Mismatch in returned embeddings count. Expected {len(batch_texts)}, got {len(embeddings_data)}.")
                    return batch_results

            # Corrected order of exception handling
            except RateLimitError as e: # Specific to rate limits - CATCH FIRST
                logger.warning(f"OpenAI Rate Limit Error on attempt {attempt + 1}: {e.message}", exc_info=False)
                if attempt >= self._max_retries:
                    logger.error("OpenAI Rate Limit Error: Max retries reached. Batch failed.")
                    return batch_results

                retry_after_seconds = self._get_retry_after(e) or current_retry_delay
                logger.info(f"Rate limited. Retrying in {retry_after_seconds:.2f} seconds...")
                await asyncio.sleep(retry_after_seconds)
                current_retry_delay = max(current_retry_delay * 1.5, retry_after_seconds)

            except APIError as e: # General API errors (non-rate limit)
                # Use getattr for safer access to status_code
                status_code = getattr(e, "status_code", None)
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {status_code or 'N/A'} - {e.message}", exc_info=True)
                if status_code == 401:
                    logger.error("OpenAI API Key is invalid or expired. Halting retries for this batch.")
                    return batch_results
                if status_code == 400:
                    logger.error(f"OpenAI Bad Request (400): {e.message}. Check input data. Halting retries for this batch.")
                    return batch_results

                if attempt >= self._max_retries:
                    logger.error("OpenAI API error: Max retries reached. Batch failed.")
                    return batch_results

                logger.info(f"Retrying in {current_retry_delay:.2f} seconds...")
                await asyncio.sleep(current_retry_delay)
                current_retry_delay *= 2

            except Exception as e: # Other unexpected errors (network, etc.)
                logger.error(f"Unexpected error during OpenAI embedding on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt >= self._max_retries:
                    logger.error("Unexpected error: Max retries reached. Batch failed.")
                    return batch_results

                logger.info(f"Retrying in {current_retry_delay:.2f} seconds...")
                await asyncio.sleep(current_retry_delay)
                current_retry_delay *= 2

        return batch_results

    def _get_retry_after(self, error: RateLimitError) -> Optional[float]:
        # This implementation assumes error.response.headers exists and is a dict-like object
        # which is typical for httpx-based libraries like openai v1.0+.
        if hasattr(error, "response") and error.response and hasattr(error.response, "headers"):
            headers = error.response.headers
            if "retry-after" in headers:
                try:
                    return float(headers["retry-after"])
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse 'retry-after' header value: {headers['retry-after']}")
            # Fallback for x-ratelimit-reset-requests (less common but possible)
            # Example: "60ms", "1s". This is a simplified parser.
            reset_val = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
            if reset_val and isinstance(reset_val, str):
                try:
                    if reset_val.endswith("ms"):
                        return float(reset_val[:-2]) / 1000.0
                    if reset_val.endswith("s"):
                        return float(reset_val[:-1])
                except ValueError:
                    logger.warning(f"Could not parse rate limit reset header value: {reset_val}")
        return None


    async def teardown(self) -> None:
        if self._client:
            try:
                await self._client.close()
                logger.debug("OpenAIEmbeddingGenerator: AsyncOpenAI client closed.")
            except Exception as e:
                logger.error(f"Error closing OpenAI client: {e}", exc_info=True)
            finally: # Ensure client is None even if close fails
                self._client = None
