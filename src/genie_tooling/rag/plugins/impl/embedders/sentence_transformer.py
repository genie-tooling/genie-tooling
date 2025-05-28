"""SentenceTransformerEmbedder: Generates embeddings using sentence-transformers library."""
import asyncio
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.rag.plugins.abc import EmbeddingGeneratorPlugin

logger = logging.getLogger(__name__)

# Attempt to import sentence-transformers, make it optional
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None # type: ignore

class SentenceTransformerEmbedder(EmbeddingGeneratorPlugin):
    plugin_id: str = "sentence_transformer_embedder_v1"
    description: str = "Generates text embeddings using models from the sentence-transformers library (local execution)."

    _model: Optional[Any] = None # Holds the SentenceTransformer model instance
    _model_name: str
    _device: Optional[str] = None # e.g., 'cpu', 'cuda'

    # No __init__ to allow PluginManager simple instantiation. Configuration via setup.

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the SentenceTransformer model.
        Config options:
            "model_name": str (default: "all-MiniLM-L6-v2")
            "device": Optional[str] (e.g., "cuda", "cpu", None for auto)
        """
        if not SentenceTransformer:
            logger.error("SentenceTransformerEmbedder Error: 'sentence-transformers' library not installed. "
                         "Please install it with: poetry install --extras embeddings")
            return # Cannot proceed without the library

        cfg = config or {}
        self._model_name = cfg.get("model_name", "all-MiniLM-L6-v2")
        self._device = cfg.get("device") # Let SentenceTransformer pick if None

        logger.info(f"SentenceTransformerEmbedder: Initializing model '{self._model_name}' on device '{self._device or 'auto'}'. This may take time...")

        # Model loading is CPU/IO bound, run in executor to avoid blocking event loop
        try:
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                None, # Default thread pool executor
                SentenceTransformer, # The callable
                self._model_name,    # Arguments to SentenceTransformer constructor
                self._device
            )
            logger.info(f"SentenceTransformerEmbedder: Model '{self._model_name}' loaded successfully.")
        except Exception as e:
            self._model = None # Ensure model is None if loading failed
            logger.error(f"SentenceTransformerEmbedder Error: Failed to load model '{self._model_name}': {e}", exc_info=True)
            # Consider re-raising or setting an internal error state

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        """
        Generates embeddings for each chunk using the loaded sentence-transformer model.
        Config options:
            "batch_size": int (default: 32) - How many chunks to embed at once.
            "show_progress_bar": bool (default: False) - For sentence_transformers.encode
        """
        if not self._model:
            logger.error("SentenceTransformerEmbedder Error: Model not loaded. Cannot generate embeddings.")
            if False: yield # type: ignore # Make it an async generator
            return

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 32))
        show_progress_bar = bool(cfg.get("show_progress_bar", False)) # Progress bar for `encode`

        # Normalize newlines for some models, though many handle them fine.
        # This is a sentence-level embedder, so paragraphs/sentences are typical inputs.
        # For simplicity, we'll pass content as is. If specific preprocessing is needed, it
        # should be part of the chunking or a dedicated pre-embedding step.

        current_batch_chunks: List[Chunk] = []
        current_batch_texts: List[str] = []

        logger.debug(f"SentenceTransformerEmbedder: Starting embedding process with batch size {batch_size}.")
        processed_chunk_count = 0

        async for chunk in chunks:
            current_batch_chunks.append(chunk)
            current_batch_texts.append(chunk.content) # Assuming chunk.content is the text to embed

            if len(current_batch_texts) >= batch_size:
                try:
                    loop = asyncio.get_running_loop()
                    # The `encode` method is CPU-bound.
                    # `convert_to_tensor=False`, `normalize_embeddings=False` are defaults usually.
                    # Some models benefit from normalize_embeddings=True for cosine similarity.
                    # For now, use defaults.
                    embeddings_np = await loop.run_in_executor(
                        None,
                        self._model.encode, # type: ignore
                        current_batch_texts,
                        show_progress_bar=show_progress_bar
                        # batch_size parameter within encode is for its internal batching if sentences list is huge
                    )

                    for i, chunk_in_batch in enumerate(current_batch_chunks):
                        processed_chunk_count +=1
                        yield chunk_in_batch, cast(List[float], embeddings_np[i].tolist()) # Convert numpy array to list of floats

                except Exception as e:
                    logger.error(f"SentenceTransformerEmbedder: Error during batch embedding: {e}", exc_info=True)
                    # Optionally, yield chunks with an error indicator or skip them
                    for chunk_in_batch_err in current_batch_chunks:
                        processed_chunk_count +=1
                        # Yield with empty embedding list to signal an error for this chunk
                        yield chunk_in_batch_err, []

                current_batch_chunks = []
                current_batch_texts = []

        # Process any remaining chunks in the last batch
        if current_batch_texts:
            try:
                loop = asyncio.get_running_loop()
                embeddings_np = await loop.run_in_executor(
                    None, self._model.encode, current_batch_texts, show_progress_bar=show_progress_bar # type: ignore
                )
                for i, chunk_in_batch in enumerate(current_batch_chunks):
                    processed_chunk_count +=1
                    yield chunk_in_batch, cast(List[float], embeddings_np[i].tolist())
            except Exception as e:
                logger.error(f"SentenceTransformerEmbedder: Error during final batch embedding: {e}", exc_info=True)
                for chunk_in_batch_err in current_batch_chunks:
                    processed_chunk_count +=1
                    yield chunk_in_batch_err, []

        logger.info(f"SentenceTransformerEmbedder: Finished embedding {processed_chunk_count} chunks.")

    async def teardown(self) -> None:
        self._model = None # Release model reference to allow garbage collection
        logger.debug("SentenceTransformerEmbedder: Model released.")
