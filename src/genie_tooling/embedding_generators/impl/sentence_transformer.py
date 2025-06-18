# src/genie_tooling/embedding_generators/impl/sentence_transformer.py
import asyncio
import functools  # Import functools
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from genie_tooling.core.types import Chunk, EmbeddingVector

# Updated import path for EmbeddingGeneratorPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin

logger = logging.getLogger(__name__)

# Attempt to import sentence-transformers, make it optional
try:
    import numpy  # To check if the result is a numpy array
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None # type: ignore
    numpy = None # type: ignore

class SentenceTransformerEmbedder(EmbeddingGeneratorPlugin):
    """
    Generates text embeddings using models from the `sentence-transformers` library.
    This executor runs models locally on the host machine.
    """
    plugin_id: str = "sentence_transformer_embedder_v1"
    description: str = "Generates text embeddings using models from the sentence-transformers library (local execution)."

    _model: Optional[Any] = None # Holds the SentenceTransformer model instance
    _model_name: str
    _device: Optional[str] = None # e.g., 'cpu', 'cuda'

    # No __init__ to allow PluginManager simple instantiation. Configuration via setup.

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the SentenceTransformer model.

        This method loads the specified model into memory. Since this can be
        a time-consuming, blocking operation, it is performed in a separate thread.

        Args:
            config: A dictionary containing optional configuration settings:
                - `model_name` (str): The name or path of the sentence-transformer
                  model to load (e.g., from Hugging Face).
                  Defaults to "all-MiniLM-L6-v2".
                - `device` (str, optional): The device to run the model on
                  (e.g., "cuda", "cpu"). If None, `sentence-transformers` will
                  auto-detect the best available device.
        """
        if not SentenceTransformer:
            logger.error("SentenceTransformerEmbedder Error: 'sentence-transformers' library not installed. "
                         "Please install it with: poetry install --extras embeddings")
            return # Cannot proceed without the library
        if not numpy: # Also check for numpy as it's used for type checking encode output
            logger.error("SentenceTransformerEmbedder Error: 'numpy' library not installed. "
                         "Required for handling embedding outputs.")
            return

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

    def _to_list_float(self, embedding_item: Any) -> List[float]:
        """Converts an embedding item (numpy array, torch tensor, or list) to List[float]."""
        if numpy and isinstance(embedding_item, numpy.ndarray):
            return embedding_item.tolist()
        elif isinstance(embedding_item, list):
            return [float(x) for x in embedding_item] # Ensure elements are float
        else:
            logger.warning(f"Unexpected embedding item type: {type(embedding_item)}. Attempting to convert.")
            try:
                return [float(x) for x in list(embedding_item)]
            except Exception:
                logger.error(f"Could not convert embedding item of type {type(embedding_item)} to List[float].")
                return []


    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        """
        Generates embeddings for each chunk using the loaded sentence-transformer model.
        Config options:
            "batch_size": int (default: 32) - How many chunks to embed at once.
            "show_progress_bar": bool (default: False) - For sentence_transformers.encode
        """
        if not self._model:
            logger.error("SentenceTransformerEmbedder Error: Model not loaded. Cannot generate embeddings.")
            if False:
                yield # type: ignore
            return
        if not numpy:
            logger.error("SentenceTransformerEmbedder Error: Numpy not available. Cannot process embeddings.")
            if False:
                yield # type: ignore
            return


        cfg = config or {}
        batch_size_for_embed = int(cfg.get("batch_size", 32))
        show_progress_bar = bool(cfg.get("show_progress_bar", False))

        current_batch_chunks: List[Chunk] = []
        current_batch_texts: List[str] = []

        logger.debug(f"SentenceTransformerEmbedder: Starting embedding process with batch size {batch_size_for_embed}.")
        processed_chunk_count = 0

        async for chunk in chunks:
            current_batch_chunks.append(chunk)
            current_batch_texts.append(chunk.content)

            if len(current_batch_texts) >= batch_size_for_embed:
                try:
                    loop = asyncio.get_running_loop()
                    encode_partial = functools.partial(
                        self._model.encode, # type: ignore
                        sentences=current_batch_texts,
                        show_progress_bar=show_progress_bar
                    )
                    embeddings_output = await loop.run_in_executor(None, encode_partial)

                    for i, chunk_in_batch in enumerate(current_batch_chunks):
                        processed_chunk_count +=1
                        vector_list = self._to_list_float(embeddings_output[i])
                        yield chunk_in_batch, vector_list

                except Exception as e:
                    logger.error(f"SentenceTransformerEmbedder: Error during batch embedding: {e}", exc_info=True)
                    for chunk_in_batch_err in current_batch_chunks:
                        processed_chunk_count +=1
                        yield chunk_in_batch_err, []

                current_batch_chunks = []
                current_batch_texts = []

        if current_batch_texts:
            try:
                loop = asyncio.get_running_loop()
                encode_partial_final = functools.partial(
                    self._model.encode, # type: ignore
                    sentences=current_batch_texts,
                    show_progress_bar=show_progress_bar
                )
                embeddings_output = await loop.run_in_executor(None, encode_partial_final)
                for i, chunk_in_batch in enumerate(current_batch_chunks):
                    processed_chunk_count +=1
                    vector_list = self._to_list_float(embeddings_output[i])
                    yield chunk_in_batch, vector_list
            except Exception as e:
                logger.error(f"SentenceTransformerEmbedder: Error during final batch embedding: {e}", exc_info=True)
                for chunk_in_batch_err in current_batch_chunks:
                    processed_chunk_count +=1
                    yield chunk_in_batch_err, []

        logger.info(f"SentenceTransformerEmbedder: Finished embedding {processed_chunk_count} chunks.")

    async def teardown(self) -> None:
        self._model = None
        logger.debug("SentenceTransformerEmbedder: Model released.")
