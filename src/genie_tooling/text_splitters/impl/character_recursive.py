import logging
from typing import Any, AsyncIterable, Callable, Dict, List, Optional, cast

from genie_tooling.core.types import Chunk, Document
from genie_tooling.text_splitters.abc import TextSplitterPlugin

logger = logging.getLogger(__name__)

class _ConcreteChunk:
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id

class CharacterRecursiveTextSplitter(TextSplitterPlugin):
    plugin_id: str = "character_recursive_text_splitter_v1"
    description: str = "Recursively splits text by a list of character separators, aiming for a target chunk size with overlap."

    def __init__(self):
        self._length_function: Callable[[str], int] = len
        self._default_separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""]
        logger.debug("CharacterRecursiveTextSplitter initialized.")

    def _recursive_split_internal(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        final_chunks: List[str] = []
        if not text.strip(): return final_chunks

        current_separator = ""
        if separators: current_separator = separators[0]
        remaining_separators = separators[1:] if separators else []

        if not current_separator: # Base case: split by character
            if not text: return []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if chunk.strip(): final_chunks.append(chunk)
            return final_chunks

        # Try splitting with the current separator
        try:
            # Use a regex that keeps the delimiter for context, but handle empty strings from re.split
            # A positive lookbehind for the separator might be better, or manual find.
            # For simplicity, let's use find and slice.
            splits_with_sep: List[str] = []
            start_idx = 0
            while start_idx < len(text):
                idx = text.find(current_separator, start_idx)
                if idx == -1:
                    splits_with_sep.append(text[start_idx:])
                    break
                splits_with_sep.append(text[start_idx : idx]) # Part before separator
                splits_with_sep.append(current_separator)    # Separator itself
                start_idx = idx + len(current_separator)

            # Filter out any purely empty strings that might arise
            splits_with_sep = [s for s in splits_with_sep if s]

        except Exception: # Fallback if re.split fails (e.g. on complex regex if used)
            splits_with_sep = [text] # Treat as one block

        # Process these splits
        current_merged_text = ""
        for part in splits_with_sep:
            # If adding this part makes the current merged text too long
            if len(current_merged_text + part) > chunk_size and current_merged_text:
                # If current_merged_text itself is too long, recurse on it with remaining separators
                if len(current_merged_text) > chunk_size and remaining_separators:
                    final_chunks.extend(self._recursive_split_internal(current_merged_text, remaining_separators, chunk_size, chunk_overlap))
                elif current_merged_text.strip(): # Otherwise, it's a valid chunk (or will be handled by base case if still too long)
                    final_chunks.append(current_merged_text)

                # Start new merged text. Consider overlap from the last *added* chunk.
                overlap_content = ""
                if final_chunks and chunk_overlap > 0:
                    max_possible_overlap = min(chunk_overlap, len(final_chunks[-1]))
                    overlap_content = final_chunks[-1][-max_possible_overlap:]
                current_merged_text = overlap_content + part
            else:
                current_merged_text += part

        # Add any remaining text in current_merged_text
        if current_merged_text.strip():
            if len(current_merged_text) > chunk_size and remaining_separators:
                final_chunks.extend(self._recursive_split_internal(current_merged_text, remaining_separators, chunk_size, chunk_overlap))
            else: # Add as is. If still too long, the next recursive call (or base case) will handle it.
                final_chunks.append(current_merged_text)

        # Final pass for any chunks that are still too large (e.g., long word, or no effective separators left)
        # This ensures the base case (character split) is applied if needed.
        truly_final_chunks: List[str] = []
        for f_chunk in final_chunks:
            if len(f_chunk) > chunk_size: # If still too big, force character split
                truly_final_chunks.extend(self._recursive_split_internal(f_chunk, [""], chunk_size, chunk_overlap))
            elif f_chunk.strip():
                truly_final_chunks.append(f_chunk.strip())

        return [c for c in truly_final_chunks if c.strip()]


    async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Chunk]:
        cfg = config or {}
        chunk_size = int(cfg.get("chunk_size", 1000))
        chunk_overlap = int(cfg.get("chunk_overlap", 200))
        separators = cfg.get("separators", self._default_separators)

        if chunk_overlap >= chunk_size:
            logger.warning(f"{self.plugin_id}: Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to 0.")
            chunk_overlap = 0

        doc_counter = 0
        async for doc in documents:
            doc_counter += 1
            if not doc.content or not doc.content.strip():
                logger.debug(f"Skipping empty document (ID: {doc.id or 'N/A'}).")
                continue

            text_chunks = self._recursive_split_internal(doc.content, separators, chunk_size, chunk_overlap)

            for i, chunk_content in enumerate(text_chunks):
                if not chunk_content.strip(): continue

                chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                chunk_metadata.update({
                    "chunk_index": i, "original_doc_id": doc.id, "split_method": self.plugin_id
                })
                chunk_id_base = doc.id if doc.id else f"doc{doc_counter}"
                chunk_id = f"{chunk_id_base}_chunk_{i}"
                yield cast(Chunk, _ConcreteChunk(content=chunk_content, metadata=chunk_metadata, id=chunk_id))
        logger.info(f"{self.plugin_id}: Finished splitting documents.")
