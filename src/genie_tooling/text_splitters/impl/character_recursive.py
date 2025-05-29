"""CharacterRecursiveTextSplitter: A common text splitting strategy inspired by LangChain."""
import logging
from typing import Any, AsyncIterable, Callable, Dict, List, Optional, cast

from genie_tooling.core.types import Chunk, Document

# Updated import path for TextSplitterPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin

logger = logging.getLogger(__name__)

class _ConcreteChunk: # Copied from file_system.py loader for this splitter too
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id


class CharacterRecursiveTextSplitter(TextSplitterPlugin):
    plugin_id: str = "character_recursive_text_splitter_v1"
    description: str = "Recursively splits text by a list of character separators, aiming for a target chunk size with overlap."

    def __init__(self):
        # Default length function: number of characters
        self._length_function: Callable[[str], int] = len
        self._default_separators: List[str] = ["\n\n", "\n", ". ", "! ", "? ", " ", ""] # Common separators
        logger.debug("CharacterRecursiveTextSplitter initialized.")

    def _split_text_with_separators(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Core recursive splitting logic.
        Tries to split by separators in order. If a split results in chunks larger than
        chunk_size, it recursively calls itself with the next separator on that oversized chunk.
        """
        final_chunks: List[str] = []

        # Get the first separator to try
        separator_to_use = separators[0]
        remaining_separators = separators[1:]

        # If current separator is empty string, split by character (base case for recursion)
        if not separator_to_use: # Base case: split by character if no effective separator left
            for i in range(0, self._length_function(text), chunk_size - chunk_overlap):
                chunk = text[i : i + chunk_size]
                if self._length_function(chunk.strip()) > 0 :
                    final_chunks.append(chunk)
            return final_chunks

        # Split by the current separator
        # Using re.split to keep the separator is complex with overlap.
        # Let's use a find and slice approach for better control.

        current_pos = 0
        current_doc_frags: List[str] = []
        while current_pos < self._length_function(text):
            idx = text.find(separator_to_use, current_pos)

            if idx == -1: # Separator not found for the rest of the text
                frag = text[current_pos:]
                current_doc_frags.append(frag)
                current_pos = self._length_function(text) # End loop
            else:
                # Add part before separator, then separator itself
                frag = text[current_pos : idx]
                current_doc_frags.append(frag)
                current_doc_frags.append(separator_to_use) # Keep separator for potential re-joining context
                current_pos = idx + self._length_function(separator_to_use)

        # Filter out empty strings that might result from adjacent separators
        current_doc_frags = [s for s in current_doc_frags if s]

        # Merge these fragments respecting chunk_size, or recurse if a fragment is too big
        merged_text = ""
        for frag in current_doc_frags:
            # If adding this fragment would make merged_text too big
            if self._length_function(merged_text + frag) > chunk_size and merged_text:
                # If merged_text itself is too big (e.g. long segment without current separator)
                # and we have more separators to try, recurse on merged_text
                if self._length_function(merged_text) > chunk_size and remaining_separators:
                    final_chunks.extend(self._split_text_with_separators(merged_text, remaining_separators, chunk_size, chunk_overlap))
                elif self._length_function(merged_text.strip()) > 0: # Otherwise, it's a good chunk (or will be handled by base case later)
                    final_chunks.append(merged_text)

                # Start new merged_text. Consider overlap.
                # If previous chunk (final_chunks[-1]) exists and overlap is needed, take its tail.
                if final_chunks and chunk_overlap > 0:
                    overlap_content = final_chunks[-1][-chunk_overlap:]
                    # Avoid re-adding the separator if overlap_content ends with it and frag starts after it.
                    if overlap_content.endswith(separator_to_use) and frag == "": # common if separator was just split
                         merged_text = overlap_content
                    else:
                         merged_text = overlap_content + frag
                else:
                    merged_text = frag
            else:
                merged_text += frag

        # Add any remaining text in merged_text
        if self._length_function(merged_text.strip()) > 0:
            if self._length_function(merged_text) > chunk_size and remaining_separators:
                final_chunks.extend(self._split_text_with_separators(merged_text, remaining_separators, chunk_size, chunk_overlap))
            else: # Add as is, will be handled by final loop if still too big
                final_chunks.append(merged_text)

        # Final pass: if any chunks are still too big (e.g., due to no effective separators or very long words)
        # and all separators have been tried (i.e., we wouldn't recurse further with more specific separators),
        # split them by the base case (character by character).
        truly_final_chunks: List[str] = []
        for f_chunk in final_chunks:
            if self._length_function(f_chunk) > chunk_size: # No 'and not remaining_separators' here; if it's too big, split it.
                truly_final_chunks.extend(self._split_text_with_separators(f_chunk, [""], chunk_size, chunk_overlap)) # Base case call
            elif self._length_function(f_chunk.strip()) > 0:
                truly_final_chunks.append(f_chunk.strip())

        return [c for c in truly_final_chunks if c]

    async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Chunk]:
        """Splits documents into chunks using recursive character splitting."""
        cfg = config or {}
        chunk_size = int(cfg.get("chunk_size", 1000))
        chunk_overlap = int(cfg.get("chunk_overlap", 200))
        separators = cfg.get("separators", self._default_separators)

        if chunk_overlap >= chunk_size:
            logger.warning(f"Chunk overlap ({chunk_overlap}) is greater than or equal to chunk size ({chunk_size}). Setting overlap to 0.")
            chunk_overlap = 0

        logger.debug(f"CharacterRecursiveTextSplitter: Config: size={chunk_size}, overlap={chunk_overlap}, separators={separators}")

        doc_counter = 0
        async for doc in documents:
            doc_counter += 1
            if not doc.content or not doc.content.strip():
                logger.debug(f"Skipping empty document (ID: {doc.id or 'N/A'}).")
                continue

            logger.debug(f"Splitting document ID: {doc.id or f'doc_{doc_counter}'}, original length: {self._length_function(doc.content)}")

            text_chunks = self._split_text_with_separators(doc.content, separators, chunk_size, chunk_overlap)

            for i, chunk_content in enumerate(text_chunks):
                if not chunk_content.strip():
                    continue

                chunk_metadata = doc.metadata.copy() if doc.metadata else {}
                chunk_metadata["chunk_index"] = i
                chunk_metadata["original_doc_id"] = doc.id
                chunk_metadata["split_method"] = self.plugin_id

                chunk_id_base = doc.id if doc.id else f"doc{doc_counter}"
                chunk_id = f"{chunk_id_base}_chunk_{i}"

                logger.debug(f"Yielding chunk {i} from doc {chunk_id_base}, length: {self._length_function(chunk_content)}")
                yield cast(Chunk, _ConcreteChunk(content=chunk_content, metadata=chunk_metadata, id=chunk_id))
        logger.info("CharacterRecursiveTextSplitter: Finished splitting documents.")
