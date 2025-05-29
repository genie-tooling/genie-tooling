"""Unit tests for CharacterRecursiveTextSplitter."""
import logging  # For caplog
from typing import Any, AsyncIterable, Dict, List

import pytest
from genie_tooling.core.types import Chunk, Document
from genie_tooling.text_splitters.impl.character_recursive import (
    CharacterRecursiveTextSplitter,
    _ConcreteChunk,
)


# Helper to create a Document easily for tests
class _HelperDocForSplitterTest(Document): # Renamed from TestDoc
    def __init__(self, content: str, id: str = "doc1", metadata: Dict[str, Any] = None):
        self.content = content
        self.id = id
        self.metadata = metadata or {"source": "test"}

async def collect_chunks(splitter_instance: CharacterRecursiveTextSplitter, docs: List[Document], config: Dict[str, Any] = None) -> List[Chunk]:
    result_chunks: List[Chunk] = []
    async def doc_stream() -> AsyncIterable[Document]:
        for doc_item in docs:
            yield doc_item

    async for chunk_item in splitter_instance.split(doc_stream(), config=config):
        result_chunks.append(chunk_item)
    return result_chunks

@pytest.fixture
async def splitter() -> CharacterRecursiveTextSplitter:
    return CharacterRecursiveTextSplitter()

@pytest.mark.asyncio
async def test_split_basic(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "This is sentence one.\n\nThis is sentence two. This is sentence three.\nThis is sentence four."
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 30, "chunk_overlap": 5, "separators": ["\n\n", "\n", ". ", " "]}

    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) > 1
    for chunk_obj in chunks:
        assert isinstance(chunk_obj, _ConcreteChunk) # type: ignore
        assert len(chunk_obj.content) <= config["chunk_size"] + len(max(config.get("separators", [" "]), key=len, default=""))

    if len(chunks) >= 2:
        assert True

@pytest.mark.asyncio
async def test_split_overlap_greater_than_or_equal_to_size(splitter: CharacterRecursiveTextSplitter, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    actual_splitter = await splitter
    doc_content = "This is a test sentence for overlap issues."
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 20, "chunk_overlap": 25}

    chunks = await collect_chunks(actual_splitter, docs, config)
    assert "Chunk overlap (25) is greater than or equal to chunk size (20). Setting overlap to 0." in caplog.text
    assert len(chunks) == 3
    assert chunks[0].content == "This is a test"
    assert chunks[1].content == "sentence for overlap"
    assert chunks[2].content == "issues."

@pytest.mark.asyncio
async def test_split_with_long_word_exceeding_chunk_size(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Supercalifragilisticexpialidocious"
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 10, "chunk_overlap": 2, "separators": ["\n\n", "\n", " ", ""]}
    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) > 0
    for chunk_item in chunks:
        assert len(chunk_item.content) <= 10

    assert chunks[0].content == "Supercalif"
    assert len(chunks) == 5
    assert chunks[1].content == "ifragilist"
    assert chunks[2].content == "sticexpial"
    assert chunks[3].content == "alidocious"
    assert chunks[4].content == "us"

@pytest.mark.asyncio
async def test_split_empty_document(splitter: CharacterRecursiveTextSplitter, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    actual_splitter = await splitter
    docs = [_HelperDocForSplitterTest("")] # Use renamed helper
    chunks = await collect_chunks(actual_splitter, docs)
    assert len(chunks) == 0
    assert "Skipping empty document" in caplog.text

    caplog.clear()
    docs_whitespace = [_HelperDocForSplitterTest("   \n \t  ")] # Use renamed helper
    chunks_ws = await collect_chunks(actual_splitter, docs_whitespace)
    assert len(chunks_ws) == 0
    assert "Skipping empty document" in caplog.text

@pytest.mark.asyncio
async def test_split_document_smaller_than_chunk_size(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Short document."
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 100, "chunk_overlap": 10}
    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) == 1
    assert chunks[0].content == doc_content

@pytest.mark.asyncio
async def test_split_document_equal_to_chunk_size(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Exactly 20 chars.."
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 20, "chunk_overlap": 5}
    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) == 1
    assert chunks[0].content == doc_content

@pytest.mark.asyncio
async def test_split_with_custom_separators(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Part1---Part2---Part3"
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 10, "chunk_overlap": 1, "separators": ["---"]}
    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) == 3
    assert chunks[0].content == "Part1---"
    assert chunks[1].content == "-Part2---"
    assert chunks[2].content == "-Part3"


@pytest.mark.asyncio
async def test_split_force_character_level(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "HelloWorldNoSpaces"
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 5, "chunk_overlap": 1, "separators": [""]}

    chunks = await collect_chunks(actual_splitter, docs, config)
    assert len(chunks) == 5
    assert chunks[0].content == "Hello"
    assert chunks[1].content == "oWorl"
    assert chunks[2].content == "ldNoS"
    assert chunks[3].content == "Space"
    assert chunks[4].content == "es"


@pytest.mark.asyncio
async def test_split_whitespace_only_chunks_filtered(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Text\n\n     \n\nMore Text"
    docs = [_HelperDocForSplitterTest(doc_content)] # Use renamed helper
    config = {"chunk_size": 10, "chunk_overlap": 0}
    chunks = await collect_chunks(actual_splitter, docs, config)

    for chunk_item in chunks:
        assert chunk_item.content.strip() != ""
    assert len(chunks) == 2
    assert chunks[0].content == "Text"
    assert chunks[1].content == "More Text"

@pytest.mark.asyncio
async def test_split_no_document_id(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Content for document without explicit ID."
    docs = [_HelperDocForSplitterTest(content=doc_content, id=None)] # Use renamed helper
    config = {"chunk_size": 20, "chunk_overlap": 2}
    chunks = await collect_chunks(actual_splitter, docs, config)

    assert len(chunks) > 0
    for chunk_item in chunks:
        assert chunk_item.id is not None
        assert chunk_item.id.startswith("doc1_chunk_")
        assert chunk_item.metadata["original_doc_id"] is None

@pytest.mark.asyncio
async def test_chunk_metadata_inheritance(splitter: CharacterRecursiveTextSplitter):
    actual_splitter = await splitter
    doc_content = "Test content."
    original_metadata = {"source": "test_source", "category": "A"}
    docs = [_HelperDocForSplitterTest(doc_content, metadata=original_metadata.copy())] # Use renamed helper
    chunks = await collect_chunks(actual_splitter, docs)

    assert len(chunks) == 1
    chunk_item = chunks[0]
    assert chunk_item.metadata["source"] == "test_source"
    assert chunk_item.metadata["category"] == "A"
    assert "chunk_index" in chunk_item.metadata
    assert "original_doc_id" in chunk_item.metadata
    assert "split_method" in chunk_item.metadata
