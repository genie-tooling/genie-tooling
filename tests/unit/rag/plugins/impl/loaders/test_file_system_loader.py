### tests/unit/rag/plugins/impl/loaders/test_file_system_loader.py
"""Unit tests for FileSystemLoader."""
import logging
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.core.types import Document
from genie_tooling.document_loaders.impl.file_system import (
    FileSystemLoader,
)

logger = logging.getLogger(__name__) # Module logger for FileSystemLoader
FS_LOADER_LOGGER_NAME = "genie_tooling.document_loaders.impl.file_system"


async def collect_docs_fs(loader_instance: FileSystemLoader, path_uri: str, config: Dict[str, Any] = None) -> List[Document]:
    results: List[Document] = []
    async for doc_item in loader_instance.load(path_uri, config=config):
        results.append(doc_item)
    return results

@pytest.fixture()
async def fs_loader() -> FileSystemLoader:
    return FileSystemLoader()

@pytest.mark.asyncio()
async def test_load_non_existent_directory(fs_loader: FileSystemLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=FS_LOADER_LOGGER_NAME)
    actual_loader = await fs_loader
    non_existent_dir = tmp_path / "i_do_not_exist"
    docs = await collect_docs_fs(actual_loader, str(non_existent_dir))
    assert len(docs) == 0
    assert any(
        f"FileSystemLoader: Source URI '{non_existent_dir}' is not a valid directory." in rec.message and rec.name == FS_LOADER_LOGGER_NAME
        for rec in caplog.records
    )


@pytest.mark.asyncio()
async def test_load_basic_txt_files(fs_loader: FileSystemLoader, tmp_path: Path):
    actual_loader = await fs_loader
    test_dir = tmp_path / "txt_files"
    test_dir.mkdir()
    file1_path = test_dir / "file1.txt"
    file1_path.write_text("Content of file1.")
    file2_path = test_dir / "file2.txt"
    file2_path.write_text("Content of file2.")
    (test_dir / "ignored.md").write_text("Markdown file.")

    docs = await collect_docs_fs(actual_loader, str(test_dir), config={"glob_pattern": "*.txt"})

    assert len(docs) == 2
    doc_ids = sorted([doc.id for doc in docs])
    expected_ids = sorted([str(file1_path.resolve()), str(file2_path.resolve())])
    assert doc_ids == expected_ids

    contents = {doc.id: doc.content for doc in docs}
    assert contents[str(file1_path.resolve())] == "Content of file1."
    assert contents[str(file2_path.resolve())] == "Content of file2."

@pytest.mark.asyncio()
async def test_load_with_glob_pattern_md(fs_loader: FileSystemLoader, tmp_path: Path):
    actual_loader = await fs_loader
    test_dir = tmp_path / "md_files"
    test_dir.mkdir()
    md_file_path = test_dir / "doc.md"
    md_file_path.write_text("# Markdown Content")
    (test_dir / "another.txt").write_text("Text file.")

    docs = await collect_docs_fs(actual_loader, str(test_dir), config={"glob_pattern": "*.md"})
    assert len(docs) == 1
    assert docs[0].id == str(md_file_path.resolve())
    assert docs[0].content == "# Markdown Content"

@pytest.mark.asyncio()
async def test_load_recursive_glob(fs_loader: FileSystemLoader, tmp_path: Path):
    actual_loader = await fs_loader
    base_dir = tmp_path / "recursive_test"
    base_dir.mkdir()
    (base_dir / "root.log").write_text("Root log.")
    sub_dir = base_dir / "sub"
    sub_dir.mkdir()
    (sub_dir / "sub.log").write_text("Sub log.")

    docs = await collect_docs_fs(actual_loader, str(base_dir), config={"glob_pattern": "**/*.log"})
    assert len(docs) == 2
    doc_names = sorted([Path(doc.id).name for doc in docs])
    assert doc_names == ["root.log", "sub.log"]

@pytest.mark.asyncio()
async def test_load_with_custom_encoding(fs_loader: FileSystemLoader, tmp_path: Path):
    actual_loader = await fs_loader
    test_dir = tmp_path / "encoding_test"
    test_dir.mkdir()
    content_utf16 = "你好世界"
    file_path_utf16 = test_dir / "utf16_file.txt"
    file_path_utf16.write_text(content_utf16, encoding="utf-16")

    docs = await collect_docs_fs(actual_loader, str(test_dir), config={"encoding": "utf-16", "glob_pattern": "*.txt"})

    assert len(docs) == 1
    assert docs[0].content == content_utf16

@pytest.mark.asyncio()
async def test_load_skip_large_file(fs_loader: FileSystemLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=FS_LOADER_LOGGER_NAME) # Capture WARNING for this logger
    actual_loader = await fs_loader
    test_dir = tmp_path / "large_file_test"
    test_dir.mkdir(exist_ok=True)

    max_size_mb_for_test = 0.000001

    small_file_to_load = test_dir / "small_ok.txt"
    small_file_to_load.write_text("a")

    large_file_to_skip = test_dir / "large_skip.txt"
    large_file_to_skip.write_text("tiny") # 4 bytes

    docs = await collect_docs_fs(actual_loader, str(test_dir), config={"max_file_size_mb": max_size_mb_for_test, "glob_pattern": "*.txt"})

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}. Docs: {[d.id for d in docs]}"
    assert docs[0].id == str(small_file_to_load.resolve())
    assert any(
        f"Skipping file {large_file_to_skip.resolve()!s} due to size" in rec.message and rec.name == FS_LOADER_LOGGER_NAME
        for rec in caplog.records
    )


@pytest.mark.asyncio()
async def test_load_skip_empty_file(fs_loader: FileSystemLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=FS_LOADER_LOGGER_NAME) # Capture DEBUG for this logger
    actual_loader = await fs_loader
    test_dir = tmp_path / "empty_file_test"
    test_dir.mkdir()
    empty_file = test_dir / "empty.txt"
    empty_file.touch()
    non_empty_file = test_dir / "content.txt"
    non_empty_file.write_text("has content")

    docs = await collect_docs_fs(actual_loader, str(test_dir))
    assert len(docs) == 1
    assert docs[0].id == str(non_empty_file.resolve())
    assert any(
        f"Skipping empty file {empty_file.resolve()!s}" in rec.message and rec.name == FS_LOADER_LOGGER_NAME
        for rec in caplog.records
    )


@pytest.mark.asyncio()
async def test_load_handles_file_read_error(fs_loader: FileSystemLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=FS_LOADER_LOGGER_NAME) # Capture ERROR for this logger

    actual_loader = await fs_loader
    test_dir: Path = tmp_path / "read_error_test_fs_debug"
    test_dir.mkdir(exist_ok=True)

    good_file_content = "Good content file data for debug"
    readable_file_path_obj = (test_dir / "good_debug.txt").resolve()
    readable_file_path_obj.write_text(good_file_content)

    error_file_path_obj = (test_dir / "bad_debug.txt").resolve()
    error_file_path_obj.write_text("Error content for debug")


    async_file_mock = AsyncMock()
    async_file_mock.read = AsyncMock(return_value=good_file_content)
    async_cm_mock = AsyncMock()
    async_cm_mock.__aenter__.return_value = async_file_mock
    async_cm_mock.__aexit__ = AsyncMock(return_value=None)

    def simple_aio_open_side_effect(p: Path, *args, **kwargs):
        if p.resolve() == error_file_path_obj:
            raise OSError(f"Mocked OSError for {p.resolve()}")
        if p.resolve() == readable_file_path_obj:
            return async_cm_mock
        raise ValueError(f"Mocked aiofiles.open received unexpected path {p.resolve()}")

    with patch("genie_tooling.document_loaders.impl.file_system.aiofiles.open",
               side_effect=simple_aio_open_side_effect) as mock_open_patch:
        docs = await collect_docs_fs(actual_loader, str(test_dir), config={"glob_pattern": "*.txt"})


    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}. Caplog: {caplog.text}"
    if docs:
        assert docs[0].id == str(readable_file_path_obj)
        assert docs[0].content == good_file_content

    mock_open_patch.assert_any_call(readable_file_path_obj, mode="r", encoding="utf-8", errors="replace")
    mock_open_patch.assert_any_call(error_file_path_obj, mode="r", encoding="utf-8", errors="replace")

    assert any(
        f"Error loading or reading file {error_file_path_obj!s}" in rec.message and rec.name == FS_LOADER_LOGGER_NAME
        for rec in caplog.records
    ), f"Expected error log for bad_debug.txt not found. Caplog: {caplog.text}"


@pytest.mark.asyncio()
async def test_load_skips_non_files_in_glob(fs_loader: FileSystemLoader, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=FS_LOADER_LOGGER_NAME) # Capture DEBUG for this logger
    actual_loader = await fs_loader
    test_dir = tmp_path / "skip_non_files"
    test_dir.mkdir()
    real_file_path = test_dir / "real_file.txt"
    real_file_path.write_text("Real text")
    dir_as_file_path = test_dir / "dir_named_like_file.txt"
    dir_as_file_path.mkdir()

    docs = await collect_docs_fs(actual_loader, str(test_dir), config={"glob_pattern": "*.txt"})

    assert len(docs) == 1
    assert docs[0].id == str(real_file_path.resolve())
    assert any(
        f"Path {dir_as_file_path.resolve()!s} matched glob but is not a file. Skipping." in rec.message and rec.name == FS_LOADER_LOGGER_NAME
        for rec in caplog.records
    )
