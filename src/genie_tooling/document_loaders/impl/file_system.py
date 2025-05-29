"""FileSystemLoader: Loads documents from local text files."""
import logging
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Optional, cast

import aiofiles  # Requires: poetry add aiofiles

from genie_tooling.core.types import Document

# Updated import path for DocumentLoaderPlugin
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin

logger = logging.getLogger(__name__)

# A concrete Document class adhering to the Document protocol for internal use by loaders
class _ConcreteDocument:
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id

class FileSystemLoader(DocumentLoaderPlugin):
    plugin_id: str = "file_system_loader_v1"
    description: str = "Loads documents from text-based files in a local directory (e.g., .txt, .md)."

    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        """
        Loads files matching a glob pattern from a directory specified by source_uri.
        Config options:
            "glob_pattern": str (e.g., "*.txt", "**/*.md", default: "*.txt")
            "encoding": str (e.g., "utf-8", default: "utf-8")
            "max_file_size_mb": float (default: 10.0, skip files larger than this)
        """
        cfg = config or {}
        directory = Path(source_uri)
        glob_pattern = cfg.get("glob_pattern", "*.txt")
        encoding = cfg.get("encoding", "utf-8")
        max_file_size_mb_val = float(cfg.get("max_file_size_mb", 10.0))
        max_file_size_bytes = int(max_file_size_mb_val * 1024 * 1024)

        # logger.debug(f"\n[LOADER DEBUG] Source URI: {source_uri}, Directory Path: {directory}, Glob: {glob_pattern}") # TEMP DEBUG

        if not directory.is_dir():
            logger.error(f"FileSystemLoader: Source URI '{source_uri}' is not a valid directory.")
            # logger.debug(f"[LOADER DEBUG] Directory {directory} is not valid.") # TEMP DEBUG
            if False:
                yield
                return

        logger.info(f"FileSystemLoader: Scanning '{directory}' for files matching '{glob_pattern}'.")
        file_count = 0
        loaded_count = 0

        logger.debug(f"[LOADER DEBUG] About to rglob in {directory} with pattern {glob_pattern}") # TEMP DEBUG
        found_files_for_debug = list(directory.rglob(glob_pattern))
        logger.debug(f"[LOADER DEBUG] rglob found: {[str(f.resolve()) for f in found_files_for_debug]}") # TEMP DEBUG

        for file_path in directory.rglob(glob_pattern): # Use original rglob here
            logger.debug(f"[LOADER DEBUG] Processing file_path from rglob: {file_path.resolve()}") # TEMP DEBUG
            file_count += 1
            if file_path.is_file():
                logger.debug(f"[LOADER DEBUG] {file_path.resolve()} is a file.") # TEMP DEBUG
                try:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        logger.debug(f"FileSystemLoader: Skipping empty file {file_path}.")
                        logger.debug(f"[LOADER DEBUG] Skipping empty file {file_path.resolve()}") # TEMP DEBUG
                        continue
                    if file_size > max_file_size_bytes:
                        logger.warning(f"FileSystemLoader: Skipping file {file_path} due to size ({file_size / (1024*1024):.2f}MB > {max_file_size_mb_val:.2f}MB).")
                        logger.debug(f"[LOADER DEBUG] Skipping large file {file_path.resolve()}") # TEMP DEBUG
                        continue

                    logger.debug(f"[LOADER DEBUG] Attempting aiofiles.open for {file_path.resolve()}") # TEMP DEBUG
                    async with aiofiles.open(file_path, mode="r", encoding=encoding, errors="replace") as f:
                        content = await f.read()
                    logger.debug(f"[LOADER DEBUG] Successfully read {file_path.resolve()}") # TEMP DEBUG

                    doc_id = str(file_path.resolve())
                    metadata = {
                        "source_type": "file_system",
                        "source_uri": source_uri,
                        "file_path": doc_id,
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "size_bytes": file_size,
                    }
                    logger.debug(f"FileSystemLoader: Loaded content from {file_path}.")
                    loaded_count +=1
                    logger.debug(f"[LOADER DEBUG] Yielding doc for {doc_id}") # TEMP DEBUG
                    yield cast(Document, _ConcreteDocument(content=content, metadata=metadata, id=doc_id))
                except Exception as e:
                    logger.error(f"FileSystemLoader: Error loading or reading file {file_path}: {e}", exc_info=True)
                    logger.debug(f"[LOADER DEBUG] Exception for {file_path.resolve()}: {e}") # TEMP DEBUG
            else:
                logger.debug(f"FileSystemLoader: Path {file_path} matched glob but is not a file. Skipping.")
                logger.debug(f"[LOADER DEBUG] {file_path.resolve()} is not a file, skipping.") # TEMP DEBUG
        logger.info(f"FileSystemLoader: Scanned {file_count} paths, loaded {loaded_count} documents from '{source_uri}'.")
        logger.debug(f"[LOADER DEBUG] Load method finished. Loaded_count: {loaded_count}") # TEMP DEBUG
