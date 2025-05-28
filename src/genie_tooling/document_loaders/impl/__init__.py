"""Implementations of DocumentLoaderPlugin."""
from .file_system import FileSystemLoader
from .web_page import WebPageLoader

__all__ = ["FileSystemLoader", "WebPageLoader"]
