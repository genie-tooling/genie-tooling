"""Document Loader Abstractions and Implementations."""

from .abc import DocumentLoaderPlugin
from .impl import FileSystemLoader, WebPageLoader

__all__ = [
    "DocumentLoaderPlugin",
    "FileSystemLoader",
    "WebPageLoader",
]
