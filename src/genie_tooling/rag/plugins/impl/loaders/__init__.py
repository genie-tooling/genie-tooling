"""Concrete implementations of DocumentLoaderPlugins."""
from .file_system import FileSystemLoader
from .web_page import WebPageLoader

# from .api_data import APIDataLoader # Example for future

__all__ = ["FileSystemLoader", "WebPageLoader"]
