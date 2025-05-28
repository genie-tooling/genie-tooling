"""Concrete implementations of ToolPlugins."""
from .calculator import CalculatorTool
from .code_execution_tool import GenericCodeExecutionTool
from .openweather import OpenWeatherMapTool

# Example: from .web_search import WebSearchTool # If you add more tools

__all__ = [
    "CalculatorTool",
    "OpenWeatherMapTool",
    "GenericCodeExecutionTool",
    # "WebSearchTool",
]
