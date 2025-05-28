"""Tool-related functionality: ToolPlugin definition, ToolManager, formatters."""
from .abc import Tool as ToolPlugin
from .formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
from .impl.calculator import CalculatorTool
from .impl.code_execution_tool import GenericCodeExecutionTool
from .impl.openweather import OpenWeatherMapTool
from .manager import ToolManager

# Add other built-in tools here if desired for direct import

__all__ = [
    "ToolPlugin", "ToolManager", "DefinitionFormatterPlugin",
    "CalculatorTool", "OpenWeatherMapTool", "GenericCodeExecutionTool"
]
