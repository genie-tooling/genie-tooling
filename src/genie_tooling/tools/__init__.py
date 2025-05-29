"""Tool-related functionality: ToolPlugin definition, ToolManager."""
from .abc import Tool as ToolPlugin
# DefinitionFormatterPlugin is now sourced from genie_tooling.definition_formatters
from .impl.calculator import CalculatorTool
from .impl.code_execution_tool import GenericCodeExecutionTool
from .impl.openweather import OpenWeatherMapTool
from .manager import ToolManager

# Add other built-in tools here if desired for direct import

__all__ = [
    "ToolPlugin", "ToolManager",
    "CalculatorTool", "OpenWeatherMapTool", "GenericCodeExecutionTool"
]