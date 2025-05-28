"""Abstract Base Class/Protocol for CodeExecutor Plugins."""
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

class CodeExecutionResult(NamedTuple):
    """Standardized result structure for code execution."""
    stdout: str
    stderr: str
    result: Optional[Any] # Capturing a structured 'return' value from script is complex and executor-dependent.
                          # Often, scripts print results to stdout.
    error: Optional[str]  # Error message from the executor itself (e.g., timeout, setup fail),
                          # distinct from stderr produced by the executed code.
    execution_time_ms: float

@runtime_checkable
class CodeExecutor(Plugin, Protocol):
    """Protocol for a plugin that executes code, ideally in a sandboxed environment."""
    # plugin_id: str (from Plugin protocol)
    executor_id: str # Specific ID for this executor instance/type (can be same as plugin_id)
    description: str # Human-readable description of this executor and its sandboxing capabilities/limitations
    supported_languages: List[str] # List of language identifiers (e.g., ["python", "javascript"])

    async def execute_code(
        self,
        language: str,
        code: str,
        timeout_seconds: int,
        input_data: Optional[Dict[str, Any]] = None # For passing structured input to the code's execution scope
    ) -> CodeExecutionResult:
        """
        Asynchronously executes the provided code string in the specified language.
        Implementations should strive for secure execution (sandboxing).
        Args:
            language: The programming language of the code.
            code: The code script to execute.
            timeout_seconds: Maximum allowed execution time in seconds.
            input_data: Optional dictionary to make available as variables within the code's scope
                        (e.g., as a global dict named `_input` or similar, executor-defined).
        Returns:
            A CodeExecutionResult NamedTuple.
        """
        logger.warning(f"CodeExecutor '{self.plugin_id}' execute_code method not fully implemented.")
        return CodeExecutionResult(stdout="", stderr="Not implemented", result=None, error="Executor not implemented", execution_time_ms=0.0)
