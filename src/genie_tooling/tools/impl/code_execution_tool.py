### src/genie_tooling/tools/impl/code_execution_tool.py
"""GenericCodeExecutionTool: A tool that uses pluggable CodeExecutorPlugins."""
import inspect
import logging
from typing import Any, Dict, List, Optional, Set

# Updated import paths for CodeExecutor and CodeExecutionResult
from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import (
    KeyProvider,  # Unused but part of interface
)
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class GenericCodeExecutionTool(Tool):
    identifier: str = "generic_code_execution_tool"
    plugin_id: str = "generic_code_execution_tool"

    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        self._available_executor_ids: List[str] = []
        self._default_executor_id: Optional[str] = None
        logger.info(f"{self.identifier}: Initialized with PluginManager.")

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Configures the tool with available executor IDs.
        The 'executor_id' key in the config can be used to set a default.
        Other executors can be made available via 'available_executor_ids' list.
        """
        cfg = config or {}
        self._default_executor_id = cfg.get("executor_id", "pysandbox_executor_stub_v1")

        # --- REVISED FIX: If the key is explicitly in the config, use its value. ---
        # --- Otherwise, default to a list containing the default executor. ---
        if "available_executor_ids" in cfg:
            self._available_executor_ids = cfg["available_executor_ids"]
        else:
            self._available_executor_ids = [self._default_executor_id] if self._default_executor_id else []

        logger.info(
            f"{self.identifier}: Setup complete. "
            f"Default executor: '{self._default_executor_id}'. "
            f"Available executors: {self._available_executor_ids}"
        )

    async def get_metadata(self) -> Dict[str, Any]:
        supported_languages_set: Set[str] = {"python", "javascript", "bash"} # Assume common languages
        executor_ids_for_schema = self._available_executor_ids or [] # Use empty list if None

        return {
            "identifier": self.identifier,
            "name": "Code Execution Engine",
            "description_human": "Executes code scripts in various supported languages using configured sandboxed executors. This tool is powerful but carries inherent security risks depending on the executor and code.",
            "description_llm": f"CodeExecutor: Runs code. Args: language (str, e.g., 'python'), code_to_run (str), executor_id (str, optional, to use a specific backend), timeout_seconds (int, optional, default 30). Supported languages depend on the chosen executor. Available executors: {executor_ids_for_schema}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language of the code (e.g., 'python').",
                        "enum": sorted(supported_languages_set) if supported_languages_set else None
                    },
                    "code_to_run": {
                        "type": "string",
                        "description": "The code script to execute."
                    },
                    "executor_id": {
                        "type": "string",
                        "description": f"Optional: ID of a specific CodeExecutorPlugin to use. If not provided, the default '{self._default_executor_id}' will be used.",
                        "enum": executor_ids_for_schema
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional: Maximum execution time in seconds.",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300
                    }
                },
                "required": ["language", "code_to_run"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "stdout": {"type": "string", "description": "Standard output from the code execution."},
                    "stderr": {"type": "string", "description": "Standard error output from the code execution."},
                    "result": {"type": ["string", "number", "boolean", "object", "array", "null"], "description": "A structured result returned by the code, if the executor supports capturing it (often part of stdout/stderr)."},
                    "error": {"type": ["string", "null"], "description": "An error message if the executor itself failed (e.g., timeout, setup error), distinct from stderr of the executed code."},
                    "execution_time_ms": {"type": "number", "description": "Duration of the code execution in milliseconds."}
                },
                "required": ["stdout", "stderr", "result", "error", "execution_time_ms"]
            },
            "key_requirements": [],
            "tags": ["code", "execution", "scripting", "development", "sandboxed", "caution"],
            "version": "1.1.0",
            "cacheable": False,
        }

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        language = params["language"]
        code_to_run = params["code_to_run"]
        requested_executor_id = params.get("executor_id", self._default_executor_id)
        timeout = params.get("timeout_seconds", 30)

        if not requested_executor_id:
            msg = "No executor_id specified and no default is configured for this tool."
            logger.error(msg)
            return CodeExecutionResult("", msg, None, "ConfigurationError", 0.0)._asdict()

        if requested_executor_id not in self._available_executor_ids:
            msg = f"Requested executor '{requested_executor_id}' is not in the list of available executors for this tool: {self._available_executor_ids}"
            logger.error(msg)
            return CodeExecutionResult("", msg, None, "ConfigurationError", 0.0)._asdict()

        executor_instance = await self._plugin_manager.get_plugin_instance(requested_executor_id)
        if not isinstance(executor_instance, CodeExecutor):
            msg = f"Failed to load a valid CodeExecutor for ID '{requested_executor_id}'."
            logger.error(msg)
            return CodeExecutionResult("", msg, None, "ExecutorLoadError", 0.0)._asdict()

        if language not in executor_instance.supported_languages:
            msg = f"Executor '{executor_instance.executor_id}' does not support the requested language '{language}'."
            logger.warning(msg)
            return CodeExecutionResult("", msg, None, "UnsupportedLanguage", 0.0)._asdict()

        logger.info(f"Executing code with language '{language}' using executor '{executor_instance.executor_id}'. Timeout: {timeout}s.")
        code_input_data = (context or {}).get("code_input_data")

        try:
            exec_result: CodeExecutionResult = await executor_instance.execute_code(
                language=language,
                code=code_to_run,
                timeout_seconds=timeout,
                input_data=code_input_data
            )
            return exec_result._asdict()
        except Exception as e:
            logger.error(f"Unhandled exception during code execution via tool '{self.identifier}' with executor '{executor_instance.executor_id}': {e}", exc_info=True)
            return CodeExecutionResult(
                stdout="", stderr=f"Tool-level execution error: {e!s}", result=None,
                error=f"Critical error in execution process: {e!s}", execution_time_ms=0.0
            )._asdict()

    async def teardown(self) -> None:
        logger.debug(f"{self.identifier}: Teardown complete.")