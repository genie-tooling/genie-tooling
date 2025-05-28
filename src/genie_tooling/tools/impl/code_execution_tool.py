### src/genie_tooling/tools/impl/code_execution_tool.py
"""GenericCodeExecutionTool: A tool that uses pluggable CodeExecutorPlugins."""
import inspect
import logging
from typing import Any, Dict, List, Optional, Set

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.executors.abc import (  # Defined later
    CodeExecutionResult,
    CodeExecutor,
)
from genie_tooling.security.key_provider import (
    KeyProvider,  # Unused but part of interface
)
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class GenericCodeExecutionTool(Tool):
    identifier: str = "generic_code_execution_tool"
    plugin_id: str = "generic_code_execution_tool"

    # This tool needs PluginManager to discover available CodeExecutorPlugins.
    # It should be injected during instantiation by ToolManager if __init__ signature allows.
    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        self._available_executors_cache: Optional[List[CodeExecutor]] = None
        logger.info(f"{self.identifier}: Initialized with PluginManager.")

    async def _get_available_executors(self) -> List[CodeExecutor]:
        """Lazily discovers and caches available CodeExecutorPlugins."""
        if self._available_executors_cache is None:
            if not self._plugin_manager: # Should have been injected
                logger.error(f"{self.identifier}: PluginManager not available. Cannot discover executors.")
                self._available_executors_cache = []
                return []

            executors: List[CodeExecutor] = []
            # Iterate all discovered plugin IDs.
            # The PluginManager.get_plugin_instance will instantiate them.
            # Then we check if the instance is a CodeExecutor.
            for plugin_id in self._plugin_manager.list_discovered_plugin_classes().keys():
                try:
                    # Attempt to get an instance. Configuration for executors during discovery
                    # is usually minimal or default.
                    # Pass an empty config for now; specific executors might need more.
                    # Ensure the plugin has setup called by get_plugin_instance.
                    instance = await self._plugin_manager.get_plugin_instance(plugin_id, config={})
                    if instance and isinstance(instance, CodeExecutor): # This uses @runtime_checkable
                        executors.append(instance)
                    elif instance:
                        logger.debug(f"Plugin '{plugin_id}' is not a CodeExecutor (type: {type(instance).__name__}). Skipping.")
                except Exception as e:
                    logger.error(f"Error instantiating or checking plugin '{plugin_id}' as CodeExecutor: {e}", exc_info=True)

            self._available_executors_cache = executors
            logger.debug(f"{self.identifier}: Discovered {len(self._available_executors_cache)} code executors.")
        return self._available_executors_cache


    async def get_metadata(self) -> Dict[str, Any]:
        executors = await self._get_available_executors()
        supported_languages_set: Set[str] = set()
        executor_descriptions_list: List[str] = []

        for exec_plugin in executors:
            supported_languages_set.update(exec_plugin.supported_languages)
            executor_descriptions_list.append(f"- {exec_plugin.executor_id} ({', '.join(exec_plugin.supported_languages)}): {exec_plugin.description}")

        supported_languages = sorted(list(supported_languages_set))
        available_executors_desc = "\nAvailable Executors:\n" + "\n".join(executor_descriptions_list) if executor_descriptions_list else "No code executors currently available."


        return {
            "identifier": self.identifier,
            "name": "Code Execution Engine",
            "description_human": f"Executes code scripts in various supported languages using configured sandboxed executors. This tool is powerful but carries inherent security risks depending on the executor and code. {available_executors_desc}",
            "description_llm": f"CodeExecutor: Runs code. Args: language (str, e.g., '{supported_languages[0] if supported_languages else 'python'}'), code_to_run (str), executor_id (str, optional, to use a specific backend), timeout_seconds (int, optional, default 30). Ensure code is safe. Supported languages: {', '.join(supported_languages) if supported_languages else 'None'}.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The programming language of the code (e.g., 'python').",
                        "enum": supported_languages if supported_languages else None # Enum if list is not empty
                    },
                    "code_to_run": {
                        "type": "string",
                        "description": "The code script to execute."
                    },
                    "executor_id": {
                        "type": "string",
                        "description": "Optional: ID of a specific CodeExecutorPlugin to use (e.g., 'pysandbox_executor_v1_stub'). If not provided, a suitable executor for the language will be chosen if available.",
                        "enum": [exec_p.executor_id for exec_p in executors] if executors else None
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional: Maximum execution time in seconds.",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300 # Example reasonable maximum
                    }
                },
                "required": ["language", "code_to_run"]
            },
            "output_schema": { # Matches CodeExecutionResult structure
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
            "key_requirements": [], # Executors themselves might have key needs (e.g., a cloud-based executor)
            "tags": ["code", "execution", "scripting", "development", "sandboxed", "caution"],
            "version": "1.0.0",
            "cacheable": False, # Code execution is generally not cacheable due to side effects and variability
        }

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider, # May be needed if an executor requires keys via context
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]: # Should match output_schema, essentially CodeExecutionResult as dict
        language = params["language"]
        code_to_run = params["code_to_run"]
        requested_executor_id = params.get("executor_id")
        timeout = params.get("timeout_seconds", 30)

        executors = await self._get_available_executors()
        chosen_executor: Optional[CodeExecutor] = None

        if requested_executor_id:
            for exec_plugin in executors:
                if exec_plugin.executor_id == requested_executor_id:
                    if language in exec_plugin.supported_languages:
                        chosen_executor = exec_plugin
                        break
                    else:
                        msg = f"Requested executor '{requested_executor_id}' does not support language '{language}'."
                        logger.warning(msg)
                        # Ensure the 'error' field contains a concise message for programmatic checks
                        # and 'stderr' contains the more descriptive 'msg' for human readability/logging.
                        return CodeExecutionResult("", msg, None, "Executor language mismatch.", 0.0)._asdict()
            if not chosen_executor:
                msg = f"Requested executor '{requested_executor_id}' not found or not available."
                logger.warning(msg)
                return CodeExecutionResult("", msg, None, "Executor not found.", 0.0)._asdict()
        else: # Auto-select executor
            for exec_plugin in executors:
                if language in exec_plugin.supported_languages:
                    chosen_executor = exec_plugin
                    logger.debug(f"Auto-selected executor '{chosen_executor.executor_id}' for language '{language}'.")
                    break
            if not chosen_executor:
                msg = f"No available executor found that supports language '{language}'."
                logger.warning(msg)
                return CodeExecutionResult("", msg, None, "No suitable executor.", 0.0)._asdict()

        logger.info(f"Executing code with language '{language}' using executor '{chosen_executor.executor_id}'. Timeout: {timeout}s.")
        # Input data for the code execution context, if any. Can come from tool params or agent context.
        # This is a conceptual pass-through; executor needs to support it.
        code_input_data = (context or {}).get("code_input_data")

        try:
            # The CodeExecutorPlugin might need the key_provider or context itself,
            # but the execute_code interface is simpler. If needed, an executor
            # can be designed to receive these during its own setup/instantiation via PluginManager.
            exec_result: CodeExecutionResult = await chosen_executor.execute_code(
                language=language,
                code=code_to_run,
                timeout_seconds=timeout,
                input_data=code_input_data
            )
            return exec_result._asdict() # Convert NamedTuple to dict for output_schema compliance
        except Exception as e:
            logger.error(f"Unhandled exception during code execution via tool '{self.identifier}' with executor '{chosen_executor.executor_id}': {e}", exc_info=True)
            return CodeExecutionResult(
                stdout="", stderr=f"Tool-level execution error: {str(e)}", result=None,
                error=f"Critical error in execution process: {str(e)}", execution_time_ms=0.0
            )._asdict()

    async def teardown(self) -> None:
        """Tear down any discovered executor plugins if they have teardown methods."""
        if self._available_executors_cache:
            for executor in self._available_executors_cache:
                if hasattr(executor, "teardown") and callable(executor.teardown):
                    try:
                        # Ensure teardown is awaited if it's an async method
                        if inspect.iscoroutinefunction(executor.teardown):
                            await executor.teardown()
                        else: # type: ignore #Should not happen based on Plugin Protocol
                            executor.teardown() # type: ignore
                    except Exception as e:
                        logger.error(f"Error tearing down executor '{executor.executor_id}': {e}", exc_info=True)
            self._available_executors_cache = None # Clear the cache after attempting teardown
        logger.debug(f"{self.identifier}: Teardown complete.")
