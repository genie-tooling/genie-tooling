"""
PySandboxExecutorStub: Executes Python code using 'exec'.
**WARNING: THIS IS A STUB AND IS EXTREMELY INSECURE. DO NOT USE IN PRODUCTION.**
It does not provide any real sandboxing. It is for demonstration and testing only
where the executed code is fully trusted.
A proper implementation would use a dedicated sandboxing library or containerization.
"""
import asyncio
import contextlib  # For redirecting stdout/stderr
import io
import logging
import time
import traceback  # For capturing exception details
from typing import Any, Dict, List, Optional

from genie_tooling.executors.abc import CodeExecutionResult, CodeExecutor

logger = logging.getLogger(__name__)

class PySandboxExecutorStub(CodeExecutor):
    plugin_id: str = "pysandbox_executor_stub_v1" # Explicitly mark as stub
    executor_id: str = "pysandbox_executor_stub_v1"
    description: str = (
        "Executes Python code using 'exec'. **STUB IMPLEMENTATION - HIGHLY INSECURE.** "
        "No actual sandboxing is performed. Only for trusted code in development."
    )
    supported_languages: List[str] = ["python"]

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.critical(f"{self.plugin_id}: Initialized. "
                        "WARNING: THIS EXECUTOR IS A STUB AND USES 'exec' WITHOUT REAL SANDBOXING. IT IS INSECURE.")

    def _execute_sync_with_capture(self, code_str: str, input_vars: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synchronously executes Python code using 'exec' and attempts to capture stdout/stderr.
        This is still fundamentally insecure.
        """
        execution_scope = {"_input": input_vars or {}}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        execution_scope["_result"] = None
        exec_error_message: Optional[str] = None

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                compiled_code = compile(code_str, "<sandboxed_string>", "exec")
                exec(compiled_code, execution_scope)

        except SyntaxError as se:
            exec_error_message = f"SyntaxError: {se.msg} (line {se.lineno}, offset {se.offset})"
            # Manually add to stderr_capture as redirect might not catch compile-time errors effectively.
            stderr_capture.write(f"{exec_error_message}\n{traceback.format_exc(limit=1)}\n")
        except Exception as e:
            exec_error_message = f"ExecutionError: {type(e).__name__}: {str(e)}"
            # Manually add to stderr_capture for runtime exceptions.
            stderr_capture.write(f"{type(e).__name__}: {str(e)}\n{traceback.format_exc(limit=5)}\n") # Limit traceback depth

        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()
        returned_result = execution_scope.get("_result")

        return {
            "stdout": stdout_val,
            "stderr": stderr_val,
            "result": returned_result,
            "error": exec_error_message
        }

    async def execute_code(
        self,
        language: str,
        code: str,
        timeout_seconds: int,
        input_data: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        if language.lower() != "python":
            msg = f"Language '{language}' not supported by PySandboxExecutorStub."
            logger.warning(msg)
            return CodeExecutionResult(stdout="", stderr=msg, result=None, error="Unsupported language", execution_time_ms=0.0)

        logger.warning(f"{self.plugin_id}: Executing Python code. "
                       "REMINDER: THIS IS INSECURE AND FOR DEVELOPMENT/TRUSTED CODE ONLY.")

        start_time = time.perf_counter()

        try:
            loop = asyncio.get_running_loop()
            sandboxed_output = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync_with_capture, code, input_data),
                timeout=float(timeout_seconds)
            )
        except asyncio.TimeoutError:
            end_time = time.perf_counter()
            logger.warning(f"{self.plugin_id}: Code execution timed out after {timeout_seconds} seconds.")
            return CodeExecutionResult(
                stdout="", stderr="Execution timed out by wrapper.", result=None, error="Timeout",
                execution_time_ms=(end_time - start_time) * 1000
            )
        except Exception as e_task:
            end_time = time.perf_counter()
            logger.error(f"{self.plugin_id}: Error running execution task: {e_task}", exc_info=True)
            return CodeExecutionResult(
                stdout="", stderr=f"Executor task error: {str(e_task)}", result=None, error=f"Executor task failed: {type(e_task).__name__}",
                execution_time_ms=(end_time - start_time) * 1000
            )

        end_time = time.perf_counter()
        execution_duration_ms = (end_time - start_time) * 1000
        logger.debug(f"{self.plugin_id}: Code execution finished in {execution_duration_ms:.2f} ms.")

        return CodeExecutionResult(
            stdout=sandboxed_output.get("stdout", ""),
            stderr=sandboxed_output.get("stderr", ""),
            result=sandboxed_output.get("result"),
            error=sandboxed_output.get("error"),
            execution_time_ms=execution_duration_ms
        )

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Torn down (no specific resources to release for this stub).")
