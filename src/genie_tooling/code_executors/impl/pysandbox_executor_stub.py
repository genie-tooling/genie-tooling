# src/genie_tooling/code_executors/impl/pysandbox_executor_stub.py
"""
PySandboxExecutorStub: Executes Python code using 'exec'.
**WARNING: THIS IS A STUB AND IS EXTREMELY INSECURE. DO NOT USE IN PRODUCTION.**
It does not provide any real sandboxing. It is for demonstration and testing only
where the executed code is fully trusted.
For secure execution, use SecureDockerExecutor.
"""
import asyncio
import contextlib
import io
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Dict, List, Optional

from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor

logger = logging.getLogger(__name__)

# A very restricted set of builtins. Extend with caution.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
    "bool": bool, "bytearray": bytearray, "bytes": bytes, "callable": callable,
    "chr": chr, "complex": complex, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "getattr": getattr, "hasattr": hasattr, "hash": hash,
    "hex": hex, "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max, "min": min,
    "next": next, "oct": oct, "ord": ord, "pow": pow, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
    "NameError": NameError, "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "None": None, "True": True, "False": False,
    "__import__": __import__, 
}
ALLOWED_MODULES = {
    "math": __import__("math"), "json": __import__("json"),
    "random": __import__("random"), "datetime": __import__("datetime"),
    "time": __import__("time"),
}

class PySandboxExecutorStub(CodeExecutor):
    """
    An insecure stub implementation of a Python code executor using `exec`.

    **WARNING**: This executor is NOT sandboxed and should NEVER be used in
    production environments or with untrusted code. It is provided for local
    testing and development convenience only.
    """
    plugin_id: str = "pysandbox_executor_stub_v1"
    executor_id: str = "pysandbox_executor_stub_v1"
    description: str = (
        "Executes Python code using 'exec'. **STUB IMPLEMENTATION - HIGHLY INSECURE.** "
        "No actual sandboxing is performed. Only for trusted code in development/testing. "
        "For secure execution, use SecureDockerExecutor."
    )
    supported_languages: ClassVar[List[str]] = ["python"]
    _allowed_builtins: Dict[str, Any]
    _allowed_modules: Dict[str, Any]
    _executor_pool: Optional[ThreadPoolExecutor] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the executor's thread pool and security settings.

        Args:
            config: A dictionary containing optional configuration settings:
                - `allowed_builtins` (Dict[str, Any]): Overrides the default
                  dictionary of built-in functions available to the executed code.
                - `allowed_modules` (Dict[str, Any]): Overrides the default
                  dictionary of modules available to the executed code.
                - `max_workers` (int): The number of worker threads to use for
                  executing code. Defaults to 1.
        """
        cfg = config or {}
        self._allowed_builtins = cfg.get("allowed_builtins", SAFE_BUILTINS.copy())
        self._allowed_modules = cfg.get("allowed_modules", ALLOWED_MODULES.copy())

        self._executor_pool = ThreadPoolExecutor(max_workers=cfg.get("max_workers", 1), thread_name_prefix=f"{self.plugin_id}_exec")

        logger.critical(
            f"{self.plugin_id}: Initialized. WARNING: THIS EXECUTOR IS A STUB AND USES 'exec' WITHOUT REAL SANDBOXING. IT IS INSECURE."
        )
        logger.info(f"{self.plugin_id}: Allowed builtins count: {len(self._allowed_builtins)}. Allowed modules: {list(self._allowed_modules.keys())}. Max worker threads for exec: {self._executor_pool._max_workers if self._executor_pool else 'N/A'}")

    def _execute_sync_with_capture(self, code_str: str, input_vars: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        execution_globals: Dict[str, Any] = {"__builtins__": self._allowed_builtins.copy()}
        execution_globals.update(self._allowed_modules)
        execution_locals: Dict[str, Any] = {"_input": input_vars or {}}
        execution_locals["_result"] = None
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exec_error_message: Optional[str] = None

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                try:
                    compiled_code = compile(code_str, "<sandboxed_string>", "exec")
                    exec(compiled_code, execution_globals, execution_locals)  # noqa: S102
                except SyntaxError as se:
                    exec_error_message = f"SyntaxError: {se.msg} (line {se.lineno}, offset {se.offset})"
                    traceback.print_exc() # Print to the redirected stderr (stderr_capture)
                except Exception as e:
                    exec_error_message = f"ExecutionError: {type(e).__name__}: {e!s}"
                    traceback.print_exc() # Print to the redirected stderr (stderr_capture)
        except Exception as e_outer:
            exec_error_message = f"OuterError: {type(e_outer).__name__}: {e_outer!s}"
            logger.error(f"Error in _execute_sync_with_capture's redirection or compilation: {e_outer}", exc_info=True)
            if not stderr_capture.getvalue():
                 traceback.print_exc(file=stderr_capture)


        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()
        returned_result = execution_locals.get("_result")

        return {
            "stdout": stdout_val, "stderr": stderr_val,
            "result": returned_result, "error": exec_error_message
        }

    async def execute_code(
        self, language: str, code: str, timeout_seconds: int,
        input_data: Optional[Dict[str, Any]] = None
    ) -> CodeExecutionResult:
        if language.lower() != "python":
            msg = f"Language '{language}' not supported by PySandboxExecutorStub."
            logger.warning(msg)
            return CodeExecutionResult("", msg, None, "Unsupported language", 0.0)

        if not self._executor_pool:
            logger.error(f"{self.plugin_id}: ThreadPoolExecutor not initialized. Cannot execute code.")
            return CodeExecutionResult("", "Executor not initialized.", None, "ExecutorSetupError", 0.0)

        logger.warning(f"{self.plugin_id}: Executing Python code via 'exec()'. REMINDER: THIS IS INSECURE.")
        start_time = time.perf_counter()
        sandboxed_output: Dict[str, Any]
        future = None

        try:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self._executor_pool, self._execute_sync_with_capture, code, input_data)
            sandboxed_output = await asyncio.wait_for(future, timeout=float(timeout_seconds))
        except asyncio.TimeoutError:
            end_time = time.perf_counter()
            logger.warning(f"{self.plugin_id}: Code execution task timed out after {timeout_seconds} seconds (wrapper timeout).")
            if future and not future.done():
                future.cancel()
            return CodeExecutionResult(
                "", "Execution timed out by wrapper. Note: CPU-bound loops in the thread may not have been interrupted.",
                None, "Timeout", (end_time - start_time) * 1000
            )
        except Exception as e_task:
            end_time = time.perf_counter()
            logger.error(f"{self.plugin_id}: Error running execution task: {e_task}", exc_info=True)
            return CodeExecutionResult(
                "", f"Executor task error: {e_task!s}", None,
                f"Executor task failed: {type(e_task).__name__}", (end_time - start_time) * 1000
            )
        end_time = time.perf_counter()
        execution_duration_ms = (end_time - start_time) * 1000
        logger.debug(f"{self.plugin_id}: Code execution finished in {execution_duration_ms:.2f} ms.")
        return CodeExecutionResult(
            stdout=sandboxed_output.get("stdout", ""), stderr=sandboxed_output.get("stderr", ""),
            result=sandboxed_output.get("result"), error=sandboxed_output.get("error"),
            execution_time_ms=execution_duration_ms
        )

    async def teardown(self) -> None:
        if self._executor_pool:
            logger.debug(f"{self.plugin_id}: Shutting down ThreadPoolExecutor...")
            self._executor_pool.shutdown(wait=False, cancel_futures=True)
            self._executor_pool = None
            logger.info(f"{self.plugin_id}: ThreadPoolExecutor shut down.")
        logger.debug(f"{self.plugin_id}: Torn down.")