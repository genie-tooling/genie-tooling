"""Unit tests for PySandboxExecutorStub."""

import pytest
from genie_tooling.executors.abc import CodeExecutionResult
from genie_tooling.executors.impl.pysandbox_executor_stub import PySandboxExecutorStub


@pytest.fixture
async def pysandbox_stub() -> PySandboxExecutorStub:
    executor = PySandboxExecutorStub()
    await executor.setup()
    return executor

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_success_with_stdout_stderr_result(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub # Await the fixture
    code = """
import sys
print("Hello to stdout")
sys.stderr.write("Error to stderr\\n")
_input_value = _input.get("val", 0) # Access input_data via _input
_result = {"status": "ok", "value": 10 + _input_value} # Set _result for capture
"""
    input_data = {"val": 5}
    expected_result_val = {"status": "ok", "value": 15}

    exec_res: CodeExecutionResult = await executor_instance.execute_code( # Use awaited instance
        language="python",
        code=code,
        timeout_seconds=5,
        input_data=input_data
    )

    assert "Hello to stdout" in exec_res.stdout
    assert "Error to stderr" in exec_res.stderr
    assert exec_res.result == expected_result_val
    assert exec_res.error is None # No executor error
    assert exec_res.execution_time_ms > 0

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_syntax_error(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub # Await the fixture
    code_with_syntax_error = "print('Hello' esto_es_un_error_de_sintaxis)"

    exec_res = await executor_instance.execute_code("python", code_with_syntax_error, 5) # Use awaited instance

    assert exec_res.error is not None
    assert "SyntaxError" in exec_res.error
    # stderr might also contain syntax error details
    assert "invalid syntax" in exec_res.stderr.lower() or "invalid syntax" in exec_res.error.lower()
    assert exec_res.stdout == "" # No stdout expected on syntax error

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_runtime_error(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub # Await the fixture
    code_with_runtime_error = "x = 1 / 0" # This will raise ZeroDivisionError

    exec_res = await executor_instance.execute_code("python", code_with_runtime_error, 5) # Use awaited instance

    assert exec_res.error is not None
    assert "ExecutionError: ZeroDivisionError: division by zero" in exec_res.error
    # Stderr should also capture the traceback or error message
    assert "ZeroDivisionError: division by zero" in exec_res.stderr
    assert exec_res.stdout == ""

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_timeout(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub # Await the fixture
    # This code will sleep longer than the timeout, allowing the thread to eventually finish,
    # but asyncio.wait_for should still raise TimeoutError.
    code_long_running = "import time\ntime.sleep(2)"  # Sleeps for 2s

    exec_res = await executor_instance.execute_code("python", code_long_running, timeout_seconds=1) # Timeout is 1s

    assert exec_res.error == "Timeout"
    assert "Execution timed out by wrapper" in exec_res.stderr
    assert exec_res.stdout == ""
    # execution_time_ms will be around the timeout value (1000ms), not the sleep duration (2000ms)
    assert exec_res.execution_time_ms >= 1000
    assert exec_res.execution_time_ms < 1500 # Allow some buffer for overhead

@pytest.mark.asyncio
async def test_pysandbox_stub_unsupported_language(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub # Await the fixture
    exec_res = await executor_instance.execute_code("javascript", "console.log('hi')", 5) # Use awaited instance
    assert exec_res.error == "Unsupported language"
    assert "not supported by PySandboxExecutorStub" in exec_res.stderr
