### tests/unit/executors/impl/test_pysandbox_executor_stub.py
"""Unit tests for PySandboxExecutorStub."""

import logging
import time
from unittest.mock import patch

import pytest
from genie_tooling.code_executors.abc import CodeExecutionResult
from genie_tooling.code_executors.impl.pysandbox_executor_stub import (
    PySandboxExecutorStub,
)

logger = logging.getLogger(__name__)


@pytest.fixture
async def pysandbox_stub() -> PySandboxExecutorStub:
    executor = PySandboxExecutorStub()
    await executor.setup()
    return executor

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_success_with_stdout_stderr_result(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub
    code = """
import sys
print("Hello to stdout")
sys.stderr.write("Error to stderr\\n")
_input_value = _input.get("val", 0) if isinstance(_input, dict) else 0
_result = {"status": "ok", "value": 10 + _input_value}
"""
    input_data = {"val": 5}
    expected_result_val = {"status": "ok", "value": 15}

    exec_res: CodeExecutionResult = await executor_instance.execute_code(
        language="python",
        code=code,
        timeout_seconds=5,
        input_data=input_data
    )

    assert "Hello to stdout" in exec_res.stdout
    assert "Error to stderr" in exec_res.stderr
    assert exec_res.result == expected_result_val
    assert exec_res.error is None
    assert exec_res.execution_time_ms > 0

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_syntax_error(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub
    code_with_syntax_error = "print('Hello' esto_es_un_error_de_sintaxis)"

    exec_res = await executor_instance.execute_code("python", code_with_syntax_error, 5)

    assert exec_res.error is not None
    assert "SyntaxError" in exec_res.error
    # stderr might contain more than just "invalid syntax", so use "in"
    assert "invalid syntax" in exec_res.stderr.lower() or "invalid syntax" in exec_res.error.lower()
    assert exec_res.stdout == ""

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_runtime_error(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub
    code_with_runtime_error = "x = 1 / 0"

    exec_res = await executor_instance.execute_code("python", code_with_runtime_error, 5)

    assert exec_res.error is not None
    assert "ExecutionError: ZeroDivisionError: division by zero" in exec_res.error
    assert "ZeroDivisionError: division by zero" in exec_res.stderr
    assert exec_res.stdout == ""

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_timeout(pysandbox_stub: PySandboxExecutorStub):
    executor_instance = await pysandbox_stub
    code_long_running = "import time\ntime.sleep(0.2)"  # Sleep for 0.2 seconds

    # Test with a timeout of 0.1 second
    timeout_seconds = 0.1
    start_time = time.perf_counter()
    # We mock _execute_sync_with_capture to simulate it taking longer than the timeout
    # The original _execute_sync_with_capture is synchronous.
    # We need to ensure that run_in_executor is what's being timed out.

    original_execute_sync = executor_instance._execute_sync_with_capture

    def slow_execute_sync_with_capture(*args, **kwargs):
        time.sleep(timeout_seconds + 0.1) # Sleep longer than the timeout
        return original_execute_sync(*args, **kwargs)

    with patch.object(executor_instance, "_execute_sync_with_capture", side_effect=slow_execute_sync_with_capture):
        exec_res = await executor_instance.execute_code("python", code_long_running, timeout_seconds=timeout_seconds)

    end_time = time.perf_counter()

    assert exec_res.error == "Timeout"
    assert "Execution timed out by wrapper." in exec_res.stderr
    # Actual execution time should be around timeout_seconds because asyncio.wait_for cancels it
    assert exec_res.execution_time_ms >= timeout_seconds * 1000
    assert exec_res.execution_time_ms < (timeout_seconds + 0.1) * 1000 # Should be close to timeout

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_unsupported_language(pysandbox_stub: PySandboxExecutorStub):
    executor = await pysandbox_stub
    result = await executor.execute_code("javascript", "console.log('hello')", 10)
    assert result.error == "Unsupported language"
    assert "Language 'javascript' not supported by PySandboxExecutorStub." in result.stderr
    assert result.execution_time_ms == 0.0

@pytest.mark.asyncio
async def test_pysandbox_stub_run_in_executor_task_fails(
    pysandbox_stub: PySandboxExecutorStub,
    caplog: pytest.LogCaptureFixture
):
    executor = await pysandbox_stub
    caplog.set_level(logging.ERROR)
    code_to_run = "print('test')"
    timeout_sec = 5
    simulated_error_type_name = "OSError" # The type of error we are simulating from run_in_executor

    # Simulate run_in_executor itself failing
    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=OSError("Executor pool closed")):
        result = await executor.execute_code("python", code_to_run, timeout_sec)

    assert result.error == f"Executor task failed: {simulated_error_type_name}" # Check against the type name
    assert "Executor task error: Executor pool closed" in result.stderr
    assert "Error running execution task: Executor pool closed" in caplog.text


@pytest.mark.asyncio
async def test_pysandbox_stub_teardown(pysandbox_stub: PySandboxExecutorStub, caplog: pytest.LogCaptureFixture):
    executor = await pysandbox_stub
    caplog.set_level(logging.DEBUG)
    await executor.teardown()
    assert f"{executor.plugin_id}: Torn down (no specific resources to release for this stub)." in caplog.text

@pytest.mark.asyncio
async def test_pysandbox_stub_execute_no_input_data(pysandbox_stub: PySandboxExecutorStub):
    executor = await pysandbox_stub
    code = "_result = _input.get('default_val') if isinstance(_input, dict) else 'no_input_dict'"
    res = await executor.execute_code("python", code, 5, input_data=None)
    assert res.result is None

    code_check_type = "_result = isinstance(_input, dict)"
    res_check_type = await executor.execute_code("python", code_check_type, 5, input_data=None)
    assert res_check_type.result is True


@pytest.mark.asyncio
async def test_pysandbox_stub_execute_code_no_result_var(pysandbox_stub: PySandboxExecutorStub):
    executor = await pysandbox_stub
    code_no_result = "a = 1 + 1\nprint(a)"
    res = await executor.execute_code("python", code_no_result, 5)
    assert res.stdout.strip() == "2"
    assert res.result is None
    assert res.error is None
