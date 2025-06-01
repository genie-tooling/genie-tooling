"""Unit tests for PySandboxExecutorStub."""

import pytest

from genie_tooling.code_executors.abc import CodeExecutionResult
from genie_tooling.code_executors.impl.pysandbox_executor_stub import (
    PySandboxExecutorStub,
)


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
print("Hello to stdout") # Changed from logger.debug to print
sys.stderr.write("Error to stderr\\n")
_input_value = _input.get("val", 0)
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
    code_long_running = "import time\ntime.sleep(2)"

    exec_res = await executor_instance.execute_code("python", code_long_running, timeout_seconds=1)

    assert exec_res.error == "Timeout"#!/bin/bash
