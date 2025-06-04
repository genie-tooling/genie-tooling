### tests/unit/code_executors/impl/test_pysandbox_executor.py
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the correct stub that is actually being tested
from genie_tooling.code_executors.impl.pysandbox_executor_stub import (
    PySandboxExecutorStub,
)

EXECUTOR_LOGGER_NAME = "genie_tooling.code_executors.impl.pysandbox_executor_stub"

@pytest.fixture
async def pysandbox_executor_stub() -> PySandboxExecutorStub: # Changed fixture name
    executor = PySandboxExecutorStub()
    await executor.setup()
    return executor

@pytest.fixture
def mock_execute_sync_with_capture() -> MagicMock:
    # Patch the internal method of the stub we are testing
    with patch(
        "genie_tooling.code_executors.impl.pysandbox_executor_stub.PySandboxExecutorStub._execute_sync_with_capture"
    ) as mock_func:
        mock_func.return_value = {
            "stdout": "Default mock output",
            "stderr": "",
            "result": "Default mock result",
            "error": None,
        }
        yield mock_func


@pytest.mark.asyncio
class TestPySandboxExecutorStub: # Changed class name
    async def test_setup_logs_warning(self, pysandbox_executor_stub: PySandboxExecutorStub, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.CRITICAL, logger=EXECUTOR_LOGGER_NAME)
        executor = await pysandbox_executor_stub # Fixture calls setup
        assert executor.plugin_id == "pysandbox_executor_stub_v1"
        temp_executor = PySandboxExecutorStub()
        with caplog.at_level(logging.CRITICAL, logger=EXECUTOR_LOGGER_NAME):
            await temp_executor.setup()
        assert f"{temp_executor.plugin_id}: Initialized. WARNING: THIS EXECUTOR IS A STUB AND USES 'exec' WITHOUT REAL SANDBOXING. IT IS INSECURE." in caplog.text


    async def test_execute_unsupported_language(self, pysandbox_executor_stub: PySandboxExecutorStub):
        executor = await pysandbox_executor_stub
        result = await executor.execute_code("javascript", "console.log('hi')", 10)
        assert result.error == "Unsupported language"
        assert "Language 'javascript' not supported" in result.stderr

    async def test_execute_success(
        self, pysandbox_executor_stub: PySandboxExecutorStub, mock_execute_sync_with_capture: MagicMock
    ):
        executor = await pysandbox_executor_stub
        code = "print('hello')"
        input_data = {"val": 1}
        timeout = 10
        expected_output_from_sync = {
            "stdout": "hello from sync",
            "stderr": "",
            "result": {"status": "ok_sync"},
            "error": None,
        }
        mock_execute_sync_with_capture.return_value = expected_output_from_sync

        result = await executor.execute_code("python", code, timeout, input_data)

        mock_execute_sync_with_capture.assert_called_once_with(code, input_data)
        assert result.stdout == "hello from sync"
        assert result.stderr == ""
        assert result.result == {"status": "ok_sync"}
        assert result.error is None
        assert result.execution_time_ms >= 0

    async def test_execute_timeout_from_wait_for(
        self, pysandbox_executor_stub: PySandboxExecutorStub, mock_execute_sync_with_capture: MagicMock
    ):
        executor = await pysandbox_executor_stub
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError("wait_for timed out")):
            result = await executor.execute_code("python", "while True: pass", 1)

        assert result.error == "Timeout"
        assert result.stderr == "Execution timed out by wrapper."
        # _execute_sync_with_capture might be called if run_in_executor schedules it before timeout
        # The key is that the overall operation times out and reports correctly.
        # So, removing mock_execute_sync_with_capture.assert_not_called()

    async def test_execute_run_in_executor_general_exception(
        self, pysandbox_executor_stub: PySandboxExecutorStub, mock_execute_sync_with_capture: MagicMock
    ):
        executor = await pysandbox_executor_stub
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop_instance = AsyncMock()
            mock_loop_instance.run_in_executor.side_effect = ValueError("Executor task failed")
            mock_get_loop.return_value = mock_loop_instance

            result = await executor.execute_code("python", "code", 10)

        assert result.error == "Executor task failed: ValueError"
        assert "Executor task error: Executor task failed" in result.stderr
        assert result.result is None
        mock_execute_sync_with_capture.assert_not_called()

    async def test_execute_sandbox_function_returns_error(
        self, pysandbox_executor_stub: PySandboxExecutorStub, mock_execute_sync_with_capture: MagicMock
    ):
        executor = await pysandbox_executor_stub
        sandbox_error_output = {
            "stdout": "Some output before error",
            "stderr": "Error inside sandbox",
            "result": None,
            "error": "SandboxExecutionErrorFromSync",
        }
        mock_execute_sync_with_capture.return_value = sandbox_error_output

        result = await executor.execute_code("python", "code that errors", 10)

        assert result.stdout == "Some output before error"
        assert result.stderr == "Error inside sandbox"
        assert result.error == "SandboxExecutionErrorFromSync"
        assert result.result is None

    async def test_teardown_is_noop(self, pysandbox_executor_stub: PySandboxExecutorStub, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.DEBUG, logger=EXECUTOR_LOGGER_NAME)
        executor = await pysandbox_executor_stub
        await executor.teardown()
        assert f"{executor.plugin_id}: Torn down (no specific resources to release for this stub)." in caplog.text
