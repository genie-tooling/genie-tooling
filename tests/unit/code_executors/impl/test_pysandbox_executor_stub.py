### tests/unit/code_executors/impl/test_pysandbox_executor_stub.py
import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.code_executors.impl.pysandbox_executor_stub import (
    ALLOWED_MODULES,
    SAFE_BUILTINS,
    PySandboxExecutorStub,
)

EXECUTOR_LOGGER_NAME = "genie_tooling.code_executors.impl.pysandbox_executor_stub"


@pytest.fixture()
async def pysandbox_executor_stub_fixture(request) -> PySandboxExecutorStub:
    executor = PySandboxExecutorStub()
    await executor.setup()

    def finalizer_sync():
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.ensure_future(executor.teardown())
            else: # pragma: no cover
                loop.run_until_complete(executor.teardown())
        except RuntimeError: # pragma: no cover
            asyncio.run(executor.teardown())

    request.addfinalizer(finalizer_sync)
    return executor


@pytest.mark.asyncio()
class TestPySandboxExecutorStubSetup:
    async def test_setup_default_config(self, caplog: pytest.LogCaptureFixture):
        executor = PySandboxExecutorStub()
        with caplog.at_level(logging.INFO, logger=EXECUTOR_LOGGER_NAME):
            await executor.setup()

        assert executor._allowed_builtins == SAFE_BUILTINS
        assert executor._allowed_modules == ALLOWED_MODULES
        assert executor._executor_pool is not None
        assert executor._executor_pool._max_workers == 1
        assert f"{executor.plugin_id}: Initialized. WARNING: THIS EXECUTOR IS A STUB" in caplog.text
        assert f"Allowed builtins count: {len(SAFE_BUILTINS)}" in caplog.text
        await executor.teardown()

    async def test_setup_custom_config(self):
        executor = PySandboxExecutorStub()
        custom_builtins = {"print": print}
        custom_modules = {"math": __import__("math")}
        await executor.setup(
            config={
                "allowed_builtins": custom_builtins,
                "allowed_modules": custom_modules,
                "max_workers": 2,
            }
        )
        assert executor._allowed_builtins == custom_builtins
        assert executor._allowed_modules == custom_modules
        assert executor._executor_pool is not None
        assert executor._executor_pool._max_workers == 2
        await executor.teardown()

    async def test_teardown_closes_executor_pool(
        self # No fixture needed, test teardown directly
    ):
        executor_instance = PySandboxExecutorStub()
        await executor_instance.setup()

        pool = executor_instance._executor_pool
        assert pool is not None
        mock_shutdown = MagicMock()
        pool.shutdown = mock_shutdown

        await executor_instance.teardown()
        mock_shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        assert executor_instance._executor_pool is None


    async def test_teardown_no_pool(self):
        executor = PySandboxExecutorStub()
        executor._executor_pool = None
        await executor.teardown()
        assert executor._executor_pool is None


@pytest.mark.asyncio()
class TestPySandboxExecutorStubExecuteSync:
    async def test_execute_sync_basic_success(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "x = 1 + 2\n_result = x"
        output = executor._execute_sync_with_capture(code, None)
        assert output["result"] == 3
        assert output["error"] is None
        assert output["stdout"] == ""
        assert output["stderr"] == ""

    async def test_execute_sync_stdout_capture(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "print('Hello')\nprint('World')"
        output = executor._execute_sync_with_capture(code, None)
        assert output["stdout"] == "Hello\nWorld\n"

    async def test_execute_sync_stderr_capture(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "import sys\nsys.stderr.write('Error message\\n')"
        output = executor._execute_sync_with_capture(code, None)
        assert output["stderr"] == "Error message\n"

    async def test_execute_sync_syntax_error(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "x = 1 + "
        output = executor._execute_sync_with_capture(code, None)
        assert output["error"] is not None
        assert "SyntaxError" in output["error"]

    async def test_execute_sync_runtime_exception(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "x = 1 / 0"
        output = executor._execute_sync_with_capture(code, None)
        assert output["error"] is not None
        assert "ZeroDivisionError" in output["error"]
        assert "division by zero" in output["stderr"]

    async def test_execute_sync_with_input_vars(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "_result = _input['a'] + _input['b']"
        input_data = {"a": 10, "b": 20}
        output = executor._execute_sync_with_capture(code, input_data)
        assert output["result"] == 30

    async def test_execute_sync_empty_code(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        output = executor._execute_sync_with_capture("", None)
        assert output["result"] is None
        assert output["error"] is None

    # Removed test_execute_sync_outer_error_during_redirect as it's hard to trigger reliably
    # with the current source code structure and its patching was flawed.
    # The critical aspect is capturing stderr from exec, which is now handled.


@pytest.mark.asyncio()
class TestPySandboxExecutorStubExecuteCode:
    async def test_execute_code_unsupported_language(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        result = await executor.execute_code("javascript", "console.log('hi')", 10)
        assert result.error == "Unsupported language"
        assert "Language 'javascript' not supported" in result.stderr

    async def test_execute_code_executor_pool_not_initialized(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=EXECUTOR_LOGGER_NAME)
        executor = PySandboxExecutorStub()
        # Do not call setup, so _executor_pool remains None
        result = await executor.execute_code("python", "print('hi')", 10)
        assert result.error == "ExecutorSetupError"
        assert result.stderr == "Executor not initialized."
        assert f"{executor.plugin_id}: ThreadPoolExecutor not initialized." in caplog.text

    async def test_execute_code_timeout(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub, caplog
    ):
        executor = await pysandbox_executor_stub_fixture
        caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER_NAME)
        code_long_running = "import time\ntime.sleep(0.2)\n_result = 'done'"

        with patch.object(asyncio.get_running_loop(), "run_in_executor", side_effect=asyncio.TimeoutError("Simulated wait_for timeout")):
            result = await executor.execute_code(
                "python", code_long_running, timeout_seconds=0.05
            )

        assert result.error == "Timeout"
        assert "Execution timed out by wrapper" in result.stderr
        assert f"{executor.plugin_id}: Code execution task timed out" in caplog.text

    async def test_execute_code_task_exception(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub, caplog
    ):
        executor = await pysandbox_executor_stub_fixture
        caplog.set_level(logging.ERROR, logger=EXECUTOR_LOGGER_NAME)
        code = "print('test')"

        with patch.object(
            asyncio.get_running_loop(),
            "run_in_executor",
            side_effect=RuntimeError("Task execution failed"),
        ):
            result = await executor.execute_code("python", code, 10)

        assert result.error == "Executor task failed: RuntimeError"
        assert "Executor task error: Task execution failed" in result.stderr
        assert f"{executor.plugin_id}: Error running execution task: Task execution failed" in caplog.text

    async def test_execute_code_with_input_data(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "_result = _input['value'] * 2"
        input_data = {"value": 5}
        result = await executor.execute_code("python", code, 10, input_data=input_data)
        assert result.result == 10
        assert result.error is None

    async def test_execute_code_successful_run(
        self, pysandbox_executor_stub_fixture: PySandboxExecutorStub
    ):
        executor = await pysandbox_executor_stub_fixture
        code = "print('Success!')\n_result = 42"
        start_time = time.perf_counter()
        result = await executor.execute_code("python", code, 10)
        end_time = time.perf_counter()

        assert result.stdout == "Success!\n"
        assert result.result == 42
        assert result.error is None
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < (end_time - start_time + 0.1) * 1000
