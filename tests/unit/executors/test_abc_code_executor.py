"""Unit tests for the default implementations in executors.abc.CodeExecutor."""
import logging
from typing import Any, Dict, List, Optional

import pytest

# Updated import paths for CodeExecutionResult and CodeExecutor
from genie_tooling.code_executors.abc import CodeExecutionResult, CodeExecutor
from genie_tooling.core.types import Plugin  # For concrete implementation


# A minimal concrete implementation of CodeExecutor for testing defaults
class DefaultImplCodeExecutor(CodeExecutor, Plugin):
    plugin_id: str = "default_impl_executor_v1"
    executor_id: str = "default_impl_executor_v1_instance"
    description: str = "A code executor using only default implementations."
    supported_languages: List[str] = ["python_default_test"]

    # setup and teardown are part of Plugin
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config)

    async def teardown(self) -> None:
        await super().teardown()

@pytest.fixture
async def default_code_executor_fixture() -> DefaultImplCodeExecutor: # Renamed for clarity
    executor = DefaultImplCodeExecutor()
    await executor.setup()
    return executor

@pytest.mark.asyncio
async def test_code_executor_default_execute_code(default_code_executor_fixture: DefaultImplCodeExecutor, caplog: pytest.LogCaptureFixture):
    """Test default CodeExecutor.execute_code() logs warning and returns default result."""
    default_code_executor = await default_code_executor_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    language = "python_test"
    code = "logger.debug('hello')"
    timeout_seconds = 10

    result = await default_code_executor.execute_code(language, code, timeout_seconds)

    assert isinstance(result, CodeExecutionResult)
    assert result.stdout == ""
    assert result.stderr == "Not implemented"
    assert result.result is None
    assert result.error == "Executor not implemented"
    assert result.execution_time_ms == 0.0

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"CodeExecutor '{default_code_executor.plugin_id}' execute_code method not fully implemented." in caplog.text
