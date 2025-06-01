"""Unit tests for the default implementations in logging_monitoring.abc."""
import logging
from typing import Any, Dict, Optional

import pytest

from genie_tooling.core.types import Plugin  # For concrete implementation

# Updated import path for LogAdapter
from genie_tooling.log_adapters.abc import LogAdapter

# Updated import path for Redactor
from genie_tooling.redactors.abc import Redactor


# Minimal concrete implementation of LogAdapter
class DefaultImplLogAdapter(LogAdapter, Plugin):
    plugin_id: str = "default_impl_log_adapter_v1"
    description: str = "A log adapter using only default implementations."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config)

    async def teardown(self) -> None:
        await super().teardown()

# Minimal concrete implementation of Redactor
class DefaultImplRedactor(Redactor, Plugin):
    plugin_id: str = "default_impl_redactor_v1"
    description: str = "A redactor using only default implementations."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config)

    async def teardown(self) -> None:
        await super().teardown()

@pytest.fixture
async def default_log_adapter_fixture() -> DefaultImplLogAdapter: # Renamed for clarity
    adapter = DefaultImplLogAdapter()
    await adapter.setup()
    return adapter

@pytest.fixture
async def default_redactor_fixture() -> DefaultImplRedactor: # Renamed for clarity
    redactor = DefaultImplRedactor()
    await redactor.setup()
    return redactor

@pytest.mark.asyncio
async def test_log_adapter_default_setup_logging(default_log_adapter_fixture: DefaultImplLogAdapter, caplog: pytest.LogCaptureFixture):
    """Test default LogAdapter.setup_logging() logs warning."""
    default_log_adapter = await default_log_adapter_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    await default_log_adapter.setup_logging(config={}) # Should not raise error

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"LogAdapter '{default_log_adapter.plugin_id}' setup_logging method not fully implemented." in caplog.text

@pytest.mark.asyncio
async def test_log_adapter_default_process_event(default_log_adapter_fixture: DefaultImplLogAdapter, caplog: pytest.LogCaptureFixture):
    """Test default LogAdapter.process_event() logs warning."""
    default_log_adapter = await default_log_adapter_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    await default_log_adapter.process_event(event_type="test_event", data={"key": "value"}) # Should not raise error

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"LogAdapter '{default_log_adapter.plugin_id}' process_event method not fully implemented." in caplog.text

@pytest.mark.asyncio # Redactor.sanitize is sync, but fixture is async
async def test_redactor_default_sanitize(default_redactor_fixture: DefaultImplRedactor, caplog: pytest.LogCaptureFixture):
    """Test default Redactor.sanitize() logs warning and returns data as is."""
    default_redactor = await default_redactor_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    original_data = {"sensitive": "info", "normal": 123}
    sanitized_data = default_redactor.sanitize(original_data)

    assert sanitized_data == original_data # Default implementation returns data as is
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"Redactor '{default_redactor.plugin_id}' sanitize method not fully implemented. Returning data as is." in caplog.text
