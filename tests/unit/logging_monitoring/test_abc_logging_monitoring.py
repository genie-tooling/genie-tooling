import logging
from typing import Any, Dict, Optional
import pytest
from genie_tooling.core.types import Plugin
from genie_tooling.log_adapters.abc import LogAdapter
from genie_tooling.redactors.abc import Redactor

class DefaultImplLogAdapter(LogAdapter, Plugin):
    plugin_id: str = "default_impl_log_adapter_v1"
    description: str = "A log adapter using only default implementations."
    # No need to override setup/teardown if they just call super()

class DefaultImplRedactor(Redactor, Plugin):
    plugin_id: str = "default_impl_redactor_v1"
    description: str = "A redactor using only default implementations."
    # No need to override setup/teardown if they just call super()

@pytest.fixture
def default_log_adapter_fixture() -> DefaultImplLogAdapter:
    return DefaultImplLogAdapter()

@pytest.fixture
def default_redactor_fixture() -> DefaultImplRedactor:
    return DefaultImplRedactor()

@pytest.mark.asyncio
async def test_log_adapter_default_setup(default_log_adapter_fixture: DefaultImplLogAdapter, caplog: pytest.LogCaptureFixture):
    default_log_adapter = default_log_adapter_fixture
    caplog.set_level(logging.WARNING)
    await default_log_adapter.setup(config={})
    assert len(caplog.records) == 0 # Default implementation should not log

@pytest.mark.asyncio
async def test_log_adapter_default_process_event(default_log_adapter_fixture: DefaultImplLogAdapter, caplog: pytest.LogCaptureFixture):
    default_log_adapter = default_log_adapter_fixture
    caplog.set_level(logging.WARNING)
    await default_log_adapter.process_event(event_type="test_event", data={"key": "value"})
    assert len(caplog.records) == 0 # Default implementation should not log
