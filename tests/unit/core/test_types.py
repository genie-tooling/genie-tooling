"""Unit tests for default behaviors in core.types, specifically the Plugin protocol."""

import pytest

from genie_tooling.core.types import Plugin


class ConcreteTestPlugin(Plugin):
    plugin_id: str = "concrete_test_plugin_v1"

@pytest.fixture
async def concrete_plugin_fixture() -> ConcreteTestPlugin:
    return ConcreteTestPlugin()

@pytest.mark.asyncio
async def test_plugin_protocol_default_setup(concrete_plugin_fixture: ConcreteTestPlugin):
    concrete_plugin = await concrete_plugin_fixture
    try:
        await concrete_plugin.setup(config={"test_key": "test_value"})
        await concrete_plugin.setup()
    except Exception as e:
        pytest.fail(f"Default Plugin.setup() raised an unexpected exception: {e}")

@pytest.mark.asyncio
async def test_plugin_protocol_default_teardown(concrete_plugin_fixture: ConcreteTestPlugin):
    concrete_plugin = await concrete_plugin_fixture
    try:
        await concrete_plugin.teardown()
    except Exception as e:
        pytest.fail(f"Default Plugin.teardown() raised an unexpected exception: {e}")

def test_plugin_type_variable():
    from genie_tooling.core.types import PluginType

    def takes_plugin_type(p: PluginType) -> PluginType:
        return p

    plugin_instance = ConcreteTestPlugin()
    returned_plugin = takes_plugin_type(plugin_instance)
    assert returned_plugin is plugin_instance
    assert True
