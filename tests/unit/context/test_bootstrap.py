from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.context.bootstrap import CqsEngineBootstrapPlugin
from genie_tooling.context.interface import ContextInterface


@pytest.mark.asyncio
async def test_bootstrap_attaches_context_interface(mock_plugin_manager_fixture):
    """
    Exercises the bootstrap plugin in isolation rather than the full Genie.create()
    pipeline. The bootstrap's only contract is: given a Genie with a config that
    has an 'context_engine' extension entry and a plugin manager, attach a
    ContextInterface as genie.context.
    """
    mock_genie = MagicMock()
    mock_genie._config = MiddlewareConfig(
        extension_configurations={
            "context_engine": {
                "context_source_plugin_id": "mock_context_source_v1",
                "context_inference_plugin_id": "mock_context_inference_v1",
                "rule_engine_plugin_id": "mock_rule_engine_v1",
            }
        }
    )
    # ContextManager now reaches for plugins via the public facade (genie.plugins).
    mock_genie.plugins = MagicMock()
    mock_genie.plugins.get_instance = AsyncMock(
        side_effect=lambda plugin_id, **kw: mock_plugin_manager_fixture._plugins.get(plugin_id)
    )
    mock_genie.conversation = MagicMock()
    mock_genie.conversation.load_state = AsyncMock(return_value={"history": []})

    await CqsEngineBootstrapPlugin().bootstrap(mock_genie)

    assert hasattr(mock_genie, "context"), "Bootstrap failed to attach 'context' attribute."
    assert isinstance(mock_genie.context, ContextInterface), "'context' attribute is not a ContextInterface instance."
