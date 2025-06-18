# tests/unit/test_bootstrap.py
import logging
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pytest
from genie_tooling.bootstrap.abc import BootstrapPlugin
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.genie import Genie
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider


# --- Mocks for Bootstrap Testing ---
class MockSuccessBootstrap(BootstrapPlugin):
    plugin_id: str = "success_bootstrap_v1"
    description: str = "A mock bootstrap plugin that succeeds."
    bootstrap = AsyncMock()

class MockFailBootstrap(BootstrapPlugin):
    plugin_id: str = "fail_bootstrap_v1"
    description: str = "A mock bootstrap plugin that fails."
    async def bootstrap(self, genie: "Genie") -> None:
        raise RuntimeError("Bootstrap task failed deliberately.")

@pytest.fixture
def mock_plugin_manager_for_bootstrap(mocker) -> MagicMock:
    """Provides a flexible mock PluginManager for bootstrap tests."""
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_all_plugin_instances_by_type = AsyncMock(return_value=[])
    pm.get_plugin_instance = AsyncMock(return_value=None)
    pm.discover_plugins = AsyncMock()
    return pm


@pytest.mark.asyncio
# Patch the RAGManager directly where it's used
@patch("genie_tooling.genie.RAGManager")
class TestBootstrapMechanism:

    async def test_genie_create_runs_bootstrap_plugin(
        self, MockRAGManager, mock_plugin_manager_for_bootstrap: MagicMock
    ):
        """
        Verify that a registered BootstrapPlugin's `bootstrap` method is called
        during `Genie.create()` and that it receives a valid Genie instance.
        """
        # --- FIX: Ensure the instance returned by the patched class is an AsyncMock ---
        MockRAGManager.return_value = AsyncMock(spec=RAGManager)
        
        # 1. Setup
        mock_success_plugin = MockSuccessBootstrap()
        mock_plugin_manager_for_bootstrap.get_all_plugin_instances_by_type.return_value = [
            mock_success_plugin
        ]

        mock_kp_instance = AsyncMock(spec=KeyProvider)
        mock_log_adapter_instance = AsyncMock(spec=LogAdapterPlugin)
        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == "environment_key_provider_v1":
                return mock_kp_instance
            if plugin_id == "default_log_adapter_v1":
                return mock_log_adapter_instance
            return AsyncMock()
        mock_plugin_manager_for_bootstrap.get_plugin_instance.side_effect = get_instance_side_effect

        test_config = MiddlewareConfig(
            features=FeatureSettings(
                rag_embedder="sentence_transformer",
                rag_vector_store="faiss"
            )
        )

        # 2. Action
        genie = await Genie.create(config=test_config, plugin_manager=mock_plugin_manager_for_bootstrap)

        # 3. Assertions
        mock_plugin_manager_for_bootstrap.get_all_plugin_instances_by_type.assert_awaited_once_with(BootstrapPlugin)
        mock_success_plugin.bootstrap.assert_awaited_once()
        call_args, _ = mock_success_plugin.bootstrap.call_args
        assert call_args[0] is genie

        # Verify that other components were available to the bootstrap plugin
        await call_args[0].rag.index_directory(path="test")
        # Assert that the mocked RAGManager instance's method was called
        MockRAGManager.return_value.index_data_source.assert_awaited_once_with(
            loader_id=ANY,
            loader_source_uri='test',
            splitter_id=ANY,
            embedder_id=ANY,
            vector_store_id=ANY,
            loader_config=ANY,
            splitter_config=ANY,
            embedder_config=ANY,
            vector_store_config=ANY
        )

    async def test_genie_create_succeeds_with_no_bootstrap_plugins(
        self, MockRAGManager, mock_plugin_manager_for_bootstrap: MagicMock
    ):
        mock_plugin_manager_for_bootstrap.get_all_plugin_instances_by_type.return_value = []
        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == "environment_key_provider_v1":
                return AsyncMock(spec=KeyProvider)
            if plugin_id == "default_log_adapter_v1":
                return AsyncMock(spec=LogAdapterPlugin)
            return AsyncMock()
        mock_plugin_manager_for_bootstrap.get_plugin_instance.side_effect = get_instance_side_effect

        try:
            await Genie.create(config=MiddlewareConfig(), plugin_manager=mock_plugin_manager_for_bootstrap)
        except Exception as e:
            pytest.fail(f"Genie.create() raised an unexpected exception with no bootstrap plugins: {e}")

        mock_plugin_manager_for_bootstrap.get_all_plugin_instances_by_type.assert_awaited_once_with(BootstrapPlugin)


    async def test_genie_create_fails_if_bootstrap_plugin_raises_error(
        self, MockRAGManager, mock_plugin_manager_for_bootstrap: MagicMock, caplog
    ):
        caplog.set_level(logging.ERROR, logger="genie_tooling.genie")
        mock_fail_plugin = MockFailBootstrap()
        mock_plugin_manager_for_bootstrap.get_all_plugin_instances_by_type.return_value = [
            mock_fail_plugin
        ]
        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == "environment_key_provider_v1":
                return AsyncMock(spec=KeyProvider)
            if plugin_id == "default_log_adapter_v1":
                return AsyncMock(spec=LogAdapterPlugin)
            return AsyncMock()
        mock_plugin_manager_for_bootstrap.get_plugin_instance.side_effect = get_instance_side_effect

        with pytest.raises(RuntimeError, match="Bootstrap task failed deliberately."):
            await Genie.create(config=MiddlewareConfig(), plugin_manager=mock_plugin_manager_for_bootstrap)

        assert "Error running bootstrap plugin 'fail_bootstrap_v1'" in caplog.text