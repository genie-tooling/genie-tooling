### tests/unit/prompts/test_prompt_manager.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.abc import PromptRegistryPlugin, PromptTemplatePlugin
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.prompts.types import PromptData, PromptIdentifier


@pytest.fixture()
def mock_plugin_manager_for_prompts() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture()
def mock_prompt_registry() -> MagicMock:
    registry = AsyncMock(spec=PromptRegistryPlugin)
    registry.get_template_content = AsyncMock(return_value="Template: {{var}}")
    registry.list_available_templates = AsyncMock(
        return_value=[
            PromptIdentifier(name="test_prompt", version="v1", description="A test prompt")
        ]
    )
    return registry


@pytest.fixture()
def mock_template_engine() -> MagicMock:
    engine = AsyncMock(spec=PromptTemplatePlugin)
    engine.render = AsyncMock(return_value="Rendered: Value")
    engine.render_chat_messages = AsyncMock(
        return_value=[{"role": "user", "content": "Rendered Chat: Value"}]
    )
    return engine


@pytest.fixture()
def mock_tracing_manager_for_prompts() -> MagicMock:
    """Provides a mock tracing manager."""
    tm = MagicMock(spec=InteractionTracingManager)
    tm.trace_event = AsyncMock()
    return tm


@pytest.fixture()
def prompt_manager(
    mock_plugin_manager_for_prompts: MagicMock,
    mock_prompt_registry: MagicMock,
    mock_template_engine: MagicMock,
    mock_tracing_manager_for_prompts: MagicMock,
) -> PromptManager:
    # Configure side_effect for get_plugin_instance
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "default_registry":
            return mock_prompt_registry
        if plugin_id == "default_engine":
            return mock_template_engine
        return None

    mock_plugin_manager_for_prompts.get_plugin_instance.side_effect = (
        get_instance_side_effect
    )

    return PromptManager(
        plugin_manager=mock_plugin_manager_for_prompts,
        default_registry_id="default_registry",
        default_template_engine_id="default_engine",
        tracing_manager=mock_tracing_manager_for_prompts,
    )


@pytest.mark.asyncio()
async def test_get_raw_template_success(
    prompt_manager: PromptManager, mock_prompt_registry: MagicMock
):
    content = await prompt_manager.get_raw_template("my_prompt", "v1")
    assert content == "Template: {{var}}"
    mock_prompt_registry.get_template_content.assert_awaited_once_with(
        "my_prompt", "v1"
    )


@pytest.mark.asyncio()
async def test_get_raw_template_registry_not_found(
    prompt_manager: PromptManager, mock_plugin_manager_for_prompts: MagicMock
):
    mock_plugin_manager_for_prompts.get_plugin_instance.return_value = (
        None  # Simulate registry not found
    )
    content = await prompt_manager.get_raw_template("my_prompt", registry_id="bad_registry")
    assert content is None


@pytest.mark.asyncio()
async def test_get_raw_template_plugin_raises_error(
    prompt_manager: PromptManager, mock_prompt_registry: MagicMock
):
    """Test that if the registry plugin fails, the error is handled and returns None."""
    mock_prompt_registry.get_template_content.side_effect = RuntimeError("Registry DB down")
    content = await prompt_manager.get_raw_template("any_prompt")
    assert content is None
    # Check that a trace event for the error was fired
    prompt_manager._tracing_manager.trace_event.assert_awaited_with(  # type: ignore
        "log.error",
        {
            "message": "Error getting raw template 'any_prompt' from registry 'default_registry': Registry DB down",
            "exc_info": True,
        },
        "PromptManager",
    )


@pytest.mark.asyncio()
async def test_render_prompt_success(
    prompt_manager: PromptManager, mock_template_engine: MagicMock
):
    data: PromptData = {"var": "Value"}
    rendered = await prompt_manager.render_prompt("my_prompt", data)
    assert rendered == "Rendered: Value"
    mock_template_engine.render.assert_awaited_once_with("Template: {{var}}", data)


@pytest.mark.asyncio()
async def test_render_prompt_template_content_not_found(
    prompt_manager: PromptManager, mock_prompt_registry: MagicMock
):
    mock_prompt_registry.get_template_content.return_value = (
        None  # Simulate template not found
    )
    rendered = await prompt_manager.render_prompt("my_prompt", {})
    assert rendered is None


@pytest.mark.asyncio()
async def test_render_prompt_engine_not_found(
    prompt_manager: PromptManager, mock_plugin_manager_for_prompts: MagicMock
):
    # Make only the engine fail to load for this specific call
    original_side_effect = mock_plugin_manager_for_prompts.get_plugin_instance.side_effect
    async def side_effect_engine_none(plugin_id, config=None):
        if plugin_id == "bad_engine":
            return None
        return await original_side_effect(plugin_id, config)  # type: ignore

    mock_plugin_manager_for_prompts.get_plugin_instance.side_effect = (
        side_effect_engine_none
    )

    rendered = await prompt_manager.render_prompt(
        "my_prompt", {}, template_engine_id="bad_engine"
    )
    assert rendered is None
    # Restore original side_effect if other tests depend on it
    mock_plugin_manager_for_prompts.get_plugin_instance.side_effect = (
        original_side_effect
    )


@pytest.mark.asyncio()
async def test_render_prompt_engine_raises_error(
    prompt_manager: PromptManager, mock_template_engine: MagicMock
):
    """Test that if the template engine's render method fails, it's handled."""
    mock_template_engine.render.side_effect = ValueError("Template rendering crashed")
    rendered = await prompt_manager.render_prompt("my_prompt", {})
    assert rendered is None
    prompt_manager._tracing_manager.trace_event.assert_awaited_with(  # type: ignore
        "log.error",
        {
            "message": "Error rendering prompt with engine 'default_engine': Template rendering crashed",
            "exc_info": True,
        },
        "PromptManager",
    )


@pytest.mark.asyncio()
async def test_render_chat_prompt_success(
    prompt_manager: PromptManager, mock_template_engine: MagicMock
):
    data: PromptData = {"var": "Value"}
    chat_messages = await prompt_manager.render_chat_prompt("my_prompt", data)
    assert chat_messages == [{"role": "user", "content": "Rendered Chat: Value"}]
    mock_template_engine.render_chat_messages.assert_awaited_once_with(
        "Template: {{var}}", data
    )


@pytest.mark.asyncio()
async def test_list_available_templates_success(
    prompt_manager: PromptManager, mock_prompt_registry: MagicMock
):
    templates = await prompt_manager.list_available_templates()
    assert len(templates) == 1
    assert templates[0]["name"] == "test_prompt"
    mock_prompt_registry.list_available_templates.assert_awaited_once()


@pytest.mark.asyncio()
async def test_list_available_templates_plugin_fails(
    prompt_manager: PromptManager, mock_prompt_registry: MagicMock
):
    """Test handling of an error during the list_available_templates call."""
    mock_prompt_registry.list_available_templates.side_effect = Exception(
        "Registry connection lost"
    )
    results = await prompt_manager.list_available_templates()
    assert results == []
    prompt_manager._tracing_manager.trace_event.assert_awaited_with(  # type: ignore
        "log.error",
        {
            "message": "Error listing templates from registry 'default_registry': Registry connection lost",
            "exc_info": True,
        },
        "PromptManager",
    )


@pytest.mark.asyncio()
async def test_render_no_default_engine_configured(
    mock_plugin_manager_for_prompts: MagicMock,
    mock_tracing_manager_for_prompts: MagicMock,
):
    """Test rendering when no default engine is set and no override is given."""
    manager = PromptManager(
        plugin_manager=mock_plugin_manager_for_prompts,
        default_registry_id=None,
        default_template_engine_id=None,  # No default engine
        tracing_manager=mock_tracing_manager_for_prompts,
    )
    # The default fallback is basic string formatting
    result = await manager.render_prompt(
        template_content="Hello {name}", data={"name": "World"}
    )
    assert result == "Hello World"

    # Test error fallback
    result_err = await manager.render_prompt(
        template_content="Hello {missing}", data={}
    )
    assert result_err == "Hello {missing}"  # Returns original on error

    # Test chat fallback
    chat_result = await manager.render_chat_prompt(template_content="Hello", data={})
    assert chat_result == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio()
async def test_teardown(prompt_manager: PromptManager):
    # Teardown is currently a no-op for PromptManager itself
    await prompt_manager.teardown()
    assert True  # Just ensure it runs without error
