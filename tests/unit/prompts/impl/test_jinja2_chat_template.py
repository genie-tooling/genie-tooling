import logging
from unittest.mock import patch

import pytest
from genie_tooling.prompts.impl.jinja2_chat_template import (
    Jinja2ChatTemplatePlugin,
)
from genie_tooling.prompts.types import PromptData

TEMPLATE_LOGGER_NAME = "genie_tooling.prompts.impl.jinja2_chat_template"

@pytest.fixture
async def jinja_template_plugin() -> Jinja2ChatTemplatePlugin:
    plugin = Jinja2ChatTemplatePlugin()
    await plugin.setup()
    return plugin

@pytest.mark.asyncio
async def test_render_string_success(jinja_template_plugin: Jinja2ChatTemplatePlugin):
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    template = "Hello, {{ name }}! Count: {{ items | length }}"
    data: PromptData = {"name": "JinjaUser", "items": [1, 2, 3]}
    rendered = await plugin.render(template, data)
    assert rendered == "Hello, JinjaUser! Count: 3"

@pytest.mark.asyncio
async def test_render_string_template_syntax_error(jinja_template_plugin: Jinja2ChatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    template = "Hello, {{ name }! {% invalid_tag %}"
    data: PromptData = {"name": "Test"}
    rendered = await plugin.render(template, data)
    assert "Error rendering template:" in rendered
    assert "Error rendering Jinja2 template (for string output)" in caplog.text

@pytest.mark.asyncio
async def test_render_chat_messages_success(jinja_template_plugin: Jinja2ChatTemplatePlugin):
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    # Template produces a JSON string representing chat messages
    template_content = """
    [
        {"role": "system", "content": "You are {{ assistant_name }}."},
        {"role": "user", "content": "My query is: {{ user_query }}"}
        {% if history %}
        ,
        {% for msg in history %}
            {"role": "{{ msg.role }}", "content": "{{ msg.content }}"}
            {{ "," if not loop.last }}
        {% endfor %}
        {% endif %}
    ]
    """
    data: PromptData = {
        "assistant_name": "HelpfulBot",
        "user_query": "How does Jinja work?",
        "history": [{"role": "user", "content": "Previous question"}]
    }
    chat_messages = await plugin.render_chat_messages(template_content, data)
    assert len(chat_messages) == 3
    assert chat_messages[0]["role"] == "system"
    assert chat_messages[0]["content"] == "You are HelpfulBot."
    assert chat_messages[1]["role"] == "user"
    assert chat_messages[1]["content"] == "My query is: How does Jinja work?"
    assert chat_messages[2]["role"] == "user"
    assert chat_messages[2]["content"] == "Previous question"

@pytest.mark.asyncio
async def test_render_chat_messages_invalid_json_output(jinja_template_plugin: Jinja2ChatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    template_content = "This is not JSON, it's just text."
    data: PromptData = {}
    chat_messages = await plugin.render_chat_messages(template_content, data)
    assert len(chat_messages) == 1
    assert chat_messages[0]["role"] == "user" # Fallback message
    assert "Error: Template output is not valid JSON." in chat_messages[0]["content"]
    assert "Failed to parse rendered Jinja2 output as JSON for chat messages" in caplog.text

@pytest.mark.asyncio
async def test_render_chat_messages_json_not_a_list(jinja_template_plugin: Jinja2ChatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    template_content = '{"role": "user", "content": "This is a dict, not a list"}' # Valid JSON, but not a list
    data: PromptData = {}
    chat_messages = await plugin.render_chat_messages(template_content, data)
    assert len(chat_messages) == 1
    assert "Error: Template did not produce a list of messages." in chat_messages[0]["content"]
    assert "Rendered Jinja2 template for chat did not produce a JSON list." in caplog.text

@pytest.mark.asyncio
async def test_render_chat_messages_invalid_message_structure_in_list(jinja_template_plugin: Jinja2ChatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=TEMPLATE_LOGGER_NAME)
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    template_content = '[{"role": "user", "content": "Valid"}, {"not_a_role": "bad"}]'
    data: PromptData = {}
    chat_messages = await plugin.render_chat_messages(template_content, data)
    assert len(chat_messages) == 1 # Only the valid message
    assert chat_messages[0]["content"] == "Valid"
    assert "Invalid message structure in rendered chat JSON" in caplog.text

@pytest.mark.asyncio
async def test_jinja2_not_available(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    with patch("genie_tooling.prompts.impl.jinja2_chat_template.JINJA2_AVAILABLE", False):
        plugin_no_jinja = Jinja2ChatTemplatePlugin()
        await plugin_no_jinja.setup()
        assert plugin_no_jinja._env is None
        assert "Jinja2 library not installed. This plugin will not function." in caplog.text

        # Test render methods when Jinja2 is not available
        rendered_str = await plugin_no_jinja.render("template", {})
        assert rendered_str == "template" # Fallback
        assert "Jinja2 environment not initialized." in caplog.text

        chat_msgs = await plugin_no_jinja.render_chat_messages("template", {})
        assert chat_msgs[0]["content"] == "Error: Jinja2 environment not ready."
        assert "Jinja2 environment not initialized for chat messages." in caplog.text

@pytest.mark.asyncio
async def test_teardown(jinja_template_plugin: Jinja2ChatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TEMPLATE_LOGGER_NAME)
    plugin = await jinja_template_plugin
    if not plugin._env: pytest.skip("Jinja2 not available or plugin setup failed")
    await plugin.teardown()
    assert plugin._env is None
    assert f"{plugin.plugin_id}: Teardown complete." in caplog.text
