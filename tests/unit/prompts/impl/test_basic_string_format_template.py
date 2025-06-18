### tests/unit/prompts/impl/test_basic_string_format_template.py
import logging
from typing import Any  

import pytest
from genie_tooling.prompts.impl.basic_string_format_template import (
    BasicStringFormatTemplatePlugin,
)
from genie_tooling.prompts.types import PromptData

TEMPLATE_LOGGER_NAME = "genie_tooling.prompts.impl.basic_string_format_template"

@pytest.fixture()
async def string_template_plugin() -> BasicStringFormatTemplatePlugin:
    plugin = BasicStringFormatTemplatePlugin()
    await plugin.setup()
    return plugin

@pytest.mark.asyncio()
async def test_render_success(string_template_plugin: BasicStringFormatTemplatePlugin):
    plugin = await string_template_plugin
    template = "Hello, {name}! You are {age} years old."
    data: PromptData = {"name": "Alice", "age": 30}
    rendered = await plugin.render(template, data)
    assert rendered == "Hello, Alice! You are 30 years old."

@pytest.mark.asyncio()
async def test_render_missing_key(string_template_plugin: BasicStringFormatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    plugin = await string_template_plugin
    template = "Value: {value}"
    data: PromptData = {"other_key": "data"} # 'value' is missing
    rendered = await plugin.render(template, data)
    assert rendered == template # Fallback to original template
    assert any(
        "Missing key ''value'' in data for template." in rec.message  # Changed 'value' to ''value''
        and rec.name == TEMPLATE_LOGGER_NAME
        for rec in caplog.records
    )

@pytest.mark.asyncio()
async def test_render_data_not_dict(string_template_plugin: BasicStringFormatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=TEMPLATE_LOGGER_NAME)
    plugin = await string_template_plugin
    template = "Template: {placeholder}"
    data_list: Any = ["this", "is", "a", "list"] # Not a dict
    rendered = await plugin.render(template, data_list)
    # Should use empty dict, so placeholder remains
    assert rendered == template
    assert any(
        "Data for rendering is not a dictionary (type: <class 'list'>). Using empty dict." in rec.message
        and rec.name == TEMPLATE_LOGGER_NAME
        for rec in caplog.records
    )


@pytest.mark.asyncio()
async def test_render_chat_messages_success(string_template_plugin: BasicStringFormatTemplatePlugin):
    plugin = await string_template_plugin
    template = "User query: {query_text}"
    data: PromptData = {"query_text": "What is the weather?"}
    chat_messages = await plugin.render_chat_messages(template, data)
    assert len(chat_messages) == 1
    assert chat_messages[0]["role"] == "user"
    assert chat_messages[0]["content"] == "User query: What is the weather?"

@pytest.mark.asyncio()
async def test_render_chat_messages_render_fails(string_template_plugin: BasicStringFormatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TEMPLATE_LOGGER_NAME)
    plugin = await string_template_plugin
    template = "Problematic: {missing_key}"
    data: PromptData = {}
    chat_messages = await plugin.render_chat_messages(template, data)
    assert len(chat_messages) == 1
    assert chat_messages[0]["role"] == "user"
    assert chat_messages[0]["content"] == template # Fallback content
    assert any(
        "Missing key ''missing_key'' in data for template." in rec.message  # Changed 'missing_key' to ''missing_key''
        and rec.name == TEMPLATE_LOGGER_NAME
        for rec in caplog.records
    )

@pytest.mark.asyncio()
async def test_teardown(string_template_plugin: BasicStringFormatTemplatePlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TEMPLATE_LOGGER_NAME)
    plugin = await string_template_plugin
    await plugin.teardown()
    assert any(
        f"{plugin.plugin_id}: Teardown complete." in rec.message
        and rec.name == TEMPLATE_LOGGER_NAME
        for rec in caplog.records
    )
