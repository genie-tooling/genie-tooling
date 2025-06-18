from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.manager import (
    LLMOutputParserManager,
)


@pytest.fixture()
def mock_plugin_manager_for_parser_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture()
def mock_llm_output_parser() -> MagicMock:
    parser_plugin = MagicMock(spec=LLMOutputParserPlugin)
    # parse is sync, so use MagicMock, not AsyncMock
    parser_plugin.parse = MagicMock(return_value={"parsed": "data"})
    return parser_plugin

@pytest.fixture()
def output_parser_manager(
    mock_plugin_manager_for_parser_mgr: MagicMock,
    mock_llm_output_parser: MagicMock
) -> LLMOutputParserManager:
    mock_plugin_manager_for_parser_mgr.get_plugin_instance.return_value = mock_llm_output_parser
    return LLMOutputParserManager(
        plugin_manager=mock_plugin_manager_for_parser_mgr,
        default_parser_id="default_parser"
    )

@pytest.mark.asyncio()
async def test_parse_success_with_default_parser(output_parser_manager: LLMOutputParserManager, mock_llm_output_parser: MagicMock):
    text_output = '{"raw": "output"}'
    schema = {"type": "object"}

    parsed_data = await output_parser_manager.parse(text_output, schema=schema)

    assert parsed_data == {"parsed": "data"}
    mock_llm_output_parser.parse.assert_called_once_with(text_output, schema)
    output_parser_manager._plugin_manager.get_plugin_instance.assert_awaited_once_with( # type: ignore
        "default_parser", config={}
    )

@pytest.mark.asyncio()
async def test_parse_success_with_specified_parser_id(output_parser_manager: LLMOutputParserManager, mock_llm_output_parser: MagicMock):
    text_output = "some text"
    parser_id = "custom_parser"

    await output_parser_manager.parse(text_output, parser_id=parser_id)

    output_parser_manager._plugin_manager.get_plugin_instance.assert_awaited_once_with( # type: ignore
        parser_id, config={}
    )
    mock_llm_output_parser.parse.assert_called_once_with(text_output, None)


@pytest.mark.asyncio()
async def test_parse_no_parser_id_and_no_default(output_parser_manager: LLMOutputParserManager, mock_plugin_manager_for_parser_mgr: MagicMock):
    output_parser_manager._default_parser_id = None 

    text_output = "data"
    # If no parser can be identified, it should return the raw text
    result = await output_parser_manager.parse(text_output)
    assert result == text_output
    mock_plugin_manager_for_parser_mgr.get_plugin_instance.assert_not_called()

@pytest.mark.asyncio()
async def test_parse_parser_not_found(output_parser_manager: LLMOutputParserManager, mock_plugin_manager_for_parser_mgr: MagicMock):
    mock_plugin_manager_for_parser_mgr.get_plugin_instance.return_value = None # Simulate parser not found

    text_output = "data to parse"
    # If specified parser not found, should return raw text
    result = await output_parser_manager.parse(text_output, parser_id="non_existent_parser")
    assert result == text_output

@pytest.mark.asyncio()
async def test_parse_parser_raises_exception(output_parser_manager: LLMOutputParserManager, mock_llm_output_parser: MagicMock):
    mock_llm_output_parser.parse.side_effect = ValueError("Parsing failed badly")

    with pytest.raises(ValueError, match="Parsing failed badly"):
        await output_parser_manager.parse("input text")

@pytest.mark.asyncio()
async def test_teardown(output_parser_manager: LLMOutputParserManager):
    # Teardown is currently a no-op for LLMOutputParserManager itself
    await output_parser_manager.teardown()
    assert True # Just ensure it runs without error
