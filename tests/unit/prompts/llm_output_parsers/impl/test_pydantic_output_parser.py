import logging
from typing import List, Optional
from unittest.mock import patch

import pytest
from genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser import (
    PYDANTIC_AVAILABLE,
    PydanticOutputParserPlugin,
)

# Mock Pydantic BaseModel if not available, or use real one
if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field, ValidationError
else:
    # Minimal mock for BaseModel if pydantic is not installed
    class BaseModel: # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        @classmethod
        def model_validate_json(cls, json_str: str): # Mock for Pydantic v2
            import json
            data = json.loads(json_str)
            return cls(**data)
        def model_dump(self): # Mock for Pydantic v2
            return self.__dict__

    class Field: # type: ignore
        def __init__(self, default=None, **kwargs): pass
    
    class ValidationError(ValueError): # type: ignore
        def __init__(self, errors):
            super().__init__(str(errors))
            self._errors = errors
        def errors(self): return self._errors


PARSER_LOGGER_NAME = "genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser"

class MyTestModel(BaseModel):
    name: str
    age: int
    tags: Optional[List[str]] = Field(default_factory=list)

@pytest.fixture
async def pydantic_parser() -> PydanticOutputParserPlugin:
    parser = PydanticOutputParserPlugin()
    await parser.setup()
    return parser

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio
async def test_parse_valid_json_to_pydantic_model(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    text = '```json\n{"name": "Alice", "age": 30, "tags": ["test", "user"]}\n```'
    parsed_model = parser.parse(text, schema=MyTestModel)
    assert isinstance(parsed_model, MyTestModel)
    assert parsed_model.name == "Alice"
    assert parsed_model.age == 30
    assert parsed_model.tags == ["test", "user"]

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio
async def test_parse_pydantic_validation_error(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    text = '{"name": "Bob", "age": "thirty"}' # Age is string, not int
    with pytest.raises(ValueError, match="Pydantic validation failed"):
        parser.parse(text, schema=MyTestModel)
    assert "Pydantic validation failed for model 'MyTestModel'" in caplog.text

@pytest.mark.asyncio
async def test_parse_no_pydantic_schema_provided(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    if not PYDANTIC_AVAILABLE: pytest.skip("Pydantic not available")
    text = '{"data": "any"}'
    with pytest.raises(ValueError, match="A Pydantic model class must be provided as the 'schema' argument."):
        parser.parse(text, schema=None) # type: ignore
    with pytest.raises(ValueError, match="A Pydantic model class must be provided as the 'schema' argument."):
        parser.parse(text, schema=dict) # type: ignore

@pytest.mark.asyncio
async def test_parse_no_json_block_found(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    if not PYDANTIC_AVAILABLE: pytest.skip("Pydantic not available")
    text = "This is plain text."
    with pytest.raises(ValueError, match="No parsable JSON block found in the input text for Pydantic parsing."):
        parser.parse(text, schema=MyTestModel)
    assert "No valid JSON block found in text_output" in caplog.text

@pytest.mark.asyncio
async def test_pydantic_not_available_runtime_error(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PARSER_LOGGER_NAME)
    with patch("genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser.PYDANTIC_AVAILABLE", False):
        parser_no_pydantic = PydanticOutputParserPlugin()
        await parser_no_pydantic.setup() # Logs error during setup
        assert "Pydantic library not installed. This plugin will not function." in caplog.text
        
        with pytest.raises(RuntimeError, match="Pydantic library not available at runtime."):
            parser_no_pydantic.parse("text", schema=MyTestModel) # type: ignore

@pytest.mark.asyncio
async def test_teardown(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    await parser.teardown()
    assert f"{parser.plugin_id}: Teardown complete." in caplog.text
