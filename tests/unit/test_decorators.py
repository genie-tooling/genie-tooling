from typing import Any, Dict, ForwardRef, List, Optional, Union

import pytest

# Assuming decorators.py is in src.genie_tooling, adjust if necessary
from genie_tooling.decorators import (
    _map_type_to_json_schema,
    _parse_docstring_for_params,
    tool,
)

# --- Test Helper Functions from decorators.py ---

def test_parse_docstring_simple():
    doc = """
    Does a thing.
    Args:
        param1 (str): The first parameter.
        param2 (int): The second one.
    """
    expected = {"param1": "The first parameter.", "param2": "The second one."}
    assert _parse_docstring_for_params(doc) == expected

def test_parse_docstring_no_args_section_keyword():
    doc = """
    This is a description.
    param1 (str): This won't be parsed as it's not under 'Args:'.
    """
    assert _parse_docstring_for_params(doc) == {}

def test_parse_docstring_different_args_keywords():
    doc_variations = [
        "Parameters:\n    key (str): Value.",
        "Arguments:\n    key (str): Value.",
    ]
    for doc_var in doc_variations:
        assert _parse_docstring_for_params(doc_var) == {"key": "Value."}


def test_parse_docstring_no_args():
    doc = "Just a description."
    assert _parse_docstring_for_params(doc) == {}

def test_parse_docstring_empty_or_none():
    assert _parse_docstring_for_params("") == {}
    assert _parse_docstring_for_params(None) == {}

def test_map_type_to_json_schema_basic_types():
    assert _map_type_to_json_schema(str) == {"type": "string"}
    assert _map_type_to_json_schema(int) == {"type": "integer"}
    assert _map_type_to_json_schema(float) == {"type": "number"}
    assert _map_type_to_json_schema(bool) == {"type": "boolean"}
    assert _map_type_to_json_schema(type(None)) == {"type": "null"}
    assert _map_type_to_json_schema(Any) == {} # Any maps to empty schema (no constraint)

def test_map_type_to_json_schema_list():
    assert _map_type_to_json_schema(list) == {"type": "array", "items": {}}
    assert _map_type_to_json_schema(List[str]) == {"type": "array", "items": {"type": "string"}}
    assert _map_type_to_json_schema(List[int]) == {"type": "array", "items": {"type": "integer"}}
    assert _map_type_to_json_schema(List[Any]) == {"type": "array", "items": {}}

def test_map_type_to_json_schema_dict():
    assert _map_type_to_json_schema(dict) == {"type": "object"}
    # Dict[str, int] would ideally map to {"type": "object", "additionalProperties": {"type": "integer"}}
    # but current simple version just gives {"type": "object"}
    assert _map_type_to_json_schema(Dict[str, int]) == {"type": "object"}

def test_map_type_to_json_schema_optional():
    assert _map_type_to_json_schema(Optional[str]) == {"type": "string"}
    assert _map_type_to_json_schema(Optional[int]) == {"type": "integer"}
    # Optionality is handled by presence in 'required' list, not by adding "null" to type array here.

def test_map_type_to_json_schema_union():
    assert _map_type_to_json_schema(Union[str, int]) == {"type": "string"} # Takes first type for simplicity
    assert _map_type_to_json_schema(Union[str, None]) == {"type": "string"} # Same as Optional[str]

# --- Test @tool Decorator ---

@tool
def sample_tool_for_test(name: str, count: int = 5, active: Optional[bool] = None) -> str:
    """
    A sample tool for testing the @tool decorator.
    It takes a name, an optional count, and an optional active flag.

    Args:
        name (str): The name to use.
        count (int): How many times.
        active (Optional[bool]): Is it active?

    Returns:
        str: A processed string.
    """
    return f"{name} {count} {active}"

@tool
async def async_sample_tool(query: str) -> Dict[str, Any]:
    """An async sample tool."""
    return {"result": query.upper()}

@tool
def tool_no_params_no_return() -> None:
    """A tool with no parameters and no return value."""
    pass

class MyCustomClassForTypeHint:
    pass

@tool
def tool_with_custom_type(custom_param: MyCustomClassForTypeHint) -> MyCustomClassForTypeHint:
    """Tool with a custom class type hint."""
    return custom_param

StrForwardRef = ForwardRef("str")

@tool
def tool_with_forward_ref(ref_param: "StrForwardRef") -> "StrForwardRef":
    """Tool with a forward reference string type hint."""
    return ref_param


def test_tool_decorator_attaches_metadata_and_schema():
    assert hasattr(sample_tool_for_test, "_tool_metadata_")
    metadata = sample_tool_for_test._tool_metadata_

    assert isinstance(metadata, dict)
    assert metadata["identifier"] == "sample_tool_for_test"
    assert metadata["name"] == "Sample Tool For Test"
    assert "A sample tool for testing the @tool decorator." in metadata["description_human"]

    input_schema = metadata["input_schema"]
    assert input_schema["type"] == "object"
    assert "name" in input_schema["properties"]
    assert input_schema["properties"]["name"]["type"] == "string"
    assert input_schema["properties"]["name"]["description"] == "The name to use."

    assert "count" in input_schema["properties"]
    assert input_schema["properties"]["count"]["type"] == "integer"
    assert input_schema["properties"]["count"]["default"] == 5

    assert "active" in input_schema["properties"]
    assert input_schema["properties"]["active"]["type"] == "boolean" # Optional[bool] -> bool
    # 'active' is not in required because it's Optional (or has a default, though None is not a JSON schema default)
    assert "active" not in input_schema.get("required", [])

    assert input_schema["required"] == ["name"] # Only 'name' is strictly required

    output_schema = metadata["output_schema"]
    assert output_schema["type"] == "object"
    assert output_schema["properties"]["result"]["type"] == "string"
    assert output_schema["required"] == ["result"]

def test_tool_decorator_async_function_metadata():
    assert hasattr(async_sample_tool, "_tool_metadata_")
    metadata = async_sample_tool._tool_metadata_
    assert metadata["identifier"] == "async_sample_tool"

    output_schema = metadata["output_schema"]
    assert output_schema["type"] == "object" # Dict[str, Any] maps to object
    # The "result" key is now standard for the output object
    assert "result" in output_schema["properties"]
    assert output_schema["properties"]["result"]["type"] == "object" # Inner type of Dict is object

def test_tool_decorator_no_params_no_return_metadata():
    assert hasattr(tool_no_params_no_return, "_tool_metadata_")
    metadata = tool_no_params_no_return._tool_metadata_
    assert metadata["input_schema"]["properties"] == {}
    assert "required" not in metadata["input_schema"]

    output_schema = metadata["output_schema"]
    assert output_schema["type"] == "object"
    assert output_schema["properties"]["result"]["type"] == "null" # None return type
    # If return is None, result might not be "required" depending on interpretation
    # Current logic makes it required if the type is not null.
    assert "required" not in output_schema # Since result type is null

def test_tool_decorator_custom_type_hint_metadata():
    assert hasattr(tool_with_custom_type, "_tool_metadata_")
    metadata = tool_with_custom_type._tool_metadata_

    input_schema = metadata["input_schema"]
    # MyCustomClassForTypeHint defaults to "string" by _map_type_to_json_schema
    assert input_schema["properties"]["custom_param"]["type"] == "string"

    output_schema = metadata["output_schema"]
    assert output_schema["properties"]["result"]["type"] == "string"

def test_tool_decorator_forward_ref_metadata():
    assert hasattr(tool_with_forward_ref, "_tool_metadata_")
    metadata = tool_with_forward_ref._tool_metadata_

    input_schema = metadata["input_schema"]
    assert input_schema["properties"]["ref_param"]["type"] == "string"

    output_schema = metadata["output_schema"]
    assert output_schema["properties"]["result"]["type"] == "string"

def test_tool_decorator_preserves_original_function():
    assert hasattr(sample_tool_for_test, "_original_function_")
    original_func = sample_tool_for_test._original_function_
    assert callable(original_func)
    # Check if it's the actual undecorated function (can be tricky if multiple wrappers)
    # For this test, we assume @tool is the outermost relevant decorator.
    assert original_func.__name__ == "sample_tool_for_test"
    assert not hasattr(original_func, "_tool_metadata_") # Original shouldn't have it

    # Test execution of the wrapped function
    # The wrapper returned by @tool should be callable and execute the original.
    # This is more of an integration test with FunctionToolWrapper, but good to have a basic check.
    result = sample_tool_for_test(name="Test", count=3, active=True)
    assert result == "Test 3 True"

@pytest.mark.asyncio
async def test_tool_decorator_async_wrapper_execution():
    result = await async_sample_tool(query="test")
    assert result == {"result": "TEST"}

