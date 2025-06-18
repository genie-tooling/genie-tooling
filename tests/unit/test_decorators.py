### tests/unit/test_decorators.py
from typing import (
    Any,
    Dict,
    ForwardRef,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import pytest

# Assuming decorators.py is in src.genie_tooling, adjust if necessary
from genie_tooling.decorators import (
    _map_type_to_json_schema,
    _parse_docstring_for_params,
    _resolve_forward_refs,
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

def test_parse_docstring_multiline_param_description():
    doc = """
    Args:
        param_multi (str): This is a
            multi-line description for
            the parameter.
        param_single (int): A single line.
    """
    # The current parser might not perfectly handle multi-line descriptions merging.
    # It takes the first line of the description. Let's adjust expectation or parser.
    # For now, assuming it takes the first line.
    # If the parser was enhanced, this test would change.
    # Current behavior:
    parsed = _parse_docstring_for_params(doc)
    assert parsed.get("param_multi") == "This is a" # Current parser takes first line
    assert parsed.get("param_single") == "A single line."


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
    assert _map_type_to_json_schema(List[Dict[str, int]]) == {"type": "array", "items": {"type": "object"}}

def test_map_type_to_json_schema_set():
    assert _map_type_to_json_schema(set) == {"type": "array", "items": {}} # Sets map to arrays
    assert _map_type_to_json_schema(Set[str]) == {"type": "array", "items": {"type": "string"}}
    assert _map_type_to_json_schema(Set[float]) == {"type": "array", "items": {"type": "number"}}


def test_map_type_to_json_schema_tuple():
    assert _map_type_to_json_schema(tuple) == {"type": "array", "items": {}} # Generic tuple
    # For Tuple[T1, T2, ...], JSON schema uses a prefixItems style if needed,
    # or just "array" with "items" being a union if all elements are same type or any.
    # Current simple mapper will likely treat Tuple[str, int] as List[str] or List[Any].
    assert _map_type_to_json_schema(Tuple[str, int]) == {"type": "array", "items": {"type": "string"}} # Takes first type
    assert _map_type_to_json_schema(Tuple[int, ...]) == {"type": "array", "items": {"type": "integer"}} # Ellipsis indicates variable length


def test_map_type_to_json_schema_dict():
    assert _map_type_to_json_schema(dict) == {"type": "object"}
    # Dict[str, int] would ideally map to {"type": "object", "additionalProperties": {"type": "integer"}}
    # but current simple version just gives {"type": "object"}
    assert _map_type_to_json_schema(Dict[str, int]) == {"type": "object"}

def test_map_type_to_json_schema_optional():
    # _map_type_to_json_schema itself does not add 'null'. It just unwraps the Optional.
    # The 'tool' decorator logic is what adds the null type later.
    assert _map_type_to_json_schema(Optional[str]) == {"type": "string"}
    assert _map_type_to_json_schema(Optional[int]) == {"type": "integer"}

def test_map_type_to_json_schema_union():

    result = _map_type_to_json_schema(Union[str, int])
    assert isinstance(result.get("type"), list)
    assert set(result["type"]) == {"integer", "string"}

    assert _map_type_to_json_schema(Union[str, None]) == {"type": "string"} # Same as Optional[str]


# --- Tests for _resolve_forward_refs ---
class _TestClassForForwardRef:
    pass

_global_dict_for_ref_test = {"_TestClassForForwardRef": _TestClassForForwardRef, "int": int, "List": List, "Optional": Optional, "Union": Union}

class TestResolveForwardRefs:
    def test_resolve_simple_forward_ref(self):
        ref = ForwardRef("_TestClassForForwardRef")
        resolved = _resolve_forward_refs(ref, globalns=_global_dict_for_ref_test)
        assert resolved is _TestClassForForwardRef

    def test_resolve_forward_ref_in_list(self):
        ref_list = List[ForwardRef("_TestClassForForwardRef")] # type: ignore
        resolved = _resolve_forward_refs(ref_list, globalns=_global_dict_for_ref_test)
        assert get_origin(resolved) is list
        assert get_args(resolved)[0] is _TestClassForForwardRef

    def test_resolve_forward_ref_in_optional(self):
        ref_optional = Optional[ForwardRef("_TestClassForForwardRef")] # type: ignore
        resolved = _resolve_forward_refs(ref_optional, globalns=_global_dict_for_ref_test)
        assert get_origin(resolved) is Union
        args = get_args(resolved)
        assert _TestClassForForwardRef in args
        assert type(None) in args

    def test_resolve_forward_ref_in_union(self):
        ref_union = Union[ForwardRef("int"), ForwardRef("_TestClassForForwardRef")] # type: ignore
        resolved = _resolve_forward_refs(ref_union, globalns=_global_dict_for_ref_test)
        assert get_origin(resolved) is Union
        args = get_args(resolved)
        assert int in args
        assert _TestClassForForwardRef in args

    def test_resolve_unresolvable_forward_ref_raises_name_error(self):
        ref = ForwardRef("NonExistentClass")
        with pytest.raises(NameError, match="name 'NonExistentClass' is not defined"):
            _resolve_forward_refs(ref, globalns=_global_dict_for_ref_test)

    def test_resolve_non_forward_ref_returns_as_is(self):
        assert _resolve_forward_refs(int, globalns=_global_dict_for_ref_test) is int
        # For generic types, direct comparison might fail due to how they are constructed.
        # Check origin and args instead.
        resolved_list_int = _resolve_forward_refs(List[int], globalns=_global_dict_for_ref_test)
        assert get_origin(resolved_list_int) is list
        assert get_args(resolved_list_int) == (int,)

        resolved_optional_str = _resolve_forward_refs(Optional[str], globalns=_global_dict_for_ref_test)
        assert get_origin(resolved_optional_str) is Union
        assert get_args(resolved_optional_str) == (str, type(None))


    def test_resolve_with_localns(self):
        class LocalClass:
            pass
        ref = ForwardRef("LocalClass")
        resolved = _resolve_forward_refs(ref, globalns={}, localns={"LocalClass": LocalClass})
        assert resolved is LocalClass

    def test_resolve_nested_forward_refs(self):
        # e.g. List[Optional[ForwardRef(...)]]
        ref_nested = List[Optional[ForwardRef("_TestClassForForwardRef")]] # type: ignore
        resolved = _resolve_forward_refs(ref_nested, globalns=_global_dict_for_ref_test)
        assert get_origin(resolved) is list
        inner_type = get_args(resolved)[0]
        assert get_origin(inner_type) is Union
        inner_args = get_args(inner_type)
        assert _TestClassForForwardRef in inner_args
        assert type(None) in inner_args

    def test_resolve_forward_ref_already_evaluated(self):
        # If a ForwardRef was somehow already evaluated (e.g. by a previous get_type_hints call)
        # it might behave like the actual type.
        # This test is more conceptual as ForwardRef usually isn't "partially" evaluated.
        # If it's evaluated, it becomes the type. If not, it's a ForwardRef.
        evaluated_ref = _TestClassForForwardRef # Simulate it's already resolved
        resolved = _resolve_forward_refs(evaluated_ref, globalns=_global_dict_for_ref_test)
        assert resolved is _TestClassForForwardRef

    def test_resolve_forward_ref_string_type(self):
        ref = ForwardRef("str")
        resolved = _resolve_forward_refs(ref, globalns={"str": str})
        assert resolved is str

        ref_list_str = List[ForwardRef("str")] # type: ignore
        resolved_list_str = _resolve_forward_refs(ref_list_str, globalns={"str": str, "List": List})
        assert get_origin(resolved_list_str) is list
        assert get_args(resolved_list_str)[0] is str


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

@tool
def tool_no_type_hints(param1, param2="default"): # No type hints
    """A tool with no type hints for its parameters."""
    return f"{param1}, {param2}"

@tool
def tool_args_kwargs(*args, **kwargs):
    """A tool with *args and **kwargs."""
    return {"args": args, "kwargs": kwargs}

@tool
def tool_complex_list_type(data: List[Dict[str, int]]) -> int:
    """A tool with a complex list type hint."""
    total = 0
    for item_dict in data:
        for value in item_dict.values():
            total += value
    return total


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
    assert input_schema["properties"]["active"]["type"] == ["boolean", "null"]
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

@pytest.mark.asyncio()
async def test_tool_decorator_async_wrapper_execution():
    result = await async_sample_tool(query="test")
    assert result == {"result": "TEST"}

def test_tool_decorator_no_type_hints_metadata():
    assert hasattr(tool_no_type_hints, "_tool_metadata_")
    metadata = tool_no_type_hints._tool_metadata_
    input_schema = metadata["input_schema"]
    assert input_schema["properties"]["param1"] == {"type": "string", "description": "Parameter 'param1'."}
    assert input_schema["properties"]["param2"] == {"type": "string", "description": "Parameter 'param2'.", "default": "default"}
    assert input_schema["required"] == ["param1"]

def test_tool_decorator_args_kwargs_metadata():
    assert hasattr(tool_args_kwargs, "_tool_metadata_")
    metadata = tool_args_kwargs._tool_metadata_
    input_schema = metadata["input_schema"]
    # *args and **kwargs are generally not represented in JSON schema for specific parameters
    # The current decorator logic will skip them.
    assert "args" not in input_schema["properties"]
    assert "kwargs" not in input_schema["properties"]
    assert input_schema["properties"] == {} # Expect empty if only *args, **kwargs

def test_tool_decorator_complex_list_type_metadata():
    assert hasattr(tool_complex_list_type, "_tool_metadata_")
    metadata = tool_complex_list_type._tool_metadata_
    input_schema = metadata["input_schema"]
    data_param_schema = input_schema["properties"]["data"]
    assert data_param_schema["type"] == "array"
    assert data_param_schema["items"]["type"] == "object" # List[Dict[str, int]] -> items: object
    assert input_schema["required"] == ["data"]

    output_schema = metadata["output_schema"]
    assert output_schema["properties"]["result"]["type"] == "integer"
