### tests/unit/utils/gbnf/test_model_factory.py
import inspect
import logging
from enum import Enum
from typing import (
    Any,
    Dict,
    ForwardRef,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
)

import pytest
from genie_tooling.utils.gbnf.model_factory import (
    _to_pascal_case_for_model_name,
    add_run_method_to_dynamic_model,
    convert_dictionary_to_pydantic_model,
    create_dynamic_model_from_function,
    create_dynamic_models_from_dictionaries,
    json_schema_to_python_types,
    list_to_enum,
    to_pascal_case,
)
from pydantic import BaseModel


# --- Test Models & Functions for Fixtures/Tests ---
class NestedForFactory(BaseModel):
    item: str

def sample_func_for_model(name: str, age: int = 30, active: Optional[bool] = True) -> Dict[str, Any]:
    """A sample function for testing model creation.

Args:
    name (str): The name of the user.
    age (int): The age of the user. Defaults to 30.
    active (Optional[bool]): User's active status. Defaults to True.

Returns:
    Dict[str, Any]: A dictionary containing processed user information.
    """
    return {"name": name, "age": age, "is_active": active}

def func_no_params() -> str:
    """A function with no parameters."""
    return "no_params_executed"

def func_only_kwargs(**kwargs: Any) -> Dict[str, Any]:
    """A function with only kwargs."""
    return kwargs

def func_with_args_and_kwargs(pos_arg: str, *args: int, keyword_arg: bool = True, **kwargs: float) -> None:
    """Function with *args and **kwargs."""
    pass

def func_no_docstring(param1: int, param2: Optional[str] = None):
    return f"{param1}-{param2}"

class MyCustomClassForHint:
    pass

StrRef = ForwardRef("str")
CustomRef = ForwardRef("MyCustomClassForHint")

def func_with_forward_refs(ref1: StrRef, ref2: "CustomRef", ref3: "List[int]") -> None:
    pass


# --- Tests for list_to_enum ---
def test_list_to_enum_simple_strings():
    MyEnum = list_to_enum("MySimpleEnum", ["apple", "banana", "cherry"])
    assert issubclass(MyEnum, Enum)
    assert MyEnum.APPLE.value == "apple"
    assert MyEnum.BANANA.value == "banana"
    assert MyEnum.CHERRY.value == "cherry"
    assert MyEnum.__name__ == "MySimpleEnum"

def test_list_to_enum_with_spaces_and_numbers():
    MixedEnum = list_to_enum("MixedEnum", ["option one", "option_2", "3rd option"])
    assert MixedEnum.OPTION_ONE.value == "option one"
    assert MixedEnum.OPTION_2.value == "option_2"
    assert MixedEnum.VALUE_3RD_OPTION.value == "3rd option"

def test_list_to_enum_duplicates_handled_by_enum():
    DupEnum = list_to_enum("DupEnum", ["a", "b", "a", "c"])
    assert len(list(DupEnum)) == 3
    assert DupEnum.A.value == "a"
    assert {member.value for member in DupEnum} == {"a", "b", "c"}

def test_list_to_enum_non_string_values():
    NumEnum = list_to_enum("NumEnum", [1, 2.5, True, False, None])
    assert NumEnum.VALUE_1.value == 1
    assert NumEnum.VALUE_2_5.value == 2.5
    assert NumEnum.TRUE.value == 1
    assert NumEnum.FALSE.value is False
    assert NumEnum.NONE.value is None


def test_list_to_enum_empty_list():
    with pytest.raises(TypeError, match="EmptyEnum: an empty enum is not allowed"):
        list_to_enum("EmptyEnum", [])

def test_list_to_enum_invalid_name_characters():
    InvalidNameEnum = list_to_enum("InvalidNameEnum", ["!@#$", "%^&*"])

    found_first = False
    found_second = False
    for member in InvalidNameEnum:
        if member.value == "!@#$":
            found_first = True
        elif member.value == "%^&*":
            found_second = True

    assert found_first, "Enum member for '!@#$' not found after sanitization"
    assert found_second, "Enum member for '%^&*' not found after sanitization"

def test_list_to_enum_keyword_name_candidate():
    KeywordEnum = list_to_enum("KeywordEnum", ["class", "def"])
    assert KeywordEnum.KEYWORD_CLASS.value == "class"
    assert KeywordEnum.KEYWORD_DEF.value == "def"

def test_list_to_enum_name_becomes_empty_or_underscore():
    EmptyNameEnum = list_to_enum("EmptyNameEnum", ["_", "___"])
    assert hasattr(EmptyNameEnum, "UNDERSCORE")
    assert EmptyNameEnum.UNDERSCORE.value == "_"
    assert hasattr(EmptyNameEnum, "UNDERSCORES_3")
    assert EmptyNameEnum.UNDERSCORES_3.value == "___"


def test_list_to_enum_name_starts_with_underscore_not_dunder():
    UnderscoreEnum = list_to_enum("UnderscoreEnum", ["_my_value"])
    assert hasattr(UnderscoreEnum, "MY_VALUE")
    assert UnderscoreEnum.MY_VALUE.value == "_my_value"

def test_list_to_enum_name_clash_resolution():
    ClashEnum = list_to_enum("ClashEnum", ["val", "val", "val_1"])
    assert ClashEnum.VAL.value == "val"
    assert ClashEnum.VAL_1.value == "val"
    assert ClashEnum.VAL_1_1.value == "val_1"


# --- Tests for add_run_method_to_dynamic_model ---
def test_add_run_method():
    class MyDataModel(BaseModel):
        x: int
        y: str

    def process_data(x: int, y: str) -> str:
        return f"{y} repeated {x} times"

    ModelWithRun = add_run_method_to_dynamic_model(MyDataModel, process_data)
    instance = ModelWithRun(x=3, y="hello")
    assert hasattr(instance, "run")
    assert instance.run() == "hello repeated 3 times"


# --- Tests for to_pascal_case and _to_pascal_case_for_model_name ---
@pytest.mark.parametrize("input_str, expected", [
    ("snake_case_string", "SnakeCaseString"),
    ("kebab-case-string", "KebabCaseString"),
    ("AlreadyPascalCase", "AlreadyPascalCase"),
    ("stringWithNumbers123", "StringWithNumbers123"),
    ("", "UnnamedModel"),
    ("  leading space", "LeadingSpace"),
    ("trailing_space  ", "TrailingSpace"),
    ("a_b-c d", "ABCD"),
    ("url_processor_tool", "UrlProcessorTool"),
    ("myHTTPClient", "MyHTTPClient"),
    ("MyURLProcessor", "MyURLProcessor"),
    ("__private_name", "PrivateName"),
    ("_leading_underscore", "LeadingUnderscore"),
    ("complex_http_API_v2_handler", "ComplexHttpAPIV2Handler"),
    ("!@#$", "UnnamedModel"),
    ("  multiple   spaces  ", "MultipleSpaces"),
    (" leading_and_trailing_chars! ", "LeadingAndTrailingChars"),
])
def test_to_pascal_case(input_str, expected):
    actual_pascal = to_pascal_case(input_str)
    actual_pascal_for_model = _to_pascal_case_for_model_name(input_str)

    assert repr(actual_pascal) == repr(expected), \
        f"to_pascal_case failed: input='{input_str}', expected repr='{expected!r}', got repr='{actual_pascal!r}'"
    assert repr(actual_pascal_for_model) == repr(expected), \
        f"_to_pascal_case_for_model_name failed: input='{input_str}', expected repr='{expected!r}', got repr='{actual_pascal_for_model!r}'"

def test_to_pascal_case_invalid_chars_cleaned():
    assert _to_pascal_case_for_model_name("my!@#$model%^&*name") == "MyModelName"
    assert _to_pascal_case_for_model_name("!@#$") == "UnnamedModel"


# --- Tests for json_schema_to_python_types ---
@pytest.mark.parametrize("json_type, py_type", [
    ("string", str),
    ("number", float),
    ("integer", int),
    ("boolean", bool),
    ("array", list),
    ("object", dict),
    ("null", type(None)),
    ("any", Any),
    ("unknown_type", Any),
    ("STRING", str),
])
def test_json_schema_to_python_types(json_type, py_type):
    assert json_schema_to_python_types(json_type) == py_type


# --- Tests for create_dynamic_model_from_function ---
def test_create_dynamic_model_from_function_basic():
    DynamicModel = create_dynamic_model_from_function(sample_func_for_model)
    assert DynamicModel.__name__ == "SampleFuncForModelInput"
    expected_doc = "A sample function for testing model creation."
    actual_doc = DynamicModel.__doc__.replace("\r\n", "\n").replace("\r", "\n") if DynamicModel.__doc__ else ""
    expected_doc_normalized = expected_doc.replace("\r\n", "\n").replace("\r", "\n")
    assert actual_doc == expected_doc_normalized

    fields = DynamicModel.model_fields
    assert "name" in fields
    assert fields["name"].annotation == str
    assert fields["name"].is_required()
    assert fields["name"].description == "The name of the user."

    assert "age" in fields
    assert fields["age"].annotation == Optional[int]
    assert not fields["age"].is_required()
    assert fields["age"].default == 30
    assert fields["age"].description == "The age of the user. Defaults to 30."

    assert "active" in fields
    assert fields["active"].annotation == Optional[bool]
    assert not fields["active"].is_required()
    assert fields["active"].default is True
    assert fields["active"].description == "User's active status. Defaults to True."

    instance = DynamicModel(name="TestUser", age=25)
    assert instance.run() == {"name": "TestUser", "age": 25, "is_active": True}

def test_create_dynamic_model_from_function_no_params():
    DynamicModel = create_dynamic_model_from_function(func_no_params)
    assert DynamicModel.__name__ == "FuncNoParamsInput"
    assert not DynamicModel.model_fields
    instance = DynamicModel()
    assert instance.run() == "no_params_executed"

def test_create_dynamic_model_from_function_only_kwargs():
    DynamicModel = create_dynamic_model_from_function(func_only_kwargs)
    assert DynamicModel.__name__ == "FuncOnlyKwargsInput"
    assert not DynamicModel.model_fields

def test_create_dynamic_model_from_function_with_args_kwargs():
    DynamicModel = create_dynamic_model_from_function(func_with_args_and_kwargs)
    assert DynamicModel.__name__ == "FuncWithArgsAndKwargsInput"
    fields = DynamicModel.model_fields
    assert "pos_arg" in fields
    assert "keyword_arg" in fields
    assert "args" not in fields
    assert "kwargs" not in fields

def test_create_dynamic_model_from_function_no_docstring():
    DynamicModel = create_dynamic_model_from_function(func_no_docstring)
    assert DynamicModel.__name__ == "FuncNoDocstringInput"
    assert DynamicModel.__doc__ is None
    fields = DynamicModel.model_fields
    assert fields["param1"].description == "Parameter 'param1'"
    assert fields["param2"].description == "Parameter 'param2'"

def test_create_dynamic_model_from_function_with_forward_refs():
    DynamicModel = create_dynamic_model_from_function(func_with_forward_refs)
    assert DynamicModel.__name__ == "FuncWithForwardRefsInput"
    fields = DynamicModel.model_fields
    assert "ref1" in fields
    assert "ref2" in fields
    assert "ref3" in fields
    assert DynamicModel.model_config.get("arbitrary_types_allowed") is True


# --- Tests for convert_dictionary_to_pydantic_model ---
class TestConvertDictionaryToPydanticModel:
    def test_simple_properties(self):
        schema = {
            "name": "SimplePropsModel",
            "description": "Model with simple properties.",
            "properties": {
                "username": {"type": "string", "description": "User's login name."},
                "visits": {"type": "integer", "default": 0},
                "score": {"type": "number"},
                "is_member": {"type": "boolean", "default": False},
                "extra_data": {"type": "null"}
            },
            "required": ["username", "score"]
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "SimplePropsModel"
        assert Model.__doc__ == "Model with simple properties."
        fields = Model.model_fields

        assert fields["username"].annotation == str
        assert fields["username"].is_required()
        assert fields["visits"].annotation == Optional[int]
        assert fields["visits"].default == 0
        assert fields["score"].annotation == float
        assert fields["score"].is_required()
        assert fields["is_member"].annotation == Optional[bool]
        assert fields["is_member"].default is False
        assert fields["extra_data"].annotation == Optional[type(None)]
        assert fields["extra_data"].default is None

    def test_enum_creation(self):
        schema = {
            "name": "EnumModel",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive", "pending"], "description": "Current status."}
            },
            "required": ["status"]
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "EnumModel"
        status_field = Model.model_fields["status"]
        assert inspect.isclass(status_field.annotation) and issubclass(status_field.annotation, Enum)
        StatusEnum = status_field.annotation
        assert StatusEnum.ACTIVE.value == "active" # type: ignore
        assert StatusEnum.__name__ == "EnumModelStatusEnum"

    def test_array_of_simple_types(self):
        schema = {"name": "ArrayModel", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "ArrayModel"
        tags_field = Model.model_fields["tags"]
        assert tags_field.annotation == Optional[List[str]]
        assert tags_field.default_factory == list

    def test_array_of_objects(self):
        schema = {
            "name": "ArrayObjectModel",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"item_id": {"type": "integer"}, "value": {"type": "string"}},
                        "required": ["item_id"]
                    }
                }
            }
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "ArrayObjectModel"
        items_field = Model.model_fields["items"]
        assert get_origin(items_field.annotation) is Union
        list_type_args = get_args(items_field.annotation)
        assert type(None) in list_type_args
        list_type = next(t for t in list_type_args if t is not type(None))

        assert get_origin(list_type) is list
        item_model = get_args(list_type)[0]
        assert issubclass(item_model, BaseModel)
        assert item_model.__name__ == "ArrayObjectModelItemsItem"
        assert "item_id" in item_model.model_fields
        assert item_model.model_fields["item_id"].annotation == int

    def test_nested_objects(self):
        schema = {
            "name": "NestedOuterModel",
            "properties": {
                "user_profile": {
                    "type": "object",
                    "description": "User profile data.",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "preferences": {
                            "type": "object",
                            "properties": {"theme": {"type": "string", "default": "dark"}}
                        }
                    },
                    "required": ["user_id"]
                }
            }
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "NestedOuterModel"
        profile_field = Model.model_fields["user_profile"]
        assert get_origin(profile_field.annotation) is Union
        ProfileModel_args = get_args(profile_field.annotation)
        assert type(None) in ProfileModel_args
        ProfileModel = next(t for t in ProfileModel_args if t is not type(None))

        assert issubclass(ProfileModel, BaseModel)
        assert ProfileModel.__name__ == "NestedOuterModelUserProfile"
        assert ProfileModel.__doc__ == "User profile data."

        prefs_field = ProfileModel.model_fields["preferences"]
        assert get_origin(prefs_field.annotation) is Union
        PrefsModel_args = get_args(prefs_field.annotation)
        assert type(None) in PrefsModel_args
        PrefsModel = next(t for t in PrefsModel_args if t is not type(None))

        assert issubclass(PrefsModel, BaseModel)
        assert PrefsModel.__name__ == "NestedOuterModelUserProfilePreferences"
        assert PrefsModel.model_fields["theme"].default == "dark"

    def test_openai_function_schema_format(self):
        schema = {
            "name": "MyFunctionTool",
            "description": "A tool for OpenAI functions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state."},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["location"]
            }
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "MyFunctionTool"
        assert Model.__doc__ == "A tool for OpenAI functions."
        fields = Model.model_fields
        assert "location" in fields
        assert "unit" in fields
        assert fields["unit"].default == "celsius"

    def test_schema_no_type_object_but_has_properties(self):
        schema = {
            "name": "ImplicitObjectModel",
            "properties": {"field_a": {"type": "string"}}
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "ImplicitObjectModel"
        assert "field_a" in Model.model_fields

    def test_empty_properties_object(self):
        schema = {"name": "EmptyPropsModel", "type": "object", "properties": {}}
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "EmptyPropsModel"
        assert not Model.model_fields

    def test_invalid_field_schema_logs_warning(self, caplog):
        schema = {"name": "InvalidFieldTest", "properties": {"bad_field": "not_a_dict"}}
        with caplog.at_level(logging.WARNING):
            Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.__name__ == "InvalidFieldTest"
        assert "bad_field" in Model.model_fields
        assert Model.model_fields["bad_field"].annotation == Any
        assert "Schema for field 'bad_field' in model 'InvalidFieldTest' is not a dictionary. Using Any." in caplog.text

    def test_array_no_items_schema(self):
        schema = {"name": "ArrayNoItemsModel", "properties": {"my_list": {"type": "array"}}}
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.model_fields["my_list"].annotation == Optional[List[Any]]

    def test_array_items_not_dict(self, caplog):
        schema = {"name": "ArrayItemsNotDict", "properties": {"bad_list": {"type": "array", "items": "not_a_dict_schema"}}}
        with caplog.at_level(logging.WARNING):
            Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.model_fields["bad_list"].annotation == Optional[List[Any]]
        assert "Items schema for array field 'bad_list' in 'ArrayItemsNotDict' is not a dictionary. Defaulting item type to Any." in caplog.text

    def test_field_optional_no_default(self):
        schema = {"name": "OptionalNoDefault", "properties": {"opt_field": {"type": "string"}}}
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.model_fields["opt_field"].annotation == Optional[str]
        assert Model.model_fields["opt_field"].default is None

    def test_field_required_and_has_default(self):
        schema = {
            "name": "RequiredWithDefault",
            "properties": {"req_def_field": {"type": "integer", "default": 100}},
            "required": ["req_def_field"]
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        assert Model.model_fields["req_def_field"].annotation == Optional[int]
        assert Model.model_fields["req_def_field"].default == 100
        assert not Model.model_fields["req_def_field"].is_required()

    def test_field_name_needs_sanitization_for_enum(self):
        schema = {
            "name": "ComplexFieldNameEnum",
            "properties": {
                "status-code": {"type": "string", "enum": ["OK", "ERROR"]}
            }
        }
        Model = convert_dictionary_to_pydantic_model(schema)
        status_code_field = Model.model_fields["status-code"]

        actual_enum_type = status_code_field.annotation
        if get_origin(actual_enum_type) is Union:
            non_none_args = [t for t in get_args(actual_enum_type) if t is not type(None)]
            if non_none_args:
                actual_enum_type = non_none_args[0]

        assert inspect.isclass(actual_enum_type) and issubclass(actual_enum_type, Enum)
        assert actual_enum_type.__name__ == "ComplexFieldNameEnumStatusCodeEnum"


# --- Tests for create_dynamic_models_from_dictionaries ---
def test_create_dynamic_models_from_dictionaries():
    dict_schemas = [
        {"name": "ModelOne", "properties": {"field1": {"type": "string"}}},
        {"name": "ModelTwoFromDict", "description": "Second model.", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}}}},
    ]
    models = create_dynamic_models_from_dictionaries(dict_schemas)
    assert len(models) == 2
    assert models[0].__name__ == "ModelOne"
    assert "field1" in models[0].model_fields
    assert models[1].__name__ == "ModelTwoFromDict"
    assert models[1].__doc__ == "Second model."
    assert "count" in models[1].model_fields

def test_create_dynamic_models_from_empty_list():
    models = create_dynamic_models_from_dictionaries([])
    assert models == []
