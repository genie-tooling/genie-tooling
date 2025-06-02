### src/genie_tooling/utils/gbnf/model_factory.py
# utils/gbnf/documentation.py
from __future__ import annotations

import inspect
import logging
import re
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from docstring_parser import parse
from pydantic import BaseModel, create_model  # Renamed Field
from pydantic import Field as PydanticField

# These imports are fine as they are within the same subpackage (utils.gbnf)

logger = logging.getLogger(__name__)

def format_multiline_description(description: str, indent_level: int) -> str:
    """Indents a multiline description string."""
    indent = "  " * indent_level # Two spaces per indent level for text doc
    return indent + description.replace("\n", f"\n{indent}")

def list_to_enum(enum_name, values):
    """Creates an Enum class from a list of values."""
    return Enum(enum_name, {str(v).replace(" ", "_").upper(): v for v in values})


def add_run_method_to_dynamic_model(model: type[BaseModel], func: Callable[..., Any]):
    """
    Add a 'run' method to a dynamic Pydantic model, using the provided function.

    Args:
        model (type[BaseModel]): Dynamic Pydantic model class.
        func (Callable): Function to be added as a 'run' method to the model.

    Returns:
        type[BaseModel]: Pydantic model class with the added 'run' method.
    """

    def run_method_wrapper(self): # self is the instance of the dynamic model
        func_args = {name: getattr(self, name) for name in model.model_fields}
        return func(**func_args)

    model.run = run_method_wrapper

    return model

def to_pascal_case(name: str) -> str:
    """Converts a snake_case or kebab-case string to PascalCase for class names."""
    if not name:
        return "UnnamedModel"
    s = name.replace("_", " ").replace("-", " ")
    s = s.title().replace(" ", "")
    if not s:
        return "UnnamedModel"
    if not s[0].isalpha():
        s = "Model" + s
    return s

def json_schema_to_python_types(schema_type: str) -> Type[Any]:
    """Maps JSON schema type strings to Python types."""
    type_map: Dict[str, Type[Any]] = {
        "any": Any,
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    return type_map.get(schema_type.lower(), Any)

def create_dynamic_model_from_function(func: Callable[..., Any]):
    """
    Creates a dynamic Pydantic model from a given function's type hints and adds the function as a 'run' method.

    Args:
        func (Callable): A function with type hints from which to create the model.

    Returns:
        A dynamic Pydantic model class with the provided function as a 'run' method.
    """

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    parsed_docstring = parse(func.__doc__ or "")

    dynamic_fields: Dict[str, Tuple[Type[Any], Any]] = {}
    param_docs: Dict[str, str] = {p.arg_name: p.description for p in parsed_docstring.params}

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        param_annotation = type_hints.get(name, Any)
        if param_annotation == inspect.Parameter.empty:
            param_annotation = Any

        param_description = param_docs.get(name, f"Parameter '{name}'")

        field_args: Dict[str, Any] = {"description": param_description}
        if param.default == inspect.Parameter.empty:
            field_args["default"] = ...
        else:
            field_args["default"] = param.default
            if not (get_origin(param_annotation) is Union and type(None) in get_args(param_annotation)):
                param_annotation = Optional[param_annotation]

        dynamic_fields[name] = (param_annotation, PydanticField(**field_args))

    model_name = to_pascal_case(func.__name__) + "Input"
    dynamic_model: Type[BaseModel] = create_model(model_name, **dynamic_fields) # type: ignore

    if parsed_docstring.short_description:
        dynamic_model.__doc__ = parsed_docstring.short_description

    dynamic_model_with_run = add_run_method_to_dynamic_model(dynamic_model, func) # type: ignore

    return dynamic_model_with_run

def _to_pascal_case_for_model_name(name: str) -> str:
    """Converts a string to PascalCase suitable for a Python class name."""
    if not name:
        return "UnnamedModel"
    name_with_spaces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name_with_spaces = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", name_with_spaces)

    parts = re.split(r"[-_\s]+", name_with_spaces)
    pascal_cased_name = "".join(p[0].upper() + p[1:].lower() if p and p[0].isalpha() else p.capitalize() if p else "" for p in parts if p)

    if not pascal_cased_name:
        return "UnnamedModel"
    if not pascal_cased_name[0].isalpha():
        pascal_cased_name = f"Model{pascal_cased_name}"
    pascal_cased_name = re.sub(r"[^a-zA-Z0-9_]", "", pascal_cased_name)
    if not pascal_cased_name:
        return "UnnamedModel"
    return pascal_cased_name

def convert_dictionary_to_pydantic_model(
    dictionary: Dict[str, Any], model_name: str = "CustomModel"
) -> Type[BaseModel]:
    """
    Converts a JSON schema-like dictionary into a Pydantic model.
    Handles nested objects and arrays of objects by recursively creating models.
    """
    fields: Dict[str, Tuple[Type[Any], Any]] = {}
    properties_dict: Optional[Dict[str, Any]] = None
    required_fields_from_schema: Set[str] = set()

    class_name_to_use = _to_pascal_case_for_model_name(model_name)
    current_schema_description: Optional[str] = dictionary.get("description")

    if "function" in dictionary and isinstance(dictionary["function"], dict):
        func_details = dictionary["function"]
        current_schema_description = func_details.get("description", current_schema_description)
        potential_fn_name = func_details.get("name")
        if potential_fn_name and isinstance(potential_fn_name, str):
            class_name_to_use = _to_pascal_case_for_model_name(potential_fn_name)

        params_schema = func_details.get("parameters")
        if isinstance(params_schema, dict) and params_schema.get("type") == "object":
            properties_dict = params_schema.get("properties")
            required_fields_from_schema = set(params_schema.get("required", []))
    elif "parameters" in dictionary and isinstance(dictionary["parameters"], dict) and \
         "name" in dictionary and isinstance(dictionary["name"], str):
        params_schema = dictionary["parameters"]
        current_schema_description = dictionary.get("description", current_schema_description)
        class_name_to_use = _to_pascal_case_for_model_name(dictionary["name"])
        if isinstance(params_schema, dict) and params_schema.get("type") == "object":
            properties_dict = params_schema.get("properties")
            required_fields_from_schema = set(params_schema.get("required", []))
    elif "properties" in dictionary and isinstance(dictionary.get("properties"), dict):
        properties_dict = dictionary["properties"]
        required_fields_from_schema = set(dictionary.get("required", []))
    elif dictionary.get("type") == "object" and not properties_dict:
        pass

    if properties_dict is None:
        properties_dict = {}

    for field_name, field_data_any in properties_dict.items():
        if not isinstance(field_data_any, dict):
            logger.warning(f"Schema for field '{field_name}' in model '{class_name_to_use}' is not a dictionary. Using Any. Value: {field_data_any}")
            fields[field_name] = (Any, PydanticField(description=f"Malformed schema for field '{field_name}'."))
            continue

        field_data: Dict[str, Any] = field_data_any
        is_field_required = field_name in required_fields_from_schema
        field_type_str = field_data.get("type", "string")
        field_description = field_data.get("description")
        pydantic_annotation: Any
        field_definition_args: Dict[str, Any] = {}
        if field_description:
            field_definition_args["description"] = field_description

        explicit_default_value = field_data.get("default")
        has_explicit_default = "default" in field_data

        if field_data.get("enum") and isinstance(field_data.get("enum"), list):
            enum_name_for_field = f"{class_name_to_use}_{_to_pascal_case_for_model_name(field_name)}_Enum"
            pydantic_annotation = list_to_enum(enum_name_for_field, field_data["enum"])
        elif field_type_str == "array":
            items_schema_any = field_data.get("items", {})
            item_py_type: Type[Any] = Any
            if isinstance(items_schema_any, dict):
                items_schema_dict = items_schema_any
                item_json_type = items_schema_dict.get("type")
                if item_json_type == "object" or "properties" in items_schema_dict:
                    item_py_type = convert_dictionary_to_pydantic_model(
                        items_schema_dict, f"{class_name_to_use}_{_to_pascal_case_for_model_name(field_name)}_Item"
                    )
                elif item_json_type:
                    item_py_type = json_schema_to_python_types(item_json_type)
            else:
                logger.warning(f"Items schema for array field '{field_name}' in '{class_name_to_use}' is not a dictionary. Defaulting item type to Any. Schema: {items_schema_any}")
            pydantic_annotation = List[item_py_type] # type: ignore
            if not has_explicit_default and not is_field_required:
                field_definition_args["default_factory"] = list
        elif field_type_str == "object" or ("properties" in field_data and field_data.get("type") is None):
            if field_data.get("type") is None and "properties" in field_data:
                logger.debug(f"Field '{field_name}' in '{class_name_to_use}' has properties but no explicit 'type: object'. Assuming object.")
            pydantic_annotation = convert_dictionary_to_pydantic_model(field_data, f"{class_name_to_use}_{_to_pascal_case_for_model_name(field_name)}")
            if not has_explicit_default and not is_field_required:
                field_definition_args["default_factory"] = dict
        else:
            pydantic_annotation = json_schema_to_python_types(field_type_str)

        if has_explicit_default:
            field_definition_args["default"] = explicit_default_value
            if not (get_origin(pydantic_annotation) is Union and type(None) in get_args(pydantic_annotation)):
                 pydantic_annotation = Optional[pydantic_annotation]
        elif not is_field_required:
            if not (get_origin(pydantic_annotation) is Union and type(None) in get_args(pydantic_annotation)):
                pydantic_annotation = Optional[pydantic_annotation]
            if "default_factory" not in field_definition_args:
                 field_definition_args["default"] = None
        else:
            field_definition_args["default"] = ...

        fields[field_name] = (pydantic_annotation, PydanticField(**field_definition_args))

    if current_schema_description and "__doc__" not in fields:
        fields["__doc__"] = current_schema_description

    custom_model: Type[BaseModel] = create_model(class_name_to_use, **fields) # type: ignore
    return custom_model


def create_dynamic_models_from_dictionaries(dictionaries: list[dict[str, Any]]):
    """
    Create a list of dynamic Pydantic model classes from a list of dictionaries.
    Each dictionary should represent a schema for a Pydantic model.
    The 'name' key in the dictionary is used as the base for the model's class name.
    """
    dynamic_models = []
    for func_dict_schema in dictionaries:
        model_base_name = func_dict_schema.get("name", "DynamicModelFromDict")
        dyn_model = convert_dictionary_to_pydantic_model(func_dict_schema, model_base_name)
        dynamic_models.append(dyn_model)
    return dynamic_models
