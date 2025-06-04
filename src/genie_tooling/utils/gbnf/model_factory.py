### src/genie_tooling/utils/gbnf/model_factory.py
# utils/gbnf/documentation.py
from __future__ import annotations

import inspect
import keyword
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
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField

logger = logging.getLogger(__name__)

def format_multiline_description(description: str, indent_level: int) -> str:
    """Indents a multiline description string."""
    indent = "  " * indent_level
    return indent + description.replace("\n", f"\n{indent}")

def list_to_enum(enum_name: str, values: List[Any]) -> Type[Enum]:
    """Creates an Enum class from a list of values.
    Ensures member names are valid Python identifiers.
    """
    if not values:
        raise TypeError(f"{enum_name}: an empty enum is not allowed")

    members: Dict[str, Any] = {}
    used_names: Set[str] = set()

    for i, v_val in enumerate(values):
        # Step 1: Determine base name candidate from value
        if isinstance(v_val, bool):
            name_candidate = "TRUE" if v_val else "FALSE"
        elif v_val is None:
            name_candidate = "NONE"
        else:
            name_candidate = str(v_val)

        # Step 2: Initial sanitization for characters and case
        # Specific handling for single underscore or all underscores *before* general regex
        if name_candidate == "_":
            name_candidate = "UNDERSCORE"
        elif all(c == "_" for c in name_candidate) and len(name_candidate) > 1:
            name_candidate = "UNDERSCORES_" + str(len(name_candidate))
        else:
            # General sanitization: Replace non-alphanumeric (excluding underscore) with underscore
            name_candidate = re.sub(r"[^a-zA-Z0-9_]", "_", name_candidate)
            name_candidate = name_candidate.upper()
            # Consolidate multiple underscores that might have been introduced
            name_candidate = re.sub(r"_+", "_", name_candidate)
            # Strip leading/trailing underscores that are not part of a dunder name
            if not (name_candidate.startswith("__") and name_candidate.endswith("__") and len(name_candidate) > 4):
                name_candidate = name_candidate.strip("_")

        # Step 3: Handle empty or digit-starting names after initial sanitization
        if not name_candidate:
            name_candidate = f"VALUE_{i}"
        elif name_candidate and name_candidate[0].isdigit(): # Check if not empty before indexing
            name_candidate = f"VALUE_{name_candidate}"

        # Step 4: Handle Python keywords.
        # Check if the *original value (if string and lowercased) or the sanitized candidate (lowercased)*
        # is a keyword. Enum members are typically uppercase.
        # Python keywords are lowercase.
        # If the sanitized uppercase name, when lowercased, is a keyword, prefix it.
        if keyword.iskeyword(name_candidate.lower()): # Check lowercase version
            name_candidate = f"KEYWORD_{name_candidate}"

        # Step 5: Ensure uniqueness and final identifier validity
        original_candidate_for_clash = name_candidate
        counter = 0

        current_test_name = name_candidate
        # Initial check before loop
        if not current_test_name.isidentifier() or keyword.iskeyword(current_test_name) or current_test_name in used_names:
            counter = 1
            current_test_name = f"{original_candidate_for_clash}_{counter}"

        while not current_test_name.isidentifier() or keyword.iskeyword(current_test_name) or current_test_name in used_names:
            counter += 1
            current_test_name = f"{original_candidate_for_clash}_{counter}"
            if counter > 50 + len(values):
                raise ValueError(f"Could not generate a unique, valid Enum member name for value '{v_val}' starting with '{original_candidate_for_clash}'")

        name_candidate = current_test_name
        used_names.add(name_candidate)
        members[name_candidate] = v_val

    return Enum(enum_name, members)


def add_run_method_to_dynamic_model(model: type[BaseModel], func: Callable[..., Any]):
    """
    Add a 'run' method to a dynamic Pydantic model, using the provided function.
    """
    def run_method_wrapper(self):
        func_args = {name: getattr(self, name) for name in model.model_fields}
        return func(**func_args)
    model.run = run_method_wrapper
    return model

def _to_pascal_case_for_model_name(name: str) -> str:
    """
    Converts a string to PascalCase suitable for a Python class name.
    Handles snake_case, kebab-case, space-separated.
    Removes invalid characters and ensures it starts with a letter.
    Attempts to preserve acronyms (e.g., HTTP, URL).
    """
    if not name:
        return "UnnamedModel"

    s = name
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)

    parts = s.split()

    if not parts:
        pascal_cased_name = ""
    else:
        processed_parts = []
        for part in parts:
            if not part:
                continue
            if part.isupper() and len(part) > 1:
                processed_parts.append(part)
            else:
                processed_parts.append(part[0].upper() + part[1:])
        pascal_cased_name = "".join(processed_parts)

    pascal_cased_name = re.sub(r"[^a-zA-Z0-9]", "", pascal_cased_name)

    if not pascal_cased_name:
        return "UnnamedModel"

    if not pascal_cased_name[0].isalpha():
        pascal_cased_name = f"Model{pascal_cased_name}"

    return pascal_cased_name

def to_pascal_case(name: str) -> str:
    """
    Converts a snake_case or kebab-case string to PascalCase.
    """
    return _to_pascal_case_for_model_name(name)


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
    """
    sig = inspect.signature(func)

    globalns = getattr(func, "__globals__", {})
    try:
        type_hints = get_type_hints(func, globalns=globalns, localns=None, include_extras=True)
    except Exception as e:
        logger.warning(f"Could not fully resolve type hints for {func.__name__} due to {e}. Some types might default to Any.")
        type_hints = {p.name: p.annotation for p in sig.parameters.values()}

    parsed_docstring = parse(func.__doc__ or "")

    dynamic_fields: Dict[str, Tuple[Type[Any], Any]] = {}
    param_docs: Dict[str, str] = {p.arg_name: (p.description or "") for p in parsed_docstring.params}

    for name, param in sig.parameters.items():
        if name in ("self", "cls") or \
           param.kind == inspect.Parameter.VAR_POSITIONAL or \
           param.kind == inspect.Parameter.VAR_KEYWORD:
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

    model_config_dict = {"arbitrary_types_allowed": True}

    dynamic_model: Type[BaseModel] = create_model(
        model_name,
        __config__=model_config_dict,
        **dynamic_fields # type: ignore
    )

    doc_to_set = parsed_docstring.short_description
    if not doc_to_set and parsed_docstring.long_description:
        if not any(marker in parsed_docstring.long_description for marker in ["Args:", "Returns:", "Yields:", "Raises:"]):
            doc_to_set = parsed_docstring.long_description
    if doc_to_set:
        dynamic_model.__doc__ = doc_to_set.strip()

    dynamic_model_with_run = add_run_method_to_dynamic_model(dynamic_model, func) # type: ignore
    return dynamic_model_with_run


def convert_dictionary_to_pydantic_model(
    dictionary: Dict[str, Any], model_name_fallback: str = "CustomModel"
) -> Type[BaseModel]:
    fields: Dict[str, Tuple[Type[Any], Any]] = {}
    properties_dict: Optional[Dict[str, Any]] = None
    required_fields_from_schema: Set[str] = set()

    potential_class_name = dictionary.get("name")
    current_schema_description = dictionary.get("description")

    if "function" in dictionary and isinstance(dictionary["function"], dict):
        func_details = dictionary["function"]
        if not potential_class_name and isinstance(func_details.get("name"), str):
            potential_class_name = func_details.get("name")
        current_schema_description = current_schema_description or func_details.get("description")
        params_schema = func_details.get("parameters")
        if isinstance(params_schema, dict) and params_schema.get("type") == "object":
            properties_dict = params_schema.get("properties")
            required_fields_from_schema = set(params_schema.get("required", []))
    elif "parameters" in dictionary and isinstance(dictionary["parameters"], dict):
        params_schema = dictionary["parameters"]
        current_schema_description = current_schema_description or dictionary.get("description")
        if isinstance(params_schema, dict) and params_schema.get("type") == "object":
            properties_dict = params_schema.get("properties")
            required_fields_from_schema = set(params_schema.get("required", []))
    elif "properties" in dictionary and isinstance(dictionary.get("properties"), dict):
        properties_dict = dictionary["properties"]
        required_fields_from_schema = set(dictionary.get("required", []))
    elif dictionary.get("type") == "object" and not properties_dict:
        pass

    class_name_to_use = _to_pascal_case_for_model_name(potential_class_name or model_name_fallback)

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

        nested_model_name_for_field = f"{class_name_to_use}{_to_pascal_case_for_model_name(field_name)}"


        if field_data.get("enum") and isinstance(field_data.get("enum"), list):
            enum_name_for_field = f"{nested_model_name_for_field}Enum"
            pydantic_annotation = list_to_enum(enum_name_for_field, field_data["enum"])
        elif field_type_str == "array":
            items_schema_any = field_data.get("items", {})
            item_py_type: Type[Any] = Any
            if isinstance(items_schema_any, dict):
                items_schema_dict = items_schema_any
                item_json_type = items_schema_dict.get("type")
                if item_json_type == "object" or "properties" in items_schema_dict:
                    item_py_type = convert_dictionary_to_pydantic_model(
                        items_schema_dict, f"{nested_model_name_for_field}Item"
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
            pydantic_annotation = convert_dictionary_to_pydantic_model(field_data, nested_model_name_for_field)
            if not has_explicit_default and not is_field_required:
                pydantic_annotation = Optional[pydantic_annotation]
                field_definition_args["default"] = None
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

    final_model_doc = current_schema_description

    model_config_for_create = {"arbitrary_types_allowed": True}
    custom_model: Type[BaseModel] = create_model(class_name_to_use, __config__=model_config_for_create, **fields) # type: ignore
    if final_model_doc:
        custom_model.__doc__ = final_model_doc.strip()

    return custom_model


def create_dynamic_models_from_dictionaries(dictionaries: list[dict[str, Any]]):
    """
    Create a list of dynamic Pydantic model classes from a list of dictionaries.
    """
    dynamic_models = []
    for func_dict_schema in dictionaries:
        model_base_name = func_dict_schema.get("name", "DynamicModelFromDict")
        dyn_model = convert_dictionary_to_pydantic_model(func_dict_schema, model_base_name)
        dynamic_models.append(dyn_model)
    return dynamic_models
