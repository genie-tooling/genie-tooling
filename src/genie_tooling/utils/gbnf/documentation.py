### src/genie_tooling/utils/gbnf/documentation.py
# src/genie_tooling/utils/gbnf/documentation.py
from __future__ import annotations

import inspect
import json
from inspect import getdoc, isclass  # For documentation generation
from typing import (
    Any,
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

from pydantic import BaseModel, RootModel
from pydantic.fields import FieldInfo
from pydantic_core import (
    PydanticUndefined,  # For checking default values in Pydantic V2
)

from .constructor import (
    generate_gbnf_grammar_from_pydantic_models,
)
from .core import format_model_and_field_name

# Import for create_dynamic_models_from_dictionaries
from .model_factory import create_dynamic_models_from_dictionaries


def format_multiline_description(description: str, indent_level: int) -> str:
    """Indents a multiline description string."""
    indent = "  " * indent_level
    return indent + description.replace("\n", f"\n{indent}")


def _get_python_type_name(py_type: Type[Any]) -> str:
    """Helper to get a readable Python type name, handling generics."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        member_names = sorted(list(set(_get_python_type_name(arg) for arg in non_none_args)))
        type_name_str = " or ".join(member_names)
        if type(None) in args:
            return f"({type_name_str} or none-type)"
        return type_name_str
    elif origin in [list, List]:
        element_type = args[0] if args else Any
        return f"list of {_get_python_type_name(element_type)}"
    elif origin in [set, Set]:
        element_type = args[0] if args else Any
        return f"set of {_get_python_type_name(element_type)}"
    elif origin in [dict, Dict]:
        key_type = args[0] if args and len(args) > 0 else Any
        val_type = args[1] if args and len(args) > 1 else Any
        return f"dict with {_get_python_type_name(key_type)} keys and {_get_python_type_name(val_type)} values"
    else:
        return getattr(py_type, "__name__", str(py_type))


def generate_field_text(
    field_name: str, field_type: Type[Any], model: Type[BaseModel], depth: int = 1,
    documentation_with_field_description: bool = True
) -> str:
    """Generates text documentation for a single Pydantic field."""
    indent = "  " * depth
    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""
    type_name_str = _get_python_type_name(field_type)

    field_text = f"{indent}{field_name} ({type_name_str})"

    is_field_required = field_info.is_required() if field_info else True
    is_explicitly_optional = get_origin(field_type) is Union and type(None) in get_args(field_type)

    if not is_field_required or is_explicitly_optional:
        field_text += " (optional"
        has_default_info = False
        if field_info and field_info.default_factory is not None:
            factory_name = getattr(field_info.default_factory, "__name__", repr(field_info.default_factory))
            field_text += f", default_factory: {factory_name}"
            has_default_info = True
        # Check for explicit default, including `None` if explicitly set
        elif field_info and field_info.default is not PydanticUndefined:
            field_text += f", default: {repr(field_info.default)}"
            has_default_info = True
        # If it's Optional[T] (is_explicitly_optional) and no other default was specified (default is PydanticUndefined)
        # then it implicitly defaults to None.
        elif is_explicitly_optional and field_info and field_info.default is PydanticUndefined:
             field_text += ", default: None"
             has_default_info = True

        if not has_default_info and not is_field_required:
             pass

        field_text += ")"
    field_text += "\n"

    if documentation_with_field_description and field_description:
        desc_indent = "  " * (depth + 1)
        field_text += f"{desc_indent}Description: {field_description}\n"

    if documentation_with_field_description and field_info:
        examples_to_show = None
        if hasattr(field_info, "examples") and field_info.examples:
            examples_to_show = field_info.examples
        elif isinstance(field_info.json_schema_extra, dict):
            if "example" in field_info.json_schema_extra:
                 examples_to_show = [field_info.json_schema_extra["example"]]
            elif "examples" in field_info.json_schema_extra and isinstance(field_info.json_schema_extra["examples"], list):
                 examples_to_show = field_info.json_schema_extra["examples"]

        if examples_to_show:
            example_indent = "  " * (depth + 1)
            for i, ex in enumerate(examples_to_show):
                example_text = f"'{ex}'" if isinstance(ex, str) else json.dumps(ex)
                field_text += f"{example_indent}Example{'' if len(examples_to_show) == 1 else f' {i+1}'}: {example_text}\n"
    return field_text


def generate_text_documentation(
    pydantic_models: List[Type[BaseModel]], model_prefix: str = "Model", fields_prefix: str = "Fields",
    documentation_with_field_description: bool = True
) -> str:
    documentation = ""
    models_to_document_queue: List[Tuple[Type[BaseModel], bool, int]] = [(model, True, 0) for model in pydantic_models]
    processed_model_names: Set[str] = set()
    idx = 0

    while idx < len(models_to_document_queue):
        model_cls, is_top_level, current_depth = models_to_document_queue[idx]
        idx += 1
        model_name = model_cls.__name__

        if model_name in processed_model_names and not is_top_level:
            continue
        if is_top_level or model_name not in processed_model_names:
            processed_model_names.add(model_name)
        else:
            continue

        indent = "  " * current_depth
        documentation += f"{indent}{model_prefix if is_top_level else 'Nested Model'}: {model_name}\n"

        class_doc = getdoc(model_cls)
        base_model_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_model_doc else ""
        if class_description:
            documentation += f"{indent}  Description:\n"
            documentation += format_multiline_description(class_description, current_depth + 2) + "\n"

        is_regular_model_with_fields = hasattr(model_cls, "model_fields") and model_cls.model_fields
        is_root_model_with_field = hasattr(model_cls, "model_fields") and "root" in model_cls.model_fields and issubclass(model_cls, RootModel)


        if is_regular_model_with_fields and not is_root_model_with_field: # Standard BaseModel
            documentation += f"{indent}  {fields_prefix if is_top_level else 'Fields'}:\n"
            type_hints_for_model = get_type_hints(model_cls)
            for field_name_str, field_info_obj in model_cls.model_fields.items():
                field_py_type = type_hints_for_model.get(field_name_str, field_info_obj.annotation)
                documentation += generate_field_text(
                    field_name_str, field_py_type, model_cls, depth=current_depth + 2,
                    documentation_with_field_description=documentation_with_field_description
                )
                origin = get_origin(field_py_type)
                args = get_args(field_py_type)
                types_to_check_for_nesting: List[Type[Any]] = []
                if origin in [list, List, set, Set, Union] and args:
                    types_to_check_for_nesting.extend(arg for arg in args if arg is not type(None))
                elif isclass(field_py_type) and issubclass(field_py_type, BaseModel):
                    types_to_check_for_nesting.append(field_py_type)

                for nested_type_candidate in types_to_check_for_nesting:
                    actual_nested_type = get_origin(nested_type_candidate) or nested_type_candidate
                    if isclass(actual_nested_type) and issubclass(actual_nested_type, BaseModel):
                        if not any(m[0] == actual_nested_type for m in models_to_document_queue[idx:]):
                             models_to_document_queue.append((actual_nested_type, False, current_depth + 1))

        if is_root_model_with_field: # Specifically for RootModel
            root_field_type = model_cls.model_fields["root"].annotation
            type_name_str = _get_python_type_name(root_field_type)
            documentation += f"{indent}  Root Type: {type_name_str}\n"

        if not is_regular_model_with_fields and not is_root_model_with_field:
             documentation += f"{indent}  (This model has no fields defined.)\n"
        documentation += "\n"

    return documentation.strip()

def _get_markdown_type_name(py_type: Type[Any]) -> str:
    """Helper to get a markdown-formatted Python type name."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none_args = [arg for arg in args if arg is not type(None)]
        member_names_formatted = sorted(list(set(
            f"`{format_model_and_field_name(getattr(get_origin(arg) or arg, '__name__', str(arg)))}`"
            if inspect.isclass(get_origin(arg) or arg) and issubclass(get_origin(arg) or arg, BaseModel)
            else f"`{getattr(get_origin(arg) or arg, '__name__', str(arg))}`"
            for arg in non_none_args
        )))
        type_name_str = " or ".join(member_names_formatted)
        if type(None) in args:
            return f"({type_name_str} or `none-type`)"
        return type_name_str
    elif origin in [list, List]:
        element_type = args[0] if args else Any
        return f"List of {_get_markdown_type_name(element_type)}"
    elif origin in [set, Set]:
        element_type = args[0] if args else Any
        return f"Set of {_get_markdown_type_name(element_type)}"
    else:
        type_name_raw = getattr(py_type, "__name__", str(py_type))
        if isclass(py_type) and issubclass(py_type, BaseModel):
            return f"`{format_model_and_field_name(type_name_raw)}`"
        return f"`{type_name_raw}`"


def generate_field_markdown(
    field_name: str, field_type: Type[Any], model: Type[BaseModel], depth: int = 0,
    documentation_with_field_description: bool = True
) -> str:
    """Generates markdown documentation for a single Pydantic field."""
    indent = "  " * depth
    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""
    type_name_str = _get_markdown_type_name(field_type)

    field_text = f"{indent}*   **`{field_name}`** ({type_name_str})"

    is_field_required = field_info.is_required() if field_info else True
    is_explicitly_optional = get_origin(field_type) is Union and type(None) in get_args(field_type)

    if not is_field_required or is_explicitly_optional:
        field_text += " (optional"
        has_default_info = False
        if field_info and field_info.default_factory is not None:
            factory_name = getattr(field_info.default_factory, "__name__", repr(field_info.default_factory))
            field_text += f", default_factory: `{factory_name}`"
            has_default_info = True
        elif field_info and field_info.default is not PydanticUndefined:
            field_text += f", default: `{repr(field_info.default)}`"
            has_default_info = True
        elif is_explicitly_optional and field_info and field_info.default is PydanticUndefined:
             field_text += ", default: `None`"
             has_default_info = True

        if not has_default_info and not is_field_required:
             pass

        field_text += ")"

    if documentation_with_field_description and field_description:
        field_text += f": {field_description}\n"
    else:
        field_text += "\n"

    if documentation_with_field_description and field_info:
        examples_to_show = None
        if hasattr(field_info, "examples") and field_info.examples:
            examples_to_show = field_info.examples
        elif isinstance(field_info.json_schema_extra, dict):
            if "example" in field_info.json_schema_extra:
                 examples_to_show = [field_info.json_schema_extra["example"]]
            elif "examples" in field_info.json_schema_extra and isinstance(field_info.json_schema_extra["examples"], list):
                 examples_to_show = field_info.json_schema_extra["examples"]

        if examples_to_show:
            example_indent = indent + "    "
            for i, ex in enumerate(examples_to_show):
                example_text = f"'{ex}'" if isinstance(ex, str) else json.dumps(ex)
                field_text += f"{example_indent}- *Example{'' if len(examples_to_show) == 1 else f' {i+1}'}*: `{example_text}`\n"
    return field_text


def generate_markdown_documentation(
    pydantic_models: List[Type[BaseModel]], model_prefix: str = "Model", fields_prefix: str = "Fields",
    documentation_with_field_description: bool = True
) -> str:
    documentation = ""
    models_to_document_queue: List[Tuple[Type[BaseModel], bool, int]] = [(model, True, 0) for model in pydantic_models]
    processed_model_names: Set[str] = set()
    idx = 0

    while idx < len(models_to_document_queue):
        model_cls, is_top_level, current_depth = models_to_document_queue[idx]
        idx += 1
        model_name = model_cls.__name__

        if model_name in processed_model_names and not is_top_level:
            continue
        if is_top_level or model_name not in processed_model_names:
            processed_model_names.add(model_name)
        else:
            continue

        header_level = 2 + current_depth
        header_hashes = "#" * header_level
        documentation += f"{header_hashes} {model_prefix if is_top_level else 'Nested Model'}: `{model_name}`\n\n"

        class_doc = getdoc(model_cls)
        base_model_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_model_doc else ""
        if class_description:
            documentation += f"**Description**: {class_description}\n\n"

        is_regular_model_with_fields = hasattr(model_cls, "model_fields") and model_cls.model_fields
        is_root_model_with_field = hasattr(model_cls, "model_fields") and "root" in model_cls.model_fields and issubclass(model_cls, RootModel)


        if is_regular_model_with_fields and not is_root_model_with_field:
            documentation += f"**{fields_prefix if is_top_level else 'Fields'}**:\n"
            type_hints_for_model = get_type_hints(model_cls)
            for field_name_str, field_info_obj in model_cls.model_fields.items():
                field_py_type = type_hints_for_model.get(field_name_str, field_info_obj.annotation)
                documentation += generate_field_markdown(
                    field_name_str, field_py_type, model_cls, depth=0,
                    documentation_with_field_description=documentation_with_field_description
                )
                origin = get_origin(field_py_type)
                args = get_args(field_py_type)
                types_to_check_for_nesting: List[Type[Any]] = []
                if origin in [list, List, set, Set, Union] and args:
                    types_to_check_for_nesting.extend(arg for arg in args if arg is not type(None))
                elif isclass(field_py_type) and issubclass(field_py_type, BaseModel):
                     types_to_check_for_nesting.append(field_py_type)

                for nested_type_candidate in types_to_check_for_nesting:
                    actual_nested_type = get_origin(nested_type_candidate) or nested_type_candidate
                    if isclass(actual_nested_type) and issubclass(actual_nested_type, BaseModel):
                        if not any(m[0] == actual_nested_type for m in models_to_document_queue[idx:]):
                             models_to_document_queue.append((actual_nested_type, False, current_depth + 1))
            documentation += "\n"

        if is_root_model_with_field:
            root_field_type = model_cls.model_fields["root"].annotation
            type_name_str = _get_markdown_type_name(root_field_type)
            documentation += f"**Root Type**: {type_name_str}\n\n"

        if not is_regular_model_with_fields and not is_root_model_with_field:
             documentation += "  *(This model has no fields defined.)*\n\n"


        model_config = getattr(model_cls, "model_config", {})
        model_config_json_schema_extra = model_config.get("json_schema_extra", {}) if isinstance(model_config, dict) else {}

        model_examples = None
        if isinstance(model_config_json_schema_extra, dict) and "example" in model_config_json_schema_extra:
            model_examples = [model_config_json_schema_extra["example"]]
        elif isinstance(model_config_json_schema_extra, dict) and "examples" in model_config_json_schema_extra:
            model_examples = model_config_json_schema_extra["examples"]

        if model_examples:
            documentation += f"**Example Output for `{format_model_and_field_name(model_name)}`**:\n"
            for i, ex_data in enumerate(model_examples):
                if len(model_examples) > 1:
                    documentation += f"*Example {i+1}:*\n"
                json_example = json.dumps(ex_data, indent=2)
                documentation += f"```json\n{json_example}\n```\n\n"
        elif not is_regular_model_with_fields and not is_root_model_with_field:
             if not (hasattr(model_cls, "root_field") and isinstance(model_cls.root_field, FieldInfo)):
                pass

    return documentation.strip()


def save_gbnf_grammar_and_documentation(
    grammar: str,
    documentation: str,
    grammar_file_path: str = "./grammar.gbnf",
    documentation_file_path: str = "./grammar_documentation.md",
):
    """Saves the generated GBNF grammar and Markdown documentation to files."""
    try:
        with open(grammar_file_path, "w", encoding="utf-8") as file:
            file.write(grammar)
        print(f"Grammar successfully saved to {grammar_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the grammar file: {e}")

    try:
        with open(documentation_file_path, "w", encoding="utf-8") as file:
            file.write(documentation)
        print(f"Documentation successfully saved to {documentation_file_path}")
    except IOError as e:
        print(f"An error occurred while saving the documentation file: {e}")


def generate_and_save_gbnf_grammar_and_documentation(
    pydantic_model_list: List[Type[BaseModel]],
    grammar_file_path: str = "./generated_grammar.gbnf",
    documentation_file_path: str = "./generated_grammar_documentation.md",
    outer_object_name: Optional[str] = None,
    outer_object_content_key: Optional[str] = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description: bool = True,
):
    """Generates and saves both GBNF grammar and Markdown documentation."""
    grammar, documentation = generate_gbnf_grammar_and_documentation(
        pydantic_model_list=pydantic_model_list,
        outer_object_name=outer_object_name,
        outer_object_content_key=outer_object_content_key,
        model_prefix=model_prefix,
        fields_prefix=fields_prefix,
        list_of_outputs=list_of_outputs,
        documentation_with_field_description=documentation_with_field_description,
    )
    save_gbnf_grammar_and_documentation(grammar, documentation, grammar_file_path, documentation_file_path)


def generate_gbnf_grammar_and_documentation(
    pydantic_model_list: List[Type[BaseModel]],
    outer_object_name: Optional[str] = None,
    outer_object_content_key: Optional[str] = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description: bool = True,
) -> Tuple[str, str]:
    """Generates GBNF grammar and Markdown documentation strings."""
    documentation = generate_markdown_documentation(
        list(pydantic_model_list), model_prefix, fields_prefix,
        documentation_with_field_description=documentation_with_field_description
    )
    grammar = generate_gbnf_grammar_from_pydantic_models(
        pydantic_model_list, outer_object_name, outer_object_content_key, list_of_outputs
    )
    return grammar, documentation


def generate_gbnf_grammar_and_documentation_from_dictionaries(
    dictionaries: List[Dict[str, Any]],
    outer_object_name: Optional[str] = None,
    outer_object_content_key: Optional[str] = None,
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    list_of_outputs: bool = False,
    documentation_with_field_description: bool = True,
) -> Tuple[str, str]:
    """Generates GBNF and docs from JSON schema-like dictionaries."""
    pydantic_model_list = create_dynamic_models_from_dictionaries(dictionaries)
    return generate_gbnf_grammar_and_documentation(
        pydantic_model_list,
        outer_object_name=outer_object_name,
        outer_object_content_key=outer_object_content_key,
        model_prefix=model_prefix,
        fields_prefix=fields_prefix,
        list_of_outputs=list_of_outputs,
        documentation_with_field_description=documentation_with_field_description,
    )

def map_grammar_names_to_pydantic_model_class(
    pydantic_model_list: List[Type[BaseModel]]
) -> Dict[str, Type[BaseModel]]:
    """Maps formatted GBNF rule names back to their Pydantic model classes."""
    return {format_model_and_field_name(model.__name__): model for model in pydantic_model_list}
