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

from pydantic import BaseModel  # Renamed PydanticField

from .constructor import (
    generate_gbnf_grammar_from_pydantic_models,
)
from .core import format_model_and_field_name

# Import for create_dynamic_models_from_dictionaries
from .model_factory import create_dynamic_models_from_dictionaries


def format_multiline_description(description: str, indent_level: int) -> str:
    """Indents a multiline description string."""
    indent = "  " * indent_level # Two spaces per indent level for text doc
    # For markdown, it's often 4 spaces for code blocks under list items.
    # This function is used by text_doc which expects 2 spaces per level.
    return indent + description.replace("\n", f"\n{indent}")


def generate_field_text(
    field_name: str, field_type: Type[Any], model: Type[BaseModel], depth: int = 1,
    documentation_with_field_description: bool = True
) -> str:
    """Generates text documentation for a single Pydantic field."""
    indent = "  " * depth # Typically starting at depth=2 for fields (4 spaces)
    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""

    # Type string generation
    type_name_str = ""
    origin_type = get_origin(field_type)
    args_type = get_args(field_type)

    if origin_type is Union:
        non_none_args = [arg for arg in args_type if arg is not type(None)]
        member_names = sorted(list(set(getattr(get_origin(arg) or arg, "__name__", str(arg)) for arg in non_none_args)))
        type_name_str = " or ".join(member_names)
        if type(None) in args_type: # Indicates Optional or Union including None
            type_name_str = f"({type_name_str} or none-type)"
    elif origin_type is list or origin_type is List:
        element_type = args_type[0] if args_type else Any
        element_name = getattr(get_origin(element_type) or element_type, "__name__", str(element_type))
        type_name_str = f"list of {element_name}"
    elif origin_type is set or origin_type is Set:
        element_type = args_type[0] if args_type else Any
        element_name = getattr(get_origin(element_type) or element_type, "__name__", str(element_type))
        type_name_str = f"set of {element_name}"
    elif origin_type is dict or origin_type is Dict:
        key_type = args_type[0] if args_type and len(args_type) > 0 else Any
        val_type = args_type[1] if args_type and len(args_type) > 1 else Any
        key_name = getattr(get_origin(key_type) or key_type, "__name__", str(key_type))
        val_name = getattr(get_origin(val_type) or val_type, "__name__", str(val_type))
        type_name_str = f"dict with {key_name} keys and {val_name} values"
    else: # Simple type or nested BaseModel
        type_name_str = getattr(field_type, "__name__", str(field_type))
        if isclass(field_type) and issubclass(field_type, BaseModel): # Format nested model names
             type_name_str = format_model_and_field_name(type_name_str) # Format model names

    field_text = f"{indent}{field_name} ({type_name_str})"
    if field_info and not field_info.is_required():
        field_text += " (optional"
        if field_info.default is not None and field_info.default is not ...: # PydanticV2 uses ... for required
            field_text += f", default: {repr(field_info.default)}"
        elif field_info.default_factory is not None:
            # Attempt to get the name of the factory function
            field_text += f", default_factory: {getattr(field_info.default_factory, '__name__', repr(field_info.default_factory))}"
        field_text += ")"
    field_text += "\n"

    if documentation_with_field_description and field_description:
        desc_indent = "  " * (depth + 1)
        field_text += f"{desc_indent}Description: {field_description}\n"

    # Field-specific example from schema if available (Pydantic v2)
    if documentation_with_field_description and field_info:
        field_json_schema_extra = getattr(field_info, "json_schema_extra", None)
        if isinstance(field_json_schema_extra, dict) and "example" in field_json_schema_extra:
            example_indent = "  " * (depth + 1)
            field_example = field_json_schema_extra["example"]
            example_text = f"'{field_example}'" if isinstance(field_example, str) else json.dumps(field_example)
            field_text += f"{example_indent}Example: {example_text}\n"
    return field_text


def generate_text_documentation(
    pydantic_models: List[Type[BaseModel]], model_prefix: str = "Model", fields_prefix: str = "Fields",
    documentation_with_field_description: bool = True
) -> str:
    """Generates text documentation for a list of Pydantic models."""
    documentation = ""
    # Queue stores (model_class, is_top_level_in_initial_list, depth_level_for_header)
    models_to_document_queue: List[Tuple[Type[BaseModel], bool, int]] = [(model, True, 0) for model in pydantic_models]
    processed_model_names: Set[str] = set() # Track by name to avoid re-documenting identical nested models
    idx = 0

    while idx < len(models_to_document_queue):
        model_cls, is_top_level, current_depth = models_to_document_queue[idx]
        idx += 1
        model_name = model_cls.__name__

        if model_name in processed_model_names and not is_top_level:
            continue
        processed_model_names.add(model_name)

        indent = "  " * current_depth
        documentation += f"{indent}{model_prefix if is_top_level else 'Nested Model'}: {model_name}\n"

        class_doc = getdoc(model_cls)
        base_model_doc = getdoc(BaseModel) # To avoid printing Pydantic's own docstring
        class_description = class_doc if class_doc and class_doc != base_model_doc else ""
        if class_description:
            documentation += f"{indent}  Description:\n"
            documentation += format_multiline_description(class_description, current_depth + 2) + "\n"

        # Fields section
        if model_cls.model_fields:
            documentation += f"{indent}  {fields_prefix if is_top_level else 'Fields'}:\n"
            type_hints_for_model = get_type_hints(model_cls) # Resolve type hints once per model
            for field_name_str, field_info_obj in model_cls.model_fields.items():
                # Use resolved type hints for better accuracy with ForwardRefs etc.
                field_py_type = type_hints_for_model.get(field_name_str, field_info_obj.annotation)
                documentation += generate_field_text(
                    field_name_str, field_py_type, model_cls, depth=current_depth + 2,
                    documentation_with_field_description=documentation_with_field_description
                )
                # Add nested Pydantic models to the queue for documentation
                origin = get_origin(field_py_type)
                args = get_args(field_py_type)
                types_to_check_for_nesting: List[Type[Any]] = []
                if origin in [list, List, set, Set, Union] and args: # Handle Optional via Union
                    types_to_check_for_nesting.extend(arg for arg in args if arg is not type(None))
                elif isclass(field_py_type) and issubclass(field_py_type, BaseModel):
                    types_to_check_for_nesting.append(field_py_type)

                for nested_type_candidate in types_to_check_for_nesting:
                    # Resolve origin again for elements of generics/unions
                    actual_nested_type = get_origin(nested_type_candidate) or nested_type_candidate
                    if isclass(actual_nested_type) and issubclass(actual_nested_type, BaseModel) and actual_nested_type.__name__ not in processed_model_names:
                        if not any(m[0] == actual_nested_type for m in models_to_document_queue[idx:]): # Avoid adding duplicates to queue
                             models_to_document_queue.append((actual_nested_type, False, current_depth + 1))
        elif not model_cls.model_fields:
             documentation += f"{indent}  (This model has no fields defined.)\n"
        documentation += "\n"

    return documentation.strip()


def generate_field_markdown(
    field_name: str, field_type: Type[Any], model: Type[BaseModel], depth: int = 0,
    documentation_with_field_description: bool = True
) -> str:
    """Generates markdown documentation for a single Pydantic field."""
    indent = "  " * depth
    field_info = model.model_fields.get(field_name)
    field_description = field_info.description if field_info and field_info.description else ""

    type_name_str = ""
    origin_type = get_origin(field_type)
    args_type = get_args(field_type)

    if origin_type is Union:
        non_none_args = [arg for arg in args_type if arg is not type(None)]
        member_names = sorted(list(set(f"`{format_model_and_field_name(getattr(get_origin(arg) or arg, '__name__', str(arg)))}`" for arg in non_none_args)))
        type_name_str = " or ".join(member_names)
        if type(None) in args_type:
            type_name_str = f"({type_name_str} or `none-type`)"
    elif origin_type is list or origin_type is List:
        element_type = args_type[0] if args_type else Any
        element_name_raw = getattr(get_origin(element_type) or element_type, "__name__", str(element_type))
        element_name = f"`{format_model_and_field_name(element_name_raw)}`" if inspect.isclass(get_origin(element_type) or element_type) and issubclass(get_origin(element_type) or element_type, BaseModel) else f"`{element_name_raw}`"
        type_name_str = f"List of {element_name}"
    elif origin_type is set or origin_type is Set:
        element_type = args_type[0] if args_type else Any
        element_name_raw = getattr(get_origin(element_type) or element_type, "__name__", str(element_type))
        element_name = f"`{format_model_and_field_name(element_name_raw)}`" if inspect.isclass(get_origin(element_type) or element_type) and issubclass(get_origin(element_type) or element_type, BaseModel) else f"`{element_name_raw}`"
        type_name_str = f"Set of {element_name}"
    else:
        type_name_raw = getattr(field_type, "__name__", str(field_type))
        type_name_str = f"`{format_model_and_field_name(type_name_raw)}`" if inspect.isclass(field_type) and issubclass(field_type, BaseModel) else f"`{type_name_raw}`"

    field_text = f"{indent}*   **`{field_name}`** ({type_name_str})"
    if field_info and not field_info.is_required():
        field_text += " (optional"
        if field_info.default is not None and field_info.default is not ...:
            field_text += f", default: `{repr(field_info.default)}`"
        elif field_info.default_factory is not None:
            field_text += f", default_factory: `{getattr(field_info.default_factory, '__name__', repr(field_info.default_factory))}`"
        field_text += ")"

    if documentation_with_field_description and field_description:
        field_text += f": {field_description}\n"
    else:
        field_text += "\n"

    if documentation_with_field_description and field_info:
        field_json_schema_extra = getattr(field_info, "json_schema_extra", None)
        if isinstance(field_json_schema_extra, dict) and "example" in field_json_schema_extra:
            example_indent = indent + "    "
            field_example = field_json_schema_extra["example"]
            example_text = f"'{field_example}'" if isinstance(field_example, str) else json.dumps(field_example)
            field_text += f"{example_indent}- *Example*: `{example_text}`\n"
    return field_text


def generate_markdown_documentation(
    pydantic_models: List[Type[BaseModel]], model_prefix: str = "Model", fields_prefix: str = "Fields",
    documentation_with_field_description: bool = True
) -> str:
    """Generates markdown documentation for a list of Pydantic models."""
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
        processed_model_names.add(model_name)

        header_hashes = "#" * (2 + current_depth)
        documentation += f"{header_hashes} {model_prefix if is_top_level else 'Nested Model'}: `{model_name}`\n\n"

        class_doc = getdoc(model_cls)
        base_model_doc = getdoc(BaseModel)
        class_description = class_doc if class_doc and class_doc != base_model_doc else ""
        if class_description:
            documentation += f"**Description**: {class_description}\n\n"

        if model_cls.model_fields:
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
                    if isclass(actual_nested_type) and issubclass(actual_nested_type, BaseModel) and \
                       actual_nested_type.__name__ not in processed_model_names:
                        if not any(m[0] == actual_nested_type for m in models_to_document_queue[idx:]):
                             models_to_document_queue.append((actual_nested_type, False, current_depth + 1))
            documentation += "\n"

        model_config = getattr(model_cls, "model_config", {})
        model_config_json_schema_extra = model_config.get("json_schema_extra", {}) if isinstance(model_config, dict) else {}
        if isinstance(model_config_json_schema_extra, dict) and "example" in model_config_json_schema_extra:
            documentation += f"**Example Output for `{format_model_and_field_name(model_name)}`**:\n"
            json_example = json.dumps(model_config_json_schema_extra["example"], indent=2)
            documentation += f"```json\n{json_example}\n```\n\n"
        elif not model_cls.model_fields:
             documentation += "  *(This model has no fields defined.)*\n\n"

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
