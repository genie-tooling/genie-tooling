### src/genie_tooling/utils/gbnf/core.py
from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import Any, Dict, Set, TYPE_CHECKING, List, Union, Type, get_origin, get_args, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from types import GenericAlias # type: ignore[attr-defined]
else:
    # python 3.8 compat
    from typing import _GenericAlias as GenericAlias # type: ignore[attr-defined]


class PydanticDataType(Enum):
    """
    Defines the data types supported by the grammar_generator.
    """

    STRING = "string"
    TRIPLE_QUOTED_STRING = "triple-quoted-string"
    MARKDOWN_CODE_BLOCK = "markdown-code-block"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    OBJECT = "object"
    ARRAY = "array"
    ENUM = "enum"
    ANY = "any" # Represents a placeholder for any JSON value
    NULL = "null"
    CUSTOM_CLASS = "custom-class" # For non-Pydantic, non-Enum classes
    CUSTOM_DICT = "custom-dict" # For Dict[K,V] where K or V is complex
    SET = "set" # Represented same as array in GBNF
    LITERAL = "literal" # For Python's typing.Literal


def format_model_and_field_name(name: str) -> str:
    """
    Converts a model or field name to a GBNF-compatible kebab-case string.
    e.g., MyModelName -> my-model-name, my_field_name -> my-field-name, URLProcessor -> url-processor.
    """
    if not name:
        return "unnamed-component"
    # Handle specific known acronyms or initialisms if needed, though general rules are better.
    # Example: if name == "URLProcessor": return "url-processor" (covered by general logic)

    s1 = name.replace("_", "-")
    # Insert hyphen before uppercase letters that are preceded by a lowercase letter or digit
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)
    # Insert hyphen before uppercase letters that are part of an acronym (e.g., "URL" in "MyURLProcessor")
    # This specifically targets sequences of uppercase letters followed by an uppercase then lowercase (e.g., XXY)
    s3 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s2)
    return s3.lower()


def map_pydantic_type_to_gbnf(pydantic_type: Type[Any]) -> str:
    """
    Maps a Python/Pydantic type to a GBNF rule name or primitive type name.
    For complex types like List[T], it returns a name like "t-name-list",
    and a specific rule definition (e.g., "my-model-my-list-list-def") will be
    created by generate_gbnf_rule_for_type.
    """
    origin_type = get_origin(pydantic_type)
    args = get_args(pydantic_type)

    check_type = origin_type if origin_type is not None else pydantic_type

    if check_type is str:
        return PydanticDataType.STRING.value
    elif check_type is bool:
        return PydanticDataType.BOOLEAN.value
    elif check_type is int:
        return PydanticDataType.INTEGER.value
    elif check_type is float:
        return PydanticDataType.FLOAT.value
    elif check_type is type(None):
        return PydanticDataType.NULL.value
    elif check_type is Any:
        # FIX: Change "any" to "unknown" to match test expectation.
        # The primitive grammar generation for "unknown" should result in a generic 'value' rule.
        return "unknown"

    elif inspect.isclass(check_type) and issubclass(check_type, Enum):
        return format_model_and_field_name(check_type.__name__)
    elif inspect.isclass(check_type) and issubclass(check_type, BaseModel):
        return format_model_and_field_name(check_type.__name__)

    elif check_type is list or check_type is List:
        element_type = args[0] if args else Any
        return f"{map_pydantic_type_to_gbnf(element_type)}-list"
    elif check_type is set or check_type is Set:
        element_type = args[0] if args else Any
        return f"{map_pydantic_type_to_gbnf(element_type)}-set"

    elif origin_type is Union:
        non_none_types = [ut for ut in args if ut is not type(None)]
        if not non_none_types:
            return PydanticDataType.NULL.value
        if len(non_none_types) == 1:
            base_type_name = map_pydantic_type_to_gbnf(non_none_types[0])
            return f"optional-{base_type_name}" if type(None) in args else base_type_name
        union_member_names = sorted(list(set(map_pydantic_type_to_gbnf(ut) for ut in non_none_types)))
        return f"union-{'-or-'.join(union_member_names)}"

    elif origin_type is Literal:
        return PydanticDataType.LITERAL.value

    elif check_type is dict or check_type is Dict:
        if args and len(args) == 2:
            key_type, value_type = args
            return f"custom-dict-key-{map_pydantic_type_to_gbnf(key_type)}-value-{map_pydantic_type_to_gbnf(value_type)}"
        return PydanticDataType.OBJECT.value

    elif inspect.isclass(check_type):
        return f"{PydanticDataType.CUSTOM_CLASS.value}-{format_model_and_field_name(check_type.__name__)}"
    else:
        return "unknown"


def regex_to_gbnf(regex_pattern: str) -> str:
    """Converts common Python regex syntax to GBNF-compatible character classes."""
    gbnf_rule = regex_pattern
    gbnf_rule = gbnf_rule.replace("\\d", "[0-9]")
    gbnf_rule = gbnf_rule.replace("\\s", "[ \t\n]")
    gbnf_rule = gbnf_rule.replace("\\w", "[a-zA-Z0-9_]")
    return gbnf_rule


def remove_empty_lines(text: str) -> str:
    """Removes empty lines from a string."""
    if not text:
        return ""
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return "\n".join(non_empty_lines)