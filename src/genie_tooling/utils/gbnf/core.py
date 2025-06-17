### src/genie_tooling/utils/gbnf/core.py
from __future__ import annotations

import inspect
import re
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

if TYPE_CHECKING:
    pass
else:
    pass


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
    ANY = "any"
    NULL = "null"
    CUSTOM_CLASS = "custom-class"
    CUSTOM_DICT = "custom-dict"
    SET = "set"
    LITERAL = "literal"


def format_model_and_field_name(name: str) -> str:
    """
    Converts a model or field name to a GBNF-compatible kebab-case string.
    e.g., MyModelName -> my-model-name, my_field_name -> my-field-name, URLProcessor -> url-processor.
    """
    if not name:
        return "unnamed-component"
    s1 = name.replace("_", "-")
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)
    s3 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s2)
    return s3.lower()


def map_pydantic_type_to_gbnf(pydantic_type: Type[Any]) -> str:
    """
    Maps a Python/Pydantic type to a GBNF rule name or primitive type name.
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
        return "unknown" # For generic 'value' rule

    elif inspect.isclass(check_type) and issubclass(check_type, Enum):
        return format_model_and_field_name(check_type.__name__)
    elif inspect.isclass(check_type) and issubclass(check_type, BaseModel):
        return format_model_and_field_name(check_type.__name__)

    elif check_type is list or check_type is List or check_type is set or check_type is Set or check_type is tuple or check_type is Tuple:
        # For GBNF list representation, we simplify. Tuples become array-like.
        return PydanticDataType.ARRAY.value

    elif origin_type is Union:
        non_none_types = [ut for ut in args if ut is not type(None)]
        if not non_none_types:
            return PydanticDataType.NULL.value
        if len(non_none_types) == 1: # Optional[T] or Union[T, None]
            base_type_name = map_pydantic_type_to_gbnf(non_none_types[0])
            return f"optional-{base_type_name}" if type(None) in args else base_type_name
        # For complex Union[A, B, C], generate a name based on sorted members
        union_member_names = sorted({map_pydantic_type_to_gbnf(ut) for ut in non_none_types})
        return f"union-{'-or-'.join(union_member_names)}"

    elif origin_type is Literal:
        return PydanticDataType.LITERAL.value # Specific rule will be generated

    elif check_type is dict or check_type is Dict:
        # Complex Dict[K,V] will be handled by generate_gbnf_rule_for_type to ensure V's rules are made
        return PydanticDataType.OBJECT.value

    elif inspect.isclass(check_type): # A class, but not BaseModel or Enum
        return "unknown" # Treat as an unknown type, will default to string or object based on context
    else:
        return "unknown" # Fallback for other unhandled types


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
