### tests/unit/utils/gbnf/test_constructor.py
import logging
import re
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import pytest
from genie_tooling.utils.gbnf.constructor import (
    generate_gbnf_float_rules,
    generate_gbnf_grammar,
    generate_gbnf_grammar_from_pydantic_models,
    generate_gbnf_integer_rules,
    generate_gbnf_rule_for_type,
    generate_list_rule,
    get_members_structure,
    get_primitive_grammar,
)
from pydantic import BaseModel, Field, RootModel

logger = logging.getLogger(__name__)
CONSTRUCTOR_LOGGER_NAME = "genie_tooling.utils.gbnf.constructor"


# --- Test Models ---
class SimpleStrEnum(Enum):
    OPTION_A = "Value A"
    OPTION_B = "Value B"


class NonStrEnum(Enum):
    INT_ONE = 1
    FLOAT_TWO = 2.0
    BOOL_TRUE = "true"


class GBNFNestedModel(BaseModel):
    """A simple nested model."""

    nested_field_int: int = Field(description="An integer in the nested model.")
    nested_field_str: Optional[str] = Field(
        None, description="An optional string."
    )


class GBNFRecursiveModel(BaseModel):
    """A model that can be nested recursively."""

    name: str
    children: List["GBNFRecursiveModel"] = Field(default_factory=list)


GBNFRecursiveModel.model_rebuild()


class GBNFRootListInt(RootModel[List[int]]):
    """A root model that is a list of integers."""

    pass


class GBNFRootListOfNested(RootModel[List[GBNFNestedModel]]):
    """A root model that is a list of GBNFNestedModel."""

    pass


class GBNFComprehensiveModel(BaseModel):
    """A comprehensive model for testing various GBNF features."""

    required_string: str = Field(description="A required string.")
    optional_int: Optional[int] = Field(None, description="An optional integer.")
    default_bool: bool = Field(True, description="A boolean with a default value.")

    list_of_str: List[str] = Field(description="A list of strings.")
    list_of_optional_int: Optional[List[Optional[int]]] = Field(
        None, description="A list that can contain integers or nulls."
    )
    list_of_nested: List[GBNFNestedModel] = Field(
        description="A list of nested models."
    )

    set_of_floats: Set[float] = Field(description="A set of unique float numbers.")

    dict_str_any: Dict[str, Any] = Field(
        description="A dictionary with string keys and any type of values."
    )
    dict_str_nested: Dict[str, GBNFNestedModel] = Field(
        description="A dictionary with GBNFNestedModel values."
    )

    simple_enum_field: SimpleStrEnum = Field(
        description="A field with a simple string enum."
    )
    non_string_enum_field: Optional[NonStrEnum] = Field(
        None, description="An enum with non-string values."
    )

    literal_field_str_int: Literal["Option1", 200, "Option3", True] = Field(
        description="A literal field with mixed types."
    )

    union_field_simple: Union[str, int, None] = Field(
        description="An optional union of simple types."
    )
    union_field_complex: Union[GBNFNestedModel, List[int], str] = Field(
        description="A union including a Pydantic model and a list."
    )

    any_field: Any = Field(description="A field that can be anything.")

    constrained_string_pattern: str = Field(
        json_schema_extra={"pattern": r"^[a-z]{3}-\d{2}$"},
        description="String matching 'abc-12'.",
    )
    constrained_int_digits: int = Field(
        json_schema_extra={"min_digit": 2, "max_digit": 3},
        description="An integer with 2 to 3 digits.",
    )
    constrained_float_precision: float = Field(
        json_schema_extra={
            "min_digit": 1,
            "max_digit": 4,
            "min_precision": 1,
            "max_precision": 2,
        },
        description="A float like 1.23 or 1234.5.",
    )

    triple_quoted_str_field: str = Field(
        json_schema_extra={"triple_quoted_string": True},
        description="A multi-line string.",
    )
    markdown_code_block_field: str = Field(
        json_schema_extra={"markdown_code_block": True},
        description="A Markdown code block.",
    )

    recursive_test: Optional[GBNFRecursiveModel] = Field(
        None, description="A recursive field."
    )
    root_model_as_field: Optional[GBNFRootListInt] = Field(
        None, description="A root model used as a field."
    )


# --- Tests for generate_list_rule ---
def test_generate_list_rule():
    rule_name = "my-element-rule"
    list_def_name = "my-list-def"
    expected = (
        rf'{list_def_name} ::= "[" ws ( {rule_name} ( ws "," ws {rule_name} )* ws )? "]"'
    )
    assert generate_list_rule(rule_name, list_def_name) == expected


# --- Tests for generate_gbnf_integer_rules ---
@pytest.mark.parametrize(
    "max_digit, min_digit, expected_rule_name, expected_rule_defs",
    [
        (None, None, "integer", []),
        (3, None, "integer-part-max3", ["integer-part-max3 ::= [0-9] [0-9]? [0-9]?"]),
        (None, 2, "integer-part-min2", ["integer-part-min2 ::= [0-9] [0-9] [0-9]*"]),
        (
            3,
            2,
            "integer-part-max3-min2",
            ["integer-part-max3-min2 ::= [0-9] [0-9] [0-9]?"],
        ),
        (0, 0, "integer-part-max0-min0", ["integer-part-max0-min0 ::= "]),
        (5, 5, "integer-part-max5-min5", ["integer-part-max5-min5 ::= [0-9] [0-9] [0-9] [0-9] [0-9]"]),
        (1, 0, "integer-part-max1-min0", ["integer-part-max1-min0 ::= [0-9]?"]),
    ],
)
def test_generate_gbnf_integer_rules(
    max_digit, min_digit, expected_rule_name, expected_rule_defs
):
    rule_name, rule_defs = generate_gbnf_integer_rules(max_digit, min_digit)
    assert rule_name == expected_rule_name
    assert sorted(rule_defs) == sorted(expected_rule_defs)


# --- Tests for generate_gbnf_float_rules ---
@pytest.mark.parametrize(
    "max_d, min_d, max_p, min_p, expected_rule_name_part, expected_additional_rules_count_min",
    [
        (None, None, None, None, "float", 0),
        (3, 1, None, None, "float-d3-1", 1),
        (None, None, 2, 1, "float-p2-1", 1),
        (
            4, 2, 3, 1,
            "float-d4-2-p3-1",
            3,
        ),
        (0, 0, 1, 1, "float-d0-0-p1-1", 3),
    ],
)
def test_generate_gbnf_float_rules(
    max_d, min_d, max_p, min_p, expected_rule_name_part, expected_additional_rules_count_min
):
    rule_name, rule_defs = generate_gbnf_float_rules(
        max_d, min_d, max_p, min_p
    )
    assert expected_rule_name_part in rule_name
    assert len(rule_defs) >= expected_additional_rules_count_min

    if max_d is not None or min_d is not None:
        assert any(r.startswith("integer-part-") for r in rule_defs)
    if max_p is not None or min_p is not None:
        assert any(r.startswith("fractional-part-") for r in rule_defs)
    if expected_rule_name_part != "float":
        assert any(r.startswith(expected_rule_name_part + " ::= ") for r in rule_defs)


# --- Tests for get_members_structure ---
def test_get_members_structure_simple_enum():
    expected_parts = sorted(['"\\"Value A\\""', '"\\"Value B\\""'])
    expected_rule = f"simple-str-enum ::= {' | '.join(expected_parts)}"
    assert get_members_structure(SimpleStrEnum, "simple-str-enum") == expected_rule

def test_get_members_structure_non_string_enum():
    expected_parts = sorted(['"\\"1\\""', '"\\"2.0\\""', '"\\"true\\""'])
    expected_rule = f"non-str-enum ::= {' | '.join(expected_parts)}"
    assert get_members_structure(NonStrEnum, "non-str-enum") == expected_rule


def test_get_members_structure_non_enum_class(caplog):
    class NotAnEnum:
        pass
    with caplog.at_level(logging.WARNING, logger=CONSTRUCTOR_LOGGER_NAME):
        rule = get_members_structure(NotAnEnum, "not-an-enum-rule")
    assert rule == "not-an-enum-rule ::= object"
    assert "get_members_structure called with non-Enum class NotAnEnum" in caplog.text


# --- Tests for generate_gbnf_grammar (single model) ---
def test_generate_gbnf_grammar_simple_model():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFNestedModel])
    assert "gbnf-nested-model ::= " in grammar_str
    assert ' ws "\\"nested_field_int\\"" ":" ws integer' in grammar_str
    assert ' ws "\\"nested_field_str\\"" ":" ws gbnf-nested-model-nested-field-str-optional-def' in grammar_str
    assert "gbnf-nested-model-nested-field-str-optional-def ::= null | string" in grammar_str

def test_generate_gbnf_grammar_recursive_model():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFRecursiveModel])
    assert 'gbnf-recursive-model-children-list-def ::= "[" ws ( gbnf-recursive-model ( ws "," ws gbnf-recursive-model )* ws )? "]"' in grammar_str
    assert "gbnf-recursive-model ::= " in grammar_str


def test_generate_gbnf_grammar_root_list_model():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFRootListInt])
    assert "gbnf-root-list-int ::= gbnf-root-list-int-root-list-def" in grammar_str
    assert 'gbnf-root-list-int-root-list-def ::= "[" ws ( integer ( ws "," ws integer )* ws )? "]"' in grammar_str

def test_generate_gbnf_grammar_root_list_of_nested_model():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFRootListOfNested])
    assert "gbnf-root-list-of-nested ::= gbnf-root-list-of-nested-root-list-def" in grammar_str
    assert 'gbnf-root-list-of-nested-root-list-def ::= "[" ws ( gbnf-nested-model ( ws "," ws gbnf-nested-model )* ws )? "]"' in grammar_str
    assert "gbnf-nested-model ::= " in grammar_str


def test_generate_gbnf_grammar_comprehensive_model():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFComprehensiveModel])
    assert "gbnf-comprehensive-model ::= " in grammar_str
    assert "simple-str-enum ::= " in grammar_str
    assert "non-str-enum ::= " in grammar_str
    assert "gbnf-comprehensive-model-literal-field-str-int-literal-def ::= " in grammar_str
    assert "gbnf-comprehensive-model-constrained-string-pattern-pattern-content ::= ^[a-z]{3}-[0-9]{2}$" in grammar_str
    assert "integer-part-max3-min2 ::= " in grammar_str
    assert "float-d4-1-p2-1 ::= " in grammar_str
    assert ' ws "\\"triple_quoted_str_field\\"" ":" ws triple-quoted-string' in grammar_str
    assert ' ws "\\"markdown_code_block_field\\"" ":" ws markdown-code-block' in grammar_str
    # The following assertion was removed as it's incorrect for Dict fields:
    # assert "gbnf-comprehensive-model-dict-str-nested ::= object" in grammar_str
    # Instead, check that the field uses 'object' and GBNFNestedModel rules are present
    assert ' ws "\\"dict_str_nested\\"" ":" ws object' in grammar_str
    assert "gbnf-nested-model ::= " in grammar_str


# --- Tests for generate_gbnf_grammar_from_pydantic_models (multi-model & wrapping) ---
def test_generate_gbnf_from_pydantic_models_list_of_outputs():
    grammar = generate_gbnf_grammar_from_pydantic_models([GBNFNestedModel, SimpleStrEnum], list_of_outputs=True) # type: ignore
    assert 'root ::= (" "| "\\n")? "[" ws ( grammar-models ( ws "," ws grammar-models )* ws )? "]"' in grammar

    grammar_models_rhs_match = re.search(r"grammar-models ::= (.*)", grammar)
    assert grammar_models_rhs_match is not None
    grammar_models_rhs_parts = {part.strip() for part in grammar_models_rhs_match.group(1).split("|")}
    assert "gbnf-nested-model" in grammar_models_rhs_parts
    assert "simple-str-enum" in grammar_models_rhs_parts

    assert "gbnf-nested-model ::= " in grammar
    assert "simple-str-enum ::= " in grammar

def test_generate_gbnf_from_pydantic_models_outer_object():
    grammar = generate_gbnf_grammar_from_pydantic_models(
        [GBNFNestedModel], outer_object_name="MyToolResponse"
    )
    assert "root ::= my-tool-response" in grammar
    assert 'my-tool-response ::= (" "| "\\n")? "{" ws "\\"MyToolResponse\\"" ws ":" ws grammar-models-for-my-tool-response ws "}"' in grammar
    assert "grammar-models-for-my-tool-response ::= gbnf-nested-model" in grammar

def test_generate_gbnf_from_pydantic_models_outer_object_with_content_key():
    grammar = generate_gbnf_grammar_from_pydantic_models(
        [SimpleStrEnum], # type: ignore
        outer_object_name="ToolOutput",
        outer_object_content_key="result"
    )
    assert "root ::= tool-output" in grammar
    assert 'tool-output ::= (" "| "\\n")? "{" ws "\\"ToolOutput\\"" ws ":" ws grammar-models-for-tool-output ws "}"' in grammar
    assert "grammar-models-for-tool-output ::= simple-str-enum-wrapper-for-tool-output" in grammar
    assert 'simple-str-enum-wrapper-for-tool-output ::= "\\"SimpleStrEnum\\"" ws "," ws "\\"result\\"" ws ":" ws simple-str-enum' in grammar
    assert "simple-str-enum ::= " in grammar


# --- Tests for get_primitive_grammar ---
def test_get_primitive_grammar_basic():
    primitives = get_primitive_grammar("root ::= string")
    assert "string ::= " in primitives
    assert "boolean ::= " in primitives
    assert "value ::= " not in primitives

def test_get_primitive_grammar_with_object_array_any():
    primitives_obj = get_primitive_grammar("root ::= my-object\nmy-object ::= object")
    assert "value ::= object | array | string | number | boolean | null" in primitives_obj
    assert "object ::= " in primitives_obj

    primitives_any = get_primitive_grammar("root ::= any-value\nany-value ::= any")
    assert "value ::= object | array | string | number | boolean | null" in primitives_any

    primitives_unknown = get_primitive_grammar("root ::= unknown-thing\nunknown-thing ::= unknown")
    assert "value ::= object | array | string | number | boolean | null" in primitives_unknown


def test_get_primitive_grammar_with_special_strings():
    primitives_triple = get_primitive_grammar("root ::= triple-quoted-string", grammar_has_special_strings=True)
    assert "triple-quoted-string ::= " in primitives_triple
    assert "markdown-code-block ::= " in primitives_triple

    primitives_markdown = get_primitive_grammar("root ::= markdown-code-block", grammar_has_special_strings=True)
    assert "markdown-code-block ::= " in primitives_markdown

    primitives_both_by_flag = get_primitive_grammar("root ::= string", grammar_has_special_strings=True)
    assert "triple-quoted-string ::= " in primitives_both_by_flag
    assert "markdown-code-block ::= " in primitives_both_by_flag

    primitives_by_direct_mention_triple = get_primitive_grammar("root ::= triple-quoted-string")
    assert "triple-quoted-string ::= " in primitives_by_direct_mention_triple

    primitives_by_direct_mention_markdown = get_primitive_grammar("root ::= markdown-code-block")
    assert "markdown-code-block ::= " in primitives_by_direct_mention_markdown

# --- Additional tests for generate_gbnf_rule_for_type and generate_gbnf_grammar ---
def test_generate_gbnf_rule_for_type_any_field():
    grammar_str = generate_gbnf_grammar_from_pydantic_models([GBNFComprehensiveModel])
    assert ' ws "\\"any_field\\"" ":" ws unknown' in grammar_str
    assert "value ::= object | array | string | number | boolean | null" in grammar_str


def test_generate_gbnf_grammar_non_pydantic_class(caplog):
    class NonPydanticClass:
        pass
    with caplog.at_level(logging.WARNING, logger=CONSTRUCTOR_LOGGER_NAME):
        rules, _ = generate_gbnf_grammar(NonPydanticClass, set(), {}) # type: ignore
    assert "generate_gbnf_grammar called with non-Pydantic class NonPydanticClass" in caplog.text
    assert rules == ["non-pydantic-class ::= object"]

def test_generate_gbnf_grammar_abstract_class(caplog):
    from abc import ABC, abstractmethod
    class AbstractTest(BaseModel, ABC):
        @abstractmethod
        def method(self): pass

    with caplog.at_level(logging.DEBUG, logger=CONSTRUCTOR_LOGGER_NAME):
        rules, _ = generate_gbnf_grammar(AbstractTest, set(), {})
    assert "generate_gbnf_grammar called with abstract class AbstractTest. Skipping." in caplog.text
    assert not rules

def test_generate_gbnf_rule_for_type_literal_non_primitive_stringified(caplog):
    class MyObjectForLiteral:
        def __str__(self): return "MyObjectInstance"

    my_obj_instance = MyObjectForLiteral()
    LiteralType = Literal[my_obj_instance, 123] # type: ignore

    created_rules: Dict[str, List[str]] = {}
    with caplog.at_level(logging.WARNING, logger=CONSTRUCTOR_LOGGER_NAME):
        gbnf_type_name, _ = generate_gbnf_rule_for_type(
            "TestModel", "literal_obj_field", LiteralType, False, set(), created_rules
        )

    assert f"Literal value {my_obj_instance} of type {type(my_obj_instance)} in field literal_obj_field will be stringified for GBNF." in caplog.text
    assert gbnf_type_name.endswith("-literal-def")
    assert any(r.startswith(gbnf_type_name + " ::= ") and '"\\"MyObjectInstance\\""' in r for r in created_rules.get(gbnf_type_name, []))
    assert any(r.startswith(gbnf_type_name + " ::= ") and '"\\"123\\""' in r for r in created_rules.get(gbnf_type_name, []))

def test_generate_gbnf_rule_for_type_dict_complex_values():
    ComplexDictType = Dict[str, GBNFNestedModel]
    created_rules: Dict[str, List[str]] = {}
    gbnf_type_name, _ = generate_gbnf_rule_for_type(
        "MyModel", "complex_dict_field", ComplexDictType, False, set(), created_rules
    )
    assert gbnf_type_name == "object"
    assert "gbnf-nested-model" in created_rules

def test_generate_gbnf_rule_for_type_unsupported_type_warning(caplog):
    class UnhandledCustomType:
        pass
    created_rules: Dict[str, List[str]] = {}
    with caplog.at_level(logging.WARNING, logger=CONSTRUCTOR_LOGGER_NAME):
        gbnf_type_name, _ = generate_gbnf_rule_for_type(
            "MyModel", "custom_field", UnhandledCustomType, False, set(), created_rules
        )
    assert gbnf_type_name == "string"
    assert f"Unknown type {UnhandledCustomType} mapping to GBNF for field 'custom_field'. Defaulting to 'string'." in caplog.text

def test_generate_gbnf_grammar_empty_model():
    class EmptyModel(BaseModel):
        pass
    rules, _ = generate_gbnf_grammar(EmptyModel, set(), {})
    assert rules == ['empty-model ::= "{" ws "}"']

def test_generate_gbnf_grammar_from_pydantic_models_no_models():
    grammar = generate_gbnf_grammar_from_pydantic_models([])
    assert 'grammar-models ::= "{" ws "}"' in grammar
    assert 'root ::= (" "| "\\n")? grammar-models' in grammar

def test_generate_gbnf_grammar_from_pydantic_models_list_of_outputs_no_models():
    grammar = generate_gbnf_grammar_from_pydantic_models([], list_of_outputs=True)
    assert 'grammar-models ::= "{" ws "}"' in grammar
    assert 'root ::= (" "| "\\n")? "[" ws ( grammar-models ( ws "," ws grammar-models )* ws )? "]"' in grammar

def test_generate_gbnf_grammar_from_pydantic_models_outer_object_no_models():
    grammar = generate_gbnf_grammar_from_pydantic_models([], outer_object_name="Wrapper")
    assert 'grammar-models-for-wrapper ::= "{" ws "}"' in grammar
    assert 'wrapper ::= (" "| "\\n")? "{" ws "\\"Wrapper\\"" ws ":" ws grammar-models-for-wrapper ws "}"' in grammar
    assert "root ::= wrapper" in grammar

def test_generate_gbnf_rule_for_type_optional_union():
    OptionalUnionType = Optional[Union[str, int]]
    created_rules: Dict[str, List[str]] = {}
    gbnf_type_name, new_rules = generate_gbnf_rule_for_type(
        "MyModel", "opt_union_field", OptionalUnionType, True, set(), created_rules
    )
    assert gbnf_type_name == "my-model-opt-union-field-union-def"
    expected_rule_parts = sorted(["integer", "null", "string"])
    expected_rule_str = f"{gbnf_type_name} ::= {' | '.join(expected_rule_parts)}"
    assert any(r == expected_rule_str for r in created_rules.get(gbnf_type_name, []))

def test_generate_gbnf_rule_for_type_simple_optional():
    created_rules: Dict[str, List[str]] = {}
    gbnf_type_name, new_rules = generate_gbnf_rule_for_type(
        "MyModel", "opt_str_field", Optional[str], True, set(), created_rules
    )
    assert gbnf_type_name == "my-model-opt-str-field-optional-def"
    expected_rule_parts = sorted(["null", "string"])
    expected_rule_str = f"{gbnf_type_name} ::= {' | '.join(expected_rule_parts)}"
    assert any(r == expected_rule_str for r in created_rules.get(gbnf_type_name, [])), \
        f"Rule not found or incorrect. Expected: '{expected_rule_str}', Got from created_rules: {created_rules.get(gbnf_type_name)}"


def test_generate_gbnf_rule_for_type_tuple():
    TupleType = Tuple[str, int]
    created_rules: Dict[str, List[str]] = {}
    gbnf_type_name, new_rules = generate_gbnf_rule_for_type(
        "MyModel", "tuple_field", TupleType, False, set(), created_rules
    )
    assert gbnf_type_name == "my-model-tuple-field-list-def"
    # Updated expectation for Tuple[str, int]
    expected_rule_str_fixed_tuple = 'my-model-tuple-field-list-def ::= "[" ws string ws "," ws integer ws "]"'
    assert any(expected_rule_str_fixed_tuple == r for r in created_rules.get(gbnf_type_name, [])), \
        f"Rule not found or incorrect for Tuple[str, int]. Expected: '{expected_rule_str_fixed_tuple}', Got: {created_rules.get(gbnf_type_name)}"

    TupleEllipsisType = Tuple[int, ...]
    created_rules_ellipsis: Dict[str, List[str]] = {}
    gbnf_type_name_ellipsis, _ = generate_gbnf_rule_for_type(
        "MyModel", "tuple_ellipsis_field", TupleEllipsisType, False, set(), created_rules_ellipsis
    )
    assert gbnf_type_name_ellipsis == "my-model-tuple-ellipsis-field-list-def"
    expected_ellipsis_rule_part = 'my-model-tuple-ellipsis-field-list-def ::= "[" ws ( integer ( ws "," ws integer )* ws )? "]"'
    assert any(expected_ellipsis_rule_part in r for r in created_rules_ellipsis.get(gbnf_type_name_ellipsis, [])), \
        f"Rule not found or incorrect for Tuple[int, ...]. Expected part: '{expected_ellipsis_rule_part}', Got: {created_rules_ellipsis.get(gbnf_type_name_ellipsis)}"
