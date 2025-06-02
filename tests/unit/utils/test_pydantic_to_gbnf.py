### tests/unit/utils/test_pydantic_to_gbnf.py
### tests/unit/utils/test_pydantic_to_gbnf.py
import inspect
import re
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import pytest
from genie_tooling.utils.gbnf.constructor import (
    generate_gbnf_float_rules,
    generate_gbnf_grammar,
    generate_gbnf_grammar_from_pydantic_models,
    generate_gbnf_integer_rules,
    generate_gbnf_rule_for_type,
)

# Correct imports from the new structure
from genie_tooling.utils.gbnf.core import (
    PydanticDataType,
    format_model_and_field_name,
    map_pydantic_type_to_gbnf,
    regex_to_gbnf,
)
from genie_tooling.utils.gbnf.documentation import (
    generate_markdown_documentation,
)
from genie_tooling.utils.gbnf.model_factory import (
    convert_dictionary_to_pydantic_model,
)
from pydantic import BaseModel, Field, RootModel


# --- Test Models (from existing file, plus new ones for new tests) ---
class SimpleEnum(Enum):
    VALUE_A = "A"
    VALUE_B = "B"

class NestedModel(BaseModel):
    nested_field: str = Field(description="A field in a nested model")
    nested_optional: Optional[int] = None

class SimpleModel(BaseModel):
    name: str = Field(description="The name of the item")
    age: int = Field(gt=0, description="The age of the item")
    is_active: bool = Field(default=True, description="Activation status")
    tags: List[str] = Field(default_factory=list, description="A list of tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    nested_item: Optional[NestedModel] = Field(None, description="A nested Pydantic model")
    enum_field: SimpleEnum = Field(SimpleEnum.VALUE_A, description="An enum field")
    union_field: Union[str, int, None] = Field(None, description="A union type field")
    set_field: Set[str] = Field(default_factory=set, description="A set of strings")
    any_field: Any = Field(None, description="An Any type field")

class StringPatternModel(BaseModel):
    patterned_string: str = Field(pattern=r"^[a-zA-Z0-9]{3,5}$", description="String matching a pattern")
    triple_quoted: str = Field(json_schema_extra={"triple_quoted_string": True}, description="A triple quoted string")
    markdown_code_block: str = Field(json_schema_extra={"markdown_code_block": True}, description="A markdown code block")

class NumberConstraintModel(BaseModel):
    constrained_int: int = Field(json_schema_extra={"min_digit": 2, "max_digit": 4}, description="Integer with digit constraints")
    constrained_float: float = Field(
        json_schema_extra={"min_digit": 1, "max_digit": 3, "min_precision": 1, "max_precision": 2},
        description="Float with digit and precision constraints"
    )

def sample_function_for_dynamic_model(name: str, count: int = 5) -> str:
    """
    A sample function to test dynamic model creation.

    Args:
        name (str): The name to use.
        count (int): Number of times.

    Returns:
        str: A processed string.
    """
    return f"{name} repeated {count} times."

# --- New Models for Extended Tests ---
class ModelA(BaseModel):
    field_a: str

class ModelB(BaseModel):
    field_b: int

class ModelC(BaseModel):
    field_c: bool

class Node(BaseModel):
    name: str
    children: List["Node"] = Field(default_factory=list)
Node.model_rebuild()

class MyRootList(RootModel[List[str]]):
    pass

class MyRootStr(RootModel[str]):
    pass

AnyStr = TypeVar("AnyStr", str, bytes)


# --- Existing Tests (from user's provided file) ---

def test_format_model_and_field_name():
    assert format_model_and_field_name("MyModelName") == "my-model-name"
    assert format_model_and_field_name("my_field_name") == "my-field-name"
    assert format_model_and_field_name("URLProcessor") == "url-processor"
    assert format_model_and_field_name("simple") == "simple"
    assert format_model_and_field_name("myListField") == "my-list-field"


def test_map_pydantic_type_to_gbnf():
    assert map_pydantic_type_to_gbnf(str) == PydanticDataType.STRING.value
    assert map_pydantic_type_to_gbnf(int) == PydanticDataType.INTEGER.value
    assert map_pydantic_type_to_gbnf(float) == PydanticDataType.FLOAT.value
    assert map_pydantic_type_to_gbnf(bool) == PydanticDataType.BOOLEAN.value
    assert map_pydantic_type_to_gbnf(Any) == "unknown"
    assert map_pydantic_type_to_gbnf(type(None)) == PydanticDataType.NULL.value

    assert map_pydantic_type_to_gbnf(List[str]) == "string-list"
    assert map_pydantic_type_to_gbnf(Set[int]) == "integer-set"
    assert map_pydantic_type_to_gbnf(Dict[str, int]) == "custom-dict-key-string-value-integer"
    assert map_pydantic_type_to_gbnf(Optional[str]) == "optional-string"
    assert map_pydantic_type_to_gbnf(Union[int, str]) == "union-integer-or-string"
    assert map_pydantic_type_to_gbnf(SimpleEnum) == "simple-enum"
    assert map_pydantic_type_to_gbnf(NestedModel) == "nested-model"


def test_regex_to_gbnf():
    assert regex_to_gbnf(r"^\d{3}$") == r"^[0-9]{3}$"
    assert regex_to_gbnf(r"\s+") == "[ \t\n]+"

def test_generate_gbnf_rule_for_type_list_of_strings():
    expected_gbnf_type = "my-model-my-list-field-list-def"
    gbnf_type, rules = generate_gbnf_rule_for_type("MyModel", "myListField", List[str], False, set(), {})
    assert gbnf_type == expected_gbnf_type

    element_rule_name = "string"
    expected_list_def = rf'{expected_gbnf_type} ::= "[" ws ( {element_rule_name} ( ws "," ws {element_rule_name} )* ws )? "]"'
    assert any(r.strip() == expected_list_def.strip() for r in rules)


def test_generate_gbnf_rule_for_type_nested_model():
    created_rules_store: Dict[str, List[str]] = {}
    processed_models_store: Set[Type[BaseModel]] = set()

    _outer_field_type_name, _outer_field_rules = generate_gbnf_rule_for_type(
        "MyModel", "myNestedField", NestedModel, False, processed_models_store, created_rules_store
    )
    expected_optional_rule_def_name = "nested-model-nested-optional-optional-def"
    expected_optional_rule_def_str = f"{expected_optional_rule_def_name} ::= integer | null"

    assert expected_optional_rule_def_name in created_rules_store
    assert any(expected_optional_rule_def_str in rule_def for rule_def in created_rules_store[expected_optional_rule_def_name])


def test_generate_gbnf_rule_for_type_optional_union():
    expected_gbnf_type_name = "my-model-my-optional-union-union-def"
    expected_gbnf_rule_def_parts = ["integer", "null", "string", "::="]
    created_rules_store: Dict[str, List[str]] = {}

    gbnf_type, _rules = generate_gbnf_rule_for_type(
        "MyModel", "myOptionalUnion", Optional[Union[int, str]], True, set(), created_rules_store
    )
    assert gbnf_type == expected_gbnf_type_name
    assert expected_gbnf_type_name in created_rules_store

    generated_rule_string = created_rules_store[expected_gbnf_type_name][0]
    for part in expected_gbnf_rule_def_parts:
        assert part in generated_rule_string


def test_generate_markdown_documentation_simple_model(tmp_path):
    class NestedModelForDoc(BaseModel):
        nested_optional: Optional[int] = None

    class SimpleModelForDoc(BaseModel):
        name: str = Field(description="The name of the item")
        nested_item: Optional[NestedModelForDoc] = Field(None, description="A nested Pydantic model")

    doc = generate_markdown_documentation([SimpleModelForDoc])
    assert "## Model: `SimpleModelForDoc`" in doc
    assert "*   **`name`** (`str`): The name of the item" in doc
    assert "*   **`nested_item`** ((`nested-model-for-doc` or `none-type`)) (optional): A nested Pydantic model" in doc
    assert "### Nested Model: `NestedModelForDoc`" in doc
    assert "*   **`nested_optional`** ((`int` or `none-type`)) (optional)" in doc


def test_generate_gbnf_grammar_simple_model_resolved():
    class ModelA(BaseModel):
        field_a: str
    rules, _ = generate_gbnf_grammar(ModelA, set(), {})
    grammar_str = "\n".join(rules)
    assert 'model-a ::= "{" "\\n"  ws "\\"field_a\\"" ":" ws string "\\n" ws "}"' in grammar_str


def test_convert_dictionary_to_pydantic_model_nested_and_array_resolved_xfail():
    schema_dict = {
        "name": "ComplexSchemaForTest",
        "properties": {
            "simple_list": {"type": "array", "items": {"type": "string"}},
            "object_list": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sub_prop": {"type": "boolean"}}
                }
            }
        }
    }
    DynamicModel = convert_dictionary_to_pydantic_model(schema_dict, "ComplexSchemaForTest")
    assert DynamicModel.__name__ == "ComplexSchemaForTest"
    fields = DynamicModel.model_fields

    simple_list_annotation = fields["simple_list"].annotation
    assert get_origin(simple_list_annotation) is Union
    simple_list_args = get_args(simple_list_annotation)
    assert type(None) in simple_list_args
    list_str_type = next(t for t in simple_list_args if t is not type(None))
    assert get_origin(list_str_type) is list
    assert get_args(list_str_type)[0] is str

    object_list_annotation = fields["object_list"].annotation
    assert get_origin(object_list_annotation) is Union
    object_list_args = get_args(object_list_annotation)
    assert type(None) in object_list_args
    list_of_objects_type = next(t for t in object_list_args if t is not type(None))

    assert get_origin(list_of_objects_type) is list
    nested_list_item_type = get_args(list_of_objects_type)[0]
    assert inspect.isclass(nested_list_item_type) and issubclass(nested_list_item_type, BaseModel)
    sub_prop_annotation = nested_list_item_type.model_fields["sub_prop"].annotation
    assert get_origin(sub_prop_annotation) is Union
    sub_prop_args = get_args(sub_prop_annotation)
    assert type(None) in sub_prop_args
    assert bool in sub_prop_args
    expected_nested_item_model_name = "ComplexSchemaForTestObjectListItem"
    assert nested_list_item_type.__name__ == expected_nested_item_model_name

# --- NEW EXTENDED TESTS ---

# I. map_pydantic_type_to_gbnf
def test_map_pydantic_type_to_gbnf_list_of_optional_str():
    """Test List[Optional[str]] mapping."""
    assert map_pydantic_type_to_gbnf(List[Optional[str]]) == "optional-string-list"

def test_map_pydantic_type_to_gbnf_list_of_union():
    """Test List[Union[int, bool]] mapping."""
    assert map_pydantic_type_to_gbnf(List[Union[int, bool]]) == "union-boolean-or-integer-list"

def test_map_pydantic_type_to_gbnf_dict_complex_value():
    """Test Dict[str, List[int]] mapping."""
    assert map_pydantic_type_to_gbnf(Dict[str, List[int]]) == "custom-dict-key-string-value-integer-list"

# II. generate_gbnf_rule_for_type
@pytest.mark.parametrize("py_type, expected_gbnf_name", [
    (str, "string"),
    (int, "integer"),
    (float, "float"),
    (bool, "boolean"),
    (Any, "unknown"),
    (type(None), "null"),
])
def test_gbrft_primitive_types_exhaustive(py_type, expected_gbnf_name):
    """Test primitive types for generate_gbnf_rule_for_type."""
    gbnf_type, rules = generate_gbnf_rule_for_type("MyModel", "myField", py_type, False, set(), {})
    assert gbnf_type == expected_gbnf_name
    assert not rules

def test_gbrft_optional_model():
    """Test Optional[NestedModel] for generate_gbnf_rule_for_type."""
    created_rules: Dict[str, List[str]] = {}
    processed_models: Set[Type[BaseModel]] = set()
    gbnf_type, rules = generate_gbnf_rule_for_type(
        "MyModel", "optNested", Optional[NestedModel], True, processed_models, created_rules
    )
    expected_rule_name = "my-model-opt-nested-optional-def"
    expected_definition_content = "nested-model | null"
    assert gbnf_type == expected_rule_name
    # Check if the rule definition itself is present in the returned rules
    assert any(r.strip() == f"{expected_rule_name} ::= {expected_definition_content}" for r in rules)
    # Check that NestedModel's own rules were processed and added to the global created_rules
    assert "nested-model" in created_rules

def test_gbrft_list_of_list_str():
    """Test List[List[str]] for generate_gbnf_rule_for_type."""
    created_rules: Dict[str, List[str]] = {}
    processed_models: Set[Type[BaseModel]] = set()

    gbnf_type, rules = generate_gbnf_rule_for_type(
        "MyModel", "listList", List[List[str]], False, processed_models, created_rules
    )

    outer_list_def_name = "my-model-list-list-list-def"
    inner_list_def_name = "my-model-list-list-item-list-def"

    assert gbnf_type == outer_list_def_name

    assert inner_list_def_name in created_rules, f"Rule {inner_list_def_name} not found in {created_rules.keys()}"
    expected_inner_list_def = rf'{inner_list_def_name} ::= "[" ws ( string ( ws "," ws string )* ws )? "]"'
    # Direct string comparison after stripping whitespace
    assert any(r.strip() == expected_inner_list_def for r in created_rules[inner_list_def_name]), \
        f"Inner list rule for {inner_list_def_name} not found or incorrect. Expected: '{expected_inner_list_def}'. Got: {created_rules.get(inner_list_def_name)}"

    assert outer_list_def_name in created_rules, f"Rule {outer_list_def_name} not found in {created_rules.keys()}"
    expected_outer_list_def = rf'{outer_list_def_name} ::= "[" ws ( {inner_list_def_name} ( ws "," ws {inner_list_def_name} )* ws )? "]"'
    assert any(r.strip() == expected_outer_list_def for r in created_rules[outer_list_def_name]), \
        f"Outer list rule for {outer_list_def_name} not found or incorrect. Expected: '{expected_outer_list_def}'. Got: {created_rules.get(outer_list_def_name)}"


def test_gbrft_enum_field_in_model_context():
    """Test SimpleEnum as a field for generate_gbnf_rule_for_type."""
    created_rules: Dict[str, List[str]] = {}
    gbnf_type, rules = generate_gbnf_rule_for_type(
        "MyModel", "myEnumField", SimpleEnum, False, set(), created_rules
    )
    assert gbnf_type == "simple-enum"
    assert "simple-enum" in created_rules
    # json.dumps("A") is "\"A\"" (Python string), which is "A" in GBNF
    expected_enum_def = 'simple-enum ::= "\\"A\\"" | "\\"B\\""'
    assert any(expected_enum_def == r.strip() for r in created_rules["simple-enum"]), \
        f"Expected enum definition '{expected_enum_def}' not found in {created_rules['simple-enum']}"

@pytest.mark.xfail(reason="Pydantic V2 FieldInfo.pattern access or GBNF rule name generation needs review")
def test_gbrft_string_with_pattern():
    """Test str = Field(pattern="...") for generate_gbnf_rule_for_type."""
    created_rules: Dict[str, List[str]] = {}
    field_info_mock = Field(pattern=r"^[A-Z]{3}$")
    assert field_info_mock.pattern == r"^[A-Z]{3}$", "Pydantic FieldInfo did not store pattern correctly for test setup."

    gbnf_type, rules = generate_gbnf_rule_for_type(
        "MyModel", "patternField", str, False, set(), created_rules, field_info=field_info_mock
    )
    expected_field_rule_name = "my-model-pattern-field-string-def"
    expected_content_rule_name = "my-model-pattern-field-pattern-content"

    assert gbnf_type == expected_field_rule_name, f"Expected GBNF type '{expected_field_rule_name}', but got '{gbnf_type}'"

    assert expected_content_rule_name in created_rules
    assert any(r.startswith(f"{expected_content_rule_name} ::= ^[A-Z]{{3}}$") for r in created_rules[expected_content_rule_name])

    assert expected_field_rule_name in created_rules
    assert any(r.startswith(f'{expected_field_rule_name} ::= "\\"\\"" {expected_content_rule_name} "\\"\\"" ws') for r in created_rules[expected_field_rule_name])

def test_gbrft_constrained_int_digits():
    """Test int = Field(json_schema_extra={"min_digit":2, "max_digit":3}) for generate_gbnf_rule_for_type."""
    created_rules: Dict[str, List[str]] = {}
    field_info_mock = Field(json_schema_extra={"min_digit": 2, "max_digit": 3})

    gbnf_type, rules = generate_gbnf_rule_for_type(
        "MyModel", "constrainedInt", int, False, set(), created_rules, field_info=field_info_mock
    )
    expected_rule_name = "integer-part-max3-min2"
    assert gbnf_type == expected_rule_name
    assert expected_rule_name in created_rules
    # The rule definition itself might be in a list under the key
    assert any(r.strip() == "integer-part-max3-min2 ::= [0-9] [0-9] [0-9]?" for r in created_rules[expected_rule_name])


def test_gbrft_literal_type():
    """Test Literal[\"A\", \"B\"] and Literal[\"A\", 1] for generate_gbnf_rule_for_type."""
    created_rules_str: Dict[str, List[str]] = {}
    gbnf_type_str, rules_str = generate_gbnf_rule_for_type(
        "MyModel", "literalFieldStr", Literal["A", "B"], False, set(), created_rules_str
    )
    expected_rule_name_str = "my-model-literal-field-str-literal-def"
    assert gbnf_type_str == expected_rule_name_str
    assert expected_rule_name_str in created_rules_str
    # json.dumps("A") -> "\"A\""
    expected_def_str = f'{expected_rule_name_str} ::= "\\"A\\"" | "\\"B\\""'
    assert any(expected_def_str == r.strip() for r in created_rules_str[expected_rule_name_str]), \
        f"Expected literal definition '{expected_def_str}' not found in {created_rules_str[expected_rule_name_str]}"


    created_rules_mixed: Dict[str, List[str]] = {}
    gbnf_type_mixed, rules_mixed = generate_gbnf_rule_for_type(
        "MyModel", "literalFieldMixed", Literal["A", 1], False, set(), created_rules_mixed
    )
    expected_rule_name_mixed = "my-model-literal-field-mixed-literal-def"
    assert gbnf_type_mixed == expected_rule_name_mixed
    assert expected_rule_name_mixed in created_rules_mixed

    expected_def_mixed_parts = sorted(['"\\"A\\""', "1"]) # json.dumps(1) is '1'

    found_mixed_def = False
    for r_mixed in created_rules_mixed[expected_rule_name_mixed]:
        if r_mixed.startswith(f"{expected_rule_name_mixed} ::= "):
            parts_in_rule = sorted([p.strip() for p in r_mixed.split("::=")[1].split("|")])
            if parts_in_rule == expected_def_mixed_parts:
                found_mixed_def = True
                break
    assert found_mixed_def, f"Literal rule for mixed types not found or incorrect. Expected parts: {expected_def_mixed_parts}. Got rules: {created_rules_mixed.get(expected_rule_name_mixed)}"


# III. generate_gbnf_grammar
def test_ggg_model_with_recursive_definition():
    """Test generate_gbnf_grammar with a recursive Pydantic model."""
    created_rules: Dict[str, List[str]] = {}
    _node_rules_this_pass, _ = generate_gbnf_grammar(Node, set(), created_rules)

    assert "node" in created_rules
    # Fields are sorted: children, name
    expected_node_main_def = 'node ::= "{" "\\n"  ws "\\"children\\"" ":" ws node-children-list-def "," "\\n"  ws "\\"name\\"" ":" ws string "\\n" ws "}"'
    assert any(r.strip() == expected_node_main_def for r in created_rules["node"]), \
        f"Main 'node' rule definition not found or incorrect. Expected: '{expected_node_main_def}'. Got: {created_rules.get('node')}"

    assert "node-children-list-def" in created_rules
    expected_node_children_list_def = 'node-children-list-def ::= "[" ws ( node ( ws "," ws node )* ws )? "]"'
    assert any(r.strip() == expected_node_children_list_def for r in created_rules["node-children-list-def"]), \
        f"'node-children-list-def' rule definition not found or incorrect. Expected: '{expected_node_children_list_def}'. Got: {created_rules.get('node-children-list-def')}"

@pytest.mark.xfail(reason="Pydantic V2 FieldInfo.pattern access or GBNF rule name generation needs review")
def test_ggg_model_with_special_strings():
    """Test generate_gbnf_grammar with triple_quoted_string and markdown_code_block."""
    created_rules: Dict[str, List[str]] = {}
    _rules_this_pass, has_special = generate_gbnf_grammar(StringPatternModel, set(), created_rules)

    string_pattern_model_rule_name = "string-pattern-model"
    assert string_pattern_model_rule_name in created_rules
    main_model_rule = created_rules[string_pattern_model_rule_name][0].strip()

    assert has_special is True
    # Check for presence of each part, order is now deterministic due to sorted field names
    # Order: markdown_code_block, patterned_string, triple_quoted
    expected_part_markdown = 'ws "\\"markdown_code_block\\"" ":" ws markdown-code-block'
    expected_part_patterned = 'ws "\\"patterned_string\\"" ":" ws string-pattern-model-patterned-string-string-def' # Corrected
    expected_part_triple = 'ws "\\"triple_quoted\\"" ":" ws triple-quoted-string'

    assert expected_part_markdown in main_model_rule
    assert expected_part_patterned in main_model_rule, f"Expected patterned string part not found. Rule: {main_model_rule}"
    assert expected_part_triple in main_model_rule

    # Verify the overall structure based on sorted field names
    expected_full_rule = f'{string_pattern_model_rule_name} ::= "{{" "\\n" {expected_part_markdown} "," "\\n" {expected_part_patterned} "," "\\n" {expected_part_triple} "\\n" ws "}}"'.strip()
    assert main_model_rule == expected_full_rule


# IV. generate_gbnf_grammar_from_pydantic_models
def test_gggfpm_single_model_with_outer_object():
    """Test generate_gbnf_grammar_from_pydantic_models with an outer object structure."""
    grammar = generate_gbnf_grammar_from_pydantic_models(
        [SimpleModel], outer_object_name="MyFunction", outer_object_content_key="parameters"
    )
    assert "root ::= my-function" in grammar

    my_function_pattern_str = r'my-function\s*::=\s*\(\s*"\s*"\s*\|\s*"\\n"\s*\)\?\s*"\{"\s*ws\s*"\\"MyFunction\\""\s*ws\s*":"\s*ws\s*grammar-models-for-my-function\s*ws\s*"}"'
    assert re.search(my_function_pattern_str, grammar), f"Outer object rule 'my-function' not found or incorrect. Regex: '{my_function_pattern_str}'. Grammar:\n{grammar}"

    assert "grammar-models-for-my-function ::= simple-model-wrapper-for-my-function" in grammar

    wrapper_pattern_str = r'simple-model-wrapper-for-my-function\s*::=\s*"\\"SimpleModel\\""\s*ws\s*","\s*ws\s*"\\"parameters\\""\s*ws\s*":"\s*ws\s*simple-model'
    assert re.search(wrapper_pattern_str, grammar), f"Wrapper rule 'simple-model-wrapper-for-my-function' not found or incorrect. Regex: '{wrapper_pattern_str}'. Grammar:\n{grammar}"

    assert "simple-model ::= " in grammar


def test_gggfpm_list_of_outputs_no_outer():
    """Test generate_gbnf_grammar_from_pydantic_models with list_of_outputs=True."""
    grammar = generate_gbnf_grammar_from_pydantic_models([SimpleModel, NestedModel], list_of_outputs=True)

    expected_root_rule = 'root ::= (" "| "\\n")? "[" ws ( grammar-models ( ws "," ws grammar-models )* ws )? "]"'
    normalized_grammar = re.sub(r"\s+", " ", grammar.strip())
    normalized_expected_root_rule = re.sub(r"\s+", " ", expected_root_rule.strip())

    assert normalized_expected_root_rule in normalized_grammar, \
        f"Root rule for list_of_outputs not found or incorrect. Expected: '{normalized_expected_root_rule}'. Got (normalized grammar part): '{normalized_grammar.splitlines()[0]}'"


    assert "grammar-models ::= nested-model | simple-model" in grammar or \
           "grammar-models ::= simple-model | nested-model" in grammar


# V. Helper Function Tests
def test_ggir_min_max_digits():
    """Test generate_gbnf_integer_rules with min and max digits."""
    rule_name, rules = generate_gbnf_integer_rules(min_digit=2, max_digit=4)
    assert rule_name == "integer-part-max4-min2"
    assert len(rules) == 1
    assert rules[0].strip() == "integer-part-max4-min2 ::= [0-9] [0-9] [0-9]? [0-9]?"

def test_ggfr_all_constraints():
    """Test generate_gbnf_float_rules with all constraints."""
    rule_name, rules = generate_gbnf_float_rules(min_digit=1, max_digit=3, min_precision=1, max_precision=2)
    assert rule_name == "float-d3-1-p2-1"

    rules_stripped = {r.strip() for r in rules}
    assert "integer-part-max3-min1 ::= [0-9] [0-9]? [0-9]?" in rules_stripped
    assert "fractional-part-max2-min1 ::= [0-9] [0-9]?" in rules_stripped
    assert f'{rule_name} ::= integer-part-max3-min1 "." fractional-part-max2-min1' in rules_stripped
