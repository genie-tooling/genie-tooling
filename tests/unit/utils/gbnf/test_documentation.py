### tests/unit/utils/gbnf/test_documentation.py
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from unittest.mock import MagicMock, mock_open, patch

from genie_tooling.utils.gbnf.core import format_model_and_field_name
from genie_tooling.utils.gbnf.documentation import (
    format_multiline_description,
    generate_and_save_gbnf_grammar_and_documentation,
    generate_field_markdown,
    generate_field_text,
    generate_gbnf_grammar_and_documentation,
    generate_gbnf_grammar_and_documentation_from_dictionaries,
    generate_markdown_documentation,
    generate_text_documentation,
    map_grammar_names_to_pydantic_model_class,
    save_gbnf_grammar_and_documentation,
)
from pydantic import BaseModel, Field, RootModel


# --- Test Models ---
class DocSimpleEnum(Enum):
    ALPHA = "alpha_value"
    BETA = "beta_value"

class DocNested(BaseModel):
    """A nested model for documentation testing."""
    nested_id: int = Field(description="Identifier for the nested object.")
    is_valid: Optional[bool] = Field(default=True, description="Validity status.")
    model_config = {"json_schema_extra": {"example": {"nested_id": 1, "is_valid": True}}}

class DocRecursive(BaseModel):
    name: str
    children: List["DocRecursive"] = Field(default_factory=list)
    model_config = {"json_schema_extra": {"example": {"name": "root", "children": [{"name": "child1"}]}}}

DocRecursive.model_rebuild()


class DocRootListStr(RootModel[List[str]]):
    """A root model that is a list of strings."""
    model_config = {"json_schema_extra": {"example": ["item1", "item2"]}}


class DocMain(BaseModel):
    """
    Main Pydantic model for comprehensive documentation testing.
    It includes various field types and configurations.
    """
    string_field: str = Field(description="A simple string field.", examples=["example string"])
    integer_field: int = Field(gt=0, description="A positive integer.")
    optional_float: Optional[float] = Field(description="An optional float, defaults to None.")
    boolean_default_true: bool = Field(default=True, description="Boolean with a default of true.")
    list_of_strings: List[str] = Field(default_factory=list, description="A list of string tags.")
    dict_any_value: Dict[str, Any] = Field(default_factory=dict, description="A dictionary with string keys and any type of values.")
    nested_object: Optional[DocNested] = Field(description="An optional nested object of type DocNested.")
    enum_choice: DocSimpleEnum = Field(default=DocSimpleEnum.ALPHA, description="A choice from DocSimpleEnum.")
    union_type: Union[int, str, DocNested] = Field(description="A field that can be an int, string, or DocNested.")
    set_of_numbers: Set[int] = Field(default_factory=set, description="A set of unique integers.")
    any_type_field: Any = Field(None, description="A field that can hold any type, defaulting to None.")
    field_no_description: float
    recursive_field: Optional[DocRecursive] = None
    root_list_field: Optional[DocRootListStr] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "string_field": "hello",
                "integer_field": 10,
                "boolean_default_true": False,
                "list_of_strings": ["tag1", "tag2"],
                "nested_object": {"nested_id": 99, "is_valid": False},
                "enum_choice": "beta_value",
                "union_type": {"nested_id": 101, "is_valid": True},
                "field_no_description": 3.14
            }
        }
    }

# --- Tests for format_multiline_description ---
def test_format_multiline_description():
    assert format_multiline_description("Line 1\nLine 2", 1) == "  Line 1\n  Line 2"
    assert format_multiline_description("Single Line", 2) == "    Single Line"
    assert format_multiline_description("", 1) == "  "

# --- Tests for generate_field_text ---
def test_generate_field_text_basic(caplog):
    caplog.set_level(logging.DEBUG)
    text_doc = generate_field_text("string_field", str, DocMain, depth=1)
    assert "string_field (str)" in text_doc
    assert "Description: A simple string field." in text_doc
    assert "Example: 'example string'" in text_doc

    text_doc_no_desc_flag = generate_field_text(
        "string_field", str, DocMain, depth=1, documentation_with_field_description=False
    )
    assert "string_field (str)" in text_doc_no_desc_flag
    assert "Description:" not in text_doc_no_desc_flag
    assert "Example:" not in text_doc_no_desc_flag


def test_generate_field_text_optional_with_default():
    text_doc = generate_field_text("boolean_default_true", bool, DocMain, depth=1)
    assert "boolean_default_true (bool) (optional, default: True)" in text_doc

def test_generate_field_text_optional_no_default():
    text_doc = generate_field_text("optional_float", Optional[float], DocMain, depth=1)
    assert "optional_float ((float or none-type)) (optional, default: None)" in text_doc

def test_generate_field_text_list():
    text_doc = generate_field_text("list_of_strings", List[str], DocMain, depth=1)
    assert "list_of_strings (list of str) (optional, default_factory: list)" in text_doc

def test_generate_field_text_nested_model():
    text_doc = generate_field_text("nested_object", Optional[DocNested], DocMain, depth=1)
    assert "nested_object ((DocNested or none-type)) (optional, default: None)" in text_doc


# --- Tests for generate_text_documentation ---
def test_generate_text_documentation_single_model():
    doc = generate_text_documentation([DocNested])
    assert "Model: DocNested" in doc
    assert "nested_id (int)" in doc
    assert "is_valid ((bool or none-type)) (optional, default: True)" in doc

def test_generate_text_documentation_with_recursive_model():
    doc = generate_text_documentation([DocRecursive])
    assert "Model: DocRecursive" in doc
    assert "name (str)" in doc
    assert "children (list of DocRecursive) (optional, default_factory: list)" in doc
    assert doc.count("Model: DocRecursive") == 1
    assert "Nested Model: DocRecursive" not in doc

def test_generate_text_documentation_with_root_model():
    doc = generate_text_documentation([DocRootListStr])
    assert "Model: DocRootListStr" in doc
    assert "A root model that is a list of strings." in doc
    assert "Root Type: list of str" in doc
    assert "Fields:" not in doc


# --- Tests for generate_field_markdown ---
def test_generate_field_markdown_basic():
    md_doc = generate_field_markdown("string_field", str, DocMain)
    assert "*   **`string_field`** (`str`): A simple string field." in md_doc
    assert "- *Example*: `'example string'`" in md_doc

def test_generate_field_markdown_optional_list_nested():
    md_doc = generate_field_markdown("nested_object", Optional[DocNested], DocMain)
    assert "*   **`nested_object`** ((`doc-nested` or `none-type`)) (optional, default: `None`): An optional nested object of type DocNested." in md_doc

# --- Tests for generate_markdown_documentation ---
def test_generate_markdown_documentation_multiple_models():
    doc_main_only = generate_markdown_documentation([DocMain])
    assert "## Model: `DocMain`" in doc_main_only
    assert "### Nested Model: `DocNested`" in doc_main_only
    assert "### Nested Model: `DocRecursive`" in doc_main_only
    assert "### Nested Model: `DocRootListStr`" in doc_main_only

    doc_both_top = generate_markdown_documentation([DocNested, DocMain])
    assert doc_both_top.count("## Model: `DocNested`") == 1
    assert "## Model: `DocMain`" in doc_both_top
    assert "### Nested Model: `DocNested`" not in doc_both_top


def test_generate_markdown_documentation_model_example():
    doc = generate_markdown_documentation([DocNested])
    assert "**Example Output for `doc-nested`**:" in doc
    assert '```json\n{\n  "nested_id": 1,\n  "is_valid": true\n}\n```' in doc

# --- Tests for save_gbnf_grammar_and_documentation ---
def test_save_gbnf_grammar_and_documentation_success(tmp_path: Path):
    grammar_content = "root ::= test"
    doc_content = "# Test Doc"
    grammar_file = tmp_path / "test.gbnf"
    doc_file = tmp_path / "test_doc.md"

    save_gbnf_grammar_and_documentation(grammar_content, doc_content, str(grammar_file), str(doc_file))

    assert grammar_file.read_text() == grammar_content
    assert doc_file.read_text() == doc_content

@patch("builtins.open", new_callable=mock_open)
def test_save_gbnf_grammar_and_documentation_io_error(mock_file_open: MagicMock, capsys):
    mock_file_open.side_effect = IOError("Permission denied")
    save_gbnf_grammar_and_documentation("grammar", "doc", "g.gbnf", "d.md")
    captured = capsys.readouterr()
    assert "An error occurred while saving the grammar file: Permission denied" in captured.out
    assert "An error occurred while saving the documentation file: Permission denied" in captured.out


# --- Tests for generate_gbnf_grammar_and_documentation ---
@patch("genie_tooling.utils.gbnf.documentation.generate_markdown_documentation")
@patch("genie_tooling.utils.gbnf.documentation.generate_gbnf_grammar_from_pydantic_models")
def test_generate_gbnf_grammar_and_documentation_calls_sub_functions(
    mock_gen_grammar: MagicMock, mock_gen_doc: MagicMock
):
    mock_gen_grammar.return_value = "mock_grammar"
    mock_gen_doc.return_value = "mock_documentation"
    models_pydantic = [DocNested]


    grammar, doc = generate_gbnf_grammar_and_documentation(
        pydantic_model_list=models_pydantic,
        outer_object_name="TestOuter",
        outer_object_content_key="params",
        model_prefix="CustomModel",
        fields_prefix="CustomFields",
        list_of_outputs=True,
        documentation_with_field_description=False
    )

    assert grammar == "mock_grammar"
    assert doc == "mock_documentation"
    mock_gen_doc.assert_called_once_with(
        models_pydantic, "CustomModel", "CustomFields", documentation_with_field_description=False
    )
    mock_gen_grammar.assert_called_once_with(
        models_pydantic, "TestOuter", "params", True
    )

# --- Tests for generate_and_save_gbnf_grammar_and_documentation ---
@patch("genie_tooling.utils.gbnf.documentation.generate_gbnf_grammar_and_documentation")
@patch("genie_tooling.utils.gbnf.documentation.save_gbnf_grammar_and_documentation")
def test_generate_and_save_integration(
    mock_save_func: MagicMock, mock_generate_func: MagicMock
):
    mock_generate_func.return_value = ("generated_g", "generated_d")
    models_pydantic = [DocMain]

    generate_and_save_gbnf_grammar_and_documentation(
        pydantic_model_list=models_pydantic,
        grammar_file_path="output.gbnf",
        documentation_file_path="output.md",
        outer_object_name="Wrapper",
        list_of_outputs=False
    )
    mock_generate_func.assert_called_once_with(
        pydantic_model_list=models_pydantic,
        outer_object_name="Wrapper",
        outer_object_content_key=None,
        model_prefix="Output Model",
        fields_prefix="Output Fields",
        list_of_outputs=False,
        documentation_with_field_description=True
    )
    mock_save_func.assert_called_once_with("generated_g", "generated_d", "output.gbnf", "output.md")

# --- Tests for generate_gbnf_grammar_and_documentation_from_dictionaries ---
@patch("genie_tooling.utils.gbnf.documentation.create_dynamic_models_from_dictionaries")
@patch("genie_tooling.utils.gbnf.documentation.generate_gbnf_grammar_and_documentation")
def test_generate_from_dictionaries(
    mock_gen_gbnf_doc: MagicMock, mock_create_models: MagicMock
):
    dict_schemas = [{"name": "MySchema", "properties": {"field": {"type": "string"}}}]
    mock_dynamic_model_class = MagicMock(spec=BaseModel)
    mock_create_models.return_value = [mock_dynamic_model_class]
    mock_gen_gbnf_doc.return_value = ("dict_grammar", "dict_doc")

    g, d = generate_gbnf_grammar_and_documentation_from_dictionaries(
        dictionaries=dict_schemas,
        outer_object_name="OuterDict",
        model_prefix="DictModel"
    )
    assert g == "dict_grammar"
    assert d == "dict_doc"
    mock_create_models.assert_called_once_with(dict_schemas)
    mock_gen_gbnf_doc.assert_called_once_with(
        [mock_dynamic_model_class],
        outer_object_name="OuterDict",
        outer_object_content_key=None,
        model_prefix="DictModel",
        fields_prefix="Output Fields",
        list_of_outputs=False,
        documentation_with_field_description=True
    )

# --- Tests for map_grammar_names_to_pydantic_model_class ---
def test_map_grammar_names_to_pydantic_model_class():
    models = [DocMain, DocNested, DocRecursive]
    mapping = map_grammar_names_to_pydantic_model_class(models)

    assert len(mapping) == 3
    assert mapping[format_model_and_field_name("DocMain")] == DocMain
    assert mapping[format_model_and_field_name("DocNested")] == DocNested
    assert mapping[format_model_and_field_name("DocRecursive")] == DocRecursive
