"""
Pydantic to GBNF E2E Test Script with Llama.cpp

This script tests the end-to-end Pydantic -> GBNF -> LLM (llama.cpp) -> Pydantic
pipeline using the genie-tooling library.

Prerequisites:
1. `genie-tooling` library installed with all extras, including `llama_cpp_internal`.
2. A GGUF model file downloaded locally.
3. **Update the `LLAMA_CPP_INTERNAL_MODEL_PATH` variable below.**
"""
import asyncio
import json
import logging
import os
from enum import Enum
from pathlib import Path
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

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import LLMCompletionResponse
from genie_tooling.utils.gbnf import generate_gbnf_grammar_from_pydantic_models
from genie_tooling.utils.gbnf.documentation import (
    generate_and_save_gbnf_grammar_and_documentation,
)
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator

# --- Configuration ---
# !!! USER ACTION REQUIRED: Update this path to your GGUF model file !!!
LLAMA_CPP_INTERNAL_MODEL_PATH = os.getenv("LLAMA_CPP_INTERNAL_MODEL_PATH", "/path/to/your/model.gguf")
# Example: LLAMA_CPP_INTERNAL_MODEL_PATH = "/home/user/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Pydantic Model Definitions (Classes renamed to not start with "Test") ---

class SimpleEnumE2E(Enum):
    ALPHA = "alpha_val"
    BETA = "beta_val"
    GAMMA_WITH_SPACE = "gamma with space"

class NestedModelE2E(BaseModel):
    item_id: int = Field(description="Nested item ID.")
    description: Optional[str] = Field(None, description="Description.")
    model_config = {"json_schema_extra": {"example": {"item_id": 1, "description": "Example nested"}}}


class ModelForLlamaE2E(BaseModel):
    """Model for testing GBNF with Llama.cpp."""
    name: str = Field(description="A descriptive name.")
    count: int = Field(gt=0, lt=100, description="A count between 1 and 99.")
    is_valid: bool = Field(default=True, description="Validity flag.")
    tags: Optional[List[str]] = Field(None, description="Optional list of tags.")
    status: SimpleEnumE2E = Field(description="Status from enum.")
    nested: Optional[NestedModelE2E] = Field(None, description="Optional nested model.")
    choice: Literal["Option1", "Option2", "Option3"] = Field(description="A literal choice.")
    constrained_num: int = Field(json_schema_extra={"min_digit": 2, "max_digit": 2}, description="A 2-digit number.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Sample Item",
                "count": 10,
                "is_valid": False,
                "tags": ["test", "example"],
                "status": "beta_val",
                "nested": {"item_id": 99, "description": "A sub-item"},
                "choice": "Option2",
                "constrained_num": 42,
            }
        }
    }

class EnumNonString(Enum):
    INT_ONE = 1
    FLOAT_TWO_POINT_FIVE = 2.5
    BOOL_TRUE = True

class NamingAndNesting(BaseModel):
    """Model to test naming conventions and nested auto-generated names."""
    field_with_number123: str = Field(description="Field with numbers.")
    field_with_underscore_: Optional[int] = Field(None, description="Field ending with underscore.")
    inner_details: Optional[List[NestedModelE2E]] = Field(None, description="List of inner details.")
    numeric_enum_val: Optional[EnumNonString] = Field(None, description="Enum with non-string values.")

    @field_validator("numeric_enum_val", mode="before")
    @classmethod
    def coerce_numeric_enum_str_to_actual_value(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v == "1": return 1
            if v == "2.5": return 2.5
            if v.lower() == "true": return True
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "field_with_number123": "test123",
                "field_with_underscore_": 5,
                "inner_details": [{"item_id": 101, "description": "Inner detail 1"}],
                "numeric_enum_val": 2.5
            }
        }
    }

class MimicDynamicFuncModel(BaseModel):
    """Mimics a model dynamically created from a function signature."""
    name: str = Field(description="The name of the user.")
    age: Optional[int] = Field(default=30, description="The age of the user. Defaults to 30.")
    active: Optional[bool] = Field(default=True, description="User's active status. Defaults to True.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Dynamo",
                "age": 25,
                "active": False
            }
        }
    }

# --- More Complex Models for Edge Case Testing ---

class RootListOfComplexDictsE2E(RootModel[List[Dict[str, Union[int, str, bool, None, SimpleEnumE2E]]]]):
    """Root model: list of dicts with complex union values including an enum."""
    model_config = {"json_schema_extra": {"example": [
        {"id": 1, "value": "alpha_val", "flag": True},
        {"id": 2, "value": None, "flag": False, "extra_enum": "beta_val"}
    ]}}

class AdvancedLiteralsE2E(BaseModel):
    """Model testing advanced Literal types."""
    lit_mixed_types: Literal["Alpha", 100, True, None, SimpleEnumE2E.BETA, "gamma with space"] = Field(description="Literal with mixed primitive types, None, and an Enum member.")
    model_config = {"json_schema_extra": {"example": {"lit_mixed_types": "beta_val"}}}

class AdvancedUnionsE2E(BaseModel):
    """Model testing advanced Union types."""
    union_with_model_and_list: Union[NestedModelE2E, List[int], str] = Field(description="Union involving a Pydantic model, a list of primitives, and a string.")
    optional_union_with_none: Optional[Union[float, bool]] = Field(None, description="Optional union of float or bool (implicitly includes None).")
    model_config = {"json_schema_extra": {"example": {
        "union_with_model_and_list": {"item_id": 202, "description": "Union as model"},
        "optional_union_with_none": True
    }}}

class SetAndTupleFieldsE2E(BaseModel):
    """Model testing Set and various Tuple field types."""
    set_of_strings: Set[str] = Field(description="A set of unique strings.")
    tuple_of_mixed: Tuple[str, int, bool] = Field(description="A fixed-length tuple with mixed types.")
    variable_tuple_of_floats: Tuple[float, ...] = Field(description="A variable-length tuple of floats.")
    model_config = {"json_schema_extra": {"example": {
        "set_of_strings": ["unique1", "unique2"],
        "tuple_of_mixed": ["text", 123, False],
        "variable_tuple_of_floats": [1.1, 2.2, 3.3]
    }}}

class SpecialStringFormatsE2E(BaseModel):
    """Model testing special string formats like markdown code blocks."""
    markdown_text: str = Field(description="A field expected to be a Markdown code block.", json_schema_extra={"markdown_code_block": True})
    triple_quoted_text: str = Field(description="A field expected to be a triple-quoted string.", json_schema_extra={"triple_quoted_string": True})
    model_config = {"json_schema_extra": {"example": {
        "markdown_text": "```python\\nprint('hello')\\n```",
        "triple_quoted_text": "'''This is\\na multi-line\\nstring.'''"
    }}}

class AnyTypeAndRecursiveE2E(BaseModel):
    """Model testing Any type and a recursive structure."""
    any_data: Any = Field(description="A field that can hold any JSON-compatible data.")
    recursive_field: Optional[ModelForLlamaE2E] = Field(None, description="A recursive field using an existing model.") # Simple recursion
    model_config = {"json_schema_extra": {"example": {
        "any_data": {"nested_dict": [1, "mixed", True]},
        "recursive_field": {
            "name": "Recursive Sample", "count": 5, "status": "alpha_val", "choice": "Option1", "constrained_num": 11
        }
    }}}

# --- End of More Complex Models ---


async def run_live_gbnf_test(genie: Genie, model_type: type[BaseModel], model_name_for_prompt: str):
    """
    Tests GBNF generation and LLM interaction for a single Pydantic model.
    """
    print(f"\\n--- Testing Model: {model_name_for_prompt} ---")
    logging.info(f"Testing GBNF generation and LLM parsing for {model_name_for_prompt}")

    try:
        gbnf_grammar_str = generate_gbnf_grammar_from_pydantic_models([model_type])
        print(f"\\n--- Generated GBNF for {model_name_for_prompt} ---")
        print(gbnf_grammar_str)
        print("--- End GBNF ---")
    except Exception as e_gbnf_gen:
        print(f"ERROR generating GBNF for {model_name_for_prompt}: {e_gbnf_gen}")
        logging.error(f"GBNF generation error for {model_name_for_prompt}", exc_info=True)
        return False # Indicate failure if GBNF generation itself fails

    example_json = "{}"
    if hasattr(model_type, "model_config") and isinstance(model_type.model_config, dict):
        schema_extra = model_type.model_config.get("json_schema_extra", {})
        if isinstance(schema_extra, dict) and "example" in schema_extra:
            try:
                example_json = json.dumps(schema_extra["example"], indent=2)
            except Exception:
                pass # Ignore if example itself is not serializable for some reason

    prompt = (
        f"Generate a valid JSON object that strictly conforms to the schema of a '{model_name_for_prompt}'.\\n"
        f"The output MUST be ONLY the JSON object itself, with no other text before or after.\\n"
        f"An example of a valid instance might look like (but generate a new, different one):\\n{example_json}\\n"
        f"Now, generate a new, valid JSON instance for '{model_name_for_prompt}':"
    )
    print(f"Prompt for LLM:\\n{prompt}")

    llm_response: Optional[LLMCompletionResponse] = None
    parsed_object: Optional[BaseModel] = None
    error_occurred = False

    try:
        llm_response = await genie.llm.generate(
            prompt=prompt,
            output_schema=model_type, # Pass the Pydantic model class
            provider_id="llama_cpp_internal", # Explicitly use internal provider
            temperature=0.3,
            max_tokens=2048 # Use max_tokens for internal provider
        )

        print(f"LLM Raw Response Text:\\n{llm_response['text']}")

        if not llm_response["text"] or not llm_response["text"].strip().startswith(("{", "[")):
            print("ERROR: LLM did not return a JSON-like string.")
            logging.error(f"LLM for {model_name_for_prompt} did not return a JSON-like string: {llm_response['text']}")
            error_occurred = True
        else:
            # Use genie.llm.parse_output which handles Pydantic parsing
            parsed_object = await genie.llm.parse_output(
                response=llm_response, # Pass the full LLMCompletionResponse
                schema=model_type      # Pass the Pydantic model class
            )

            if isinstance(parsed_object, model_type):
                print(f"SUCCESS: LLM output successfully parsed into {model_name_for_prompt} instance.")
                print(f"Parsed object:\\n{parsed_object.model_dump_json(indent=2)}")
                logging.info(f"Successfully parsed {model_name_for_prompt} from LLM output.")
            else:
                print(f"ERROR: Parsed output is not an instance of {model_name_for_prompt}. Type: {type(parsed_object)}")
                logging.error(f"Parsed output for {model_name_for_prompt} is type {type(parsed_object)}, not {model_name_for_prompt}.")
                if parsed_object is not None: print(f"Parsed data: {parsed_object}")
                error_occurred = True

    except ValidationError as ve:
        print(f"VALIDATION ERROR for {model_name_for_prompt}:\\n{ve}")
        logging.error(f"Pydantic validation failed for {model_name_for_prompt}: {ve}", exc_info=True)
        if llm_response and llm_response.get("text"):
             logging.error(f"LLM text that failed Pydantic validation:\\n{llm_response['text']}")
        error_occurred = True
    except json.JSONDecodeError as je:
        print(f"JSON DECODE ERROR for {model_name_for_prompt}:\\n{je}")
        logging.error(f"JSON decoding failed for {model_name_for_prompt}: {je}", exc_info=True)
        if llm_response and llm_response.get("text"):
             logging.error(f"LLM text that failed JSON decoding:\\n{llm_response['text']}")
        error_occurred = True
    except Exception as e:
        print(f"UNEXPECTED ERROR during LLM call or parsing for {model_name_for_prompt}:\\n{e}")
        logging.error(f"Unexpected error for {model_name_for_prompt}: {e}", exc_info=True)
        error_occurred = True

    if error_occurred:
        print(f"--- Test FAILED for Model: {model_name_for_prompt} ---")
    else:
        print(f"--- Test PASSED for Model: {model_name_for_prompt} ---")
    print("-" * 50)
    return not error_occurred


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(module)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    model_path = Path(LLAMA_CPP_INTERNAL_MODEL_PATH)
    if LLAMA_CPP_INTERNAL_MODEL_PATH == "/path/to/your/model.gguf" or not model_path.exists():
        print("\\nERROR: Model path not configured or file does not exist.")
        print("Please edit this script and set 'LLAMA_CPP_INTERNAL_MODEL_PATH' to your GGUF model file path.")
        print(f"Current path: '{LLAMA_CPP_INTERNAL_MODEL_PATH}' (Exists: {model_path.exists()})\\n")
        return

    print("--- Genie Tooling: Pydantic to GBNF E2E Test with Llama.cpp (Internal) ---")
    print(f"Attempting to use Llama.cpp internal provider with model: {model_path.name}")

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(model_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1,
            llm_llama_cpp_internal_chat_format="mistral" # Adjust if needed
        )
    )

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config)
        print("Genie instance created successfully.")

        models_to_test = [
            (ModelForLlamaE2E, "ModelForLlamaE2E"),
            (NestedModelE2E, "NestedModelE2E"),
            (NamingAndNesting, "NamingAndNesting"),
            (MimicDynamicFuncModel, "MimicDynamicFuncModel"),
            (RootListOfComplexDictsE2E, "RootListOfComplexDictsE2E"),
            (AdvancedLiteralsE2E, "AdvancedLiteralsE2E"),
            (AdvancedUnionsE2E, "AdvancedUnionsE2E"),
            (SetAndTupleFieldsE2E, "SetAndTupleFieldsE2E"),
            (SpecialStringFormatsE2E, "SpecialStringFormatsE2E"),
            (AnyTypeAndRecursiveE2E, "AnyTypeAndRecursiveE2E"),
        ]

        all_passed = True
        for model_cls, model_name_str in models_to_test:
            success = await run_live_gbnf_test(genie, model_cls, model_name_str) # type: ignore
            if not success:
                all_passed = False
            await asyncio.sleep(1) # Small delay between tests

        if all_passed:
            print("\\nAll Pydantic model GBNF tests with Llama.cpp PASSED!")
        else:
            print("\\nOne or more Pydantic model GBNF tests with Llama.cpp FAILED.")

        # Demonstrate GBNF and documentation generation for one of the complex models
        output_dir = Path("./gbnf_test_output")
        output_dir.mkdir(exist_ok=True, parents=True)
        complex_model_for_doc_gen = AdvancedUnionsE2E
        print(f"\\nGenerating GBNF and Markdown for {complex_model_for_doc_gen.__name__} as a demonstration...")
        generate_and_save_gbnf_grammar_and_documentation(
            pydantic_model_list=[complex_model_for_doc_gen],
            grammar_file_path=str(output_dir / f"{complex_model_for_doc_gen.__name__}_demo.gbnf"),
            documentation_file_path=str(output_dir / f"{complex_model_for_doc_gen.__name__}_demo_doc.md"),
        )
        print(f"Demo GBNF and Docs saved to {output_dir.resolve()}")


    except Exception as e:
        print(f"A critical error occurred in the main script: {e}")
        logging.critical("Main script execution error", exc_info=True)
    finally:
        if genie:
            await genie.close()
            print("Genie instance closed.")

if __name__ == "__main__":
    asyncio.run(main())