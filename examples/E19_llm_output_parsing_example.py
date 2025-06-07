# examples/E19_llm_output_parsing_example.py
"""
Example: Using LLM Output Parsing (`genie.llm.parse_output`)
------------------------------------------------------------
This example demonstrates how to use `genie.llm.parse_output` to parse
structured data (like JSON) from an LLM's text response. It also shows
parsing into a Pydantic model.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   (Pydantic is a core dependency).
2. Ensure Ollama is running and 'mistral:latest' is pulled (or configure a different LLM).
3. Run from the root of the project:
   `poetry run python examples/E19_llm_output_parsing_example.py`
"""
import asyncio
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import LLMChatResponse
from pydantic import BaseModel, Field


class ExtractedInfo(BaseModel):
    name: str = Field(description="The full name of the person.")
    age: Optional[int] = Field(None, description="The person's age, if mentioned.")
    city: Optional[str] = Field(None, description="The city the person lives in, if mentioned.")

async def run_llm_output_parsing_demo():
    print("--- LLM Output Parsing Example ---")
    logging.basicConfig(level=logging.INFO)

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            default_llm_output_parser="json_output_parser" # Set default
        )
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie for LLM output parsing...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        print("\n--- Parsing to Dictionary (JSON) ---")
        prompt_for_json = (
            "Extract the name, age, and city from the following text "
            "and return it as a JSON object: "
            "'John Doe is 42 years old and lives in New York. He enjoys programming.' "
            "Ensure your output is ONLY the JSON object."
        )
        print(f"Sending prompt for JSON: {prompt_for_json}")

        try:
            llm_response_json: LLMChatResponse = await genie.llm.chat(
                [{"role": "user", "content": prompt_for_json}]
            )
            print(f"LLM Raw Text Output: {llm_response_json['message']['content']}")

            parsed_dict = await genie.llm.parse_output(llm_response_json) # Uses default JSON parser
            print(f"Parsed Dictionary: {parsed_dict}")
            if isinstance(parsed_dict, dict):
                print(f"  Name from dict: {parsed_dict.get('name')}")
        except Exception as e_json:
            print(f"Error during JSON parsing: {e_json}")


        print("\n--- Parsing to Pydantic Model ---")
        prompt_for_pydantic = (
            "From the text 'Alice is thirty and resides in London.', "
            "extract the name, age, and city. Format as a JSON object "
            "suitable for a model with fields: name (str), age (int, optional), city (str, optional)."
            "Output ONLY the JSON object."
        )
        print(f"Sending prompt for Pydantic: {prompt_for_pydantic}")

        try:
            llm_response_pydantic: LLMChatResponse = await genie.llm.chat(
                 [{"role": "user", "content": prompt_for_pydantic}]
            )
            print(f"LLM Raw Text Output: {llm_response_pydantic['message']['content']}")

            parsed_model_instance = await genie.llm.parse_output(
                llm_response_pydantic,
                parser_id="pydantic_output_parser_v1",
                schema=ExtractedInfo
            )

            if isinstance(parsed_model_instance, ExtractedInfo):
                print(f"Parsed Pydantic Model: {parsed_model_instance.model_dump()}")
                print(f"  Name from model: {parsed_model_instance.name}")
                print(f"  Age from model: {parsed_model_instance.age}")
            else:
                print(f"Parsing did not return an ExtractedInfo instance. Got: {type(parsed_model_instance)}")

        except Exception as e_pydantic:
            print(f"Error during Pydantic parsing: {e_pydantic}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("LLM output parsing demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_llm_output_parsing_demo())
