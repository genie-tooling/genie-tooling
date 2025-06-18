# examples/E25_llama_cpp_internal_gbnf_parsing.py
# (Originally tst.py)
import asyncio
import json
import logging
import os  
from pathlib import Path
from typing import List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage
from pydantic import BaseModel, Field


class ExtractedItemInfo(BaseModel):
    item_name: str = Field(description="The name of the item mentioned.")
    quantity: int = Field(description="The quantity of the item.", gt=0)
    color: Optional[str] = Field(None, description="The color of the item, if specified.")

async def run_llama_cpp_internal_gbnf_test():
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.llm_providers.impl.llama_cpp_internal_provider").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.utils.gbnf").setLevel(logging.DEBUG)

    print("--- Llama.cpp Internal Provider GBNF Parsing Test ---")

    # !!! USER ACTION REQUIRED: Update this path to your GGUF model file !!!
    # You can also set the LLAMA_CPP_INTERNAL_MODEL_PATH environment variable.
    default_model_path = "/path/to/your/model.gguf" # Placeholder
    model_path_str = os.getenv("LLAMA_CPP_INTERNAL_MODEL_PATH", default_model_path)
    # Example: model_path_str = "/home/user/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    model_path = Path(model_path_str)
    if model_path_str == default_model_path or not model_path.exists():
        print("\nERROR: Model path not configured or file does not exist.")
        print(f"Please edit this script and set 'model_path_str' (currently '{model_path_str}')")
        print("or set the LLAMA_CPP_INTERNAL_MODEL_PATH environment variable.")
        print(f"Current path check: '{model_path.resolve()}' (Exists: {model_path.exists()})\n")
        return

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(model_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1, # Offload all layers if possible
            llm_llama_cpp_internal_chat_format="mistral", # Adjust to your model if needed
            token_usage_recorder="in_memory_token_recorder",
            default_llm_output_parser="pydantic_output_parser"
        )
    )

    genie: Optional[Genie] = None
    try:
        print(f"\nInitializing Genie with Llama.cpp internal provider for GBNF (Model: {model_path.name})...")
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized successfully!")

        print("\n--- LLM Chat with GBNF Example (Llama.cpp Internal) ---")
        user_prompt_for_gbnf = (
            "From the sentence: 'I need 5 red apples for the pie.', "
            "extract the item name, quantity, and color. "
            "Please provide the output as a JSON object with keys: 'item_name', 'quantity', and 'color'."
            "Output ONLY the JSON object."
        )
        messages: List[ChatMessage] = [{"role": "user", "content": user_prompt_for_gbnf}]
        print(f"Sending message to LLM for GBNF generation: '{messages[0]['content']}'")

        chat_response = await genie.llm.chat(
            messages,
            output_schema=ExtractedItemInfo,
            temperature=0.1,
            max_tokens=256
        )
        response_content_str = chat_response.get("message", {}).get("content")
        print(f"\nLLM (Llama.cpp Internal) raw JSON string output:\n{response_content_str}")

        if response_content_str:
            try:
                parsed_info: Optional[ExtractedItemInfo] = await genie.llm.parse_output(
                    chat_response,
                    schema=ExtractedItemInfo
                )
                if isinstance(parsed_info, ExtractedItemInfo):
                    print("\nSuccessfully parsed into Pydantic model:")
                    print(json.dumps(parsed_info.model_dump(), indent=2))
                    assert parsed_info.item_name.lower() == "apples"
                    assert parsed_info.quantity == 5
                    assert parsed_info.color and parsed_info.color.lower() == "red"
                    print("\nGBNF Test with Pydantic Model PASSED!")
                else:
                    print(f"\nERROR: Parsing did not return an ExtractedItemInfo instance. Got: {type(parsed_info)}")
            except ValueError as ve:
                print(f"\nERROR: Failed to parse or validate LLM output against Pydantic model: {ve}")
                print(f"LLM's raw JSON string was: {response_content_str}")
            except Exception as e_parse:
                print(f"\nERROR: An unexpected error occurred during parsing: {e_parse}")
        else:
            print("\nERROR: LLM did not return any content for GBNF parsing.")

        usage_info = chat_response.get("usage")
        if usage_info:
            print(f"\nToken usage: {usage_info}")
        elif genie and genie.usage:
            summary = await genie.usage.get_summary()
            print(f"\nToken usage summary from recorder: {summary}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Llama.cpp internal GBNF test error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")

if __name__ == "__main__":
    asyncio.run(run_llama_cpp_internal_gbnf_test())
