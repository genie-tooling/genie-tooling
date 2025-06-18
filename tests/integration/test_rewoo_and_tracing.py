"""
Genie Tooling - End-to-End Test for ReWOO Agent and @traceable Decorator

This script demonstrates the integration of two advanced features:
1.  **ReWOO Agent**: A Plan-and-Execute style agent that uses an LLM to create a
    plan of tool calls, executes them, and then uses another LLM call to
    synthesize a final answer from the results.
2.  **@traceable Decorator**: A decorator from `genie_tooling.observability` that
    automatically creates OpenTelemetry spans for instrumented functions, linking
    them to the parent trace.

**Prerequisites:**
1.  **Genie Tooling Installed**: `poetry install --all-extras`
2.  **Local GGUF Model**: You must download a GGUF model file (e.g., from Hugging
    Face) and update the `LLAMA_CPP_INTERNAL_MODEL_PATH` variable below.
    An instruction-tuned model like Mistral or Llama-3 is recommended.
3.  **OpenTelemetry Collector (e.g., Jaeger)**: For this test to be fully
    verifiable, you need an OTel collector running to receive the traces.
    A simple way to start one is with Docker:
    ```bash
    docker run -d --name jaeger \
      -p 16686:16686 \
      -p 4318:4318 \
      jaegertracing/all-in-one:latest
    ```
    After running this script, you can view the detailed trace at http://localhost:16686.
"""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from genie_tooling import tool
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.observability import traceable

# --- Configuration ---
# !!! USER ACTION REQUIRED: Update this path to your GGUF model file !!!
LLAMA_CPP_INTERNAL_MODEL_PATH = "/home/kal/code/models/Qwen3-8B.Q4_K_M.gguf"
# Example: LLAMA_CPP_INTERNAL_MODEL_PATH = "/home/user/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Tool Definition with @traceable helper ---

# This is the key part for testing the decorator. The `execute` method of the tool
# will call this helper function, passing the `context` dictionary it receives.
# The decorator will then use the OTel context within that dictionary to create
# a child span, correctly linking it to the main trace.
@traceable
async def _read_file_content_traceable(file_path: Path, context: Dict[str, Any]) -> str:
    """A helper function wrapped with @traceable to generate a child span."""
    logging.info(f"Executing traceable helper to read '{file_path}'...")
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@tool
async def get_file_line_count(file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reads a file from the local filesystem and returns the number of lines.
    Args:
        file_path (str): The path to the file to be read.
        context (Dict[str, Any]): The invocation context, passed automatically.
    """
    try:
        # We call our traceable helper function here, passing the context through.
        content = await _read_file_content_traceable(Path(file_path), context=context)
        line_count = len(content.splitlines())
        return {"file_path": file_path, "line_count": line_count, "status": "success"}
    except FileNotFoundError:
        return {"file_path": file_path, "error": "File not found.", "status": "error"}
    except Exception as e:
        return {"file_path": file_path, "error": str(e), "status": "error"}


async def run_rewoo_and_traceable_test():
    """Main function to run the E2E test."""
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("genie_tooling").setLevel(logging.DEBUG) # Set to DEBUG to see all trace events

    print("--- Genie Tooling: ReWOO + @traceable E2E Test ---")

    # --- 1. Check for Model Path ---
    model_path = Path(LLAMA_CPP_INTERNAL_MODEL_PATH)
    if not model_path.exists() or "path/to/your/model.gguf" in str(model_path):
        print("\nERROR: Model path not configured or file does not exist.")
        print("Please edit this script and set 'LLAMA_CPP_INTERNAL_MODEL_PATH' to your GGUF model file path.")
        print(f"Current path: '{LLAMA_CPP_INTERNAL_MODEL_PATH}'\n")
        return

    # --- 2. Create a dummy file for the tool to read ---
    test_file_dir = Path("./tst6_sandbox")
    test_file_dir.mkdir(exist_ok=True)
    test_file = test_file_dir / "test_doc.txt"
    test_file.write_text("This is line one.\nThis is line two.\nThis is line three.\n")

    # --- 3. Configure Genie with ReWOO and OpenTelemetry ---
    app_config = MiddlewareConfig(
        # Use FeatureSettings for quick setup
        features=FeatureSettings(
            # LLM: Use the internal Llama.cpp provider for a self-contained setup
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(model_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1,
            llm_llama_cpp_internal_chat_format="mistral", # Adjust if using a different model type
            logging_adapter="pyvider_log_adapter", # Select Pyvider
            # COMMAND PROCESSOR: Enable the ReWOO agent
            command_processor="rewoo",
            # PROMPT SYSTEM: Ensure Jinja2 is the default engine for ReWOO's prompts
            prompt_template_engine="jinja2_chat_formatter",
            # PARSER: Ensure Pydantic parser is available for ReWOO's structured plan
            default_llm_output_parser="pydantic_output_parser",
            observability_tracer="console_tracer",
        ),
        log_adapter_configurations={
         "pyvider_telemetry_log_adapter_v1": { # Canonical ID for Pyvider adapter
            "enable_key_name_redaction": False, # Disable key name redaction
            # "enable_schema_redaction": True, # Schema redaction can remain if desired
            # "service_name": "my-genie-rewoo-app" # Optional: specific service name for Pyvider
        }
    },
        auto_enable_registered_tools=False,
        tool_configurations={
            "get_file_line_count": {}, # The key must match the tool's identifier
        },
    )

    # --- 4. Initialize Genie and Run the Command ---
    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with ReWOO and OTel tracing...")
        genie = await Genie.create(config=app_config)

        # Register our custom tool with Genie
        await genie.register_tool_functions([get_file_line_count])
        print("Genie initialized and custom tool registered.")

        # Define the command for the ReWOO agent
        command = f"How many lines are in the file located at '{test_file.resolve()!s}'?"
        print(f"\nSending command to ReWOO agent: '{command}'")
        print("Agent will now PLAN, EXECUTE (calling the tool), and SOLVE.")
        print("Check your OTel backend (e.g., Jaeger at http://localhost:16686) for the 'genie-rewoo-traceable-test' service to see the full trace.")

        # Run the command
        result = await genie.run_command(command)

        # --- 5. Print the results ---
        print("\n--- ReWOO Agent Final Output ---")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("final_answer"):
            print(f"Final Answer: {result['final_answer']}")

        print("\n--- ReWOO Agent Thought Process (Plan & Evidence) ---")
        if result.get("llm_thought_process"):
            # Pretty-print the JSON thought process
            try:
                thought_data = json.loads(result["llm_thought_process"])
                print(json.dumps(thought_data, indent=2))
            except json.JSONDecodeError:
                print(result["llm_thought_process"])

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        logging.exception("E2E test error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")
        # Clean up the dummy file
        if test_file.exists():
            test_file.unlink()
        if test_file_dir.exists():
            test_file_dir.rmdir()


if __name__ == "__main__":
    asyncio.run(run_rewoo_and_traceable_test())
