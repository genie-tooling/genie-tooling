# examples/E24_local_e2e_llama_cpp_internal.py
"""
End-to-End Test for Genie Tooling with Llama.cpp Internal Provider
-------------------------------------------------------------------
This example demonstrates a comprehensive flow using the Genie facade,
specifically targeting the Llama.cpp internal provider for LLM operations.
It covers LLM chat/generate with Pydantic parsing, RAG, custom tool
execution, command processing with HITL, prompt management, conversation
state, guardrails, and a simple ReActAgent.

Prerequisites:
1. `genie-tooling` installed (`poetry install --all-extras --extras llama_cpp_internal`).
2. A GGUF-format LLM model file downloaded locally.
3. **Update `LLAMA_CPP_INTERNAL_MODEL_PATH` below to point to your model file.**
"""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from genie_tooling import tool
from genie_tooling.agents.react_agent import ReActAgent
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from pydantic import BaseModel
from pydantic import Field as PydanticField

# --- Configuration ---
# !!! USER ACTION REQUIRED: Update this path to your GGUF model file !!!
LLAMA_CPP_INTERNAL_MODEL_PATH = os.getenv("LLAMA_CPP_INTERNAL_MODEL_PATH", "/path/to/your/model.gguf")
# Example: LLAMA_CPP_INTERNAL_MODEL_PATH = "/home/user/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Pydantic Model for LLM Output Parsing ---
class ExtractedDetailsInternal(BaseModel):
    item_name: str = PydanticField(description="The name of the item.")
    quantity: int = PydanticField(gt=0, description="The quantity of the item.")
    notes: Optional[str] = PydanticField(None, description="Optional notes about the item.")

# --- Custom Tool Definition ---
@tool
async def get_file_metadata_internal(file_path: str) -> Dict[str, Any]:
    """
    Retrieves metadata for a specified file within the agent's sandbox.
    Args:
        file_path (str): The relative path to the file within the agent's sandbox.
    Returns:
        Dict[str, Any]: A dictionary containing file metadata (name, size, exists) or an error.
    """
    sandbox_base = Path("./e24_agent_sandbox")
    try:
        prospective_path = (sandbox_base / file_path).resolve()
        if not str(prospective_path).startswith(str(sandbox_base.resolve())):
            return {"error": "Path traversal attempt detected.", "path_resolved": str(prospective_path)}
        full_path = prospective_path
    except Exception as e:
        return {"error": f"Path resolution error: {str(e)}"}

    if full_path.exists() and full_path.is_file():
        return {"file_name": full_path.name, "size_bytes": full_path.stat().st_size, "exists": True}
    else:
        return {"error": "File not found or is not a file.", "path_checked": str(full_path), "exists": False}

async def run_local_e2e_llama_cpp_internal():
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.llm_providers.impl.llama_cpp_internal_provider").setLevel(logging.DEBUG)

    model_path_obj = Path(LLAMA_CPP_INTERNAL_MODEL_PATH)
    if LLAMA_CPP_INTERNAL_MODEL_PATH == "/path/to/your/model.gguf" or not model_path_obj.exists():
        print("\nERROR: LLAMA_CPP_INTERNAL_MODEL_PATH is not correctly set or file does not exist.")
        print(f"Please edit this script ('{__file__}') and update LLAMA_CPP_INTERNAL_MODEL_PATH,")
        print("or set the LLAMA_CPP_INTERNAL_MODEL_PATH environment variable.")
        print(f"Current path: '{LLAMA_CPP_INTERNAL_MODEL_PATH}' (Exists: {model_path_obj.exists()})\n")
        return

    print(f"--- Genie Tooling Local E2E Test (Llama.cpp Internal with model: {model_path_obj.name}) ---")

    sandbox_dir = Path("./e24_agent_sandbox")
    rag_data_dir = sandbox_dir / "rag_docs"
    prompt_dir = Path("./e24_prompts")

    for p in [sandbox_dir, rag_data_dir, prompt_dir]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    (rag_data_dir / "doc1.txt").write_text("Internal Llama.cpp provides local GGUF model execution.")
    (rag_data_dir / "doc2.txt").write_text("Genie Tooling integrates llama-cpp-python for this.")
    (prompt_dir / "greeting_template_internal.j2").write_text(
        '[{"role": "system", "content": "You are {{ bot_name }} running on Llama.cpp internal."},\n'
        ' {"role": "user", "content": "Hello! My name is {{ user_name }}."}]'
    )
    (sandbox_dir / "testfile_internal.txt").write_text("This is a test file for internal metadata.")
    (prompt_dir / "react_agent_system_prompt_internal_v1.j2").write_text(
        "You are ReActBot (Internal Llama). Your goal is: {{ goal }}.\n"
        "Available tools:\n{{ tool_definitions }}\n"
        "Scratchpad (Thought/Action/Observation cycles):\n{{ scratchpad }}\nThought:"
    )

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(model_path_obj.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1, # Offload all to GPU if possible
            llm_llama_cpp_internal_n_ctx=2048,      # Context size
            llm_llama_cpp_internal_chat_format="mistral", # Example, adjust if your model needs a different one
            llm_llama_cpp_internal_model_name_for_logging=model_path_obj.stem, # Use model file stem for logging

            command_processor="llm_assisted",
            command_processor_formatter_id_alias="compact_text_formatter",
            tool_lookup="embedding",
            tool_lookup_embedder_id_alias="st_embedder",
            tool_lookup_formatter_id_alias="compact_text_formatter",
            rag_loader="file_system",
            rag_embedder="sentence_transformer",
            rag_vector_store="faiss",
            cache="in-memory",
            observability_tracer="console_tracer",
            hitl_approver="cli_hitl_approver",
            token_usage_recorder="in_memory_token_recorder",
            input_guardrails=["keyword_blocklist_guardrail"],
            prompt_registry="file_system_prompt_registry",
            prompt_template_engine="jinja2_chat_formatter",
            conversation_state_provider="in_memory_convo_provider",
            default_llm_output_parser="pydantic_output_parser"
        ),
        # The @tool decorated `get_file_metadata_internal` is auto-enabled by default.
        # Class-based tools like calculator and sandboxed_fs still need to be enabled.
        tool_configurations={
            "calculator_tool": {},
            "sandboxed_fs_tool_v1": {"sandbox_base_path": str(sandbox_dir.resolve())},
        },
        guardrail_configurations={
            "keyword_blocklist_guardrail_v1": {
                "blocklist": ["top_secret_internal", "classified_gguf"],
                "action_on_match": "block"
            }
        },
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(prompt_dir.resolve()),
                "template_suffix": ".j2"
            }
        },
        observability_tracer_configurations={
            "console_tracer_plugin_v1": {"log_level": "INFO"}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\n[1] Initializing Genie facade (Llama.cpp Internal)...")
        genie = await Genie.create(config=app_config)
        await genie.register_tool_functions([get_file_metadata_internal])
        print("Genie facade initialized and custom tool registered!")

        print("\n[2] Testing LLM Chat and Generate (with Pydantic parsing)...")
        try:
            chat_resp = await genie.llm.chat([{"role": "user", "content": "Hello Llama.cpp internal! Write a short, friendly greeting."}])
            print(f"  LLM Chat Response: {chat_resp['message']['content'][:100]}...")

            gen_prompt = "Extract details: Item is 'LocalWidget', quantity is 77, notes: 'Needs local processing'."
            gen_resp = await genie.llm.generate(
                prompt=f"You must output ONLY a valid JSON object. {gen_prompt}",
                output_schema=ExtractedDetailsInternal, # For GBNF
                temperature=0.1,
                max_tokens=256 # Llama.cpp internal provider maps max_tokens to n_predict
            )
            print(f"  LLM Generate (raw text for Pydantic): {gen_resp['text']}")
            parsed_details = await genie.llm.parse_output(gen_resp, schema=ExtractedDetailsInternal)
            if isinstance(parsed_details, ExtractedDetailsInternal):
                print(f"  Parsed Pydantic: Name='{parsed_details.item_name}', Qty='{parsed_details.quantity}', Notes='{parsed_details.notes}'")
                assert parsed_details.item_name == "LocalWidget"
                assert parsed_details.quantity == 77
            else:
                assert False, f"Pydantic parsing failed, got type: {type(parsed_details)}"
        except Exception as e_llm:
            print(f"  LLM Error: {e_llm}")
            raise

        print("\n[3] Testing RAG (indexing and search)...")
        try:
            rag_collection_name = "e24_internal_llama_docs"
            index_result = await genie.rag.index_directory(str(rag_data_dir.resolve()), collection_name=rag_collection_name)
            print(f"  Indexed documents from '{rag_data_dir.resolve()}'. Result: {index_result}")
            assert index_result.get("status") == "success"
            rag_results = await genie.rag.search("What is Llama.cpp internal?", collection_name=rag_collection_name, top_k=1)
            if rag_results:
                print(f"  RAG Search Result: '{rag_results[0].content[:100]}...' (Score: {rag_results[0].score:.2f})")
                assert "Llama.cpp" in rag_results[0].content
            else:
                assert False, "RAG search returned no results"
        except Exception as e_rag:
            print(f"  RAG Error: {e_rag}")
            raise

        print("\n[4] Testing direct custom tool execution (get_file_metadata_internal)...")
        try:
            metadata_result = await genie.execute_tool("get_file_metadata_internal", file_path="testfile_internal.txt")
            print(f"  Metadata for 'testfile_internal.txt': {metadata_result}")
            assert metadata_result.get("file_name") == "testfile_internal.txt"
            assert metadata_result.get("exists") is True
        except Exception as e_tool_direct:
            print(f"  Direct tool execution error: {e_tool_direct}")
            raise

        print("\n[5] Testing `run_command` (LLM-assisted, HITL)...")
        try:
            command_text = "What is the size of the file named testfile_internal.txt in the sandbox?"
            print(f"  Sending command: '{command_text}' (Approval may be requested on CLI)")
            command_result = await genie.run_command(command_text)
            print(f"  `run_command` result: {json.dumps(command_result, indent=2, default=str)}")
            assert command_result.get("tool_result", {}).get("size_bytes") is not None or \
                   command_result.get("hitl_decision", {}).get("status") != "approved", \
                   f"Tool did not run or HITL was not approved. Result: {command_result}"
        except Exception as e_run_cmd:
            print(f"  `run_command` error: {e_run_cmd}")
            raise

        print("\n[6] Testing ReActAgent with Llama.cpp Internal...")
        try:
            # Use the internal-specific prompt for the agent
            react_agent = ReActAgent(genie=genie, agent_config={"max_iterations": 3, "system_prompt_id": "react_agent_system_prompt_internal_v1"})
            agent_goal = "What is 25 times 4 using the calculator?"
            print(f"  Agent Goal: '{agent_goal}'")
            agent_result = await react_agent.run(goal=agent_goal)
            print(f"  ReActAgent Result Status: {agent_result['status']}")
            print(f"  ReActAgent Output: {str(agent_result['output'])[:200]}...")
            assert agent_result["status"] == "success"
            assert "100" in str(agent_result["output"])
        except Exception as e_agent:
            print(f"  ReActAgent Error: {e_agent}")
            raise

        print("\n--- E2E Test with Llama.cpp Internal PASSED ---")

    except Exception as e_main:
        print(f"\nE2E Test FAILED with critical error: {e_main}")
        logging.error("E2E Llama.cpp Internal Main Error", exc_info=True)
        raise
    finally:
        if genie:
            print("\n[X] Tearing down Genie facade...")
            await genie.close()
            print("Genie facade torn down.")

        print("\n[Y] Cleaning up test files/directories...")
        for p_cleanup in [sandbox_dir, prompt_dir]:
            if p_cleanup.exists(): shutil.rmtree(p_cleanup, ignore_errors=True)
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(run_local_e2e_llama_cpp_internal())
