# examples/E23_local_e2e_llama_cpp_server.py
"""
End-to-End Test for Genie Tooling with Llama.cpp Server Provider
-----------------------------------------------------------------
This example demonstrates a comprehensive flow using the Genie facade,
targeting a Llama.cpp server (Ollama-compatible API) for LLM operations.
It covers LLM chat/generate with Pydantic parsing, RAG, custom tool
execution, command processing with HITL, prompt management, conversation
state, guardrails, and a simple ReActAgent.

It also demonstrates the use of the `@traceable` decorator for adding
custom application logic to the observability trace.

Prerequisites:
1. `genie-tooling` installed (`poetry install --all-extras`).
2. A Llama.cpp server running and accessible, typically at `http://localhost:8080`.
   The server should be configured with a GBNF-compatible model (e.g., Mistral).
   You can set the model alias used by the server via the `LLAMA_CPP_MODEL_ALIAS`
   environment variable (defaults to "mistral:latest" if not set).
   Example server command:
   `./server -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -c 4096 --host 0.0.0.0 --port 8080 --api-key mysecretkey --model-alias mistral:latest --cont-batching --embedding --gbnf-enabled`
   (Adjust model path and other server parameters as needed. If using an API key,
    set `LLAMA_CPP_API_KEY="mysecretkey"` in your environment for this script.)
"""
import asyncio
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from genie_tooling import tool
from genie_tooling.agents.react_agent import ReActAgent
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.observability import traceable  # Import the new decorator
from pydantic import BaseModel
from pydantic import Field as PydanticField

# --- Configuration ---
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
LLAMA_CPP_MODEL_ALIAS = os.getenv("LLAMA_CPP_MODEL_ALIAS", "mistral:latest") # Model alias server uses
LLAMA_CPP_API_KEY_NAME = "LLAMA_CPP_API_KEY" # Env var name for the API key if server needs one

# --- Pydantic Model for LLM Output Parsing ---
class ExtractedDetails(BaseModel):
    item_name: str = PydanticField(description="The name of the item.")
    quantity: int = PydanticField(gt=0, description="The quantity of the item.")
    notes: Optional[str] = PydanticField(None, description="Optional notes about the item.")

# --- Helper function decorated with @traceable ---
@traceable
async def _get_size_from_path(file_path: Path, context: Dict[str, Any]) -> int:
    """A helper function to demonstrate custom tracing."""
    # This function's execution will appear as a nested span in the trace.
    # The 'file_path' argument will be automatically added as a span attribute.
    logging.info(f"[_get_size_from_path] Getting size for: {file_path}")
    return file_path.stat().st_size

# --- Custom Tool Definition ---
@tool
async def get_file_metadata(file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves metadata for a specified file within the agent's sandbox.
    Args:
        file_path (str): The relative path to the file within the agent's sandbox.
        context (Dict[str, Any]): The invocation context, used for tracing.
    Returns:
        Dict[str, Any]: A dictionary containing file metadata (name, size, exists) or an error.
    """
    sandbox_base = Path("./e23_agent_sandbox")
    try:
        prospective_path = (sandbox_base / file_path).resolve()
        if not str(prospective_path).startswith(str(sandbox_base.resolve())):
            return {"error": "Path traversal attempt detected.", "path_resolved": str(prospective_path)}
        full_path = prospective_path
    except Exception as e:
        return {"error": f"Path resolution error: {str(e)}"}

    if full_path.exists() and full_path.is_file():
        # Call the traceable helper function, passing the context through
        file_size = await _get_size_from_path(file_path=full_path, context=context)
        return {"file_name": full_path.name, "size_bytes": file_size, "exists": True}
    else:
        return {"error": "File not found or is not a file.", "path_checked": str(full_path), "exists": False}

async def run_local_e2e_llama_cpp_server():
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.llm_providers.impl.llama_cpp_provider").setLevel(logging.DEBUG)


    print(f"--- Genie Tooling Local E2E Test (Llama.cpp Server @ {LLAMA_CPP_BASE_URL} with model {LLAMA_CPP_MODEL_ALIAS}) ---")
    if os.getenv(LLAMA_CPP_API_KEY_NAME):
        print(f"Using API Key from environment variable: {LLAMA_CPP_API_KEY_NAME}")
    else:
        print(f"No API Key ({LLAMA_CPP_API_KEY_NAME}) found in environment. Assuming server does not require one.")


    sandbox_dir = Path("./e23_agent_sandbox")
    rag_data_dir = sandbox_dir / "rag_docs"
    prompt_dir = Path("./e23_prompts")

    for p in [sandbox_dir, rag_data_dir, prompt_dir]:
        if p.exists(): shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    (rag_data_dir / "doc1.txt").write_text("Llama.cpp is a C/C++ port of Llama for fast inference.")
    (rag_data_dir / "doc2.txt").write_text("Genie Tooling uses llama.cpp via its Ollama-compatible API for structured output.")
    (prompt_dir / "greeting_template.j2").write_text(
        '[{"role": "system", "content": "You are {{ bot_name }}."},\n'
        ' {"role": "user", "content": "Hello! My name is {{ user_name }}."}]'
    )
    (sandbox_dir / "testfile.txt").write_text("This is a test file for metadata.")
    (prompt_dir / "react_agent_system_prompt_v1.j2").write_text(
        "You are ReActBot. Your goal is: {{ goal }}.\n"
        "Available tools:\n{{ tool_definitions }}\n"
        "Scratchpad (Thought/Action/Observation cycles):\n{{ scratchpad }}\nThought:"
    )

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="llama_cpp", # Target Llama.cpp server
            llm_llama_cpp_base_url=LLAMA_CPP_BASE_URL,
            llm_llama_cpp_model_name=LLAMA_CPP_MODEL_ALIAS,
            llm_llama_cpp_api_key_name=LLAMA_CPP_API_KEY_NAME if os.getenv(LLAMA_CPP_API_KEY_NAME) else None,

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
        # The @tool decorated `get_file_metadata` is auto-enabled by default.
        # Class-based tools like calculator and sandboxed_fs still need to be enabled.
        tool_configurations={
            "calculator_tool": {},
            "sandboxed_fs_tool_v1": {"sandbox_base_path": str(sandbox_dir.resolve())},
        },
        guardrail_configurations={
            "keyword_blocklist_guardrail_v1": {
                "blocklist": ["super_secret_project_X", "highly_classified_info"],
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
        print("\n[1] Initializing Genie facade...")
        genie = await Genie.create(config=app_config)
        await genie.register_tool_functions([get_file_metadata])
        print("Genie facade initialized and custom tool registered!")

        print("\n[2] Testing LLM Chat and Generate (with Pydantic parsing)...")
        try:
            chat_resp = await genie.llm.chat([{"role": "user", "content": "Hello Llama.cpp server! Write a short, friendly greeting."}])
            print(f"  LLM Chat Response: {chat_resp['message']['content'][:100]}...")

            gen_prompt = "Extract details: Item is 'SuperWidget', quantity is 55, notes: 'Handle with care'."
            gen_resp = await genie.llm.generate(
                prompt=f"You must output ONLY a valid JSON object. {gen_prompt}",
                output_schema=ExtractedDetails, # For GBNF
                temperature=0.1,
                n_predict=256 # Llama.cpp server needs n_predict for GBNF with /v1/completions
            )
            print(f"  LLM Generate (raw text for Pydantic): {gen_resp['text']}")
            parsed_details = await genie.llm.parse_output(gen_resp, schema=ExtractedDetails)
            if isinstance(parsed_details, ExtractedDetails):
                print(f"  Parsed Pydantic: Name='{parsed_details.item_name}', Qty='{parsed_details.quantity}', Notes='{parsed_details.notes}'")
                assert parsed_details.item_name == "SuperWidget"
                assert parsed_details.quantity == 55
            else:
                assert False, f"Pydantic parsing failed, got type: {type(parsed_details)}"
        except Exception as e_llm:
            print(f"  LLM Error: {e_llm} (Is Llama.cpp server running with model '{LLAMA_CPP_MODEL_ALIAS}' at {LLAMA_CPP_BASE_URL} and GBNF enabled?)")
            raise

        print("\n[3] Testing RAG (indexing and search)...")
        try:
            rag_collection_name = "e23_llama_docs"
            index_result = await genie.rag.index_directory(str(rag_data_dir.resolve()), collection_name=rag_collection_name)
            print(f"  Indexed documents from '{rag_data_dir.resolve()}'. Result: {index_result}")
            assert index_result.get("status") == "success"
            rag_results = await genie.rag.search("What is Llama.cpp?", collection_name=rag_collection_name, top_k=1)
            if rag_results:
                print(f"  RAG Search Result: '{rag_results[0].content[:100]}...' (Score: {rag_results[0].score:.2f})")
                assert "Llama.cpp" in rag_results[0].content
            else:
                assert False, "RAG search returned no results"
        except Exception as e_rag:
            print(f"  RAG Error: {e_rag}")
            raise

        print("\n[4] Testing direct custom tool execution (get_file_metadata)...")
        try:
            metadata_result = await genie.execute_tool("get_file_metadata", file_path="testfile.txt")
            print(f"  Metadata for 'testfile.txt': {metadata_result}")
            assert metadata_result.get("file_name") == "testfile.txt"
            assert metadata_result.get("exists") is True
        except Exception as e_tool_direct:
            print(f"  Direct tool execution error: {e_tool_direct}")
            raise

        print("\n[5] Testing `run_command` (LLM-assisted, HITL)...")
        try:
            command_text = "What is the size of the file named testfile.txt in the sandbox?"
            print(f"  Sending command: '{command_text}' (Approval may be requested on CLI)")
            command_result = await genie.run_command(command_text)
            print(f"  `run_command` result: {json.dumps(command_result, indent=2, default=str)}")
            assert command_result.get("tool_result", {}).get("size_bytes") is not None or \
                   command_result.get("hitl_decision", {}).get("status") != "approved", \
                   f"Tool did not run or HITL was not approved. Result: {command_result}"
        except Exception as e_run_cmd:
            print(f"  `run_command` error: {e_run_cmd}")
            raise

        print("\n[6] Testing Prompt Management...")
        try:
            prompt_data = {"bot_name": "E23-Bot", "user_name": "Tester"}
            rendered_chat_prompt = await genie.prompts.render_chat_prompt(name="greeting_template", data=prompt_data)
            assert rendered_chat_prompt is not None and len(rendered_chat_prompt) == 2
            print(f"  Rendered chat prompt: {rendered_chat_prompt}")
        except Exception as e_prompt:
            print(f"  Prompt management error: {e_prompt}")
            raise

        print("\n[7] Testing Conversation State...")
        try:
            session_id = f"e23_session_{uuid.uuid4().hex[:6]}"
            await genie.conversation.add_message(session_id, {"role": "user", "content": "First turn in e23 test."})
            await genie.conversation.add_message(session_id, {"role": "assistant", "content": "Acknowledged first turn."})
            state = await genie.conversation.load_state(session_id)
            assert state is not None and len(state["history"]) == 2
            print(f"  Conversation history for {session_id} (last 2): {state['history'][-2:]}")
        except Exception as e_convo:
            print(f"  Conversation state error: {e_convo}")
            raise

        print("\n[8] Testing Input Guardrail...")
        try:
            blocked_input = "Tell me about super_secret_project_X."
            print(f"  Sending potentially blocked input: '{blocked_input}'")
            await genie.llm.chat([{"role": "user", "content": blocked_input}])
            assert False, "Input guardrail did not block as expected."
        except PermissionError as e_perm:
            print(f"  Guardrail test: Input successfully blocked: {e_perm}")
        except Exception as e_guard:
            print(f"  Guardrail test error (possibly unrelated to guardrail): {e_guard}")
            raise

        print("\n[9] Testing ReActAgent...")
        try:
            react_agent = ReActAgent(genie=genie, agent_config={"max_iterations": 3})
            agent_goal = "What is 15 plus 7 using the calculator?"
            print(f"  Agent Goal: '{agent_goal}' (Tool use by agent does not trigger interactive HITL here)")
            agent_result = await react_agent.run(goal=agent_goal)
            print(f"  ReActAgent Result Status: {agent_result['status']}")
            print(f"  ReActAgent Output: {str(agent_result['output'])[:200]}...")
            assert agent_result["status"] == "success"
            assert "22" in str(agent_result["output"])
        except Exception as e_agent:
            print(f"  ReActAgent Error: {e_agent}")
            raise

        print("\n[10] Testing Token Usage Summary...")
        try:
            usage_summary = await genie.usage.get_summary()
            print(f"  Token Usage: {json.dumps(usage_summary, indent=2)}")
            recorder_id = "in_memory_token_usage_recorder_v1"
            assert recorder_id in usage_summary, f"Recorder '{recorder_id}' not found in usage summary."
            assert usage_summary[recorder_id]["total_records"] > 0
        except Exception as e_usage:
            print(f"  Token usage error: {e_usage}")
            raise

        print("\n--- E2E Test PASSED ---")

    except Exception as e_main:
        print(f"\nE2E Test FAILED with critical error: {e_main}")
        logging.error("E2E Main Error", exc_info=True)
        raise
    finally:
        if genie:
            print("\n[11] Tearing down Genie facade...")
            await genie.close()
            print("Genie facade torn down.")

        print("\n[12] Cleaning up test files/directories...")
        for p_cleanup in [sandbox_dir, prompt_dir]:
            if p_cleanup.exists(): shutil.rmtree(p_cleanup, ignore_errors=True)
        print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(run_local_e2e_llama_cpp_server())
