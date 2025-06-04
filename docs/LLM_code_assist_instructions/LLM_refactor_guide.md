# LLM_GUIDE_METADATA_V0.2
# THIS FILE IS A HYPER-COMPRESSED SEMANTIC REPRESENTATION OF GENIE TOOLING FOR LLM CONSUMPTION.
# LLM: IF YOU UPDATE THE GENIE TOOLING PROJECT CODE (CLASSES, METHODS, PARAMS, CORE LOGIC),
# YOU MUST UPDATE THIS FILE TO REFLECT THOSE CHANGES ACCURATELY.
# MAINTAIN THE COMPRESSED FORMAT. HUMAN READABILITY IS NOT A PRIORITY.

**Project**: Genie Tooling. Python Agentic/LLM Middleware. Async. MIT.

**Facade**: `genie_tooling.genie.Genie`
*   `init`: `Genie.create(cfg:MiddlewareConfig, kp_inst?:KeyProvider)`
*   `llm`: `LLMInterface`
    *   `chat(msgs:List[ChatMessage], prov_id?:str, stream?:bool, **kw) -> LLMChatResp|AsyncIter[LLMChatChunk]`
    *   `generate(prompt:str, prov_id?:str, stream?:bool, **kw) -> LLMCompResp|AsyncIter[LLMCompChunk]`
    *   `parse_output(resp:LLMChatResp|LLMCompResp, parser_id?:str, schema?:Any) -> ParsedOutput`
*   `rag`: `RAGInterface`
    *   `index_directory(path:str, collection_name?:str, loader_id?:str, splitter_id?:str, embedder_id?:str, vector_store_id?:str, loader_config?:Dict, splitter_config?:Dict, embedder_config?:Dict, vector_store_config?:Dict, **kw) -> Dict[str,Any]`
    *   `index_web_page(url:str, collection_name?:str, loader_id?:str, splitter_id?:str, embedder_id?:str, vector_store_id?:str, loader_config?:Dict, splitter_config?:Dict, embedder_config?:Dict, vector_store_config?:Dict, **kw) -> Dict[str,Any]`
    *   `search(query:str, collection_name?:str, top_k?:int, retriever_id?:str, retriever_config?:Dict, **kw) -> List[RetrievedChunk]`
*   `tools`: (Methods on `Genie` instance)
    *   `execute_tool(tool_identifier:str, **params:Any) -> Any` (Tool must be enabled in `tool_configurations`)
    *   `run_command(command:str, processor_id?:str, conversation_history?:List[ChatMessage]) -> CmdProcResp` (Integrates HITL if configured; tools used must be enabled in `tool_configurations`)
*   `tool_reg`: (Methods on `Genie` instance)
    *   `@genie_tooling.tool` (Decorator for functions. Auto-generates metadata: id, name, desc_human, desc_llm, input_schema, output_schema).
    *   `await genie.register_tool_functions(functions:List[Callable])` (Registers decorated functions. Invalidates tool lookup index. Tools still need to be enabled in `tool_configurations` to be used by `execute_tool` or `run_command`).
*   `prompts`: `PromptInterface`
    *   `get_prompt_template_content(name:str, version?:str, registry_id?:str) -> str?`
    *   `render_prompt(name:str, data:PromptData, version?:str, registry_id?:str, template_engine_id?:str) -> FormattedPrompt?`
    *   `render_chat_prompt(name:str, data:PromptData, version?:str, registry_id?:str, template_engine_id?:str) -> List[ChatMessage]?`
    *   `list_templates(registry_id?:str) -> List[PromptIdentifier]`
*   `conversation`: `ConversationInterface`
    *   `load_state(session_id:str, provider_id?:str) -> ConversationState?`
    *   `save_state(state:ConversationState, provider_id?:str)`
    *   `add_message(session_id:str, message:ChatMessage, provider_id?:str)`
    *   `delete_state(session_id:str, provider_id?:str) -> bool`
*   `observability`: `ObservabilityInterface`
    *   `trace_event(event_name:str, data:Dict, component?:str, correlation_id?:str)`
*   `human_in_loop`: `HITLInterface`
    *   `request_approval(request:ApprovalRequest, approver_id?:str) -> ApprovalResponse`
*   `usage`: `UsageTrackingInterface`
    *   `record_usage(record:TokenUsageRecord)`
    *   `get_summary(recorder_id?:str, filter_criteria?:Dict) -> Dict`
*   `task_queue`: `TaskQueueInterface`
    *   `submit_task(task_name:str, args?:Tuple, kwargs?:Dict, queue_id?:str, task_options?:Dict) -> str?`
    *   `get_task_status(task_id:str, queue_id?:str) -> TaskStatus?`
    *   `get_task_result(task_id:str, queue_id?:str, timeout_seconds?:float) -> Any?`
    *   `revoke_task(task_id:str, queue_id?:str, terminate?:bool) -> bool`
*   `teardown`: `await genie.close()` (Cleans up all managers and plugins)

**Agent Classes** (in `genie_tooling.agents`):
*   `BaseAgent(genie:Genie, agent_cfg?:Dict)`
    *   `async run(goal:str, **kw) -> AgentOutput` (Abstract method)
*   `ReActAgent(BaseAgent)`
    *   `cfg`: `max_iterations:int` (Def:7), `system_prompt_id:str` (Def: `react_agent_system_prompt_v1`), `llm_provider_id:str?`, `tool_formatter_id:str` (Def: `compact_text_formatter_plugin_v1`), `stop_sequences:List[str]` (Def: `["Observation:"]`), `llm_retry_attempts:int` (Def:1), `llm_retry_delay_seconds:float` (Def:2.0)
    *   `async run(goal:str, **kw) -> AgentOutput` (Implements ReAct loop: Thought, Action, Observation)
*   `PlanAndExecuteAgent(BaseAgent)`
    *   `cfg`: `planner_system_prompt_id:str` (Def: `plan_and_execute_planner_prompt_v1`), `planner_llm_provider_id:str?`, `tool_formatter_id:str` (Def: `compact_text_formatter_plugin_v1`), `max_plan_retries:int` (Def:1), `max_step_retries:int` (Def:0), `replan_on_step_failure:bool` (Def:False)
    *   `async run(goal:str, **kw) -> AgentOutput` (Implements Plan-then-Execute loop)

**Config**: `genie_tooling.config.models.MiddlewareConfig` (`MWCfg`)
*   `features: FeatureSettings` -> `ConfigResolver` (`CfgResolver`) populates `MWCfg`.
    *   (Feature settings as before, but their impact on `tool_configurations` is now indirect or non-existent for merely enabling tools. Tools are enabled via `tool_configurations`.)
*   `tool_configurations: Dict[str_id_or_alias, Dict[str, Any]]` (Primary way to enable tools. Key presence enables tool. Value is tool-specific config.)
*   `ConfigResolver` (`genie_tooling.config.resolver.py`): `features` + aliases -> canonical IDs & cfgs. `PLUGIN_ID_ALIASES` dict.
*   `key_provider_id: str?` Def: `env_keys` if no `key_provider_instance`.
*   `key_provider_instance: KeyProvider?` -> Passed to `Genie.create()`.
*   `*_configurations: Dict[str_id_or_alias, Dict[str, Any]]` (e.g., `llm_provider_configurations`).
*   `plugin_dev_dirs: List[str]`.

**Plugins**: `PluginManager`. IDs/paths: `pyproject.toml` -> `[tool.poetry.plugins."genie_tooling.plugins"]`.
**Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES`.

**Key Plugins (ID | Alias | Cfg/Notes)**:
*   (Plugin list remains largely the same, but remember tools are only active if in `tool_configurations`)
*   `KeyProv`: `environment_key_provider_v1`|`env_keys`.
*   `LLMProv`:
    *   `ollama_llm_provider_v1`|`ollama`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`.
    *   `openai_llm_provider_v1`|`openai`. Cfg: `model_name`, `api_key_name`, `openai_api_base`, `openai_organization`. Needs KP.
    *   `gemini_llm_provider_v1`|`gemini`. Cfg: `model_name`, `api_key_name`, `system_instruction`, `safety_settings`. Needs KP.
    *   `llama_cpp_llm_provider_v1`|`llama_cpp`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`, `api_key_name`. Needs KP if `api_key_name` set.
*   `CmdProc`:
    *   `simple_keyword_processor_v1`|`simple_keyword_cmd_proc`. Cfg: `keyword_map`, `keyword_priority`.
    *   `llm_assisted_tool_selection_processor_v1`|`llm_assisted_cmd_proc`. Cfg: `llm_provider_id`, `tool_formatter_id`, `tool_lookup_top_k`, `system_prompt_template`, `max_llm_retries`.
*   `Tools`: (Examples: `calculator_tool`, `sandboxed_fs_tool_v1`, etc. **Must be listed in `tool_configurations` to be active.**)
*   (Other plugin categories like DefFormatters, RAG, ToolLookupProv, CodeExec, CacheProv, Observability, HITL, TokenUsage, Guardrails, Prompts, Conversation, LLMOutputParsers, TaskQueues, InvocationStrategies remain structurally similar but their instances are loaded based on configuration.)

**Types**:
*   (Types remain the same as listed in V0.3 of the API guide)
*   `ChatMessage`: `{role:Literal["system"|"user"|"assistant"|"tool"], content?:str, tool_calls?:List[ToolCall], tool_call_id?:str, name?:str}`
*   `ToolCall`: `{id:str, type:Literal["function"], function:{name:str, arguments:str_json}}`
*   `LLMCompResp`: `{text:str, finish_reason?:str, usage?:LLMUsageInfo, raw_response:Any}`
*   `LLMChatResp`: `{message:ChatMessage, finish_reason?:str, usage?:LLMUsageInfo, raw_response:Any}`
*   `LLMUsageInfo`: `{prompt_tokens?:int, completion_tokens?:int, total_tokens?:int}`
*   `LLMCompChunk`: `{text_delta?:str, finish_reason?:str, usage_delta?:LLMUsageInfo, raw_chunk:Any}`
*   `LLMChatChunkDeltaMsg`: `{role?:"assistant", content?:str, tool_calls?:List[ToolCall]}`
*   `LLMChatChunk`: `{message_delta?:LLMChatChunkDeltaMsg, finish_reason?:str, usage_delta?:LLMUsageInfo, raw_chunk:Any}`
*   `CmdProcResp`: `{chosen_tool_id?:str, extracted_params?:Dict, llm_thought_process?:str, error?:str, raw_response?:Any}`
*   `RetrievedChunk`: `{content:str, metadata:Dict, id?:str, score:float, rank?:int}`
*   `CodeExecRes`: `(stdout:str, stderr:str, result?:Any, error?:str, exec_time_ms:float)` (NamedTuple)
*   `PromptData`: `Dict[str,Any]`
*   `FormattedPrompt`: `Union[str, List[ChatMessage]]`
*   `PromptIdentifier`: `{name:str, version?:str, description?:str}`
*   `ConversationState`: `{session_id:str, history:List[ChatMessage], metadata?:Dict}`
*   `TraceEvent`: `{event_name:str, data:Dict, timestamp:float, component?:str, correlation_id?:str}`
*   `ApprovalRequest`: `{request_id:str, prompt:str, data_to_approve:Dict, context?:Dict, timeout_seconds?:int}`
*   `ApprovalResponse`: `{request_id:str, status:ApprovalStatus, approver_id?:str, reason?:str, timestamp?:float}`
*   `ApprovalStatus: Literal["pending"|"approved"|"denied"|"timeout"|"error"]`
*   `TokenUsageRecord`: `{provider_id:str, model_name:str, prompt_tokens?:int, completion_tokens?:int, total_tokens?:int, timestamp:float, call_type?:str, user_id?:str, session_id?:str, custom_tags?:dict}`
*   `GuardrailViolation`: `{action:Literal["allow"|"block"|"warn"], reason?:str, guardrail_id?:str, details?:Dict}`
*   `ParsedOutput`: `Any`
*   `AgentOutput`: `{status:Literal["success"|"error"|"max_iterations_reached"|"user_stopped"], output:Any, history?:List[Any], plan?:List[PlannedStep]}`
*   `PlannedStep`: `{step_number:int, tool_id:str, params:Dict, reasoning?:str}`
*   `ReActObservation`: `{thought:str, action:str, observation:str}`
*   `TaskStatus: Literal["pending", "running", "success", "failure", "revoked", "unknown"]`
```

---
Now, I will update `docs/LLM_code_assist_instructions/LLM_refactor_guide.md`:

```markdown
## Genie Tooling Refactoring & Migration Guide for LLMs

**Primary LLM Directive:** Your goal is to analyze an existing Python codebase and systematically refactor its components to utilize the `genie-tooling` middleware. The primary interaction point will be the `genie_tooling.genie.Genie` facade. Prioritize modularity, configurability, and leveraging Genie's built-in plugin ecosystem. **Key Change**: Tools are now explicitly enabled via `MiddlewareConfig.tool_configurations`.

**Core Reference Documents for LLM:**
1.  **`docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This is your **primary technical schema** for Genie Tooling's API, plugin types, configuration structure, and core concepts. Internalize this deeply, noting the change in tool enablement.
2.  **`README.md`**: For high-level vision, core concepts (Genie facade, plugins, `@tool`, explicit tool enablement), and quick start examples.
3.  **`pyproject.toml` (`[tool.poetry.plugins."genie_tooling.plugins"]` section)**: Canonical source for plugin IDs and their Python module paths.
4.  **`src/genie_tooling/config/resolver.py` (`PLUGIN_ID_ALIASES` dict)**: Mapping of user-friendly aliases to canonical plugin IDs.
5.  **`examples/` directory**: Practical usage patterns and configuration examples, now reflecting explicit tool enablement.
6.  **`docs/guides/`**: Detailed human-readable guides for specific features (e.g., RAG, LLM Providers, Configuration, Using Tools).

---

### Phase 0: Deep Ingestion & Semantic Understanding of Genie Tooling

**LLM Instructions:**

1.  **Master `docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This document is specifically structured for your consumption. Treat it as the ground truth for Genie's API, available plugin types, configuration parameters (via `MiddlewareConfig` and `FeatureSettings`), and core data structures (e.g., `ChatMessage`, `ToolCall`, `AgentOutput`). **Pay special attention to the `tool_configurations` field in `MiddlewareConfig` as the method for enabling tools.**
2.  **Internalize Core Concepts (`README.md`)**:
    *   **`Genie` Facade**: Understand its role as the central interaction point.
    *   **Plugin Architecture**: Grasp that most functionalities are swappable plugins.
    *   **Explicit Tool Enablement**: Recognize that tools are only loaded if their ID is a key in `MiddlewareConfig.tool_configurations`.
    *   **Managers**: Note their role in orchestrating plugin types (though direct interaction is rare when using the facade).
    *   **`@tool` Decorator**: Recognize its utility for quickly turning Python functions into tools. These decorated tools also need to be enabled via `tool_configurations` after registration.
    *   **Configuration (`MiddlewareConfig`, `FeatureSettings`, `ConfigResolver`)**: Understand the layered configuration approach.
3.  **Plugin Identification & Aliasing**:
    *   Correlate plugin IDs from `pyproject.toml` with aliases in `PLUGIN_ID_ALIASES` from `resolver.py`.
4.  **Facade & Interface Study**:
    *   Analyze `src/genie_tooling/genie.py` to see how the `Genie` class initializes and exposes its sub-interfaces.
    *   Thoroughly review `src/genie_tooling/interfaces.py`.
5.  **Practical Application Patterns**:
    *   Examine all files in the `examples/` directory. Pay close attention to how `MiddlewareConfig` (especially `tool_configurations`) and `FeatureSettings` are used.

---

### Phase 1: Target Project Analysis & Genie Capability Mapping

**LLM Instructions:**

1.  **Deconstruct Target Project**: (Same as before)
2.  **Create a Refactoring Map**: (Same as before, but when mapping to a Genie Tool, note that it will need to be added to `tool_configurations`).
3.  **Assess API Key Handling**: (Same as before)
4.  **Identify Custom Plugin Candidates**: (Same as before)

---

### Phase 2: Core Genie Integration & Configuration Setup

**LLM Instructions:**

1.  **Add Dependency**: (Same as before)
2.  **Design `MiddlewareConfig`**:
    *   Create a central configuration point.
    *   Instantiate `genie_tooling.config.models.MiddlewareConfig`.
    *   Prioritize using `genie_tooling.config.features.FeatureSettings`.
    *   **Crucially, populate `app_config.tool_configurations` with entries for *every tool* that needs to be active.** For tools without specific settings, an empty dictionary is sufficient (e.g., `app_config.tool_configurations = {"calculator_tool": {}}`).
        *   Example:
            ```python
            from genie_tooling.config.models import MiddlewareConfig
            from genie_tooling.config.features import FeatureSettings

            app_features = FeatureSettings(
                llm="ollama", 
                # ... other features ...
            )
            app_config = MiddlewareConfig(
                features=app_features,
                tool_configurations={
                    "calculator_tool": {}, # Enable calculator
                    "my_existing_api_tool_id": {"api_base_url": "https://service.com/api"}, # Enable & configure
                    "another_simple_tool_id": {} 
                }
            )
            ```
    *   For functionalities not covered by `FeatureSettings` or requiring specific overrides, populate the relevant `*_configurations` dictionaries.
    *   If custom plugins reside in project-specific directories, add these paths to `app_config.plugin_dev_dirs`.
3.  **Implement/Configure `KeyProvider`**: (Same as before)
4.  **Instantiate `Genie` Facade**: (Same as before)
5.  **Integrate Teardown**: (Same as before)

---

### Phase 3: Iterative Refactoring of Mapped Components

**LLM Instructions:** Systematically replace existing functionalities with their Genie Tooling equivalents.

1.  **LLM Interactions (`genie.llm`)**: (Same as before)
2.  **Tool Definition & Execution (`@tool`, `genie.execute_tool`, `genie.run_command`)**:
    *   **Refactor Functions to Tools**: Apply `@genie_tooling.tool`.
    *   **Register Tools**: Call `await genie.register_tool_functions([...])`.
    *   **Enable Registered Tools**: **Ensure the identifiers of these registered tools are added as keys to `MiddlewareConfig.tool_configurations` (e.g., `{"my_decorated_function_name": {}}`).**
    *   **Replace Direct Calls**: Change to `await genie.execute_tool("tool_name_as_string", ...)`.
    *   **Refactor Command Parsing**: Replace with `await genie.run_command(user_query_string)`. Ensure any tools the command processor might select are enabled in `tool_configurations`.
3.  **Command Processing (`genie.run_command`)**:
    *   (Same as before, but reiterate that any tools the processor might select must be enabled in `tool_configurations`).
4.  **RAG Pipeline (`genie.rag`)**: (Same as before)
5.  **Prompt Management (`genie.prompts`)**: (Same as before)
6.  **Conversation State (`genie.conversation`)**: (Same as before)
7.  **Observability (`genie.observability`)**: (Same as before)
8.  **Human-in-the-Loop (`genie.human_in_loop`)**: (Same as before)
9.  **Token Usage Tracking (`genie.usage`)**: (Same as before)
10. **Guardrails**: (Same as before)
11. **LLM Output Parsing (`genie.llm.parse_output`)**: (Same as before)
12. **Agentic Loops (`genie_tooling.agents`)**:
    *   (Same as before, ensuring any tools used by the agent are enabled in `tool_configurations`).
13. **Distributed Task Offloading (`genie.task_queue`)**:
    *   (Same as before. If the remote task involves executing a Genie tool, that tool must be enabled in the *worker's* Genie configuration.)

---

### Phase 4: Testing, Validation, and Refinement

**LLM Instructions:**

1.  **Unit Tests**: (Same as before)
2.  **Integration Tests**:
    *   Ensure `MiddlewareConfig` in tests correctly enables all necessary tools via `tool_configurations`.
3.  **Behavioral Verification**: (Same as before)
4.  **Configuration Robustness**: (Same as before)
5.  **Observability Review**: (Same as before)

---

### Phase 5: Finalization - Documentation & Code Cleanup

**LLM Instructions:**

1.  **Update Project Documentation**:
    *   Clearly document that tools must be enabled via `MiddlewareConfig.tool_configurations`.
2.  **Code Cleanup**: (Same as before)
3.  **Configuration Review**:
    *   Verify that `tool_configurations` correctly lists all active tools.
4.  **Update `LLM_api_guide.md` (If Genie Tooling Itself Was Modified)**:
    *   **CRITICAL LLM SELF-REFLECTION**: (Same as before. Ensure the API guide reflects the explicit tool enablement via `tool_configurations`.)

---
