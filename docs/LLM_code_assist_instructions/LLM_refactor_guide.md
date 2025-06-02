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
    *   `execute_tool(tool_identifier:str, **params:Any) -> Any`
    *   `run_command(command:str, processor_id?:str, conversation_history?:List[ChatMessage]) -> CmdProcResp` (Integrates HITL if configured)
*   `tool_reg`: (Methods on `Genie` instance)
    *   `@genie_tooling.tool` (Decorator for functions. Auto-generates metadata: id, name, desc_human, desc_llm, input_schema, output_schema).
    *   `await genie.register_tool_functions(functions:List[Callable])` (Registers decorated functions. Invalidates tool lookup index).
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
    *   `llm: ollama|openai|gemini|llama_cpp|none` (Def: `none`) -> `def_llm_prov_id`, sets model in `llm_prov_cfgs[llm_id]`.
        *   `llm_ollama_model_name:str?` (Def: `mistral:latest`)
        *   `llm_openai_model_name:str?` (Def: `gpt-3.5-turbo`)
        *   `llm_gemini_model_name:str?` (Def: `gemini-1.5-flash-latest`)
        *   `llm_llama_cpp_model_name:str?` (Def: `mistral:latest`)
        *   `llm_llama_cpp_base_url:str?` (Def: `http://localhost:8080`)
        *   `llm_llama_cpp_api_key_name:str?` (Optional env var name for key, enables KP injection for llama_cpp)
    *   `cache: in-memory|redis|none` (Def: `none`) -> `cache_prov_cfgs`.
        *   `cache_redis_url:str?` (Def: `redis://localhost:6379/0`)
    *   `rag_embedder: sentence_transformer|openai|none` (Def: `none`) -> `def_rag_embed_id`, sets model in `embed_gen_cfgs[embed_id]`.
        *   `rag_embedder_st_model_name:str?` (Def: `all-MiniLM-L6-v2`)
    *   `rag_vector_store: faiss|chroma|qdrant|none` (Def: `none`) -> `def_rag_vs_id`, sets path/coll in `vec_store_cfgs[vs_id]`.
        *   `rag_vector_store_chroma_path:str?` (Path for ChromaDB. Def: None -> plugin default)
        *   `rag_vector_store_chroma_collection_name:str?` (Def: `genie_rag_collection`)
        *   `rag_vector_store_qdrant_url:str?` (URL for Qdrant. Def: None)
        *   `rag_vector_store_qdrant_path:str?` (Path for local Qdrant. Def: None)
        *   `rag_vector_store_qdrant_api_key_name:str?` (API key name for Qdrant, enables KP injection)
        *   `rag_vector_store_qdrant_collection_name:str?` (Def: `genie_qdrant_rag`)
        *   `rag_vector_store_qdrant_embedding_dim:int?` (Required if creating Qdrant collection)
    *   `tool_lookup: embedding|keyword|none` (Def: `none`) -> `def_tool_lookup_prov_id`.
        *   `tool_lookup_formatter_id_alias:str?` (Def: `compact_text_formatter`) -> `def_tool_idx_formatter_id`.
        *   `tool_lookup_embedder_id_alias:str?` (Def: `st_embedder`) -> embedder for `embedding_lookup`.
        *   `tool_lookup_chroma_path:str?` (Path for ChromaDB if `embedding_lookup` uses Chroma. Def: None -> plugin default)
        *   `tool_lookup_chroma_collection_name:str?` (Def: `genie_tool_lookup_embeddings`)
    *   `command_processor: llm_assisted|simple_keyword|none` (Def: `none`) -> `def_cmd_proc_id`.
        *   `command_processor_formatter_id_alias:str?` (Def: `compact_text_formatter`) -> formatter for `llm_assisted`.
    *   `observability_tracer: console_tracer|otel_tracer|none` (Def: `none`) -> `def_obs_tracer_id`.
        *   `observability_otel_endpoint:str?` (OTLP exporter endpoint. Def: None)
    *   `hitl_approver: cli_hitl_approver|none` (Def: `none`) -> `def_hitl_approver_id`.
    *   `token_usage_recorder: in_memory_token_recorder|otel_metrics_recorder|none` (Def: `none`) -> `def_token_usage_rec_id`.
    *   `input_guardrails:List[str_alias_or_id]` (Def: `[]`), `output_guardrails:List[str_alias_or_id]` (Def: `[]`), `tool_usage_guardrails:List[str_alias_or_id]` (Def: `[]`).
    *   `prompt_registry: file_system_prompt_registry|none` (Def: `none`) -> `def_prompt_reg_id`.
    *   `prompt_template_engine: basic_string_formatter|jinja2_chat_formatter|none` (Def: `none`) -> `def_prompt_tmpl_id`.
    *   `conversation_state_provider: in_memory_convo_provider|redis_convo_provider|none` (Def: `none`) -> `def_convo_state_prov_id`.
    *   `default_llm_output_parser: json_output_parser|pydantic_output_parser|none` (Def: `none`) -> `def_llm_out_parser_id`.
    *   `task_queue: celery|rq|none` (Def: `none`) -> `def_dist_task_q_id`.
        *   `task_queue_celery_broker_url:str?` (Def: `redis://localhost:6379/1`)
        *   `task_queue_celery_backend_url:str?` (Def: `redis://localhost:6379/2`)
*   `ConfigResolver` (`genie_tooling.config.resolver.py`): `features` + aliases -> canonical IDs & cfgs. `PLUGIN_ID_ALIASES` dict.
*   `key_provider_id: str?` Def: `env_keys` if no `key_provider_instance`.
*   `key_provider_instance: KeyProvider?` -> Passed to `Genie.create()`.
*   `*_configurations: Dict[str_id_or_alias, Dict[str, Any]]` (e.g., `llm_provider_configurations`).
*   `plugin_dev_dirs: List[str]`.

**Plugins**: `PluginManager`. IDs/paths: `pyproject.toml` -> `[tool.poetry.plugins."genie_tooling.plugins"]`.
**Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES`.

**Key Plugins (ID | Alias | Cfg/Notes)**:
*   `KeyProv`: `environment_key_provider_v1`|`env_keys`.
*   `LLMProv`:
    *   `ollama_llm_provider_v1`|`ollama`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`.
    *   `openai_llm_provider_v1`|`openai`. Cfg: `model_name`, `api_key_name`, `openai_api_base`, `openai_organization`. Needs KP.
    *   `gemini_llm_provider_v1`|`gemini`. Cfg: `model_name`, `api_key_name`, `system_instruction`, `safety_settings`. Needs KP.
    *   `llama_cpp_llm_provider_v1`|`llama_cpp`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`, `api_key_name`. Needs KP if `api_key_name` set.
*   `CmdProc`:
    *   `simple_keyword_processor_v1`|`simple_keyword_cmd_proc`. Cfg: `keyword_map`, `keyword_priority`.
    *   `llm_assisted_tool_selection_processor_v1`|`llm_assisted_cmd_proc`. Cfg: `llm_provider_id`, `tool_formatter_id`, `tool_lookup_top_k`, `system_prompt_template`, `max_llm_retries`.
*   `Tools`: `calculator_tool`, `sandboxed_fs_tool_v1` (Cfg: `sandbox_base_path`), `google_search_tool_v1` (Needs KP: `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`), `open_weather_map_tool` (Needs KP: `OPENWEATHERMAP_API_KEY`), `generic_code_execution_tool`.
*   `DefFormatters`: `compact_text_formatter_plugin_v1`|`compact_text_formatter`, `openai_function_formatter_plugin_v1`|`openai_func_formatter`, `human_readable_json_formatter_plugin_v1`|`hr_json_formatter`.
*   `RAG`:
    *   Loaders: `file_system_loader_v1` (Cfg: `glob_pattern`, `encoding`, `max_file_size_mb`), `web_page_loader_v1` (Cfg: `timeout_seconds`, `headers`, `use_trafilatura`, `trafilatura_include_comments`, `trafilatura_include_tables`, `trafilatura_no_fallback`).
    *   Splitters: `character_recursive_text_splitter_v1` (Cfg: `chunk_size`, `chunk_overlap`, `separators`).
    *   Embedders: `sentence_transformer_embedder_v1`|`st_embedder` (Cfg: `model_name`, `device`), `openai_embedding_generator_v1`|`openai_embedder` (Cfg: `model_name`, `api_key_name`, `max_retries`, `initial_retry_delay`, `dimensions`). Needs KP.
    *   VS: `faiss_vector_store_v1`|`faiss_vs` (Cfg: `embedding_dim`, `index_file_path`, `doc_store_file_path`, `faiss_index_factory_string`, `persist_by_default`, `collection_name`), `chromadb_vector_store_v1`|`chroma_vs` (Cfg: `collection_name`, `path`, `host`, `port`, `use_hnsw_indexing`, `hnsw_space`), `qdrant_vector_store_v1`|`qdrant_vs` (Cfg: `collection_name`, `embedding_dim`, `distance_metric`, `url`, `host`, `port`, `path`, `api_key_name`, `prefer_grpc`, `timeout_seconds`). Needs KP if `api_key_name` set for Qdrant.
    *   Retrievers: `basic_similarity_retriever_v1`. Cfg: `embedder_id`, `embedder_config`, `vector_store_id`, `vector_store_config`.
*   `ToolLookupProv`:
    *   `embedding_similarity_lookup_v1`|`embedding_lookup`. Cfg: `embedder_id`, `embedder_config`, `vector_store_id`, `vector_store_config`, `tool_embeddings_collection_name`, `tool_embeddings_path`.
    *   `keyword_match_lookup_v1`|`keyword_lookup`.
*   `CodeExec`: `secure_docker_executor_v1` (Cfg: `python_docker_image`, `node_docker_image`, `bash_docker_image`, `pull_images_on_setup`, `default_network_mode`, `default_mem_limit`, `default_cpu_shares`, `default_pids_limit`), `pysandbox_executor_stub_v1`.
*   `CacheProv`: `in_memory_cache_provider_v1`|`in_memory_cache` (Cfg: `max_size`, `default_ttl_seconds`, `cleanup_interval_seconds`), `redis_cache_provider_v1`|`redis_cache` (Cfg: `redis_url`, `default_ttl_seconds`, `json_serialization`).
*   `Observability`:
    *   `console_tracer_plugin_v1`|`console_tracer`. Cfg: `log_level`.
    *   `otel_tracer_plugin_v1`|`otel_tracer`. Cfg: `otel_service_name`, `otel_service_version`, `exporter_type` (`console`|`otlp_http`|`otlp_grpc`), `otlp_http_endpoint`, `otlp_http_headers`, `otlp_http_timeout`, `otlp_grpc_endpoint`, `otlp_grpc_insecure`, `otlp_grpc_timeout`, `resource_attributes`.
*   `HITL`: `cli_approval_plugin_v1`|`cli_hitl_approver`.
*   `TokenUsage`:
    *   `in_memory_token_usage_recorder_v1`|`in_memory_token_recorder`.
    *   `otel_metrics_token_recorder_v1`|`otel_metrics_recorder`. (Emits OTel metrics, needs OTel SDK setup e.g. via `otel_tracer`).
*   `Guardrails`: `keyword_blocklist_guardrail_v1`|`keyword_blocklist_guardrail`. Cfg: `blocklist`, `case_sensitive`, `action_on_match`.
*   `Prompts`:
    *   Registry: `file_system_prompt_registry_v1`|`file_system_prompt_registry`. Cfg: `base_path`, `template_suffix`.
    *   Template: `basic_string_format_template_v1`|`basic_string_formatter`, `jinja2_chat_template_v1`|`jinja2_chat_formatter`.
*   `Conversation`:
    *   `in_memory_conversation_state_v1`|`in_memory_convo_provider`.
    *   `redis_conversation_state_v1`|`redis_convo_provider`. Cfg: `redis_url`, `key_prefix`, `default_ttl_seconds`.
*   `LLMOutputParsers`:
    *   `json_output_parser_v1`|`json_output_parser`. Cfg: `strict_parsing`.
    *   `pydantic_output_parser_v1`|`pydantic_output_parser`.
*   `TaskQueues`:
    *   `celery_task_queue_v1`|`celery_task_queue`. Cfg: `celery_app_name`, `celery_broker_url`, `celery_backend_url`, `celery_include_task_paths`.
    *   `redis_queue_task_plugin_v1`|`rq_task_queue`. (STUB)
*   `InvocationStrategies`:
    *   `default_async_invocation_strategy_v1`|`default_invocation_strategy`.
    *   `distributed_task_invocation_strategy_v1`|`distributed_task_strategy`. Cfg: `task_queue_plugin_id`.

**Types**:
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

**Primary LLM Directive:** Your goal is to analyze an existing Python codebase and systematically refactor its components to utilize the `genie-tooling` middleware. The primary interaction point will be the `genie_tooling.genie.Genie` facade. Prioritize modularity, configurability, and leveraging Genie's built-in plugin ecosystem.

**Core Reference Documents for LLM:**
1.  **`docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This is your **primary technical schema** for Genie Tooling's API, plugin types, configuration structure, and core concepts. Internalize this deeply.
2.  **`README.md`**: For high-level vision, core concepts (Genie facade, plugins, `@tool`), and quick start examples.
3.  **`pyproject.toml` (`[tool.poetry.plugins."genie_tooling.plugins"]` section)**: Canonical source for plugin IDs and their Python module paths.
4.  **`src/genie_tooling/config/resolver.py` (`PLUGIN_ID_ALIASES` dict)**: Mapping of user-friendly aliases to canonical plugin IDs.
5.  **`examples/` directory**: Practical usage patterns and configuration examples.
6.  **`docs/guides/`**: Detailed human-readable guides for specific features (e.g., RAG, LLM Providers, Configuration).

---

### Phase 0: Deep Ingestion & Semantic Understanding of Genie Tooling

**LLM Instructions:**

1.  **Master `docs/LLM_code_assist_instructions/LLM_api_guide.md`**: This document is specifically structured for your consumption. Treat it as the ground truth for Genie's API, available plugin types, configuration parameters (via `MiddlewareConfig` and `FeatureSettings`), and core data structures (e.g., `ChatMessage`, `ToolCall`, `AgentOutput`).
2.  **Internalize Core Concepts (`README.md`)**:
    *   **`Genie` Facade**: Understand its role as the central interaction point.
    *   **Plugin Architecture**: Grasp that most functionalities are swappable plugins.
    *   **Managers**: Note their role in orchestrating plugin types (though direct interaction is rare when using the facade).
    *   **`@tool` Decorator**: Recognize its utility for quickly turning Python functions into tools.
    *   **Configuration (`MiddlewareConfig`, `FeatureSettings`, `ConfigResolver`)**: Understand the layered configuration approach, from high-level features to specific plugin settings.
3.  **Plugin Identification & Aliasing**:
    *   Correlate plugin IDs from `pyproject.toml` with aliases in `PLUGIN_ID_ALIASES` from `resolver.py`. This is vital for interpreting `FeatureSettings` and user configurations.
4.  **Facade & Interface Study**:
    *   Analyze `src/genie_tooling/genie.py` to see how the `Genie` class initializes and exposes its sub-interfaces.
    *   Thoroughly review `src/genie_tooling/interfaces.py` to understand the methods available under `genie.llm`, `genie.rag`, `genie.prompts`, `genie.task_queue`, etc.
5.  **Practical Application Patterns**:
    *   Examine all files in the `examples/` directory. Pay close attention to how `MiddlewareConfig` and `FeatureSettings` are used to initialize `Genie` and how various facade methods are invoked.
    *   Note how different plugin types are configured and utilized in these examples.

---

### Phase 1: Target Project Analysis & Genie Capability Mapping

**LLM Instructions:**

1.  **Deconstruct Target Project**: Analyze the existing codebase to identify components and patterns that can be mapped to Genie Tooling features. Create a mental (or actual, if helpful for your process) inventory:
    *   **LLM Interactions**: Identify direct calls to `openai`, `google.generativeai`, `httpx` (for Ollama/Llama.cpp), etc. Note models used, parameters (temp, max_tokens), and error handling.
    *   **Tool/Function Logic**: Pinpoint functions or classes that perform discrete, reusable actions (e.g., calculations, API calls, file operations).
    *   **Command Interpretation**: Find any logic that parses natural language into function calls or tool selections.
    *   **RAG Components**: Locate code for document loading, text splitting, embedding generation, vector storage, and similarity search.
    *   **Distributed Tasks**: Identify usage of Celery, RQ, or other task queue systems for background processing.
    *   **Configuration**: Identify how API keys, model names, URLs, and other settings are managed.
    *   **Prompt Management**: Find hardcoded prompts, f-string templates, or custom templating systems.
    *   **Conversation History**: Analyze how chat history is stored and passed to LLMs.
    *   **Agentic Loops**: If present, identify patterns like ReAct or Plan-and-Execute.
    *   **Safety/Validation**: Note any input/output validation or content filtering.
    *   **Observability**: Current logging or tracing for LLM/tool interactions.
2.  **Create a Refactoring Map**: For each identified component in the target project, map it to a corresponding Genie Tooling feature or plugin type. Refer heavily to `docs/LLM_code_assist_instructions/LLM_api_guide.md` and `README.md` for this.
    *   *Example Mapping:*
        *   `requests.get("some_api")` -> Potential `@tool` function, or a custom `ToolPlugin`.
        *   Manual OpenAI API calls -> `genie.llm.chat(provider_id="openai", ...)`
        *   Direct Llama.cpp server calls -> `genie.llm.generate(provider_id="llama_cpp", ...)`
        *   Custom vector search logic (e.g., with Qdrant) -> `genie.rag.search()` with `features.rag_vector_store="qdrant"`.
        *   Celery `send_task` calls -> `genie.task_queue.submit_task(...)`.
        *   Environment variables for API keys -> `EnvironmentKeyProvider` (default) or custom `KeyProvider`.
        *   Hardcoded system prompts -> `FileSystemPromptRegistryPlugin` + `genie.prompts.render_chat_prompt()`.
        *   Existing ReAct loop -> `genie_tooling.agents.ReActAgent`.
3.  **Assess API Key Handling**: Determine the current strategy for API key management. Plan to integrate with Genie's `KeyProvider` system. If keys are in environment variables, the default `EnvironmentKeyProvider` might suffice. Otherwise, a custom `KeyProvider` implementation will be necessary.
4.  **Identify Custom Plugin Candidates**: Beyond simple `@tool` functions, determine if any complex components from the target project would be better implemented as full Genie plugins (e.g., a custom `VectorStorePlugin` for an unsupported database, a specialized `CommandProcessorPlugin`).

---

### Phase 2: Core Genie Integration & Configuration Setup

**LLM Instructions:**

1.  **Add Dependency**: Ensure `genie-tooling` is added to the target project's dependencies (e.g., in `pyproject.toml` or `requirements.txt`).
2.  **Design `MiddlewareConfig`**:
    *   Create a central configuration point for Genie, typically where the application initializes.
    *   Instantiate `genie_tooling.config.models.MiddlewareConfig`.
    *   Prioritize using `genie_tooling.config.features.FeatureSettings` for high-level configuration based on the analysis in Phase 1.
        *   Example:
            ```python
            from genie_tooling.config.models import MiddlewareConfig
            from genie_tooling.config.features import FeatureSettings

            app_features = FeatureSettings(
                llm="ollama", # or "openai", "gemini", "llama_cpp"
                llm_ollama_model_name="identified_ollama_model",
                # llm_llama_cpp_model_name="identified_llama_model",
                # llm_llama_cpp_base_url="http://localhost:8080",
                # llm_llama_cpp_api_key_name="MY_LLAMA_CPP_KEY_ENV_VAR", # Optional

                rag_embedder="sentence_transformer", # or "openai"
                rag_embedder_st_model_name="identified_st_model",

                rag_vector_store="faiss", # or "chroma", "qdrant"
                # rag_vector_store_qdrant_url="http://localhost:6333", # If using Qdrant
                # rag_vector_store_qdrant_collection_name="my_project_rag",
                # rag_vector_store_qdrant_embedding_dim=768, # Match your embedder

                command_processor="llm_assisted",
                tool_lookup="embedding",

                observability_tracer="otel_tracer", # or "console_tracer"
                observability_otel_endpoint="http://localhost:4318/v1/traces", # If using otel_tracer

                token_usage_recorder="in_memory_token_recorder", # or "otel_metrics_recorder"

                task_queue="celery", # or "rq", "none"
                task_queue_celery_broker_url="redis://localhost:6379/1",
                task_queue_celery_backend_url="redis://localhost:6379/2",
                # ... other features based on Phase 1 analysis ...
            )
            app_config = MiddlewareConfig(features=app_features)
            ```
    *   For functionalities not covered by `FeatureSettings` or requiring specific overrides, populate the relevant `*_configurations` dictionaries in `MiddlewareConfig` (e.g., `tool_configurations`, `command_processor_configurations`). Use canonical plugin IDs or recognized aliases (refer to `PLUGIN_ID_ALIASES`).
    *   If custom plugins (developed in Phase 3) will reside in project-specific directories, add these paths to `app_config.plugin_dev_dirs`.
3.  **Implement/Configure `KeyProvider`**:
    *   If using environment variables that match Genie's defaults (e.g., `OPENAI_API_KEY`), no explicit `KeyProvider` instance needs to be passed to `Genie.create()`.
    *   If custom key names or sources are used, implement a class inheriting from `genie_tooling.security.key_provider.KeyProvider` and plan to pass an instance to `Genie.create()`.
4.  **Instantiate `Genie` Facade**:
    *   At an appropriate point in the application's startup sequence (e.g., in `main()` or an initialization function), create the `Genie` instance:
        ```python
        # from genie_tooling.genie import Genie
        # from my_project_key_provider import MyCustomKeyProvider # If applicable

        # key_provider_instance = MyCustomKeyProvider() if using_custom_kp else None
        # global_genie_instance = await Genie.create(
        #     config=app_config,
        #     key_provider_instance=key_provider_instance
        # )
        ```
    *   Plan for how this `genie_instance` will be accessed by other parts of the refactored application (e.g., passed via dependency injection, global singleton if appropriate for the project structure).
5.  **Integrate Teardown**: Ensure `await global_genie_instance.close()` is called when the application shuts down to release resources held by plugins.

---

### Phase 3: Iterative Refactoring of Mapped Components

**LLM Instructions:** Systematically replace existing functionalities with their Genie Tooling equivalents, using the `genie_instance` created in Phase 2. Refer to `docs/LLM_code_assist_instructions/LLM_api_guide.md` for precise method signatures and `examples/` for usage patterns.

1.  **LLM Interactions (`genie.llm`)**:
    *   Replace direct SDK calls (e.g., `openai.ChatCompletion.create(...)`) with `await genie.llm.chat(...)` or `await genie.llm.generate(...)`.
    *   Pass `provider_id` (e.g., `"openai"`, `"ollama"`, `"gemini"`, `"llama_cpp"`) if needing to switch between multiple configured LLMs.
    *   Migrate LLM parameters (temperature, max_tokens) to the `**kwargs` of Genie's methods or configure them as defaults in `MiddlewareConfig.llm_provider_configurations`.

2.  **Tool Definition & Execution (`@tool`, `genie.execute_tool`, `genie.run_command`)**:
    *   **Refactor Functions to Tools**: Apply `@genie_tooling.tool` to identified Python functions. Ensure type hints are accurate and docstrings are descriptive (especially `Args:` section for parameter descriptions).
    *   **Register Tools**: After `Genie` instantiation, call `await genie.register_tool_functions([list_of_decorated_functions])`.
    *   **Replace Direct Calls**: Change existing direct function calls (that are now tools) to `await genie.execute_tool("tool_name_as_string", arg1=val1, ...)`.
    *   **Refactor Command Parsing**: If the old code parsed natural language to call tools, replace this logic with `await genie.run_command(user_query_string)`. This requires configuring a `CommandProcessorPlugin` (see next point).

3.  **Command Processing (`genie.run_command`)**:
    *   Configure `features.command_processor` (e.g., `"llm_assisted"`).
    *   If using `"llm_assisted"`:
        *   Ensure `features.llm` is set.
        *   Configure `features.tool_lookup` (e.g., `"embedding"`) and its associated `tool_lookup_formatter_id_alias` and `tool_lookup_embedder_id_alias`.
        *   Set `command_processor_configurations` for `llm_assisted_tool_selection_processor_v1` if non-default `tool_lookup_top_k` or system prompt is needed.
    *   If using `"simple_keyword"`, configure `command_processor_configurations["simple_keyword_processor_v1"]["keyword_map"]`.

4.  **RAG Pipeline (`genie.rag`)**:
    *   **Indexing**: Replace custom document loading, splitting, embedding, and vector store ingestion with `await genie.rag.index_directory(...)` or `await genie.rag.index_web_page(...)`.
    *   **Search**: Replace custom similarity search logic with `await genie.rag.search(...)`.
    *   **Configuration**:
        *   Set `features.rag_embedder` and `features.rag_vector_store` (e.g., `"faiss"`, `"chroma"`, `"qdrant"`).
        *   Provide necessary paths, collection names, API keys (for cloud vector stores via `KeyProvider`), or embedding dimensions via `features` (e.g., `features.rag_vector_store_qdrant_url`, `features.rag_vector_store_qdrant_embedding_dim`) or `MiddlewareConfig`'s `embedding_generator_configurations` and `vector_store_configurations`.
        *   If custom RAG components are needed, implement `DocumentLoaderPlugin`, `TextSplitterPlugin`, `EmbeddingGeneratorPlugin`, or `VectorStorePlugin` and register/configure them.

5.  **Prompt Management (`genie.prompts`)**:
    *   Move hardcoded prompts or templates to external files (e.g., `.txt`, `.j2`).
    *   Configure `features.prompt_registry` (e.g., `"file_system_prompt_registry"`) and set `prompt_registry_configurations` in `MiddlewareConfig` (e.g., `base_path`).
    *   Configure `features.prompt_template_engine` (e.g., `"basic_string_formatter"`, `"jinja2_chat_formatter"`).
    *   Replace old templating logic with `await genie.prompts.render_prompt(...)` or `await genie.prompts.render_chat_prompt(...)`.

6.  **Conversation State (`genie.conversation`)**:
    *   Replace custom chat history management with `await genie.conversation.load_state(...)`, `await genie.conversation.add_message(...)`, etc.
    *   Configure `features.conversation_state_provider` (e.g., `"in_memory_convo_provider"`, `"redis_convo_provider"`) and its settings in `conversation_state_provider_configurations`.

7.  **Observability (`genie.observability`)**:
    *   Configure `features.observability_tracer` (e.g., `"console_tracer"`, `"otel_tracer"`). If using `"otel_tracer"`, also configure `features.observability_otel_endpoint`.
    *   Rely on automatic tracing from Genie facade methods.
    *   Insert `await genie.observability.trace_event(...)` for custom application-specific events.

8.  **Human-in-the-Loop (`genie.human_in_loop`)**:
    *   If `genie.run_command()` is used, HITL for tool execution is automatic if `features.hitl_approver` is configured (e.g., to `"cli_hitl_approver"`).
    *   For other custom approval points, replace existing logic with `await genie.human_in_loop.request_approval(...)`.

9.  **Token Usage Tracking (`genie.usage`)**:
    *   Configure `features.token_usage_recorder` (e.g., `"in_memory_token_recorder"`, `"otel_metrics_recorder"`).
    *   Remove custom token counting logic; rely on automatic recording by `genie.llm` calls.
    *   Use `await genie.usage.get_summary()` for reporting (primarily for in-memory recorder; OTel metrics are viewed in an OTel backend).

10. **Guardrails**:
    *   Configure `features.input_guardrails`, `output_guardrails`, `tool_usage_guardrails` with aliases/IDs of guardrail plugins (e.g., `keyword_blocklist_guardrail`).
    *   Set specific configurations for these guardrails in `MiddlewareConfig.guardrail_configurations`.
    *   Replace existing validation/filtering logic with Genie's guardrail system where appropriate.

11. **LLM Output Parsing (`genie.llm.parse_output`)**:
    *   Replace custom JSON/structured data extraction from LLM string responses with `await genie.llm.parse_output(llm_response, schema=MyPydanticModelOrJsonSchema)`.
    *   Configure `features.default_llm_output_parser` or specify `parser_id` in the call.

12. **Agentic Loops (`genie_tooling.agents`)**:
    *   If the target project has ReAct or Plan-and-Execute style agents, refactor them to use `genie_tooling.agents.ReActAgent` or `genie_tooling.agents.PlanAndExecuteAgent`.
    *   Pass the `genie_instance` to the agent's constructor.
    *   Configure agent-specific parameters (like system prompt IDs, max iterations) via the `agent_config` dictionary passed to the agent constructor. Ensure these prompts are managed by `genie.prompts`.

13. **Distributed Task Offloading (`genie.task_queue`)**:
    *   **Identify**: Existing Celery, RQ, or other task queue client code used for offloading computations or long-running operations.
    *   **Refactor**:
        *   Replace direct task submission calls (e.g., `celery_app.send_task(...)`, `rq_queue.enqueue(...)`) with `await genie.task_queue.submit_task(task_name="your_worker_task_name", args=..., kwargs=...)`.
        *   Replace status checking logic with `await genie.task_queue.get_task_status(task_id)`.
        *   Replace result fetching logic with `await genie.task_queue.get_task_result(task_id)`.
        *   Replace task revocation logic with `await genie.task_queue.revoke_task(task_id)`.
    *   **Configuration**:
        *   Set `features.task_queue` in `FeatureSettings` (e.g., `"celery"` or `"rq"`).
        *   If using Celery, configure `features.task_queue_celery_broker_url` and `features.task_queue_celery_backend_url`.
        *   (Note: Worker-side task definitions (e.g., Celery tasks) still need to be defined and accessible to your workers. Genie's `DistributedTaskInvocationStrategy` can be used for a more integrated way to offload tool executions specifically, but direct `genie.task_queue` usage is for general task offloading.)

---

### Phase 4: Testing, Validation, and Refinement

**LLM Instructions:**

1.  **Unit Tests**:
    *   For any new custom plugins (`Tool`, `KeyProvider`, `RAGPlugin`, `DistributedTaskQueuePlugin`, etc.), write comprehensive unit tests.
    *   For application logic now using `Genie`, mock the relevant `genie.facade.method()` calls to test the logic in isolation.
2.  **Integration Tests**:
    *   Set up integration tests that initialize a `Genie` instance with a minimal but functional configuration (e.g., using Ollama, in-memory RAG components, console tracer, in-memory task queue if testing that flow without external broker).
    *   Test key end-to-end flows:
        *   `genie.run_command()` -> tool selection -> tool execution -> result.
        *   `genie.rag.index_directory()` followed by `genie.rag.search()`.
        *   `genie.llm.chat()` and `genie.llm.generate()` with different providers if configured.
        *   Agent runs (if `ReActAgent` or `PlanAndExecuteAgent` are used).
        *   Distributed task submission, status checking, and result retrieval if `genie.task_queue` is used.
3.  **Behavioral Verification**: Compare the behavior and outputs of the refactored system against the original system for critical user scenarios. Identify and address any regressions or unintended changes.
4.  **Configuration Robustness**: Test the application with varied `MiddlewareConfig` settings (e.g., switching LLM providers via `features`, enabling/disabling tool lookup, changing task queue providers) to ensure the refactored system adapts correctly.
5.  **Observability Review**: If tracing is enabled, review the trace outputs to ensure they provide clear and useful insights into the application's operation with Genie.

---

### Phase 5: Finalization - Documentation & Code Cleanup

**LLM Instructions:**

1.  **Update Project Documentation**:
    *   Modify `README.md` and any other developer documentation to reflect the use of Genie Tooling.
    *   Explain the new configuration process using `MiddlewareConfig` and `FeatureSettings`.
    *   Document how to add new tools (using `@tool` or custom plugins).
    *   Describe how to switch between different LLMs, RAG backends, task queues, etc., via configuration.
2.  **Code Cleanup**:
    *   Remove all old code modules, classes, and utility functions that have been made redundant by Genie Tooling.
    *   Ensure all new code (custom plugins, application logic using Genie) adheres to the project's coding standards and style guides (e.g., run `black .` and `ruff check . --fix` if used).
3.  **Configuration Review**: Perform a final review of the primary `MiddlewareConfig` setup in the application. Ensure it's clear, well-commented, and leverages `FeatureSettings` effectively for defaults, with explicit overrides only where necessary for custom behavior.
4.  **Update `LLM_api_guide.md` (If Genie Tooling Itself Was Modified)**:
    *   **CRITICAL LLM SELF-REFLECTION**: If, during the refactoring process, *you* (the LLM) identified a need to modify or extend the Genie Tooling library itself (e.g., adding a new core feature, changing a plugin interface), you *must* update `docs/LLM_code_assist_instructions/LLM_api_guide.md` to accurately reflect these changes. This ensures future LLM interactions with Genie Tooling remain consistent.

---
