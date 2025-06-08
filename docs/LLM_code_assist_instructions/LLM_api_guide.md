###docs/LLM_code_assist_instructions/LLM_api_guide.md###
# LLM_GUIDE_METADATA_V0.5

# THIS FILE IS A HYPER-COMPRESSED SEMANTIC REPRESENTATION OF GENIE TOOLING FOR LLM CONSUMPTION.
# LLM: IF YOU UPDATE THE GENIE TOOLING PROJECT CODE (CLASSES, METHODS, PARAMS, CORE LOGIC),
# YOU MUST UPDATE THIS FILE TO REFLECT THOSE CHANGES ACCURATELY.
# MAINTAIN THE COMPRESSED FORMAT. HUMAN READABILITY IS NOT A PRIORITY.

**Project**: Genie Tooling. Python Agentic/LLM Middleware. Async. MIT.

**Facade**: `genie_tooling.genie.Genie`
*   `init`: `Genie.create(cfg:MiddlewareConfig, kp_inst?:KeyProvider)`
*   `llm`: `LLMInterface`
    *   `chat(msgs:List[ChatMessage], prov_id?:str, stream?:bool, **kw) -> LLMChatResp|AsyncIter[LLMChatChunk]` (kwargs can include `output_schema` for GBNF with Llama.cpp)
    *   `generate(prompt:str, prov_id?:str, stream?:bool, **kw) -> LLMCompResp|AsyncIter[LLMCompChunk]` (kwargs can include `output_schema` for GBNF with Llama.cpp)
    *   `parse_output(resp:LLMChatResp|LLMCompResp, parser_id?:str, schema?:Any) -> ParsedOutput`
*   `rag`: `RAGInterface`
    *   `index_directory(path:str, collection_name?:str, loader_id?:str, splitter_id?:str, embedder_id?:str, vector_store_id?:str, loader_config?:Dict, splitter_config?:Dict, embedder_config?:Dict, vector_store_config?:Dict, **kw) -> Dict[str,Any]`
    *   `index_web_page(url:str, collection_name?:str, loader_id?:str, splitter_id?:str, embedder_id?:str, vector_store_id?:str, loader_config?:Dict, splitter_config?:Dict, embedder_config?:Dict, vector_store_config?:Dict, **kw) -> Dict[str,Any]`
    *   `search(query:str, collection_name?:str, top_k?:int, retriever_id?:str, retriever_config?:Dict, **kw) -> List[RetrievedChunk]`
*   `tools`: (Methods on `Genie` instance)
    *   `execute_tool(tool_identifier:str, **params:Any) -> Any` (Tool must be enabled)
    *   `run_command(command:str, processor_id?:str, conversation_history?:List[ChatMessage]) -> CmdProcResp` (Integrates HITL; tools used must be enabled)
*   `tool_reg`: (Methods on `Genie` instance)
    *   `@genie_tooling.tool` (Decorator for functions. Auto-generates metadata).
    *   `await genie.register_tool_functions(functions:List[Callable])` (Registers decorated functions. Enablement depends on `auto_enable_registered_tools` flag).
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
*   `auto_enable_registered_tools: bool` (Default: `True`. **IMPORTANT**: Set to `False` for production security.)
*   `features: FeatureSettings` -> `ConfigResolver` (`CfgResolver`) populates `MWCfg`.
    *   `llm: Literal["ollama", "openai", "gemini", "llama_cpp", "llama_cpp_internal", "none"]`
    *   `llm_ollama_model_name: str?`
    *   `llm_openai_model_name: str?`
    *   `llm_gemini_model_name: str?`
    *   `llm_llama_cpp_model_name: str?`
    *   `llm_llama_cpp_base_url: str?`
    *   `llm_llama_cpp_api_key_name: str?`
    *   `llm_llama_cpp_internal_model_path: str?`
    *   `llm_llama_cpp_internal_n_gpu_layers: int` (Def: 0)
    *   `llm_llama_cpp_internal_n_ctx: int` (Def: 2048)
    *   `llm_llama_cpp_internal_chat_format: str?`
    *   `llm_llama_cpp_internal_model_name_for_logging: str?`
    *   `cache: Literal["in-memory", "redis", "none"]`
    *   `cache_redis_url: str?`
    *   `rag_embedder: Literal["sentence_transformer", "openai", "none"]`
    *   `rag_embedder_st_model_name: str?`
    *   `rag_vector_store: Literal["faiss", "chroma", "qdrant", "none"]`
    *   `rag_vector_store_chroma_path: str?`
    *   `rag_vector_store_chroma_collection_name: str?`
    *   `rag_vector_store_qdrant_url: str?`
    *   `rag_vector_store_qdrant_path: str?`
    *   `rag_vector_store_qdrant_api_key_name: str?`
    *   `rag_vector_store_qdrant_collection_name: str?`
    *   `rag_vector_store_qdrant_embedding_dim: int?`
    *   `tool_lookup: Literal["embedding", "keyword", "hybrid", "none"]`
    *   `tool_lookup_formatter_id_alias: str?`
    *   `tool_lookup_chroma_path: str?`
    *   `tool_lookup_chroma_collection_name: str?`
    *   `tool_lookup_embedder_id_alias: str?`
    *   `command_processor: Literal["llm_assisted", "simple_keyword", "none"]`
    *   `command_processor_formatter_id_alias: str?`
    *   `logging_adapter: Literal["default_log_adapter", "pyvider_log_adapter", "none"]`
    *   `logging_pyvider_service_name: str?`
    *   `observability_tracer: Literal["console_tracer", "otel_tracer", "none"]`
    *   `observability_otel_endpoint: str?`
    *   `hitl_approver: Literal["cli_hitl_approver", "none"]`
    *   `token_usage_recorder: Literal["in_memory_token_recorder", "otel_metrics_recorder", "none"]`
    *   `input_guardrails: List[str]`
    *   `output_guardrails: List[str]`
    *   `tool_usage_guardrails: List[str]`
    *   `prompt_registry: Literal["file_system_prompt_registry", "none"]`
    *   `prompt_template_engine: Literal["basic_string_formatter", "jinja2_chat_formatter", "none"]`
    *   `conversation_state_provider: Literal["in_memory_convo_provider", "redis_convo_provider", "none"]`
    *   `default_llm_output_parser: Literal["json_output_parser", "pydantic_output_parser", "none"]`
    *   `task_queue: Literal["celery", "rq", "none"]`
    *   `task_queue_celery_broker_url: str?`
    *   `task_queue_celery_backend_url: str?`
*   `tool_configurations: Dict[str_id_or_alias, Dict[str, Any]]` (Provides tool-specific config. If `auto_enable_registered_tools=False`, this dict also serves as the explicit enablement list.)
*   `ConfigResolver` (`genie_tooling.config.resolver.py`): `features` + aliases -> canonical IDs & cfgs. `PLUGIN_ID_ALIASES` dict.
*   `key_provider_id: str?` Def: `env_keys` if no `key_provider_instance`.
*   `key_provider_instance: KeyProvider?` -> Passed to `Genie.create()`.
*   `*_configurations: Dict[str_id_or_alias, Dict[str, Any]]` (e.g., `llm_provider_configurations`, `log_adapter_configurations`).
*   `plugin_dev_dirs: List[str]`.
*   `default_log_adapter_id: str?`

**Plugins**: `PluginManager`. IDs/paths: `pyproject.toml` -> `[tool.poetry.plugins."genie_tooling.plugins"]`.
**Aliases**: `genie_tooling.config.resolver.PLUGIN_ID_ALIASES`.

**Key Plugins (ID | Alias | Cfg/Notes)**:
*   `KeyProv`: `environment_key_provider_v1`|`env_keys`.
*   `LLMProv`:
    *   `ollama_llm_provider_v1`|`ollama`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`.
    *   `openai_llm_provider_v1`|`openai`. Cfg: `model_name`, `api_key_name`, `openai_api_base`, `openai_organization`. Needs KP.
    *   `gemini_llm_provider_v1`|`gemini`. Cfg: `model_name`, `api_key_name`, `system_instruction`, `safety_settings`. Needs KP.
    *   `llama_cpp_llm_provider_v1`|`llama_cpp`. Cfg: `base_url`, `model_name`, `request_timeout_seconds`, `api_key_name`. Needs KP if `api_key_name` set.
    *   `llama_cpp_internal_llm_provider_v1`|`llama_cpp_internal`. Cfg: `model_path`, `n_gpu_layers`, `n_ctx`, `chat_format`, `model_name_for_logging`, etc. Does not use KP for keys.
*   `CmdProc`:
    *   `simple_keyword_processor_v1`|`simple_keyword_cmd_proc`. Cfg: `keyword_map`, `keyword_priority`.
    *   `llm_assisted_tool_selection_processor_v1`|`llm_assisted_cmd_proc`. Cfg: `llm_provider_id`, `tool_formatter_id`, `tool_lookup_top_k`, `system_prompt_template`, `max_llm_retries`.
*   `Tools`: (Examples: `calculator_tool`, `sandboxed_fs_tool_v1`, etc. **Enablement controlled by `auto_enable_registered_tools` flag and `tool_configurations` dict.**)
*   `LogAdapter`:
    *   `default_log_adapter_v1`|`default_log_adapter`. Cfg: `log_level`, `library_logger_name`, `redactor_plugin_id`, `redactor_config`, `enable_schema_redaction`, `enable_key_name_redaction`.
    *   `pyvider_telemetry_log_adapter_v1`|`pyvider_log_adapter`. Cfg: `service_name`, `default_level`, `module_levels`, `console_formatter`, emoji settings, `redactor_plugin_id`, etc.
*   `Observability`:
    *   `console_tracer_plugin_v1`|`console_tracer`. Cfg: `log_adapter_instance_for_console_tracer` (or similar for ID/PM), `log_level` (for its own direct logs if LogAdapter fails).
    *   `otel_tracer_plugin_v1`|`otel_tracer`. Cfg: `otel_service_name`, `exporter_type` (console, otlp_http, otlp_grpc), endpoints, headers, etc.
*   `TokenUsage`:
    *   `in_memory_token_usage_recorder_v1`|`in_memory_token_recorder`.
    *   `otel_metrics_token_recorder_v1`|`otel_metrics_recorder`. Emits OTel metrics.
*   `TaskQueues`:
    *   `celery_task_queue_v1`|`celery_task_queue`. Cfg: `celery_app_name`, `celery_broker_url`, `celery_backend_url`.
    *   `redis_queue_task_plugin_v1`|`rq_task_queue`. Cfg: `redis_url`, `default_queue_name`.
*   (Other plugin categories like DefFormatters, RAG, ToolLookupProv, CodeExec, CacheProv, HITL, Guardrails, Prompts, Conversation, LLMOutputParsers, InvocationStrategies remain structurally similar but their instances are loaded based on configuration.)

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