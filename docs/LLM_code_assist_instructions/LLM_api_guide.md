# LLM_GUIDE_METADATA_V0.3
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
    *   Loaders: `file_system_loader_v1`, `web_page_loader_v1`.
    *   Splitters: `character_recursive_text_splitter_v1`.
    *   Embedders: `sentence_transformer_embedder_v1`|`st_embedder` (Cfg: `model_name`, `device`), `openai_embedding_generator_v1`|`openai_embedder` (Needs KP).
    *   VS: `faiss_vector_store_v1`|`faiss_vs` (Cfg: `embedding_dim`, `index_file_path`, `doc_store_file_path`), `chromadb_vector_store_v1`|`chroma_vs` (Cfg: `collection_name`, `path`, `host`, `port`), `qdrant_vector_store_v1`|`qdrant_vs` (Cfg: `collection_name`, `embedding_dim`, `url`, `path`, `api_key_name`).
    *   Retrievers: `basic_similarity_retriever_v1`.
*   `ToolLookupProv`:
    *   `embedding_similarity_lookup_v1`|`embedding_lookup`. Cfg: `embedder_id`, `embedder_config`, `vector_store_id`, `vector_store_config`.
    *   `keyword_match_lookup_v1`|`keyword_lookup`.
*   `CodeExec`: `secure_docker_executor_v1` (Cfg: `python_docker_image`, `node_docker_image`, `bash_docker_image`, `pull_images_on_setup`), `pysandbox_executor_stub_v1`.
*   `CacheProv`: `in_memory_cache_provider_v1`|`in_memory_cache` (Cfg: `max_size`, `default_ttl_seconds`, `cleanup_interval_seconds`), `redis_cache_provider_v1`|`redis_cache` (Cfg: `redis_url`, `default_ttl_seconds`, `json_serialization`).
*   `Observability`:
    *   `console_tracer_plugin_v1`|`console_tracer`. Cfg: `log_level`.
    *   `otel_tracer_plugin_v1`|`otel_tracer`. Cfg: `otel_service_name`, `exporter_type` (`console`|`otlp_http`|`otlp_grpc`), `otlp_http_endpoint`, `otlp_grpc_endpoint`, etc.
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