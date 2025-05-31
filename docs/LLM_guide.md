# GenieTooling LLMGuide vHyperCompressed-APIPrecise
Facade:genie_tooling.genie.Genie
Init:Genie.create(config:MiddlewareConfig,key_provider_instance?:KeyProvider)
LLM(genie.llm):
 chat(messages:List[ChatMessage],provider_id?:str,stream?:bool,**kwargs)->Union[LLMChatResponse,AsyncIterable[LLMChatChunk]]
 generate(prompt:str,provider_id?:str,stream?:bool,**kwargs)->Union[LLMCompletionResponse,AsyncIterable[LLMCompletionChunk]]
 parse_output(response:Union[LLMChatResponse,LLMCompletionResponse],parser_id?:str,schema?:Any)->ParsedOutput
RAG(genie.rag):
 index_directory(path:str,collection_name?:str,loader_id?:str,splitter_id?:str,embedder_id?:str,vector_store_id?:str,loader_config?:Dict,splitter_config?:Dict,embedder_config?:Dict,vector_store_config?:Dict,**kwargs)->Dict
 index_web_page(url:str,collection_name?:str,...same_as_idx_dir...)->Dict
 search(query:str,collection_name?:str,top_k?:int,retriever_id?:str,retriever_config?:Dict,**kwargs)->List[RetrievedChunk]
Tools(genie):
 execute_tool(tool_identifier:str,**params:Any)->Any
 run_command(command:str,processor_id?:str,conversation_history?:List[ChatMessage])->CommandProcessorResponse
ToolReg:@genie_tooling.tool(func)->Callable(adds:_tool_metadata_:Dict,_original_function_:Callable). await genie.register_tool_functions(functions:List[Callable])->None(invalidates_lookup_idx)
Prompts(genie.prompts):
 get_prompt_template_content(name:str,version?:str,registry_id?:str)->Optional[str]
 render_prompt(name:str,data:PromptData,version?:str,registry_id?:str,template_engine_id?:str)->Optional[FormattedPrompt]
 render_chat_prompt(name:str,data:PromptData,version?:str,registry_id?:str,template_engine_id?:str)->Optional[List[ChatMessage]]
 list_templates(registry_id?:str)->List[PromptIdentifier]
Conversation(genie.conversation):
 load_state(session_id:str,provider_id?:str)->Optional[ConversationState]
 save_state(state:ConversationState,provider_id?:str)->None
 add_message(session_id:str,message:ChatMessage,provider_id?:str)->None
 delete_state(session_id:str,provider_id?:str)->bool
Observability(genie.observability):
 trace_event(event_name:str,data:Dict,component?:str,correlation_id?:str)->None
HITL(genie.human_in_loop):
 request_approval(request:ApprovalRequest,approver_id?:str)->ApprovalResponse
Usage(genie.usage):
 record_usage(record:TokenUsageRecord)->None
 get_summary(recorder_id?:str,filter_criteria?:Dict)->Dict
Teardown:genie.close()->None

Config:genie_tooling.config.models.MiddlewareConfig (MWCfg)
 FeatSets:genie_tooling.config.features.FeatureSettings->CfgResolver:genie_tooling.config.resolver.ConfigResolver
  llm:str(ollama|openai|gemini|none)->MWCfg.default_llm_provider_id,sets_model_name(e.g.MWCfg.features.llm_ollama_model_name)
  cache:str(in-memory|redis|none)->MWCfg.cache_provider_configurations(e.g.MWCfg.features.cache_redis_url)
  rag_embedder:str(sentence_transformer|openai|none)->MWCfg.default_rag_embedder_id,sets_model(e.g.MWCfg.features.rag_embedder_st_model_name)
  rag_vector_store:str(faiss|chroma|none)->MWCfg.default_rag_vector_store_id,sets_path/coll(e.g.MWCfg.features.rag_vector_store_chroma_path)
  tool_lookup:str(embedding|keyword|none)->MWCfg.default_tool_lookup_provider_id
   tool_lookup_formatter_id_alias:str->MWCfg.default_tool_indexing_formatter_id
   tool_lookup_embedder_id_alias:str->emb_for_emb_lkp
   tool_lookup_chroma_path:str,tool_lookup_chroma_collection_name:str->Chroma_for_emb_lkp
  command_processor:str(llm_assisted|simple_keyword|none)->MWCfg.default_command_processor_id
   command_processor_formatter_id_alias:str->fmt_for_llm_assist
  observability_tracer:str(console_tracer|otel_tracer|none)->MWCfg.default_observability_tracer_id
  hitl_approver:str(cli_hitl_approver|none)->MWCfg.default_hitl_approver_id
  token_usage_recorder:str(in_memory_token_recorder|none)->MWCfg.default_token_usage_recorder_id
  input_guardrails:List[str];output_guardrails:List[str];tool_usage_guardrails:List[str]->MWCfg.default_*_guardrail_ids
 CfgResolver:feat+aliases->canon_ids&cfgs.PLUGIN_ID_ALIASES:Dict.Prec:expl_cfg>expl_def>feat.
 key_provider_id:Optional[str].Def:env_keys if no key_provider_instance.
 key_provider_instance:Optional[KeyProvider]->Genie.create().
 *_configurations:Dict[str_id_or_alias,Dict[str,Any]](e.g.llm_provider_configurations,tool_configurations).Override feat.
 plugin_dev_dirs:List[str].
 default_prompt_registry_id:Optional[str];prompt_registry_configurations:Dict.
 default_prompt_template_plugin_id:Optional[str];prompt_template_configurations:Dict.
 default_conversation_state_provider_id:Optional[str];conversation_state_provider_configurations:Dict.
 default_llm_output_parser_id:Optional[str];llm_output_parser_configurations:Dict.

Plugins:genie_tooling.core.plugin_manager.PluginManager.IDs/paths:pyproject.toml->[tool.poetry.plugins."genie_tooling.plugins"]
Aliases:genie_tooling.config.resolver.PLUGIN_ID_ALIASES

KeyPlugins(ID|Alias|Cfg/Notes):
 KeyProvider:environment_key_provider_v1|env_keys|Env.
 LLMProviderPlugin:
  ollama_llm_provider_v1|ollama|Cfg:model_name,base_url.Opt:format="json".
  openai_llm_provider_v1|openai|Cfg:model_name,openai_api_base.Needs:OPENAI_API_KEY.
  gemini_llm_provider_v1|gemini|Cfg:model_name.Needs:GOOGLE_API_KEY.FuncCall.
 CommandProcessorPlugin:
  simple_keyword_processor_v1|simple_keyword_cmd_proc|Cfg:keyword_map,keyword_priority.Prompts.
  llm_assisted_tool_selection_processor_v1|llm_assisted_cmd_proc|LLM.Cfg:tool_formatter_id,tool_lookup_top_k.Uses:ToolLookupService.
 ToolPlugin:calculator_tool. sandboxed_fs_tool_v1(Cfg:sandbox_base_path). google_search_tool_v1(Needs:GOOGLE_API_KEY,GOOGLE_CSE_ID). open_weather_map_tool(Needs:OPENWEATHERMAP_API_KEY). generic_code_execution_tool(Uses:CodeExecutorPlugin).
 DefinitionFormatterPlugin:compact_text_formatter_plugin_v1|compact_text_formatter. openai_function_formatter_plugin_v1|openai_func_formatter. human_readable_json_formatter_plugin_v1|hr_json_formatter.
 RAG:
  DocLoad:file_system_loader_v1,web_page_loader_v1(Cfg:use_trafilatura).
  TxtSplit:character_recursive_text_splitter_v1.
  EmbGen:sentence_transformer_embedder_v1|st_embedder(Cfg:model_name),openai_embedding_generator_v1|openai_embedder(Needs:OPENAI_API_KEY).
  VecStore:faiss_vector_store_v1|faiss_vs,chromadb_vector_store_v1|chroma_vs(Cfg:path,collection_name).
  Retriever:basic_similarity_retriever_v1.
 ToolLookupProviderPlugin:emb_sim_lkp_v1|embedding_lookup(Uses:emb(tool_lookup_embedder_id_alias)&optVS). kw_match_lkp_v1|keyword_lookup.
 CodeExecutorPlugin:secure_docker_executor_v1|Docker(Cfg:*_docker_image,pull_images_on_setup). pysandbox_executor_stub_v1|STUB,INSECURE(exec()).
 CacheProviderPlugin:in_mem_cache_v1|in_memory_cache(Cfg:max_size,default_ttl_seconds). redis_cache_v1|redis_cache(Cfg:redis_url,default_ttl_seconds).
 InteractionTracerPlugin:console_tracer_plugin_v1|console_tracer. otel_tracer_plugin_v1|otel_tracer(STUB).
 HumanApprovalRequestPlugin:cli_approval_plugin_v1|cli_hitl_approver.
 TokenUsageRecorderPlugin:in_memory_token_usage_recorder_v1|in_memory_token_recorder.
 GuardrailPlugin:keyword_blocklist_guardrail_v1|keyword_blocklist_guardrail(Cfg:blocklist,case_sensitive,action_on_match).
 PromptRegistryPlugin:file_system_prompt_registry_v1|file_system_prompt_registry(Cfg:base_path,template_suffix).
 PromptTemplatePlugin:basic_string_format_template_v1|basic_string_formatter. jinja2_chat_template_v1|jinja2_chat_formatter.
 ConversationStateProviderPlugin:in_memory_conversation_state_v1|in_memory_convo_provider. redis_conversation_state_v1|redis_convo_provider(Cfg:redis_url,key_prefix,default_ttl_seconds).
 LLMOutputParserPlugin:json_output_parser_v1|json_output_parser(Cfg:strict_parsing). pydantic_output_parser_v1|pydantic_output_parser.

Types:
 ChatMessage:Dict{role:str,content?:str,tool_calls?:List[ToolCall],tool_call_id?:str,name?:str}
 ToolCall:Dict{id:str,type:Literal["function"],function:ToolCallFunction}
 ToolCallFunction:Dict{name:str,arguments:str(*JSON_string*)}
 LLMCompletionResponse:Dict{text:str,finish_reason?:str,usage?:LLMUsageInfo,raw_response:Any}
 LLMChatResponse:Dict{message:ChatMessage,finish_reason?:str,usage?:LLMUsageInfo,raw_response:Any}
 LLMCompletionChunk:Dict{text_delta?:str,finish_reason?:str,usage_delta?:LLMUsageInfo,raw_chunk:Any}
 LLMChatChunk:Dict{message_delta?:LLMChatChunkDeltaMessage,finish_reason?:str,usage_delta?:LLMUsageInfo,raw_chunk:Any}
 LLMChatChunkDeltaMessage:Dict{role?:str,content?:str,tool_calls?:List[ToolCall]}
 CommandProcessorResponse:Dict{chosen_tool_id?:str,extracted_params?:Dict,llm_thought_process?:str,error?:str,raw_response?:Any}
 RetrievedChunk:Dict{content:str,metadata:Dict,id?:str,score:float}
 CodeExecutionResult:NamedTuple(stdout:str,stderr:str,result?:Any,error?:str,execution_time_ms:float)
 PromptData:Dict[str,Any]
 FormattedPrompt:Union[str,List[ChatMessage]]
 PromptIdentifier:Dict{name:str,version?:str,description?:str}
 ConversationState:Dict{session_id:str,history:List[ChatMessage],metadata?:Dict}
 ApprovalRequest:Dict{request_id:str,prompt:str,data_to_approve:Dict,context?:Dict,timeout_seconds?:int}
 ApprovalResponse:Dict{request_id:str,status:str,approver_id?:str,reason?:str,timestamp?:float}
 TokenUsageRecord:Dict{provider_id:str,model_name:str,prompt_tokens?:int,completion_tokens?:int,total_tokens?:int,timestamp:float,call_type?:str,...}
 ParsedOutput:Any