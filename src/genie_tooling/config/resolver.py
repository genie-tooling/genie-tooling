# src/genie_tooling/config/resolver.py
import json
import logging
from copy import deepcopy
from typing import Any, Dict, Optional

from .models import MiddlewareConfig

logger = logging.getLogger(__name__)

PLUGIN_ID_ALIASES: Dict[str, str] = {
    "ollama": "ollama_llm_provider_v1",
    "openai": "openai_llm_provider_v1",
    "gemini": "gemini_llm_provider_v1",
    "llama_cpp": "llama_cpp_llm_provider_v1",
    "llama_cpp_internal": "llama_cpp_internal_llm_provider_v1",
    "env_keys": "environment_key_provider_v1",
    "in_memory_cache": "in_memory_cache_provider_v1",
    "redis_cache": "redis_cache_provider_v1",
    "st_embedder": "sentence_transformer_embedder_v1",
    "openai_embedder": "openai_embedding_generator_v1",
    "faiss_vs": "faiss_vector_store_v1",
    "chroma_vs": "chromadb_vector_store_v1",
    "qdrant_vs": "qdrant_vector_store_v1",
    "embedding_lookup": "embedding_similarity_lookup_v1",
    "keyword_lookup": "keyword_match_lookup_v1",
    "hybrid_lookup": "hybrid_search_lookup_v1",
    "compact_text_formatter": "compact_text_formatter_plugin_v1",
    "openai_func_formatter": "openai_function_formatter_plugin_v1",
    "hr_json_formatter": "human_readable_json_formatter_plugin_v1",
    "llm_assisted": "llm_assisted_tool_selection_processor_v1",
    "simple_keyword": "simple_keyword_processor_v1",
    "rewoo": "rewoo_command_processor_v1",
    "default_log_adapter": "default_log_adapter_v1",
    "pyvider_log_adapter": "pyvider_telemetry_log_adapter_v1",
    "noop_redactor": "noop_redactor_v1",
    "default_error_handler": "default_error_handler_v1",
    "llm_error_formatter": "llm_error_formatter_v1",
    "json_error_formatter": "json_error_formatter_v1",
    "default_invocation_strategy": "default_async_invocation_strategy_v1",
    "distributed_task_strategy": "distributed_task_invocation_strategy_v1",
    "jsonschema_validator": "jsonschema_input_validator_v1",
    "passthrough_transformer": "passthrough_output_transformer_v1",
    "console_tracer": "console_tracer_plugin_v1",
    "otel_tracer": "otel_tracer_plugin_v1",
    "cli_hitl_approver": "cli_approval_plugin_v1",
    "in_memory_token_recorder": "in_memory_token_usage_recorder_v1",
    "otel_metrics_recorder": "otel_metrics_token_recorder_v1",
    "keyword_blocklist_guardrail": "keyword_blocklist_guardrail_v1",
    "file_system_prompt_registry": "file_system_prompt_registry_v1",
    "basic_string_formatter": "basic_string_format_template_v1",
    "jinja2_chat_formatter": "jinja2_chat_template_v1", # Canonical ID for Jinja2
    "in_memory_convo_provider": "in_memory_conversation_state_v1",
    "redis_convo_provider": "redis_conversation_state_v1",
    "json_output_parser": "json_output_parser_v1",
    "pydantic_output_parser": "pydantic_output_parser_v1",
    "celery_task_queue": "celery_task_queue_v1",
    "rq_task_queue": "redis_queue_task_plugin_v1",
}

class ConfigResolver:
    def _merge_plugin_specific_config(
        self,
        target_dict: Dict[str, Dict[str, Any]],
        canonical_plugin_id: str,
        config_from_features: Dict[str, Any],
        config_from_user_explicit: Dict[str, Any]
    ):
        merged_conf = deepcopy(config_from_features)
        merged_conf.update(deepcopy(config_from_user_explicit))
        target_dict[canonical_plugin_id] = merged_conf


    def resolve(self, user_config: MiddlewareConfig, key_provider_instance: Optional[Any] = None) -> MiddlewareConfig:  # noqa: C901
        resolved_config = MiddlewareConfig()
        if "features" in user_config.model_fields_set:
            resolved_config.features = user_config.features.model_copy(deep=True)
        features = resolved_config.features

        # LLM
        if features.llm != "none" and PLUGIN_ID_ALIASES.get(features.llm):
            llm_id = PLUGIN_ID_ALIASES[features.llm]
            resolved_config.default_llm_provider_id = llm_id
            conf = {}
            if features.llm == "ollama":
                conf["model_name"] = features.llm_ollama_model_name
            elif features.llm == "openai":
                conf["model_name"] = features.llm_openai_model_name
            elif features.llm == "gemini":
                conf["model_name"] = features.llm_gemini_model_name
            elif features.llm == "llama_cpp":
                conf["model_name"] = features.llm_llama_cpp_model_name
                conf["base_url"] = features.llm_llama_cpp_base_url
                if features.llm_llama_cpp_api_key_name:
                    conf["api_key_name"] = features.llm_llama_cpp_api_key_name
            elif features.llm == "llama_cpp_internal":
                conf["model_path"] = features.llm_llama_cpp_internal_model_path
                conf["n_gpu_layers"] = features.llm_llama_cpp_internal_n_gpu_layers
                conf["n_ctx"] = features.llm_llama_cpp_internal_n_ctx
                if features.llm_llama_cpp_internal_chat_format:
                    conf["chat_format"] = features.llm_llama_cpp_internal_chat_format
                if features.llm_llama_cpp_internal_model_name_for_logging:
                    conf["model_name_for_logging"] = features.llm_llama_cpp_internal_model_name_for_logging
            if features.llm in ["openai", "gemini"] and key_provider_instance:
                conf["key_provider"] = key_provider_instance
            elif features.llm == "llama_cpp" and key_provider_instance and features.llm_llama_cpp_api_key_name:
                conf["key_provider"] = key_provider_instance
            if conf or features.llm in ["ollama", "openai", "gemini", "llama_cpp", "llama_cpp_internal"]:
                 resolved_config.llm_provider_configurations.setdefault(llm_id, {}).update(conf)

        # Cache
        if features.cache != "none":
            cache_alias = {"in-memory": "in_memory_cache", "redis": "redis_cache"}.get(features.cache)
            if cache_alias and PLUGIN_ID_ALIASES.get(cache_alias):
                cache_id = PLUGIN_ID_ALIASES[cache_alias]
                conf = {}
                if features.cache == "redis":
                    conf["redis_url"] = features.cache_redis_url
                resolved_config.cache_provider_configurations.setdefault(cache_id, {}).update(conf)

        # RAG Embedder
        if features.rag_embedder != "none":
            alias = {"sentence_transformer": "st_embedder", "openai": "openai_embedder"}.get(features.rag_embedder)
            if alias and PLUGIN_ID_ALIASES.get(alias):
                embed_id = PLUGIN_ID_ALIASES[alias]
                resolved_config.default_rag_embedder_id = embed_id
                conf = {}
                if features.rag_embedder == "sentence_transformer":
                    conf["model_name"] = features.rag_embedder_st_model_name
                if features.rag_embedder == "openai" and key_provider_instance:
                    conf["key_provider"] = key_provider_instance
                if conf or features.rag_embedder in ["sentence_transformer", "openai"]:
                     resolved_config.embedding_generator_configurations.setdefault(embed_id, {}).update(conf)

        # RAG Vector Store
        if features.rag_vector_store != "none":
            alias = {"faiss": "faiss_vs", "chroma": "chroma_vs", "qdrant": "qdrant_vs"}.get(features.rag_vector_store)
            if alias and PLUGIN_ID_ALIASES.get(alias):
                vs_id = PLUGIN_ID_ALIASES[alias]
                resolved_config.default_rag_vector_store_id = vs_id
                conf = {}
                if features.rag_vector_store == "chroma":
                    conf["collection_name"] = features.rag_vector_store_chroma_collection_name
                    if features.rag_vector_store_chroma_mode == "persistent":
                        conf["path"] = features.rag_vector_store_chroma_path
                    elif features.rag_vector_store_chroma_mode == "http":
                        conf["host"] = features.rag_vector_store_chroma_host
                        conf["port"] = features.rag_vector_store_chroma_port
                    elif features.rag_vector_store_chroma_mode == "ephemeral":
                        # The ChromaDBVectorStore plugin should interpret path=None as ephemeral
                        conf["path"] = None
                elif features.rag_vector_store == "qdrant":
                    conf["collection_name"] = features.rag_vector_store_qdrant_collection_name
                    if features.rag_vector_store_qdrant_url:
                        conf["url"] = features.rag_vector_store_qdrant_url
                    if features.rag_vector_store_qdrant_path:
                        conf["path"] = features.rag_vector_store_qdrant_path
                    if features.rag_vector_store_qdrant_api_key_name and key_provider_instance:
                        conf["api_key_name"] = features.rag_vector_store_qdrant_api_key_name
                        conf["key_provider"] = key_provider_instance
                    if features.rag_vector_store_qdrant_embedding_dim:
                        conf["embedding_dim"] = features.rag_vector_store_qdrant_embedding_dim
                if conf or features.rag_vector_store == "faiss":
                     resolved_config.vector_store_configurations.setdefault(vs_id, {}).update(conf)

        # Tool Lookup
        if features.tool_lookup != "none":
            lookup_id = PLUGIN_ID_ALIASES.get(features.tool_lookup + "_lookup")
            if lookup_id:
                resolved_config.default_tool_lookup_provider_id = lookup_id
                resolved_config.tool_lookup_provider_configurations.setdefault(lookup_id, {})
            else:
                logger.warning(f"Could not resolve tool lookup provider ID for feature setting: '''{features.tool_lookup}'''.")
            if features.tool_lookup_formatter_id_alias:
                 resolved_config.default_tool_indexing_formatter_id = PLUGIN_ID_ALIASES.get(features.tool_lookup_formatter_id_alias, features.tool_lookup_formatter_id_alias)
            if lookup_id and features.tool_lookup in ["embedding", "hybrid"]:
                top_level_provider_cfg = resolved_config.tool_lookup_provider_configurations.get(lookup_id, {})
                target_config_for_dense_details = {}
                if features.tool_lookup == "hybrid":
                    target_config_for_dense_details = top_level_provider_cfg.setdefault("dense_provider_config", {})
                else:
                    target_config_for_dense_details = top_level_provider_cfg
                embed_alias = features.tool_lookup_embedder_id_alias or "st_embedder"
                embed_id = PLUGIN_ID_ALIASES.get(embed_alias)
                if embed_id:
                    target_config_for_dense_details["embedder_id"] = embed_id
                    emb_tl_conf = {}
                    if embed_alias == "st_embedder" and features.rag_embedder_st_model_name:
                        emb_tl_conf["model_name"] = features.rag_embedder_st_model_name
                    elif embed_alias == "openai_embedder" and key_provider_instance:
                        emb_tl_conf["key_provider"] = key_provider_instance
                    if emb_tl_conf:
                        target_config_for_dense_details.setdefault("embedder_config", {}).update(emb_tl_conf)
                if features.tool_lookup_chroma_collection_name is not None:
                    target_config_for_dense_details["vector_store_id"] = PLUGIN_ID_ALIASES.get("chroma_vs")
                    vs_tl_conf = {"collection_name": features.tool_lookup_chroma_collection_name}
                    if features.tool_lookup_chroma_path is not None:
                        vs_tl_conf["path"] = features.tool_lookup_chroma_path
                    target_config_for_dense_details.setdefault("vector_store_config", {}).update(vs_tl_conf)
                resolved_config.tool_lookup_provider_configurations[lookup_id] = top_level_provider_cfg

        # Command Processor
        if features.command_processor != "none":
            cmd_proc_id = PLUGIN_ID_ALIASES.get(features.command_processor)
            if cmd_proc_id:
                resolved_config.default_command_processor_id = cmd_proc_id
                cmd_proc_conf = {}
                if features.command_processor == "llm_assisted":
                    if features.command_processor_formatter_id_alias:
                        cmd_proc_conf["tool_formatter_id"] = PLUGIN_ID_ALIASES.get(
                            features.command_processor_formatter_id_alias, features.command_processor_formatter_id_alias
                        )
                    if features.tool_lookup_top_k is not None:
                        cmd_proc_conf["tool_lookup_top_k"] = features.tool_lookup_top_k
                resolved_config.command_processor_configurations.setdefault(cmd_proc_id, {}).update(cmd_proc_conf)
                logger.debug(f"Resolved command processor '''{features.command_processor}''' to ID '''{cmd_proc_id}''' with config: {resolved_config.command_processor_configurations.get(cmd_proc_id)}")
            else:
                logger.warning(f"Could not resolve command processor ID for feature setting: '''{features.command_processor}'''. Default command processor will not be set by features.")

        # Logging Adapter
        if features.logging_adapter != "none":
            log_adapter_alias = features.logging_adapter
            log_adapter_id = PLUGIN_ID_ALIASES.get(log_adapter_alias)
            if log_adapter_id:
                resolved_config.default_log_adapter_id = log_adapter_id
                conf = {}
                if log_adapter_alias == "pyvider_log_adapter":
                    if features.logging_pyvider_service_name:
                        conf["service_name"] = features.logging_pyvider_service_name
                resolved_config.log_adapter_configurations.setdefault(log_adapter_id, {}).update(conf)
            else:
                logger.warning(f"Unknown logging_adapter alias '''{log_adapter_alias}''' in FeatureSettings.")
        elif features.logging_adapter == "none":
             resolved_config.default_log_adapter_id = None

        # Observability
        if features.observability_tracer != "none":
            tracer_id = PLUGIN_ID_ALIASES.get(features.observability_tracer)
            if tracer_id:
                resolved_config.default_observability_tracer_id = tracer_id
                conf = {}
                if features.observability_tracer == "otel_tracer" and features.observability_otel_endpoint:
                    if "4318" in features.observability_otel_endpoint or "http" in features.observability_otel_endpoint.lower():
                        conf["exporter_type"] = "otlp_http"
                        conf["otlp_http_endpoint"] = features.observability_otel_endpoint
                    elif "4317" in features.observability_otel_endpoint:
                        conf["exporter_type"] = "otlp_grpc"
                        conf["otlp_grpc_endpoint"] = features.observability_otel_endpoint
                    else:
                        conf["exporter_type"] = "otlp_http"
                        conf["otlp_http_endpoint"] = features.observability_otel_endpoint
                resolved_config.observability_tracer_configurations.setdefault(tracer_id, {}).update(conf)

        # HITL
        if features.hitl_approver != "none":
            approver_id = PLUGIN_ID_ALIASES.get(features.hitl_approver)
            if approver_id:
                resolved_config.default_hitl_approver_id = approver_id
                resolved_config.hitl_approver_configurations.setdefault(approver_id, {})

        # Token Usage Recorder
        if features.token_usage_recorder != "none":
            recorder_id = PLUGIN_ID_ALIASES.get(features.token_usage_recorder)
            if recorder_id:
                resolved_config.default_token_usage_recorder_id = recorder_id
                resolved_config.token_usage_recorder_configurations.setdefault(recorder_id, {})

        # Guardrails
        resolved_config.default_input_guardrail_ids = [PLUGIN_ID_ALIASES.get(g_alias, g_alias) for g_alias in features.input_guardrails]
        resolved_config.default_output_guardrail_ids = [PLUGIN_ID_ALIASES.get(g_alias, g_alias) for g_alias in features.output_guardrails]
        resolved_config.default_tool_usage_guardrail_ids = [PLUGIN_ID_ALIASES.get(g_alias, g_alias) for g_alias in features.tool_usage_guardrails]
        all_feature_guardrails = set(resolved_config.default_input_guardrail_ids + resolved_config.default_output_guardrail_ids + resolved_config.default_tool_usage_guardrail_ids)
        for gr_id in all_feature_guardrails:
            resolved_config.guardrail_configurations.setdefault(gr_id, {})

        # Prompt Registry
        if features.prompt_registry != "none":
            reg_alias = features.prompt_registry
            reg_id = PLUGIN_ID_ALIASES.get(reg_alias)
            if reg_id:
                resolved_config.default_prompt_registry_id = reg_id
                resolved_config.prompt_registry_configurations.setdefault(reg_id, {})
            else:
                logger.warning(f"Unknown prompt_registry alias '''{reg_alias}''' in FeatureSettings. Default not set.")

        # Prompt Template Engine
        if features.prompt_template_engine != "none":
            engine_alias = features.prompt_template_engine
            engine_id = PLUGIN_ID_ALIASES.get(engine_alias) # This should resolve "jinja2_chat_formatter" to "jinja2_chat_template_v1"
            if engine_id:
                resolved_config.default_prompt_template_plugin_id = engine_id
                resolved_config.prompt_template_configurations.setdefault(engine_id, {})
                logger.debug(f"Resolved prompt_template_engine '''{engine_alias}''' to ID '''{engine_id}'''.")
            else:
                logger.warning(f"Unknown prompt_template_engine alias '''{engine_alias}''' in FeatureSettings. Default not set.")
        elif features.prompt_template_engine == "none": # Explicitly none
            resolved_config.default_prompt_template_plugin_id = None


        # Conversation State Provider
        if features.conversation_state_provider != "none":
            convo_alias = features.conversation_state_provider
            convo_id = PLUGIN_ID_ALIASES.get(convo_alias)
            if convo_id:
                resolved_config.default_conversation_state_provider_id = convo_id
                resolved_config.conversation_state_provider_configurations.setdefault(convo_id, {})
            else:
                logger.warning(f"Unknown conversation_state_provider alias '''{convo_alias}''' in FeatureSettings. Default not set.")

        # LLM Output Parser
        if features.default_llm_output_parser is not None and features.default_llm_output_parser != "none":
            parser_alias = features.default_llm_output_parser
            parser_id = PLUGIN_ID_ALIASES.get(parser_alias)
            if parser_id:
                resolved_config.default_llm_output_parser_id = parser_id
                resolved_config.llm_output_parser_configurations.setdefault(parser_id, {})
            else:
                logger.warning(f"Unknown default_llm_output_parser alias '''{parser_alias}''' in FeatureSettings. Default not set.")
        elif features.default_llm_output_parser is None:
             resolved_config.default_llm_output_parser_id = None

        # Distributed Task Queue
        if features.task_queue != "none":
            task_q_alias = {"celery": "celery_task_queue", "rq": "rq_task_queue"}.get(features.task_queue)
            if task_q_alias and PLUGIN_ID_ALIASES.get(task_q_alias):
                task_q_id = PLUGIN_ID_ALIASES[task_q_alias]
                resolved_config.default_distributed_task_queue_id = task_q_id
                conf = {}
                if features.task_queue == "celery":
                    conf["celery_broker_url"] = features.task_queue_celery_broker_url
                    conf["celery_backend_url"] = features.task_queue_celery_backend_url
                resolved_config.distributed_task_queue_configurations.setdefault(task_q_id, {}).update(conf)

        # Merge user's explicit config
        user_explicit_copy = user_config.model_copy(deep=True)
        for field_name in user_explicit_copy.model_fields_set:
            if field_name == "features": continue
            user_value = getattr(user_explicit_copy, field_name)
            if field_name.endswith("_configurations") and isinstance(user_value, dict):
                target_dict_in_resolved = getattr(resolved_config, field_name)
                for key_alias_from_user, conf_from_user in user_value.items():
                    canonical_plugin_id = PLUGIN_ID_ALIASES.get(key_alias_from_user, key_alias_from_user)
                    base_conf_from_features = target_dict_in_resolved.get(canonical_plugin_id, {})
                    user_plugin_conf = deepcopy(conf_from_user or {})
                    final_merged_plugin_conf = {**base_conf_from_features, **user_plugin_conf}
                    if key_provider_instance and "key_provider" not in user_plugin_conf:
                        is_openai_llm = canonical_plugin_id == PLUGIN_ID_ALIASES.get("openai")
                        is_gemini_llm = canonical_plugin_id == PLUGIN_ID_ALIASES.get("gemini")
                        is_llama_cpp_llm_with_key = (canonical_plugin_id == PLUGIN_ID_ALIASES.get("llama_cpp") and final_merged_plugin_conf.get("api_key_name") is not None)
                        is_openai_embed = canonical_plugin_id == PLUGIN_ID_ALIASES.get("openai_embedder")
                        is_qdrant_vs_with_key = (canonical_plugin_id == PLUGIN_ID_ALIASES.get("qdrant_vs") and final_merged_plugin_conf.get("api_key_name") is not None)
                        if (field_name == "llm_provider_configurations" and (is_openai_llm or is_gemini_llm or is_llama_cpp_llm_with_key)) or                            (field_name == "embedding_generator_configurations" and is_openai_embed) or                            (field_name == "vector_store_configurations" and is_qdrant_vs_with_key):
                            final_merged_plugin_conf["key_provider"] = key_provider_instance
                        if field_name == "tool_lookup_provider_configurations" and                            isinstance(final_merged_plugin_conf.get("embedder_config"), dict) and                            final_merged_plugin_conf.get("embedder_id") == PLUGIN_ID_ALIASES.get("openai_embedder") and                            "key_provider" not in final_merged_plugin_conf["embedder_config"]:
                            final_merged_plugin_conf["embedder_config"]["key_provider"] = key_provider_instance
                    target_dict_in_resolved[canonical_plugin_id] = final_merged_plugin_conf
                    if key_alias_from_user != canonical_plugin_id and key_alias_from_user in target_dict_in_resolved:
                        del target_dict_in_resolved[key_alias_from_user]
            elif field_name.startswith("default_") and field_name.endswith("_id") or field_name.endswith("_ids"):
                if isinstance(user_value, list):
                    setattr(resolved_config, field_name, [PLUGIN_ID_ALIASES.get(str(uv_item), uv_item) for uv_item in user_value])
                elif user_value is not None:
                    setattr(resolved_config, field_name, PLUGIN_ID_ALIASES.get(str(user_value), user_value))
                else:
                    setattr(resolved_config, field_name, None)
            elif hasattr(resolved_config, field_name):
                setattr(resolved_config, field_name, user_value)

        if "key_provider_id" in user_config.model_fields_set and user_config.key_provider_id is not None:
            resolved_config.key_provider_id = PLUGIN_ID_ALIASES.get(user_config.key_provider_id, user_config.key_provider_id)
        elif key_provider_instance and hasattr(key_provider_instance, "plugin_id") and              not ("key_provider_id" in user_config.model_fields_set and user_config.key_provider_id is not None):
            resolved_config.key_provider_id = key_provider_instance.plugin_id
        elif resolved_config.key_provider_id is None:
             resolved_config.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]

        try:
            loggable_dump = resolved_config.model_dump(exclude={"features"}, exclude_none=True)
            for map_name in list(loggable_dump.keys()):
                if map_name.endswith("_configurations") and isinstance(loggable_dump[map_name], dict):
                    for p_id, p_conf in list(loggable_dump[map_name].items()):
                        if isinstance(p_conf, dict) and "key_provider" in p_conf and not isinstance(p_conf["key_provider"], (str, int, float, bool, type(None))):
                            loggable_dump[map_name][p_id]["key_provider"] = f"<KeyProvider instance {type(p_conf['key_provider']).__name__}>"
                        if map_name == "tool_lookup_provider_configurations" and                            isinstance(p_conf.get("embedder_config"), dict) and                            "key_provider" in p_conf["embedder_config"] and                            not isinstance(p_conf["embedder_config"]["key_provider"], (str, int, float, bool, type(None))):
                            loggable_dump[map_name][p_id]["embedder_config"]["key_provider"] = f"<KeyProvider instance {type(p_conf['embedder_config']['key_provider']).__name__}>"
            logger.debug(f"Resolved MiddlewareConfig (excluding features): {json.dumps(loggable_dump, indent=2, default=str)}")
        except Exception as e_log:
            logger.error(f"Error serializing resolved_config for logging: {e_log}")
        return resolved_config
