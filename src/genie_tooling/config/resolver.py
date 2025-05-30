### src/genie_tooling/config/resolver.py
# src/genie_tooling/config/resolver.py
import json
import logging
from copy import deepcopy
from typing import Any, Dict, Optional

from .models import MiddlewareConfig

logger = logging.getLogger(__name__)

PLUGIN_ID_ALIASES: Dict[str, str] = {
    "ollama": "ollama_llm_provider_v1", "openai": "openai_llm_provider_v1", "gemini": "gemini_llm_provider_v1",
    "env_keys": "environment_key_provider_v1",
    "in_memory_cache": "in_memory_cache_provider_v1", "redis_cache": "redis_cache_provider_v1",
    "st_embedder": "sentence_transformer_embedder_v1", "openai_embedder": "openai_embedding_generator_v1",
    "faiss_vs": "faiss_vector_store_v1", "chroma_vs": "chromadb_vector_store_v1",
    "embedding_lookup": "embedding_similarity_lookup_v1", "keyword_lookup": "keyword_match_lookup_v1",
    "compact_text_formatter": "compact_text_formatter_plugin_v1",
    "openai_func_formatter": "openai_function_formatter_plugin_v1",
    "hr_json_formatter": "human_readable_json_formatter_plugin_v1",
    "llm_assisted_cmd_proc": "llm_assisted_tool_selection_processor_v1",
    "simple_keyword_cmd_proc": "simple_keyword_processor_v1",
    "default_log_adapter": "default_log_adapter_v1", "noop_redactor": "noop_redactor_v1",
    "default_error_handler": "default_error_handler_v1",
    "llm_error_formatter": "llm_error_formatter_v1", "json_error_formatter": "json_error_formatter_v1",
    "default_invocation_strategy": "default_async_invocation_strategy_v1",
    "jsonschema_validator": "jsonschema_input_validator_v1",
    "passthrough_transformer": "passthrough_output_transformer_v1",
}

class ConfigResolver:
    def _merge_plugin_specific_config(
        self,
        target_dict: Dict[str, Dict[str, Any]], # e.g., resolved_config.llm_provider_configurations
        canonical_plugin_id: str,
        config_from_features: Dict[str, Any],
        config_from_user_explicit: Dict[str, Any]
    ):
        # Start with feature-derived config as base
        merged_conf = deepcopy(config_from_features)
        # User's explicit config for this plugin overrides/adds to feature-derived
        merged_conf.update(deepcopy(config_from_user_explicit))

        target_dict[canonical_plugin_id] = merged_conf


    def resolve(self, user_config: MiddlewareConfig, key_provider_instance: Optional[Any] = None) -> MiddlewareConfig:
        # 1. Create a new MiddlewareConfig. Its `features` will be default.
        resolved_config = MiddlewareConfig()

        # 2. If user provided features, use them. This updates resolved_config.features.
        if "features" in user_config.model_fields_set:
            resolved_config.features = user_config.features.model_copy(deep=True)

        # Use the (potentially user-overridden) features from resolved_config for deriving defaults.
        features = resolved_config.features

        # 3. Populate resolved_config fields based on these features.
        #    This sets defaults for IDs and their configurations in the `*_configurations` dicts.
        if features.llm != "none" and PLUGIN_ID_ALIASES.get(features.llm):
            llm_id = PLUGIN_ID_ALIASES[features.llm]
            resolved_config.default_llm_provider_id = llm_id
            conf = {}
            if features.llm == "ollama": conf["model_name"] = features.llm_ollama_model_name
            elif features.llm == "openai": conf["model_name"] = features.llm_openai_model_name
            elif features.llm == "gemini": conf["model_name"] = features.llm_gemini_model_name

            # Inject key_provider for relevant LLMs if instance is available
            if features.llm in ["openai", "gemini"] and key_provider_instance:
                conf["key_provider"] = key_provider_instance
            if conf or features.llm in ["ollama", "openai", "gemini"]: # Ensure config dict is created even if conf is empty
                resolved_config.llm_provider_configurations.setdefault(llm_id, {}).update(conf)


        if features.cache != "none":
            cache_alias = {"in-memory": "in_memory_cache", "redis": "redis_cache"}.get(features.cache)
            if cache_alias and PLUGIN_ID_ALIASES.get(cache_alias):
                cache_id = PLUGIN_ID_ALIASES[cache_alias]
                conf = {}
                if features.cache == "redis":
                    conf["redis_url"] = features.cache_redis_url
                resolved_config.cache_provider_configurations.setdefault(cache_id, {}).update(conf)


        if features.rag_embedder != "none":
            alias = {"sentence_transformer": "st_embedder", "openai": "openai_embedder"}.get(features.rag_embedder)
            if alias and PLUGIN_ID_ALIASES.get(alias):
                embed_id = PLUGIN_ID_ALIASES[alias]
                resolved_config.default_rag_embedder_id = embed_id
                conf = {}
                if features.rag_embedder == "sentence_transformer": conf["model_name"] = features.rag_embedder_st_model_name
                if features.rag_embedder == "openai" and key_provider_instance: conf["key_provider"] = key_provider_instance
                if conf or features.rag_embedder in ["sentence_transformer", "openai"]:
                     resolved_config.embedding_generator_configurations.setdefault(embed_id, {}).update(conf)

        if features.rag_vector_store != "none":
            alias = {"faiss": "faiss_vs", "chroma": "chroma_vs"}.get(features.rag_vector_store)
            if alias and PLUGIN_ID_ALIASES.get(alias):
                vs_id = PLUGIN_ID_ALIASES[alias]
                resolved_config.default_rag_vector_store_id = vs_id
                conf = {}
                if features.rag_vector_store == "chroma":
                    conf["collection_name"] = features.rag_vector_store_chroma_collection_name
                    if features.rag_vector_store_chroma_path is not None: conf["path"] = features.rag_vector_store_chroma_path
                if conf or features.rag_vector_store == "faiss": # Ensure dict created for faiss too
                     resolved_config.vector_store_configurations.setdefault(vs_id, {}).update(conf)

        if features.tool_lookup != "none":
            lookup_id_key_part = features.tool_lookup # "embedding" or "keyword"
            lookup_id = PLUGIN_ID_ALIASES.get(f"{lookup_id_key_part}_lookup") # e.g. embedding_lookup or keyword_lookup

            if lookup_id:
                resolved_config.default_tool_lookup_provider_id = lookup_id
                # Ensure provider config dict exists if feature is on
                resolved_config.tool_lookup_provider_configurations.setdefault(lookup_id, {})


            if features.tool_lookup_formatter_id_alias:
                 resolved_config.default_tool_indexing_formatter_id = PLUGIN_ID_ALIASES.get(features.tool_lookup_formatter_id_alias, features.tool_lookup_formatter_id_alias)


            if lookup_id and features.tool_lookup == "embedding":
                tl_prov_cfg = resolved_config.tool_lookup_provider_configurations.setdefault(lookup_id, {})

                embed_alias = features.tool_lookup_embedder_id_alias or "st_embedder"
                embed_id = PLUGIN_ID_ALIASES.get(embed_alias)
                if embed_id:
                    tl_prov_cfg["embedder_id"] = embed_id
                    emb_tl_conf = {}
                    if embed_alias == "st_embedder" and features.rag_embedder_st_model_name:
                        emb_tl_conf["model_name"] = features.rag_embedder_st_model_name
                    elif embed_alias == "openai_embedder" and key_provider_instance:
                        emb_tl_conf["key_provider"] = key_provider_instance
                    if emb_tl_conf:
                        tl_prov_cfg.setdefault("embedder_config", {}).update(emb_tl_conf)

                if features.tool_lookup_chroma_collection_name is not None: # Chroma is specified for tool lookup
                    tl_prov_cfg["vector_store_id"] = PLUGIN_ID_ALIASES.get("chroma_vs")
                    vs_tl_conf = {"collection_name": features.tool_lookup_chroma_collection_name}
                    if features.tool_lookup_chroma_path is not None:
                        vs_tl_conf["path"] = features.tool_lookup_chroma_path
                    tl_prov_cfg.setdefault("vector_store_config", {}).update(vs_tl_conf)


        if features.command_processor != "none":
            cmd_proc_key_part = features.command_processor # "llm_assisted" or "simple_keyword"
            cmd_proc_id = PLUGIN_ID_ALIASES.get(f"{cmd_proc_key_part}_cmd_proc")

            if cmd_proc_id:
                resolved_config.default_command_processor_id = cmd_proc_id
                cmd_proc_conf = {}
                if features.command_processor == "llm_assisted":
                    if features.command_processor_formatter_id_alias:
                        cmd_proc_conf["tool_formatter_id"] = PLUGIN_ID_ALIASES.get(
                            features.command_processor_formatter_id_alias, features.command_processor_formatter_id_alias
                        )
                # Ensure config dict is created even if specific feature-derived sub-configs are empty
                resolved_config.command_processor_configurations.setdefault(cmd_proc_id, {}).update(cmd_proc_conf)


        # 4. Merge user's explicit config on top of feature-derived defaults.
        user_explicit_copy = user_config.model_copy(deep=True)

        for field_name in user_explicit_copy.model_fields_set:
            if field_name == "features": continue

            user_value = getattr(user_explicit_copy, field_name)

            if field_name.endswith("_configurations") and isinstance(user_value, dict):
                target_dict_in_resolved = getattr(resolved_config, field_name)
                for key_alias_from_user, conf_from_user in user_value.items():
                    canonical_plugin_id = PLUGIN_ID_ALIASES.get(key_alias_from_user, key_alias_from_user)
                    base_conf = target_dict_in_resolved.get(canonical_plugin_id, {})
                    user_plugin_conf = deepcopy(conf_from_user or {})
                    final_merged_plugin_conf = {**base_conf, **user_plugin_conf}

                    # Key provider injection for specific plugin types if instance is available
                    # and user hasn't explicitly set it or set it to None in their specific config.
                    # This logic might need refinement based on which plugins should get KP by default.
                    if key_provider_instance and "key_provider" not in user_plugin_conf:
                        # Only inject if the plugin type typically needs it (e.g., OpenAI, Gemini LLMs/Embedders)
                        is_openai_llm = canonical_plugin_id == PLUGIN_ID_ALIASES.get("openai")
                        is_gemini_llm = canonical_plugin_id == PLUGIN_ID_ALIASES.get("gemini")
                        is_openai_embed = canonical_plugin_id == PLUGIN_ID_ALIASES.get("openai_embedder")

                        if (field_name == "llm_provider_configurations" and (is_openai_llm or is_gemini_llm)) or \
                           (field_name == "embedding_generator_configurations" and is_openai_embed) or \
                           (field_name == "tool_lookup_provider_configurations" and isinstance(final_merged_plugin_conf.get("embedder_config"),dict) and final_merged_plugin_conf.get("embedder_id") == PLUGIN_ID_ALIASES.get("openai_embedder")):
                            # If embedder_config exists within tool_lookup_provider_configurations
                            if field_name == "tool_lookup_provider_configurations" and "embedder_config" in final_merged_plugin_conf:
                                final_merged_plugin_conf["embedder_config"]["key_provider"] = key_provider_instance
                            else:
                                final_merged_plugin_conf["key_provider"] = key_provider_instance


                    target_dict_in_resolved[canonical_plugin_id] = final_merged_plugin_conf
                    if key_alias_from_user != canonical_plugin_id and key_alias_from_user in target_dict_in_resolved:
                        del target_dict_in_resolved[key_alias_from_user]

            elif hasattr(resolved_config, field_name):
                final_value = PLUGIN_ID_ALIASES.get(str(user_value), user_value) if user_value is not None else None
                setattr(resolved_config, field_name, final_value)

        if "key_provider_id" in user_config.model_fields_set and user_config.key_provider_id is not None:
            resolved_config.key_provider_id = PLUGIN_ID_ALIASES.get(user_config.key_provider_id, user_config.key_provider_id)
        elif key_provider_instance and hasattr(key_provider_instance, "plugin_id") and \
             not ("key_provider_id" in user_config.model_fields_set and user_config.key_provider_id is not None):
            resolved_config.key_provider_id = key_provider_instance.plugin_id
        elif resolved_config.key_provider_id is None:
             resolved_config.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]

        try:
            loggable_dump = resolved_config.model_dump(exclude={"features"}, exclude_none=True)
            for map_name in list(loggable_dump.keys()): # Iterate over keys that are config maps
                if map_name.endswith("_configurations") and isinstance(loggable_dump[map_name], dict):
                    for p_id, p_conf in list(loggable_dump[map_name].items()):
                        if isinstance(p_conf, dict) and "key_provider" in p_conf and not isinstance(p_conf["key_provider"], (str, int, float, bool, type(None))):
                            loggable_dump[map_name][p_id]["key_provider"] = f"<KeyProvider instance {type(p_conf['key_provider']).__name__}>"
                        # Also check for embedder_config inside tool_lookup_provider_configurations
                        if map_name == "tool_lookup_provider_configurations" and isinstance(p_conf.get("embedder_config"), dict):
                            if "key_provider" in p_conf["embedder_config"] and not isinstance(p_conf["embedder_config"]["key_provider"], (str, int, float, bool, type(None))):
                                loggable_dump[map_name][p_id]["embedder_config"]["key_provider"] = f"<KeyProvider instance {type(p_conf['embedder_config']['key_provider']).__name__}>"

            logger.debug(f"Resolved MiddlewareConfig (excluding features): {json.dumps(loggable_dump, indent=2, default=str)}")
        except Exception as e_log:
            logger.error(f"Error serializing resolved_config for logging: {e_log}")

        return resolved_config
