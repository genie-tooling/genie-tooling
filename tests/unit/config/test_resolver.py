# tests/unit/config/test_resolver.py
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.security.key_provider import KeyProvider


@pytest.fixture
def mock_kp_instance_for_resolver() -> MagicMock:
    kp = MagicMock(spec=KeyProvider)
    kp.plugin_id = "mock_kp_instance_id_for_resolver"
    return kp

@pytest.fixture
def config_resolver() -> ConfigResolver:
    return ConfigResolver()

def test_resolver_llm_feature_ollama(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="ollama", llm_ollama_model_name="test-ollama-model")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    assert resolved.default_llm_provider_id == PLUGIN_ID_ALIASES["ollama"]
    assert resolved.llm_provider_configurations[PLUGIN_ID_ALIASES["ollama"]]["model_name"] == "test-ollama-model"

def test_resolver_llm_feature_openai_with_key_provider(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="openai", llm_openai_model_name="test-gpt")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_id = PLUGIN_ID_ALIASES["openai"]
    assert resolved.default_llm_provider_id == openai_id
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "test-gpt"
    assert resolved.llm_provider_configurations[openai_id]["key_provider"] is mock_kp_instance_for_resolver

def test_resolver_rag_embedder_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(rag_embedder="openai") # This should map to "openai_embedder" alias
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]
    assert resolved.default_rag_embedder_id == openai_embed_id
    assert resolved.embedding_generator_configurations[openai_embed_id]["key_provider"] is mock_kp_instance_for_resolver

def test_resolver_rag_vector_store_chroma_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            rag_vector_store="chroma", # This should map to "chroma_vs" alias
            rag_vector_store_chroma_collection_name="my_rag_docs",
            rag_vector_store_chroma_path="/data/chroma_rag"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    chroma_vs_id = PLUGIN_ID_ALIASES["chroma_vs"]
    assert resolved.default_rag_vector_store_id == chroma_vs_id
    vs_config = resolved.vector_store_configurations[chroma_vs_id]
    assert vs_config["collection_name"] == "my_rag_docs"
    assert vs_config["path"] == "/data/chroma_rag"

def test_resolver_tool_lookup_embedding_feature(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            tool_lookup="embedding",
            tool_lookup_formatter_id_alias="compact_text_formatter",
            tool_lookup_embedder_id_alias="st_embedder", # Refers to an alias for embedder plugin_id
            tool_lookup_chroma_collection_name="tools_lookup_test",
            tool_lookup_chroma_path="/data/tool_lookup_chroma"
        )
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    embedding_lookup_id = PLUGIN_ID_ALIASES["embedding_lookup"]
    st_embedder_id = PLUGIN_ID_ALIASES["st_embedder"]
    chroma_vs_id = PLUGIN_ID_ALIASES["chroma_vs"] # For the VS used by tool lookup
    compact_text_id = PLUGIN_ID_ALIASES["compact_text_formatter"]

    assert resolved.default_tool_lookup_provider_id == embedding_lookup_id
    assert resolved.default_tool_indexing_formatter_id == compact_text_id

    lookup_provider_cfg = resolved.tool_lookup_provider_configurations.get(embedding_lookup_id, {})
    assert lookup_provider_cfg.get("embedder_id") == st_embedder_id
    assert lookup_provider_cfg.get("vector_store_id") == chroma_vs_id

    assert "embedder_config" in lookup_provider_cfg # embedder_config is for the *embedder* used by lookup
    assert lookup_provider_cfg["embedder_config"].get("model_name") == FeatureSettings().rag_embedder_st_model_name # Check default ST model

    assert "vector_store_config" in lookup_provider_cfg # vs_config is for the *vector store* used by lookup
    assert lookup_provider_cfg["vector_store_config"].get("collection_name") == "tools_lookup_test"
    assert lookup_provider_cfg["vector_store_config"].get("path") == "/data/tool_lookup_chroma"

def test_resolver_explicit_config_overrides_features(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="ollama", llm_ollama_model_name="mistral-from-features"),
        default_llm_provider_id=PLUGIN_ID_ALIASES["ollama"], # User explicitly uses canonical ID
        llm_provider_configurations={PLUGIN_ID_ALIASES["ollama"]: {"model_name": "llama2-explicit-override"}}
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    assert resolved.default_llm_provider_id == ollama_id
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "llama2-explicit-override"

def test_resolver_alias_in_user_config_keys(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="none"), # Turn off feature-based LLM
        llm_provider_configurations={"openai": {"model_name": "gpt-4-via-alias"}} # User uses alias as key
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    openai_canonical_id = PLUGIN_ID_ALIASES["openai"]
    assert openai_canonical_id in resolved.llm_provider_configurations
    assert resolved.llm_provider_configurations[openai_canonical_id]["model_name"] == "gpt-4-via-alias"
    assert "openai" not in resolved.llm_provider_configurations # Alias key should be replaced by canonical

def test_resolver_feature_none_value(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="none", cache="none", rag_embedder="none", tool_lookup="none")
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    assert resolved.default_llm_provider_id is None
    assert not resolved.llm_provider_configurations # No configs should be populated by 'none' features
    assert resolved.default_rag_embedder_id is None
    assert not resolved.embedding_generator_configurations

def test_resolver_default_key_provider_id_behavior(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    # Case 1: No KP ID provided by user, no instance passed -> defaults to env_keys
    user_config_no_kp = MiddlewareConfig()
    resolved1 = config_resolver.resolve(user_config_no_kp, key_provider_instance=None)
    assert resolved1.key_provider_id == PLUGIN_ID_ALIASES["env_keys"]

    # Case 2: User provides an alias for key_provider_id
    with patch.dict(PLUGIN_ID_ALIASES, {"my_custom_kp_alias": "my_custom_kp_canonical_v1"}, clear=False):
        user_config_alias_kp = MiddlewareConfig(key_provider_id="my_custom_kp_alias")
        resolved2 = config_resolver.resolve(user_config_alias_kp, key_provider_instance=None)
        assert resolved2.key_provider_id == "my_custom_kp_canonical_v1"

    # Case 3: User provides a canonical ID for key_provider_id
    user_config_canonical_kp = MiddlewareConfig(key_provider_id="some_canonical_key_provider_id_v1")
    resolved3 = config_resolver.resolve(user_config_canonical_kp, key_provider_instance=None)
    assert resolved3.key_provider_id == "some_canonical_key_provider_id_v1"

    # Case 4: Instance provided, its ID should be used if user doesn't explicitly set key_provider_id
    resolved4 = config_resolver.resolve(user_config_no_kp, mock_kp_instance_for_resolver)
    assert resolved4.key_provider_id == mock_kp_instance_for_resolver.plugin_id

    # Case 5: Instance provided AND user explicitly sets key_provider_id (user's explicit ID wins)
    user_config_explicit_kp_id = MiddlewareConfig(key_provider_id="user_chosen_kp_id")
    resolved5 = config_resolver.resolve(user_config_explicit_kp_id, mock_kp_instance_for_resolver)
    assert resolved5.key_provider_id == "user_chosen_kp_id"


def test_resolver_complex_merge_and_override(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", llm_ollama_model_name="mistral-feature",
            rag_embedder="sentence_transformer", # This maps to "st_embedder" alias
            rag_embedder_st_model_name="st-feature-model"
        ),
        default_llm_provider_id="openai", # Explicit default using alias
        llm_provider_configurations={ # Explicit configs
            PLUGIN_ID_ALIASES["ollama"]: {"model_name": "ollama-explicit-model", "temperature": 0.7},
            "openai": {"model_name": "gpt-explicit-model", "max_tokens": 100} # Using alias as key
        },
        embedding_generator_configurations={
            PLUGIN_ID_ALIASES["openai_embedder"]: {"model_name": "ada-explicit-model"} # User provides explicit for OpenAI embedder
        },
        default_rag_embedder_id=None # User explicitly sets the *default* RAG embedder to None
    )
    resolved = config_resolver.resolve(user_config, mock_kp_instance_for_resolver)

    ollama_id = PLUGIN_ID_ALIASES["ollama"]
    openai_id = PLUGIN_ID_ALIASES["openai"]
    st_embed_id = PLUGIN_ID_ALIASES["st_embedder"]
    openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]

    # LLM Checks
    assert resolved.default_llm_provider_id == openai_id # User's explicit default_llm_provider_id (after alias resolution)
    assert resolved.llm_provider_configurations[ollama_id]["model_name"] == "ollama-explicit-model"
    assert resolved.llm_provider_configurations[openai_id]["model_name"] == "gpt-explicit-model"

    # RAG Embedder Checks
    assert resolved.default_rag_embedder_id is None # User explicitly set to None

    # Check that ST embedder config (from features) is present
    assert st_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[st_embed_id].get("model_name") == "st-feature-model"

    # Check that OpenAI embedder config (from user's explicit config) is also present
    assert openai_embed_id in resolved.embedding_generator_configurations
    assert resolved.embedding_generator_configurations[openai_embed_id].get("model_name") == "ada-explicit-model"

def test_resolver_logging_of_mock(config_resolver: ConfigResolver, mock_kp_instance_for_resolver: MagicMock):
    user_config = MiddlewareConfig(
        features=FeatureSettings(llm="openai", llm_openai_model_name="test-gpt")
    )
    try:
        config_resolver.resolve(user_config, mock_kp_instance_for_resolver)
    except Exception as e:
        pytest.fail(f"Resolver logging failed with mock: {e}")
