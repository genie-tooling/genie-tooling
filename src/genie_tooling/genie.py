# src/genie_tooling/genie.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from .config.models import MiddlewareConfig
from .core.plugin_manager import PluginManager
from .core.types import (
    Plugin as CorePluginType,  # Alias to avoid conflict with local Plugin
)
from .core.types import RetrievedChunk  # For RAGInterface.search return type
from .invocation.invoker import ToolInvoker
from .lookup.service import ToolLookupService
from .rag.manager import RAGManager
from .security.key_provider import KeyProvider
from .tools.manager import ToolManager

# Attempt to import managers, use placeholders if not yet available
try:
    from .llm_providers.manager import LLMProviderManager
    from .llm_providers.types import ChatMessage, LLMChatResponse, LLMCompletionResponse
except ImportError:
    LLMProviderManager = type("LLMProviderManager", (), {}) # Placeholder
    ChatMessage = Dict # Placeholder type
    LLMChatResponse = Any
    LLMCompletionResponse = Any
    logging.debug("LLMProviderManager or its types not found, using placeholders.")

try:
    from .command_processors.manager import CommandProcessorManager
    from .command_processors.types import CommandProcessorResponse
except ImportError:
    CommandProcessorManager = type("CommandProcessorManager", (), {}) # Placeholder
    CommandProcessorResponse = Any # Placeholder type
    logging.debug("CommandProcessorManager or CommandProcessorResponse not found, using placeholders.")

if TYPE_CHECKING: # To avoid circular import issues at runtime but allow type checking
    pass


logger = logging.getLogger(__name__)

class LLMInterface:
    """
    Provides a simplified interface for LLM interactions, managed by Genie.
    """
    def __init__(self, llm_provider_manager: LLMProviderManager, default_provider_id: Optional[str]):
        self._llm_provider_manager = llm_provider_manager
        self._default_provider_id = default_provider_id
        logger.debug(f"LLMInterface initialized with default provider ID: {self._default_provider_id}")

    async def generate(self, prompt: str, provider_id: Optional[str] = None, **kwargs: Any) -> LLMCompletionResponse:
        """Generates text completion using the specified or default LLM provider."""
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for generate.")

        logger.debug(f"LLMInterface: Requesting generation from provider '{provider_to_use}'.")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use) # type: ignore
        if not provider:
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")

        return await provider.generate(prompt, **kwargs) # type: ignore

    async def chat(self, messages: List[ChatMessage], provider_id: Optional[str] = None, **kwargs: Any) -> LLMChatResponse:
        """Generates a chat completion using the specified or default LLM provider."""
        provider_to_use = provider_id or self._default_provider_id
        if not provider_to_use:
            raise ValueError("No LLM provider ID specified and no default is set for chat.")

        logger.debug(f"LLMInterface: Requesting chat completion from provider '{provider_to_use}'.")
        provider = await self._llm_provider_manager.get_llm_provider(provider_to_use) # type: ignore
        if not provider:
            raise RuntimeError(f"LLM Provider '{provider_to_use}' not found or failed to load.")

        return await provider.chat(messages, **kwargs) # type: ignore


class RAGInterface:
    """
    Provides a simplified interface for RAG operations, managed by Genie.
    """
    def __init__(self, rag_manager: RAGManager, config: MiddlewareConfig, key_provider: KeyProvider):
        self._rag_manager = rag_manager
        self._config = config
        self._key_provider = key_provider
        logger.debug("RAGInterface initialized.")

    async def index_directory(
        self,
        path: str,
        collection_name: Optional[str] = None,
        loader_id: Optional[str] = None,
        splitter_id: Optional[str] = None,
        embedder_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Indexes a directory using configured or default RAG components."""
        final_loader_id = loader_id or self._config.default_rag_loader_id or "file_system_loader_v1"
        final_splitter_id = splitter_id or self._config.default_rag_splitter_id or "character_recursive_text_splitter_v1"
        final_embedder_id = embedder_id or self._config.default_rag_embedder_id or "sentence_transformer_embedder_v1"
        final_vector_store_id = vector_store_id or self._config.default_rag_vector_store_id or "faiss_vector_store_v1"

        final_loader_config = loader_config if loader_config is not None else kwargs.get("loader_config_override", {})
        final_splitter_config = splitter_config if splitter_config is not None else kwargs.get("splitter_config_override", {})

        final_embedder_config = embedder_config if embedder_config is not None else kwargs.get("embedder_config_override", {})
        if "key_provider" not in final_embedder_config:
            final_embedder_config["key_provider"] = self._key_provider

        final_vector_store_config = vector_store_config if vector_store_config is not None else kwargs.get("vector_store_config_override", {})
        if collection_name and "collection_name" not in final_vector_store_config:
            final_vector_store_config["collection_name"] = collection_name

        logger.info(f"RAGInterface: Indexing directory '{path}' with "
                    f"Loader='{final_loader_id}', Splitter='{final_splitter_id}', "
                    f"Embedder='{final_embedder_id}', Store='{final_vector_store_id}'.")

        return await self._rag_manager.index_data_source(
            loader_id=final_loader_id,
            loader_source_uri=path,
            splitter_id=final_splitter_id,
            embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config,
            splitter_config=final_splitter_config,
            embedder_config=final_embedder_config,
            vector_store_config=final_vector_store_config,
        )

    async def index_web_page(
        self,
        url: str,
        collection_name: Optional[str] = None,
        loader_id: Optional[str] = None,
        splitter_id: Optional[str] = None,
        embedder_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Indexes a web page using configured or default RAG components."""
        final_loader_id = loader_id or self._config.default_rag_loader_id or "web_page_loader_v1"
        final_splitter_id = splitter_id or self._config.default_rag_splitter_id or "character_recursive_text_splitter_v1"
        final_embedder_id = embedder_id or self._config.default_rag_embedder_id or "sentence_transformer_embedder_v1"
        final_vector_store_id = vector_store_id or self._config.default_rag_vector_store_id or "faiss_vector_store_v1"

        final_loader_config = loader_config if loader_config is not None else kwargs.get("loader_config_override", {})
        final_splitter_config = splitter_config if splitter_config is not None else kwargs.get("splitter_config_override", {})

        final_embedder_config = embedder_config if embedder_config is not None else kwargs.get("embedder_config_override", {})
        if "key_provider" not in final_embedder_config:
            final_embedder_config["key_provider"] = self._key_provider

        final_vector_store_config = vector_store_config if vector_store_config is not None else kwargs.get("vector_store_config_override", {})
        if collection_name and "collection_name" not in final_vector_store_config:
            final_vector_store_config["collection_name"] = collection_name

        logger.info(f"RAGInterface: Indexing web page '{url}' with Loader='{final_loader_id}'.")
        return await self._rag_manager.index_data_source(
            loader_id=final_loader_id,
            loader_source_uri=url,
            splitter_id=final_splitter_id,
            embedder_id=final_embedder_id,
            vector_store_id=final_vector_store_id,
            loader_config=final_loader_config,
            splitter_config=final_splitter_config,
            embedder_config=final_embedder_config,
            vector_store_config=final_vector_store_config,
        )

    async def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = 5,
        retriever_id: Optional[str] = None,
        retriever_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[RetrievedChunk]:
        """Searches the RAG system using configured or default retriever."""
        final_retriever_id = retriever_id or self._config.default_rag_retriever_id or "basic_similarity_retriever_v1"

        final_retriever_config = retriever_config.copy() if retriever_config is not None else {} # type: ignore

        for k, v in kwargs.items():
            if k not in final_retriever_config:
                final_retriever_config[k] = v

        if "embedder_config" not in final_retriever_config:
            final_retriever_config["embedder_config"] = {}
        if "key_provider" not in final_retriever_config["embedder_config"]:
             final_retriever_config["embedder_config"]["key_provider"] = self._key_provider

        if "vector_store_config" not in final_retriever_config:
            final_retriever_config["vector_store_config"] = {}
        if collection_name and "collection_name" not in final_retriever_config["vector_store_config"]:
            final_retriever_config["vector_store_config"]["collection_name"] = collection_name

        logger.info(f"RAGInterface: Searching with query '{query[:50]}...' using retriever '{final_retriever_id}'.")
        return await self._rag_manager.retrieve_from_query(
            query_text=query,
            retriever_id=final_retriever_id,
            retriever_config=final_retriever_config,
            top_k=top_k
        )


class Genie:
    """
    The central facade for interacting with the Genie Tooling middleware.
    It provides simplified access to tools, RAG, LLMs, and command processing.
    """

    def __init__(
        self,
        plugin_manager: PluginManager,
        key_provider: KeyProvider,
        config: MiddlewareConfig,
        tool_manager: ToolManager,
        tool_invoker: ToolInvoker,
        rag_manager: RAGManager,
        tool_lookup_service: ToolLookupService,
        llm_provider_manager: LLMProviderManager,
        command_processor_manager: CommandProcessorManager,
        llm_interface: LLMInterface,
        rag_interface: RAGInterface,
    ):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._config = config
        self._tool_manager = tool_manager
        self._tool_invoker = tool_invoker
        self._rag_manager = rag_manager
        self._tool_lookup_service = tool_lookup_service
        self._llm_provider_manager = llm_provider_manager
        self._command_processor_manager = command_processor_manager

        self.llm = llm_interface
        self.rag = rag_interface

        self._config._genie_instance = self # Allows interfaces to access Genie if needed

        logger.info("Genie facade initialized.")

    @classmethod
    async def create(
        cls,
        config: MiddlewareConfig,
        key_provider_instance: Optional[KeyProvider] = None,
        key_provider_id: Optional[str] = None,
    ) -> Genie:
        """
        Asynchronous factory method to create and initialize a Genie instance.
        """
        if not key_provider_instance and not key_provider_id:
            raise ValueError("Either key_provider_instance or key_provider_id must be provided.")
        if key_provider_instance and key_provider_id:
            logger.warning("Both key_provider_instance and key_provider_id provided. Instance will be used.")

        pm = PluginManager(plugin_dev_dirs=config.plugin_dev_dirs)
        await pm.discover_plugins()

        final_key_provider: KeyProvider
        if key_provider_instance:
            final_key_provider = key_provider_instance
            if isinstance(key_provider_instance, CorePluginType):
                plugin_id = getattr(key_provider_instance, "plugin_id", None)
                if plugin_id and plugin_id not in pm.list_discovered_plugin_classes():
                    pm._discovered_plugin_classes[plugin_id] = type(key_provider_instance) # type: ignore
                    pm._plugin_instances[plugin_id] = key_provider_instance # type: ignore
                    logger.debug(f"Provided KeyProvider instance '{plugin_id}' registered with PluginManager.")
        elif key_provider_id:
            kp_instance_any = await pm.get_plugin_instance(key_provider_id)
            if not kp_instance_any or not isinstance(kp_instance_any, KeyProvider):
                raise RuntimeError(f"Failed to load KeyProvider with ID '{key_provider_id}'.")
            final_key_provider = cast(KeyProvider, kp_instance_any)
        else:
            raise ValueError("Internal error: KeyProvider could not be resolved.")

        logger.info(f"Using KeyProvider: {type(final_key_provider).__name__}")

        tool_manager = ToolManager(plugin_manager=pm)
        tool_configurations = getattr(config, "tool_configurations", {})
        await tool_manager.initialize_tools(tool_configurations=tool_configurations)

        tool_invoker = ToolInvoker(tool_manager=tool_manager, plugin_manager=pm)
        rag_manager = RAGManager(plugin_manager=pm)

        default_lookup_id = config.default_tool_lookup_provider_id or "embedding_similarity_lookup_v1"
        default_formatter_id = config.default_tool_indexing_formatter_id or "llm_compact_text_v1"
        tool_lookup_service = ToolLookupService(
            tool_manager=tool_manager,
            plugin_manager=pm,
            default_provider_id=default_lookup_id,
            default_indexing_formatter_id=default_formatter_id
        )

        llm_provider_manager = LLMProviderManager(pm, final_key_provider, config) # type: ignore

        command_processor_manager = CommandProcessorManager(pm, final_key_provider, config) # type: ignore

        llm_interface = LLMInterface(llm_provider_manager, config.default_llm_provider_id) # type: ignore
        rag_interface = RAGInterface(rag_manager, config, final_key_provider)

        genie_instance = cls(
            plugin_manager=pm,
            key_provider=final_key_provider,
            config=config,
            tool_manager=tool_manager,
            tool_invoker=tool_invoker,
            rag_manager=rag_manager,
            tool_lookup_service=tool_lookup_service,
            llm_provider_manager=llm_provider_manager, # type: ignore
            command_processor_manager=command_processor_manager, # type: ignore
            llm_interface=llm_interface,
            rag_interface=rag_interface
        )

        # Post-initialization update for CommandProcessorManager if it needs the Genie instance
        # The CommandProcessorManager's get_command_processor passes genie_facade to plugin's setup
        # so direct setting on manager might not be strictly needed if plugins are self-contained post-setup.

        logger.info("Genie instance created and core managers initialized.")
        return genie_instance

    async def execute_tool(self, tool_identifier: str, **params: Any) -> Any:
        """
        Invokes a tool by its identifier with the given parameters.
        """
        if not self._tool_invoker:
            raise RuntimeError("ToolInvoker not initialized in Genie.")
        if not self._key_provider:
            raise RuntimeError("KeyProvider not initialized in Genie.")

        logger.info(f"Genie: Executing tool '{tool_identifier}' with params: {params}")
        return await self._tool_invoker.invoke(
            tool_identifier=tool_identifier,
            params=params,
            key_provider=self._key_provider
        )

    async def run_command(
        self,
        command: str,
        processor_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> Any:
        """
        Processes a natural language command using a CommandProcessorPlugin.
        """
        if not self._command_processor_manager:
            logger.error("Genie.run_command: CommandProcessorManager not initialized.")
            return {"error": "CommandProcessorManager not initialized", "tool_result": None}

        target_processor_id = processor_id or self._config.default_command_processor_id
        if not target_processor_id:
            logger.error("Genie.run_command: No command processor ID specified and no default is set in configuration.")
            return {"error": "No command processor configured", "tool_result": None}

        try:
            processor_plugin = await self._command_processor_manager.get_command_processor( # type: ignore
                target_processor_id,
                genie_facade=self
            )
            if not processor_plugin:
                err_msg = f"CommandProcessor plugin '{target_processor_id}' not found or failed to load."
                logger.error(f"Genie.run_command: {err_msg}")
                return {"error": err_msg, "tool_result": None}

            logger.info(f"Genie: Running command '{command[:50]}...' with processor '{target_processor_id}'.")

            cmd_proc_response: CommandProcessorResponse = await processor_plugin.process_command( # type: ignore
                command=command,
                conversation_history=conversation_history
            )

            error_val = cmd_proc_response.get("error") # type: ignore
            thought_val = cmd_proc_response.get("llm_thought_process") # type: ignore
            chosen_tool_id_val = cmd_proc_response.get("chosen_tool_id") # type: ignore
            extracted_params_val = cmd_proc_response.get("extracted_params") # type: ignore

            if error_val:
                logger.error(f"Genie.run_command: Command processing by '{target_processor_id}' failed: {error_val}")
                return {"error": error_val, "tool_result": None, "thought_process": thought_val}

            if chosen_tool_id_val and extracted_params_val is not None:
                logger.info(f"Genie.run_command: Processor '{target_processor_id}' selected tool '{chosen_tool_id_val}' "
                            f"with params: {extracted_params_val}. "
                            f"Thought: {thought_val or 'N/A'}")
                try:
                    tool_result = await self.execute_tool(
                        chosen_tool_id_val,
                        **extracted_params_val
                    )
                    return {"tool_result": tool_result, "thought_process": thought_val}
                except Exception as e_tool_exec:
                    logger.error(f"Genie.run_command: Error executing tool '{chosen_tool_id_val}' after command processing: {e_tool_exec}", exc_info=True)
                    return {"error": str(e_tool_exec), "tool_result": None, "thought_process": thought_val}
            else:
                logger.info(f"Genie.run_command: Processor '{target_processor_id}' did not select a tool or extract parameters. "
                            f"Thought: {thought_val or 'N/A'}")
                return {"message": "No tool selected by command processor.", "thought_process": thought_val}

        except Exception as e_outer:
            logger.error(f"Genie.run_command: Unexpected error during command processing pipeline: {e_outer}", exc_info=True)
            return {"error": f"Unexpected error in run_command: {str(e_outer)}", "tool_result": None}


    async def close(self) -> None:
        """
        Tears down all managed components.
        """
        logger.info("Genie: Initiating teardown...")
        if self._plugin_manager:
            await self._plugin_manager.teardown_all_plugins()
            logger.debug("PluginManager teardown_all_plugins completed.")

        managers_to_teardown = [
            self._llm_provider_manager,
            self._command_processor_manager,
        ]
        for manager in managers_to_teardown:
            if manager and hasattr(manager, "teardown") and callable(getattr(manager, "teardown", None)):
                try:
                    logger.debug(f"Tearing down manager: {type(manager).__name__}")
                    await manager.teardown() # type: ignore
                except Exception as e:
                    logger.error(f"Error tearing down manager {type(manager).__name__}: {e}", exc_info=True)

        self._plugin_manager = None # type: ignore
        self._key_provider = None # type: ignore
        self._config = None # type: ignore
        self._tool_manager = None # type: ignore
        self._tool_invoker = None # type: ignore
        self._rag_manager = None # type: ignore
        self._tool_lookup_service = None # type: ignore
        self._llm_provider_manager = None # type: ignore
        self._command_processor_manager = None # type: ignore
        self.llm = None # type: ignore
        self.rag = None # type: ignore

        logger.info("Genie: Teardown complete.")
