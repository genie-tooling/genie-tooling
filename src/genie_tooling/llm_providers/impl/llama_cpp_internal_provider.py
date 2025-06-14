import asyncio
import functools
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Union, cast

from pydantic import BaseModel as PydanticBaseModel

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatChunkDeltaMessage,
    LLMChatResponse,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.llm_providers.types import (
    ToolCall as GenieToolCall,
)
from genie_tooling.security.key_provider import (
    KeyProvider,
)
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.utils.gbnf import (
    create_dynamic_models_from_dictionaries,
    generate_gbnf_grammar_from_pydantic_models,
)

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama, LlamaGrammar
    from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry
    LLAMA_CPP_PYTHON_AVAILABLE = True
except ImportError:
    Llama = None # type: ignore
    LlamaGrammar = None # type: ignore
    LlamaChatCompletionHandlerRegistry = None # type: ignore
    LLAMA_CPP_PYTHON_AVAILABLE = False
    logger.warning(
        "LlamaCppInternalLLMProviderPlugin: 'llama-cpp-python' library not installed. "
        "This plugin will not be functional. Please install it."
    )

class LlamaCppInternalLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "llama_cpp_internal_llm_provider_v1"
    description: str = "LLM provider for running GGUF models locally using the llama-cpp-python library."

    _model_client: Optional[Llama] = None
    _model_path: Optional[str] = None
    _model_name_for_logging: str = "local_llama_cpp_model"
    _key_provider: Optional[KeyProvider] = None
    _token_usage_manager: Optional[TokenUsageManager] = None

    _n_gpu_layers: int
    _n_ctx: int
    _n_batch: int
    _main_gpu: int
    _tensor_split: Optional[List[float]]
    _seed: int
    _verbose_llama_cpp: bool
    _chat_format: Optional[str]
    _lora_path: Optional[str]
    _lora_base: Optional[str]
    _num_threads: Optional[int]
    _embedding_mode: bool

    # Heuristic mapping from common model name keywords to valid chat format strings
    _MODEL_PATH_TO_CHAT_FORMAT_MAP = {
        "mistral": "mistral-instruct",
        "llama-3": "llama-3",
        "llama-2": "llama-2",
        "gemma": "gemma",
        "qwen": "qwen",
        "chatml": "chatml",
        "zephyr": "zephyr",
        "vicuna": "vicuna",
        "openchat": "openchat",
        "functionary-v2": "functionary-v2",
        "functionary": "functionary", # Catches v1 as well
    }
    _VALID_CHAT_FORMATS = list(LlamaChatCompletionHandlerRegistry()._chat_handlers.keys()) if LLAMA_CPP_PYTHON_AVAILABLE and LlamaChatCompletionHandlerRegistry else []


    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        if not LLAMA_CPP_PYTHON_AVAILABLE or not Llama:
            logger.error(f"{self.plugin_id}: 'llama-cpp-python' library is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._model_path = cfg.get("model_path")
        if not self._model_path:
            logger.info(f"{self.plugin_id}: 'model_path' not provided in configuration. Plugin will be disabled until configured by LLMProviderManager.")
            self._model_client = None
            return

        self._model_name_for_logging = cfg.get("model_name_for_logging", self._model_path.split("/")[-1])
        self._key_provider = cfg.get("key_provider")
        self._token_usage_manager = cfg.get("token_usage_manager")
        self._n_gpu_layers = int(cfg.get("n_gpu_layers", 0))
        self._n_ctx = int(cfg.get("n_ctx", 2048))
        self._n_batch = int(cfg.get("n_batch", 512))
        self._main_gpu = int(cfg.get("main_gpu", 0))
        self._tensor_split = cfg.get("tensor_split")
        self._seed = int(cfg.get("seed", -1))
        self._verbose_llama_cpp = bool(cfg.get("verbose_llama_cpp", False))
        self._lora_path = cfg.get("lora_path")
        self._lora_base = cfg.get("lora_base")
        self._num_threads = cfg.get("num_threads")
        self._embedding_mode = bool(cfg.get("embedding_mode", False))

        # --- Intelligent Chat Format Detection ---
        user_chat_format = cfg.get("chat_format")
        final_chat_format: Optional[str] = None
        model_path_lower = self._model_path.lower()

        if user_chat_format:
            if user_chat_format in self._VALID_CHAT_FORMATS:
                final_chat_format = user_chat_format
                logger.info(f"{self.plugin_id}: Using user-provided valid chat_format: '{final_chat_format}'.")
            else:
                logger.warning(f"{self.plugin_id}: User-provided chat_format '{user_chat_format}' is invalid. Attempting to auto-determine from model path. Valid formats: {self._VALID_CHAT_FORMATS}")

        if not final_chat_format:
            for keyword, valid_format in self._MODEL_PATH_TO_CHAT_FORMAT_MAP.items():
                if keyword in model_path_lower:
                    final_chat_format = valid_format
                    logger.info(f"{self.plugin_id}: Auto-determined chat_format to be '{final_chat_format}' based on model path.")
                    break

        if not final_chat_format:
            fallback_format = "chatml"
            logger.warning(f"{self.plugin_id}: Could not auto-determine chat format from model path. Falling back to '{fallback_format}'. This may not be optimal for your model.")
            final_chat_format = fallback_format

        self._chat_format = final_chat_format
        # --- End of Chat Format Detection ---

        try:
            logger.info(f"{self.plugin_id}: Initializing Llama model from path: {self._model_path}")
            loop = asyncio.get_running_loop()
            self._model_client = await loop.run_in_executor(None, functools.partial(Llama, model_path=self._model_path, n_gpu_layers=self._n_gpu_layers, n_ctx=self._n_ctx, n_batch=self._n_batch, main_gpu=self._main_gpu, tensor_split=self._tensor_split, seed=self._seed, verbose=self._verbose_llama_cpp, chat_format=self._chat_format, lora_path=self._lora_path, lora_base=self._lora_base, n_threads=self._num_threads, embedding=self._embedding_mode))
            logger.info(f"{self.plugin_id}: Llama model '{self._model_path}' loaded successfully.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Llama model: {e}", exc_info=True)
            self._model_client = None

    async def _record_usage_if_manager(self, usage_data: Optional[Dict[str, int]], call_type: str):
        if self._token_usage_manager and usage_data:
            from genie_tooling.token_usage.types import TokenUsageRecord
            record = TokenUsageRecord(provider_id=self.plugin_id, model_name=self._model_name_for_logging, prompt_tokens=usage_data.get("prompt_tokens"), completion_tokens=usage_data.get("completion_tokens"), total_tokens=usage_data.get("total_tokens"), timestamp=asyncio.get_event_loop().time(), call_type=call_type)
            await self._token_usage_manager.record_usage(record)

    async def generate(self, prompt: str, stream: bool = False, **kwargs: Any) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")
        gen_params: Dict[str, Any] = {"max_tokens": kwargs.get("max_tokens", kwargs.get("n_predict", -1)), "temperature": kwargs.get("temperature", 0.8), "top_p": kwargs.get("top_p", 0.95), "top_k": kwargs.get("top_k", 40), "stop": kwargs.get("stop_sequences"), "seed": kwargs.get("seed", self._seed), "repeat_penalty": kwargs.get("repeat_penalty", 1.1)}
        gen_params = {k: v for k, v in gen_params.items() if v is not None}
        output_schema = kwargs.get("output_schema")
        if output_schema and LlamaGrammar:
            try:
                gbnf_grammar_str: Optional[str] = None
                if isinstance(output_schema, type) and issubclass(output_schema, PydanticBaseModel):
                    gbnf_grammar_str = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict):
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    gbnf_grammar_str = generate_gbnf_grammar_from_pydantic_models(dynamic_models) if dynamic_models else None
                if gbnf_grammar_str:
                    gen_params["grammar"] = LlamaGrammar.from_string(gbnf_grammar_str)
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured output via generate().")
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate or apply GBNF grammar: {e_gbnf}", exc_info=True)
        loop = asyncio.get_running_loop()
        if stream:
            async def stream_generator() -> AsyncIterable[LLMCompletionChunk]:
                completion_stream = await loop.run_in_executor(None, functools.partial(self._model_client.create_completion, prompt=prompt, stream=True, **gen_params)) # type: ignore
                full_text = ""
                final_usage_info: Optional[LLMUsageInfo] = None
                for chunk_data in completion_stream: # type: ignore
                    text_delta = chunk_data["choices"][0].get("text", "")
                    full_text += text_delta
                    finish_reason_str = chunk_data["choices"][0].get("finish_reason")
                    current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_data}
                    if finish_reason_str:
                        current_chunk["finish_reason"] = finish_reason_str
                    raw_usage = chunk_data.get("usage")
                    final_usage_info = {"prompt_tokens": raw_usage.get("prompt_tokens"), "completion_tokens": raw_usage.get("completion_tokens"), "total_tokens": raw_usage.get("total_tokens")} if raw_usage else None
                    current_chunk["usage_delta"] = final_usage_info
                    yield current_chunk
                await self._record_usage_if_manager(final_usage_info, "generate_stream_end")
            return stream_generator()
        else:
            completion = await loop.run_in_executor(None, functools.partial(self._model_client.create_completion, prompt=prompt, stream=False, **gen_params)) # type: ignore
            text_content = completion["choices"][0].get("text", "")
            finish_reason = completion["choices"][0].get("finish_reason", "unknown")
            usage_data = completion.get("usage")
            usage_info: Optional[LLMUsageInfo] = None
            if usage_data:
                usage_info = {"prompt_tokens": usage_data.get("prompt_tokens"), "completion_tokens": usage_data.get("completion_tokens"), "total_tokens": usage_data.get("total_tokens")}
            await self._record_usage_if_manager(usage_info, "generate")
            return {"text": text_content, "finish_reason": finish_reason, "usage": usage_info, "raw_response": completion}

    async def chat(self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")
        chat_params: Dict[str, Any] = {"max_tokens": kwargs.get("max_tokens", kwargs.get("n_predict", -1)), "temperature": kwargs.get("temperature", 0.8), "top_p": kwargs.get("top_p", 0.95), "top_k": kwargs.get("top_k", 40), "stop": kwargs.get("stop_sequences"), "seed": kwargs.get("seed", self._seed), "repeat_penalty": kwargs.get("repeat_penalty", 1.1)}
        chat_params = {k: v for k, v in chat_params.items() if v is not None}
        if "tools" in kwargs:
            chat_params["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            chat_params["tool_choice"] = kwargs["tool_choice"]
        output_schema = kwargs.get("output_schema")
        if output_schema and LlamaGrammar:
            try:
                gbnf_grammar_str: Optional[str] = None
                if isinstance(output_schema, type) and issubclass(output_schema, PydanticBaseModel):
                    gbnf_grammar_str = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict):
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    gbnf_grammar_str = generate_gbnf_grammar_from_pydantic_models(dynamic_models) if dynamic_models else None
                if gbnf_grammar_str:
                    chat_params["grammar"] = LlamaGrammar.from_string(gbnf_grammar_str)
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured output via chat().")
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate or apply GBNF grammar for chat: {e_gbnf}", exc_info=True)
        loop = asyncio.get_running_loop()
        if stream:
            async def stream_chat_chunks() -> AsyncIterable[LLMChatChunk]:
                completion_stream = await loop.run_in_executor(None, functools.partial(self._model_client.create_chat_completion, messages=messages, stream=True, **chat_params)) # type: ignore
                final_usage_info: Optional[LLMUsageInfo] = None
                for chunk_data in completion_stream: # type: ignore
                    delta_raw = chunk_data["choices"][0].get("delta", {})
                    delta_message: LLMChatChunkDeltaMessage = {}
                    if "role" in delta_raw:
                        delta_message["role"] = delta_raw["role"]
                    if delta_raw.get("content") is not None:
                        delta_message["content"] = delta_raw["content"]
                    if delta_raw.get("tool_calls"):
                        genie_tool_calls_delta: List[GenieToolCall] = []
                        for tc_delta in delta_raw["tool_calls"]:
                            if tc_delta.get("type") == "function" and tc_delta.get("function"):
                                genie_tool_calls_delta.append({"id": tc_delta.get("id", f"call_{tc_delta.get('index', 'unk')}_{asyncio.get_event_loop().time()}"), "type": "function", "function": {"name": tc_delta["function"].get("name", ""), "arguments": tc_delta["function"].get("arguments", "")}})
                        if genie_tool_calls_delta:
                            delta_message["tool_calls"] = genie_tool_calls_delta
                    current_chunk: LLMChatChunk = {"message_delta": delta_message, "raw_chunk": chunk_data}
                    finish_reason_str = chunk_data["choices"][0].get("finish_reason")
                    if finish_reason_str:
                        current_chunk["finish_reason"] = finish_reason_str
                        raw_usage = chunk_data.get("usage")
                        final_usage_info = {"prompt_tokens": raw_usage.get("prompt_tokens"), "completion_tokens": raw_usage.get("completion_tokens"), "total_tokens": raw_usage.get("total_tokens")} if raw_usage else None
                        current_chunk["usage_delta"] = final_usage_info
                    yield current_chunk
                await self._record_usage_if_manager(final_usage_info, "chat_stream_end")
            return stream_chat_chunks()
        else:
            completion = await loop.run_in_executor(None, functools.partial(self._model_client.create_chat_completion, messages=messages, stream=False, **chat_params)) # type: ignore
            choice = completion["choices"][0]
            raw_assistant_message = choice["message"]
            assistant_message: ChatMessage = {"role": cast(Any, raw_assistant_message.get("role", "assistant"))}
            if raw_assistant_message.get("content") is not None:
                assistant_message["content"] = raw_assistant_message.get("content")
            if raw_assistant_message.get("tool_calls"):
                genie_tool_calls_list: List[GenieToolCall] = []
                for tc_raw in raw_assistant_message["tool_calls"]:
                    if tc_raw.get("type") == "function" and tc_raw.get("function"):
                        genie_tool_calls_list.append({"id": tc_raw.get("id", str(asyncio.get_event_loop().time())), "type": "function", "function": {"name": tc_raw["function"].get("name", "unknown_function"), "arguments": tc_raw["function"].get("arguments", "{}")}})
                if genie_tool_calls_list:
                    assistant_message["tool_calls"] = genie_tool_calls_list
            finish_reason = choice.get("finish_reason", "unknown")
            usage_data = completion.get("usage")
            usage_info: Optional[LLMUsageInfo] = None
            if usage_data:
                usage_info = {"prompt_tokens": usage_data.get("prompt_tokens"), "completion_tokens": usage_data.get("completion_tokens"), "total_tokens": usage_data.get("total_tokens")}
            await self._record_usage_if_manager(usage_info, "chat")
            return {"message": assistant_message, "finish_reason": finish_reason, "usage": usage_info, "raw_response": completion}

    async def get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"provider": "llama.cpp (internal)", "model_path": self._model_path or "Not specified", "n_ctx": self._n_ctx, "n_gpu_layers": self._n_gpu_layers, "chat_format": self._chat_format or "Not specified (model default or raw prompting)"}
        if self._model_client and hasattr(self._model_client, "model_params"):
            try:
                info["llama_cpp_model_params_available"] = True
            except Exception:
                info["llama_cpp_model_params_available"] = False
        return info

    async def teardown(self) -> None:
        if self._model_client:
            logger.info(f"{self.plugin_id}: Releasing Llama model '{self._model_path}'.")
            del self._model_client
            self._model_client = None
        self._key_provider = None
        self._token_usage_manager = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
