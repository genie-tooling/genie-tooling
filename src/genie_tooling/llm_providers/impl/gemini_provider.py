import asyncio
import json
import logging
from typing import Any, AsyncIterable, Dict, List, Literal, Optional, Tuple, Union, cast

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
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

_GenerativeModelTypePlaceholder: Any = Any
_GenerationConfigType: Any = Any
_ContentDictType: Any = Any
_GeminiSDKToolType: Any = Any
_FunctionDeclarationType: Any = Any
_GenerateContentResponseType: Any = Any
_CandidateType: Any = Any
_AsyncGenerateContentResponseType: Any = Any

try:
    import google.generativeai as genai
    from google.generativeai.types import (
        AsyncGenerateContentResponse,
        ContentDict,
        FunctionDeclaration,
        GenerationConfig,
    )
    from google.generativeai.types import Tool as GeminiSDKTool
    from google.generativeai.types.generation_types import (
        Candidate,
        GenerateContentResponse,
    )

    _GenerativeModelTypePlaceholder = genai.GenerativeModel
    _GenerationConfigType = GenerationConfig
    _ContentDictType = ContentDict
    _GeminiSDKToolType = GeminiSDKTool
    _FunctionDeclarationType = FunctionDeclaration
    _GenerateContentResponseType = GenerateContentResponse
    _CandidateType = Candidate
    _AsyncGenerateContentResponseType = AsyncGenerateContentResponse
except ImportError:
    genai = None
    logger.warning(
        "GeminiLLMProviderPlugin: 'google-generativeai' library not installed. "
        "This plugin will not be functional. Please install it: pip install google-generativeai"
    )

GEMINI_ROLE_MAP = {
    "user": "user", "assistant": "model", "system": "user", "tool": "tool",
}
GEMINI_FINISH_REASON_MAP = {
    1: "stop", 2: "length", 3: "safety", 4: "recitation",
    0: "unknown", 5: "other", 6: "tool_calls"
}

class GeminiLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "gemini_llm_provider_v1"
    description: str = "LLM provider for Google's Gemini models using the google-generativeai library."

    _model_client: Optional[_GenerativeModelTypePlaceholder] = None
    _model_name: str
    _api_key_name: str = "GOOGLE_API_KEY"
    _key_provider: Optional[KeyProvider] = None


    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        if not genai:
            logger.error(f"{self.plugin_id}: 'google-generativeai' library is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._key_provider = cfg.get("key_provider")
        if not self._key_provider or not isinstance(self._key_provider, KeyProvider):
            logger.error(f"{self.plugin_id}: KeyProvider not found in config or is invalid. Cannot fetch API key.")
            return

        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gemini-1.5-flash-latest")

        api_key = await self._key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.info(f"{self.plugin_id}: API key '{self._api_key_name}' not found. Plugin will be disabled.")
            self._model_client = None
            return

        try:
            genai.configure(api_key=api_key)
            system_instruction_text = cfg.get("system_instruction")
            safety_settings_config = cfg.get("safety_settings")
            self._model_client = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system_instruction_text,
                safety_settings=safety_settings_config
            )
            logger.info(f"{self.plugin_id}: Initialized Gemini client for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Gemini client: {e}", exc_info=True)
            self._model_client = None

    def _convert_messages_to_gemini(self, messages: List[ChatMessage]) -> List[_ContentDictType]:
        gemini_messages: List[_ContentDictType] = []
        for msg_idx, msg in enumerate(messages):
            role = GEMINI_ROLE_MAP.get(msg["role"])
            if not role and msg["role"] == "system":
                logger.warning(f"{self.plugin_id}: Converting 'system' role in history to 'user' for Gemini.")
                role = "user"
            if not role:
                logger.warning(f"{self.plugin_id}: Unsupported role '{msg['role']}' for Gemini (msg_idx: {msg_idx}), skipping.")
                continue

            current_message_parts: list = []
            raw_content = msg.get("content")
            if raw_content is not None:
                content_str = str(raw_content).strip()
                if content_str:
                    if not (msg["role"] == "assistant" and msg.get("tool_calls") and raw_content is None):
                        current_message_parts.append({"text": content_str})

            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc_idx, tc in enumerate(msg["tool_calls"]): # type: ignore
                    try:
                        args_str = tc["function"]["arguments"]
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        current_message_parts.append({"function_call": {"name": tc["function"]["name"], "args": args}})
                    except json.JSONDecodeError:
                        logger.error(f"{self.plugin_id}: Could not parse tool_call args for {tc['function']['name']} (msg_idx: {msg_idx}, tc_idx: {tc_idx}). Using empty args.")
                        current_message_parts.append({"function_call": {"name": tc["function"]["name"], "args": {}}})
            elif msg["role"] == "tool" and msg.get("tool_call_id"):
                tool_name = msg.get("name", msg.get("tool_call_id"))
                tool_content = msg.get("content")
                response_value_for_gemini_sdk: Dict[str, Any]
                if isinstance(tool_content, str):
                    try:
                        parsed_json_content = json.loads(tool_content)
                        if not isinstance(parsed_json_content, dict):
                            response_value_for_gemini_sdk = {"output": parsed_json_content}
                        else:
                            response_value_for_gemini_sdk = parsed_json_content
                    except json.JSONDecodeError:
                        response_value_for_gemini_sdk = {"output": tool_content}
                elif isinstance(tool_content, dict):
                    response_value_for_gemini_sdk = tool_content
                else:
                    response_value_for_gemini_sdk = {"output": tool_content}
                current_message_parts = [{"function_response": {"name": str(tool_name), "response": response_value_for_gemini_sdk}}]
                logger.debug(f"{self.plugin_id}: Processed tool message (msg_idx: {msg_idx}). Parts: {current_message_parts}")

            if current_message_parts:
                gemini_messages.append(cast(_ContentDictType, {"role": role, "parts": current_message_parts}))
            else:
                logger.warning(f"{self.plugin_id}: Message (msg_idx: {msg_idx}, role: {msg['role']}) resulted in empty parts. Skipping.")
        return gemini_messages

    def _parse_gemini_candidate(self, candidate: _CandidateType) -> Tuple[ChatMessage, Optional[str]]:
        role: Literal["assistant"] = "assistant"
        content: Optional[str] = None
        tool_calls: Optional[List[GenieToolCall]] = None
        fn_calls_from_gemini = []

        candidate_content = getattr(candidate, "content", None)
        if candidate_content and getattr(candidate_content, "parts", None):
            text_parts = [part.text for part in candidate_content.parts if hasattr(part, "text") and part.text is not None]
            if text_parts: content = " ".join(text_parts)
            fn_calls_from_gemini = [p.function_call for p in candidate_content.parts if hasattr(p, "function_call") and p.function_call]
            if fn_calls_from_gemini:
                tool_calls = []
                for i, fc in enumerate(fn_calls_from_gemini):
                    fc_name = getattr(fc, "name", "unknown_fn")
                    fc_args = getattr(fc, "args", {})
                    tool_calls.append({"id": f"call_{fc_name}_{i}", "type": "function", "function": {"name": fc_name, "arguments": json.dumps(fc_args or {})}})
        chat_msg: ChatMessage = {"role": role}
        if content is not None: chat_msg["content"] = content
        elif tool_calls: chat_msg["content"] = None
        if tool_calls: chat_msg["tool_calls"] = tool_calls
        finish_reason_val = getattr(candidate, "finish_reason", None)
        finish_reason_enum = getattr(finish_reason_val, "value", 0) if finish_reason_val else 0
        finish_reason_str = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum)
        if finish_reason_enum == 6: finish_reason_str = "tool_calls"
        elif fn_calls_from_gemini and finish_reason_enum not in [3,4]: finish_reason_str = "tool_calls"
        return chat_msg, finish_reason_str

    async def _execute_gemini_request(self, gemini_formatted_messages: List[_ContentDictType], generation_config_args: Dict[str, Any], tools_arg: Optional[List[Any]], safety_settings_arg: Optional[List[Any]], stream: bool) -> Union[_GenerateContentResponseType, _AsyncGenerateContentResponseType]:
        if not self._model_client: raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")
        if not genai: raise RuntimeError(f"{self.plugin_id}: Google Generative AI library not available at runtime.")
        generation_config_instance: Optional[_GenerationConfigType] = None
        if _GenerationConfigType is not Any and generation_config_args: generation_config_instance = _GenerationConfigType(**generation_config_args)
        try:
            response = await self._model_client.generate_content_async(contents=gemini_formatted_messages, generation_config=generation_config_instance, tools=tools_arg, safety_settings=safety_settings_arg, stream=stream)
            return response
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during Gemini API call: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e

    async def generate(self, prompt: str, stream: bool = False, **kwargs: Any) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        if not self._model_client: raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        msgs = self._convert_messages_to_gemini([{"role": "user", "content": prompt}])
        cfg_args = {k:v for k,v in kwargs.items() if k in ["temperature","top_p","top_k","max_output_tokens","stop_sequences","candidate_count"]}
        api_resp_any = await self._execute_gemini_request(msgs, cfg_args, None, kwargs.get("safety_settings"), stream)
        if stream:
            if not isinstance(api_resp_any, AsyncIterable): raise RuntimeError("Expected AsyncIterable for streaming generate from Gemini")
            api_resp_stream = cast(_AsyncGenerateContentResponseType, api_resp_any)
            async def stream_completion_chunks() -> AsyncIterable[LLMCompletionChunk]:
                final_usage: Optional[LLMUsageInfo] = None; final_raw_response: Optional[Dict[str, Any]] = None
                async for chunk_resp_item in api_resp_stream:
                    text_delta = getattr(chunk_resp_item, "text", "")
                    chunk_finish_reason: Optional[str] = None
                    candidates = getattr(chunk_resp_item, "candidates", [])
                    if candidates:
                        finish_reason_val = getattr(candidates[0], "finish_reason", None)
                        finish_reason_enum = getattr(finish_reason_val, "value", 0) if finish_reason_val else 0
                        chunk_finish_reason = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum)
                    current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_resp_item.to_dict() if hasattr(chunk_resp_item, "to_dict") else str(chunk_resp_item)}
                    if chunk_finish_reason:
                        current_chunk["finish_reason"] = chunk_finish_reason
                        if hasattr(chunk_resp_item, "usage_metadata"):
                            um = getattr(chunk_resp_item, "usage_metadata", None)
                            if um: final_usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count}; current_chunk["usage_delta"] = final_usage
                            final_raw_response = chunk_resp_item.to_dict() if hasattr(chunk_resp_item, "to_dict") else str(chunk_resp_item)
                    yield current_chunk
            return stream_completion_chunks()
        else:
            api_resp_non_stream = cast(_GenerateContentResponseType, api_resp_any)
            text, reason, usage, raw = "", "unknown", None, {}
            candidates = getattr(api_resp_non_stream, "candidates", [])
            if candidates: chat_msg, reason = self._parse_gemini_candidate(candidates[0]); text = chat_msg.get("content") or ""
            elif pf := getattr(api_resp_non_stream, "prompt_feedback", None):
                if br := getattr(pf, "block_reason", None): br_name = getattr(br, "name", "UNKNOWN_BLOCK"); reason, text = f"blocked: {br_name}", f"[Blocked: {br_name}]"
            if um := getattr(api_resp_non_stream, "usage_metadata", None): usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count}
            raw = api_resp_non_stream.to_dict() if hasattr(api_resp_non_stream, "to_dict") else str(api_resp_non_stream)
            return {"text": text, "finish_reason": reason, "usage": usage, "raw_response": raw}

    async def chat(self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        if not self._model_client: raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        msgs = self._convert_messages_to_gemini(messages)
        cfg_args = {k:v for k,v in kwargs.items() if k in ["temperature","top_p","top_k","max_output_tokens","stop_sequences","candidate_count"]}
        tools_for_api = kwargs.get("tools"); safety_settings_for_api = kwargs.get("safety_settings")
        api_resp_any = await self._execute_gemini_request(msgs, cfg_args, tools_for_api, safety_settings_for_api, stream)
        if stream:
            if not isinstance(api_resp_any, AsyncIterable): raise RuntimeError("Expected AsyncIterable for streaming chat from Gemini")
            api_resp_stream = cast(_AsyncGenerateContentResponseType, api_resp_any)
            async def stream_chat_chunks() -> AsyncIterable[LLMChatChunk]:
                final_usage: Optional[LLMUsageInfo] = None; final_raw_response: Optional[Dict[str, Any]] = None
                async for chunk_resp_item in api_resp_stream:
                    delta_msg_content: Optional[str] = None; delta_tool_calls: Optional[List[GenieToolCall]] = None
                    candidates = getattr(chunk_resp_item, "candidates", []); chunk_finish_reason: Optional[str] = None
                    if candidates:
                        candidate_content = getattr(candidates[0], "content", None)
                        if candidate_content and getattr(candidate_content, "parts", None):
                            text_deltas = [part.text for part in candidate_content.parts if hasattr(part, "text") and part.text is not None]
                            if text_deltas: delta_msg_content = "".join(text_deltas)
                            fn_calls_from_gemini = [p.function_call for p in candidate_content.parts if hasattr(p, "function_call") and p.function_call]
                            if fn_calls_from_gemini:
                                delta_tool_calls = []
                                for i, fc in enumerate(fn_calls_from_gemini):
                                    fc_name = getattr(fc, "name", "unknown_fn"); fc_args = getattr(fc, "args", {})
                                    delta_tool_calls.append({"id": f"call_{fc_name}_{i}", "type": "function", "function": {"name": fc_name, "arguments": json.dumps(fc_args or {})}})
                        finish_reason_val = getattr(candidates[0], "finish_reason", None)
                        finish_reason_enum = getattr(finish_reason_val, "value", 0) if finish_reason_val else 0
                        chunk_finish_reason = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum)
                        if finish_reason_enum == 6: chunk_finish_reason = "tool_calls"
                        elif fn_calls_from_gemini and finish_reason_enum not in [3,4]: chunk_finish_reason = "tool_calls"
                    delta_message_obj: LLMChatChunkDeltaMessage = {"role": "assistant"}
                    if delta_msg_content is not None: delta_message_obj["content"] = delta_msg_content
                    if delta_tool_calls: delta_message_obj["tool_calls"] = delta_tool_calls
                    current_chunk: LLMChatChunk = {"message_delta": delta_message_obj, "raw_chunk": chunk_resp_item.to_dict() if hasattr(chunk_resp_item, "to_dict") else str(chunk_resp_item)}
                    if chunk_finish_reason:
                        current_chunk["finish_reason"] = chunk_finish_reason
                        if hasattr(chunk_resp_item, "usage_metadata"):
                            um = getattr(chunk_resp_item, "usage_metadata", None)
                            if um: final_usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count}; current_chunk["usage_delta"] = final_usage
                            final_raw_response = chunk_resp_item.to_dict() if hasattr(chunk_resp_item, "to_dict") else str(chunk_resp_item)
                    yield current_chunk
            return stream_chat_chunks()
        else:
            api_resp_non_stream = cast(_GenerateContentResponseType, api_resp_any)
            chat_msg: ChatMessage = {"role": "assistant", "content": ""}; reason, usage, raw = "unknown", None, {}
            candidates = getattr(api_resp_non_stream, "candidates", [])
            if candidates: chat_msg, reason = self._parse_gemini_candidate(candidates[0])
            elif pf := getattr(api_resp_non_stream, "prompt_feedback", None):
                if br := getattr(pf, "block_reason", None): br_name = getattr(br, "name", "UNKNOWN_BLOCK"); reason = f"blocked: {br_name}"; chat_msg["content"] = f"[Chat blocked: {br_name}]"
            if um := getattr(api_resp_non_stream, "usage_metadata", None): usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count}
            raw = api_resp_non_stream.to_dict() if hasattr(api_resp_non_stream, "to_dict") else str(api_resp_non_stream)
            return {"message": chat_msg, "finish_reason": reason, "usage": usage, "raw_response": raw}

    async def get_model_info(self) -> Dict[str, Any]:
        if not genai or not self._model_client: return {"error": "Gemini client not initialized or library not available."}
        info: Dict[str, Any] = {"provider": "Google Gemini", "configured_model_name": self._model_name}
        try:
            loop = asyncio.get_running_loop()
            model_info_sdk = await loop.run_in_executor(None, genai.get_model, f"models/{self._model_name}")
            if model_info_sdk: info.update({"display_name": getattr(model_info_sdk, "display_name", None), "version": getattr(model_info_sdk, "version", None), "input_token_limit": getattr(model_info_sdk, "input_token_limit", None), "output_token_limit": getattr(model_info_sdk, "output_token_limit", None), "supported_generation_methods": getattr(model_info_sdk, "supported_generation_methods", None)})
            else: info["error"] = "Could not retrieve model info from SDK."
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not retrieve detailed model info for '{self._model_name}': {e}", exc_info=False)
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        self._model_client = None
        self._key_provider = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
