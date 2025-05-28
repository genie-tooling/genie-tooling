### src/genie_tooling/llm_providers/impl/gemini_provider.py
# src/genie_tooling/llm_providers/impl/gemini_provider.py
import asyncio
import functools
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
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

try:
    import google.generativeai as genai
    from google.generativeai.types import (
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

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        await super().setup(config, key_provider)
        if not genai:
            logger.error(f"{self.plugin_id}: 'google-generativeai' library is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gemini-1.5-flash-latest")

        api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"{self.plugin_id}: API key '{self._api_key_name}' not found via KeyProvider.")
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
        for msg_idx, msg in enumerate(messages): # Add index for debugging
            role = GEMINI_ROLE_MAP.get(msg["role"])
            if not role and msg["role"] == "system":
                logger.warning(f"{self.plugin_id}: Converting 'system' role in history to 'user' for Gemini.")
                role = "user"
            if not role:
                logger.warning(f"{self.plugin_id}: Unsupported role '{msg['role']}' for Gemini (msg_idx: {msg_idx}), skipping.")
                continue

            current_message_parts: list = []

            # Handle text content first, common to user, assistant (if no tool_calls), system (converted to user)
            if msg.get("content") is not None:
                # Only add text part if it's not an assistant message that *only* has tool_calls
                if not (msg["role"] == "assistant" and msg.get("tool_calls") and msg.get("content") is None):
                    current_message_parts.append({"text": str(msg["content"])}) # Ensure content is string

            # Handle assistant tool calls
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc_idx, tc in enumerate(msg["tool_calls"]): # type: ignore
                    try:
                        args_str = tc["function"]["arguments"]
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        current_message_parts.append({"function_call": {"name": tc["function"]["name"], "args": args}})
                    except json.JSONDecodeError:
                        logger.error(f"{self.plugin_id}: Could not parse tool_call args for {tc['function']['name']} (msg_idx: {msg_idx}, tc_idx: {tc_idx}).")

            # Handle tool responses
            elif msg["role"] == "tool" and msg.get("tool_call_id") and "content" in msg:
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
                        response_value_for_gemini_sdk = {"output": tool_content} # Store as {"output": "plain string"}
                elif isinstance(tool_content, dict):
                    response_value_for_gemini_sdk = tool_content
                else:
                    response_value_for_gemini_sdk = {"output": tool_content}

                # This part MUST be the only thing added to current_message_parts for a tool role.
                # Ensure no prior text part was added if role is "tool".
                # The structure `if msg.get("content") is not None:` is general.
                # For role "tool", "content" holds the *result* of the tool, not text from the "tool" itself.
                # So, if role is "tool", we should *only* build the function_response part.

                # Overwrite current_message_parts if it's a tool message to ensure only function_response
                current_message_parts = [{
                    "function_response": {
                        "name": str(tool_name),
                        "response": response_value_for_gemini_sdk
                    }
                }]
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
                    tool_calls.append({
                        "id": f"call_{fc_name}_{i}", "type": "function",
                        "function": {"name": fc_name, "arguments": json.dumps(fc_args or {})}
                    })

        chat_msg: ChatMessage = {"role": role}
        if content is not None: chat_msg["content"] = content
        if tool_calls: chat_msg["tool_calls"] = tool_calls

        finish_reason_val = getattr(candidate, "finish_reason", None)
        finish_reason_enum = getattr(finish_reason_val, "value", 0) if finish_reason_val else 0
        finish_reason_str = GEMINI_FINISH_REASON_MAP.get(finish_reason_enum)

        if finish_reason_enum == 6:
            finish_reason_str = "tool_calls"
        elif fn_calls_from_gemini and finish_reason_enum not in [3,4]:
             finish_reason_str = "tool_calls"
        return chat_msg, finish_reason_str

    async def _execute_gemini_request(
        self,
        gemini_formatted_messages: List[_ContentDictType],
        generation_config_args: Dict[str, Any],
        tools_arg: Optional[List[Any]],
        safety_settings_arg: Optional[List[Any]],
        stream: bool
    ) -> _GenerateContentResponseType:
        if not self._model_client:
            raise RuntimeError(f"{self.plugin_id}: Model client not initialized.")
        if not genai:
            raise RuntimeError(f"{self.plugin_id}: Google Generative AI library not available at runtime.")

        generation_config_instance: Optional[_GenerationConfigType] = None
        if _GenerationConfigType is not Any and generation_config_args: # type: ignore
            generation_config_instance = _GenerationConfigType(**generation_config_args) # type: ignore

        partial_func = functools.partial(
            self._model_client.generate_content, # type: ignore
            contents=gemini_formatted_messages,
            generation_config=generation_config_instance,
            tools=tools_arg,
            safety_settings=safety_settings_arg,
            stream=stream
        )
        try:
            loop = asyncio.get_running_loop()
            response: _GenerateContentResponseType = await loop.run_in_executor(
                None,
                partial_func
            )
            return response
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during Gemini API call: {e}", exc_info=True)
            raise RuntimeError(f"Gemini API call failed: {str(e)}") from e


    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        if not self._model_client: raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        msgs = self._convert_messages_to_gemini([{"role": "user", "content": prompt}])
        cfg_args = {k:v for k,v in kwargs.items() if k in ["temperature","top_p","top_k","max_output_tokens","stop_sequences","candidate_count"]}
        api_resp = await self._execute_gemini_request(msgs, cfg_args, None, kwargs.get("safety_settings"), False)

        text, reason, usage, raw = "", "unknown", None, {}
        candidates = getattr(api_resp, "candidates", [])
        if candidates:
            chat_msg, reason = self._parse_gemini_candidate(candidates[0])
            text = chat_msg.get("content") or ""
        elif pf := getattr(api_resp, "prompt_feedback", None):
            if br := getattr(pf, "block_reason", None):
                br_name = getattr(br, "name", "UNKNOWN_BLOCK")
                reason, text = f"blocked: {br_name}", f"[Blocked: {br_name}]"

        if um := getattr(api_resp, "usage_metadata", None):
            usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count} # type: ignore

        raw = api_resp.to_dict() if hasattr(api_resp, "to_dict") else str(api_resp) # type: ignore
        return {"text": text, "finish_reason": reason, "usage": usage, "raw_response": raw} # type: ignore

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        if not self._model_client: raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        msgs = self._convert_messages_to_gemini(messages)
        cfg_args = {k:v for k,v in kwargs.items() if k in ["temperature","top_p","top_k","max_output_tokens","stop_sequences","candidate_count"]}
        api_resp = await self._execute_gemini_request(msgs, cfg_args, kwargs.get("tools"), kwargs.get("safety_settings"), False)

        chat_msg: ChatMessage = {"role": "assistant", "content": ""}
        reason, usage, raw = "unknown", None, {}
        candidates = getattr(api_resp, "candidates", [])
        if candidates:
            chat_msg, reason = self._parse_gemini_candidate(candidates[0])
        elif pf := getattr(api_resp, "prompt_feedback", None):
            if br := getattr(pf, "block_reason", None):
                br_name = getattr(br, "name", "UNKNOWN_BLOCK")
                reason = f"blocked: {br_name}"
                chat_msg["content"] = f"[Chat blocked: {br_name}]"

        if um := getattr(api_resp, "usage_metadata", None):
            usage = {"prompt_tokens": um.prompt_token_count, "completion_tokens": um.candidates_token_count, "total_tokens": um.total_token_count} # type: ignore

        raw = api_resp.to_dict() if hasattr(api_resp, "to_dict") else str(api_resp) # type: ignore
        return {"message": chat_msg, "finish_reason": reason, "usage": usage, "raw_response": raw} # type: ignore

    async def get_model_info(self) -> Dict[str, Any]:
        if not genai or not self._model_client:
            return {"error": "Gemini client not initialized or library not available."}
        info: Dict[str, Any] = {"provider": "Google Gemini", "configured_model_name": self._model_name}
        try:
            loop = asyncio.get_running_loop()
            model_info_sdk = await loop.run_in_executor(None, genai.get_model, f"models/{self._model_name}")
            if model_info_sdk:
                info.update({
                    "display_name": getattr(model_info_sdk, "display_name", None),
                    "version": getattr(model_info_sdk, "version", None),
                    "input_token_limit": getattr(model_info_sdk, "input_token_limit", None),
                    "output_token_limit": getattr(model_info_sdk, "output_token_limit", None),
                    "supported_generation_methods": getattr(model_info_sdk, "supported_generation_methods", None)
                })
            else: info["error"] = "Could not retrieve model info from SDK."
        except Exception as e:
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        self._model_client = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
###<END-OF-FILE>###
