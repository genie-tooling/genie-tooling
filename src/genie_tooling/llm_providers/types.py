# src/genie_tooling/llm_providers/types.py
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# --- Tool Calling Structures (used by ChatMessage and LLMChatResponse) ---
class ToolCallFunction(TypedDict):
    """Represents the function to be called within a ToolCall."""
    name: str
    arguments: str # Typically a JSON string of arguments

class ToolCall(TypedDict):
    """
    Represents a tool call requested by the LLM.
    Compatible with OpenAI's tool_calls structure.
    """
    id: str
    type: Literal["function"] # Currently, only 'function' type is widely supported
    function: ToolCallFunction

# --- Chat Message Structure ---
class ChatMessage(TypedDict, total=False):
    """
    Represents a single message in a chat conversation.
    Compatible with OpenAI's ChatCompletion message structure.
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] # Optional for assistant messages with tool_calls, or tool messages
    name: Optional[str] # Optional: The name of the author of this message (e.g. tool function name)
    
    # For assistant messages requesting tool calls
    tool_calls: Optional[List[ToolCall]]
    
    # For tool messages providing results
    tool_call_id: Optional[str] # Required if role is 'tool'

# --- LLM Response Structures ---
class LLMUsageInfo(TypedDict, total=False):
    """Represents token usage information from an LLM response."""
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]

class LLMCompletionResponse(TypedDict):
    """Standardized response for text completion LLM calls."""
    text: str
    finish_reason: Optional[str] # e.g., "stop", "length", "tool_calls"
    usage: Optional[LLMUsageInfo]
    raw_response: Any # The original, unprocessed response from the provider

class LLMChatResponse(TypedDict):
    """Standardized response for chat completion LLM calls."""
    message: ChatMessage # The assistant's response message (or tool message if applicable)
    finish_reason: Optional[str] # e.g., "stop", "length", "tool_calls"
    usage: Optional[LLMUsageInfo]
    raw_response: Any # The original, unprocessed response from the provider