"""Native Slack tool (Phase 6B.3.1).

Why native instead of wrapping the official MCP server: the SRE/dev
weekly-async use case needs **threaded progress updates** from a
long-running agent (see ProgressSinkPlugin and a forthcoming Slack-thread
sink). The MCP server's post-message tool is fire-and-forget per call,
which makes weaving an evolving thread into agent progress streaming
awkward. Owning the boundary here lets the Slack tool, the progress
sink, and the audit-attestation field cooperate cleanly.

Implements a small surface area focused on the recurring corporate
flows:

* ``post_message`` — post a message (optionally as a threaded reply).
* ``add_reaction`` — emoji react to a message.
* ``get_user_profile`` — look up a user by id (for OOO detection).
* ``list_channels`` — list channels (read-only, cacheable).
* ``get_channel_history`` — read recent messages from a channel.

All operations route through ``https://slack.com/api/...`` with a Bot
Token retrieved via the configured ``KeyProvider`` (key name
``SLACK_BOT_TOKEN`` by default, overridable via config). HTTP errors
raise; Slack `ok: false` responses are surfaced as ``{"error": ...}``.

Side-effects metadata:

* ``post_message`` / ``add_reaction``: ``write``, ``requires_approval``
  defers to the policy (defaults to ``ask`` under the Claude-Code model).
* ``list_channels`` / ``get_user_profile`` / ``get_channel_history``:
  ``read``, idempotent, cacheable.

Each writeable operation returns an ``audit_artifact`` containing the
exact API endpoint and the request body (with the bot token redacted)
so Phase 6C.6 captures the ground-truth artifact.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

SLACK_API = "https://slack.com/api"


# ---------------------------------------------------------------------------
# Per-operation tool classes — each one is a Genie Tool with declared
# side-effects so the policy plugin can gate them appropriately.
# ---------------------------------------------------------------------------


class _SlackToolBase(Tool, Plugin):
    """Shared HTTP / auth machinery for Slack tools."""

    _key_name: str = "SLACK_BOT_TOKEN"
    _client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._key_name = cfg.get("api_key_name", "SLACK_BOT_TOKEN")
        self._client = httpx.AsyncClient(timeout=float(cfg.get("timeout_seconds", 30.0)))

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _post(
        self, endpoint: str, body: Dict[str, Any], key_provider: KeyProvider
    ) -> Dict[str, Any]:
        token = await key_provider.get_key(self._key_name)
        if not token:
            return {"error": f"Slack bot token '{self._key_name}' is not configured."}
        if self._client is None:
            return {"error": "Slack tool not initialized."}
        url = f"{SLACK_API}/{endpoint}"
        try:
            resp = await self._client.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )
            data = resp.json() if resp.content else {}
        except Exception as e:
            logger.error(f"Slack {endpoint}: HTTP error: {e}", exc_info=True)
            return {"error": f"HTTP error: {e!s}"}
        if not data.get("ok", False):
            return {"error": data.get("error", "unknown"), "raw": data}
        return data

    async def _get(
        self, endpoint: str, params: Dict[str, Any], key_provider: KeyProvider
    ) -> Dict[str, Any]:
        token = await key_provider.get_key(self._key_name)
        if not token:
            return {"error": f"Slack bot token '{self._key_name}' is not configured."}
        if self._client is None:
            return {"error": "Slack tool not initialized."}
        url = f"{SLACK_API}/{endpoint}"
        try:
            resp = await self._client.get(
                url, params=params, headers={"Authorization": f"Bearer {token}"}
            )
            data = resp.json() if resp.content else {}
        except Exception as e:
            logger.error(f"Slack {endpoint}: HTTP error: {e}", exc_info=True)
            return {"error": f"HTTP error: {e!s}"}
        if not data.get("ok", False):
            return {"error": data.get("error", "unknown"), "raw": data}
        return data


class SlackPostMessageTool(_SlackToolBase):
    plugin_id: str = "slack_post_message_v1"

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return "slack_post_message"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Slack: Post Message",
            "description_human": "Post a message to a Slack channel (optionally in a thread).",
            "description_llm": "Post a Slack message to `channel` with `text`. Optional `thread_ts` to reply in a thread.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string", "description": "Channel ID or name (e.g. C0123 or #engineering)."},
                    "text": {"type": "string", "description": "Message text."},
                    "thread_ts": {"type": ["string", "null"], "description": "Parent message ts to reply in a thread."},
                    "blocks": {"type": ["array", "null"], "description": "Optional Slack Block Kit blocks."},
                },
                "required": ["channel", "text"],
            },
            "output_schema": {"type": "object"},
            "key_requirements": [{"name": "SLACK_BOT_TOKEN", "description": "Slack Bot User OAuth token."}],
            "tags": ["slack", "chat", "write"],
            "version": "1.0.0",
            "side_effects": "write",
            "requires_approval": None,  # defer to policy
            "idempotent": False,
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        body: Dict[str, Any] = {"channel": params["channel"], "text": params["text"]}
        if params.get("thread_ts"):
            body["thread_ts"] = params["thread_ts"]
        if params.get("blocks"):
            body["blocks"] = params["blocks"]
        result = await self._post("chat.postMessage", body, key_provider)
        # Phase 6C.6 audit attestation: capture the exact API request sent.
        result["audit_artifact"] = {"endpoint": "chat.postMessage", "request_body": body}
        return result


class SlackAddReactionTool(_SlackToolBase):
    plugin_id: str = "slack_add_reaction_v1"

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return "slack_add_reaction"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Slack: Add Reaction",
            "description_human": "Add an emoji reaction to a Slack message.",
            "description_llm": "React to a Slack message with an emoji name (without colons).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "timestamp": {"type": "string", "description": "Message ts."},
                    "name": {"type": "string", "description": "Emoji name without colons (e.g. 'thumbsup')."},
                },
                "required": ["channel", "timestamp", "name"],
            },
            "output_schema": {"type": "object"},
            "key_requirements": [{"name": "SLACK_BOT_TOKEN", "description": "Slack Bot OAuth token."}],
            "tags": ["slack", "chat"],
            "version": "1.0.0",
            "side_effects": "write",
            "requires_approval": None,
            "idempotent": True,
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        body = {"channel": params["channel"], "timestamp": params["timestamp"], "name": params["name"]}
        result = await self._post("reactions.add", body, key_provider)
        result["audit_artifact"] = {"endpoint": "reactions.add", "request_body": body}
        return result


class SlackGetUserProfileTool(_SlackToolBase):
    plugin_id: str = "slack_get_user_profile_v1"

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return "slack_get_user_profile"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Slack: Get User Profile",
            "description_human": "Look up a Slack user's profile by user id.",
            "description_llm": "Fetch a Slack user's profile (display_name, real_name, status_text).",
            "input_schema": {
                "type": "object",
                "properties": {"user": {"type": "string", "description": "Slack user id (e.g. U0123)."}},
                "required": ["user"],
            },
            "output_schema": {"type": "object"},
            "key_requirements": [{"name": "SLACK_BOT_TOKEN", "description": "Slack Bot OAuth token."}],
            "tags": ["slack", "read"],
            "version": "1.0.0",
            "side_effects": "read",
            "requires_approval": False,
            "idempotent": True,
            "cacheable": True,
            "cache_ttl_seconds": 600,
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        return await self._get("users.profile.get", {"user": params["user"]}, key_provider)


class SlackListChannelsTool(_SlackToolBase):
    plugin_id: str = "slack_list_channels_v1"

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return "slack_list_channels"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Slack: List Channels",
            "description_human": "List Slack channels the bot has access to.",
            "description_llm": "List Slack channels (id, name). Use `cursor` for pagination.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Page size (default 200, max 1000).", "default": 200},
                    "cursor": {"type": ["string", "null"]},
                    "types": {"type": "string", "description": "Comma-separated channel types.", "default": "public_channel"},
                },
            },
            "output_schema": {"type": "object"},
            "key_requirements": [{"name": "SLACK_BOT_TOKEN", "description": "Slack Bot OAuth token."}],
            "tags": ["slack", "read"],
            "version": "1.0.0",
            "side_effects": "read",
            "requires_approval": False,
            "idempotent": True,
            "cacheable": True,
            "cache_ttl_seconds": 300,
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        api_params = {
            "limit": int(params.get("limit", 200)),
            "types": params.get("types", "public_channel"),
        }
        if params.get("cursor"):
            api_params["cursor"] = params["cursor"]
        return await self._get("conversations.list", api_params, key_provider)


class SlackGetChannelHistoryTool(_SlackToolBase):
    plugin_id: str = "slack_get_channel_history_v1"

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return "slack_get_channel_history"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Slack: Get Channel History",
            "description_human": "Read recent messages from a Slack channel.",
            "description_llm": "Read up to `limit` recent messages from a Slack channel.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "limit": {"type": "integer", "default": 50},
                    "oldest": {"type": ["string", "null"], "description": "Unix timestamp (sec) lower bound."},
                    "latest": {"type": ["string", "null"], "description": "Unix timestamp (sec) upper bound."},
                },
                "required": ["channel"],
            },
            "output_schema": {"type": "object"},
            "key_requirements": [{"name": "SLACK_BOT_TOKEN", "description": "Slack Bot OAuth token."}],
            "tags": ["slack", "read"],
            "version": "1.0.0",
            "side_effects": "read",
            "requires_approval": False,
            "idempotent": True,
            "cacheable": False,  # History changes fast — cache by caller, not here.
        }

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        api_params: Dict[str, Any] = {
            "channel": params["channel"],
            "limit": int(params.get("limit", 50)),
        }
        if params.get("oldest"):
            api_params["oldest"] = params["oldest"]
        if params.get("latest"):
            api_params["latest"] = params["latest"]
        return await self._get("conversations.history", api_params, key_provider)


# ---------------------------------------------------------------------------
# Slack thread progress sink — pairs nicely with the Slack tools above.
# (See genie_tooling/progress/impl for the sink-side; this is just a notice.)
# ---------------------------------------------------------------------------


class SlackThreadProgressSinkPlugin(_SlackToolBase, Plugin):
    """ProgressSinkPlugin that pushes agent progress events as threaded
    Slack messages. Configure with the channel + thread root timestamp;
    each progress event becomes a thread reply.

    This plugin implements ``emit(event)`` (the ProgressSinkPlugin
    protocol) but also subclasses _SlackToolBase to reuse the HTTP/auth
    machinery. The Genie plugin registration treats it as a normal
    plugin; it's loaded via the ``ProgressSinkPlugin`` protocol check
    by callers.
    """

    plugin_id: str = "slack_thread_progress_sink_v1"
    description: str = "ProgressSinkPlugin that pushes agent progress as threaded Slack messages."

    _channel: Optional[str] = None
    _thread_ts: Optional[str] = None
    _key_provider: Optional[KeyProvider] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        await super().setup(cfg)
        self._channel = cfg.get("channel")
        self._thread_ts = cfg.get("thread_ts")
        self._key_provider = cfg.get("key_provider")
        if not self._channel:
            logger.warning(f"{self.plugin_id}: no 'channel' configured; events will be dropped.")

    async def emit(self, event: Any) -> None:
        if not self._channel or not self._key_provider:
            return
        text = f"`{event.phase}` {event.message}"
        if event.iteration is not None:
            text = f"[i{event.iteration}] " + text
        body: Dict[str, Any] = {"channel": self._channel, "text": text}
        if self._thread_ts:
            body["thread_ts"] = self._thread_ts
        await self._post("chat.postMessage", body, self._key_provider)
