"""WebhookProgressSinkPlugin — POSTs each event to a configured URL.

The receiver is expected to render the progress somewhere the operator sees
(Slack interactive message thread, a UI element, a chat channel, etc.).
Failures are logged and dropped — progress is best-effort, not a hard
dependency of the agent loop.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import httpx

from genie_tooling.progress.abc import ProgressSinkPlugin
from genie_tooling.progress.types import ProgressEvent

logger = logging.getLogger(__name__)


class WebhookProgressSinkPlugin(ProgressSinkPlugin):
    plugin_id: str = "webhook_progress_sink_v1"
    description: str = (
        "POSTs progress events to a configured URL. Useful for Slack interactive "
        "message thread updates, custom dashboards, or any HTTP-receiving operator UI."
    )

    _url: Optional[str] = None
    _headers: Dict[str, str]
    _timeout_seconds: float
    _client: Optional[httpx.AsyncClient] = None
    _fire_and_forget: bool

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._url = cfg.get("url")
        self._headers = dict(cfg.get("headers", {}))
        self._timeout_seconds = float(cfg.get("timeout_seconds", 10.0))
        self._fire_and_forget = bool(cfg.get("fire_and_forget", True))
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        if not self._url:
            logger.warning(f"{self.plugin_id}: no 'url' configured; events will be dropped.")

    async def emit(self, event: ProgressEvent) -> None:
        if not self._url or not self._client:
            return
        payload = asdict(event)
        if self._fire_and_forget:
            # Don't block the agent on slow webhook receivers.
            asyncio.create_task(self._post_safely(payload))
        else:
            await self._post_safely(payload)

    async def _post_safely(self, payload: Dict[str, Any]) -> None:
        try:
            await self._client.post(self._url, json=payload, headers=self._headers)
        except Exception as e:
            logger.error(f"{self.plugin_id}: webhook POST failed: {e}")

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
