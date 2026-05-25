"""ProgressEmitter — agent-facing helper that fans events to configured sinks."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .abc import ProgressSinkPlugin
from .types import ProgressEvent

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """Lightweight wrapper agents use to emit progress events.

    Usage inside an agent::

        progress = ProgressEmitter(
            run_id=run_id,
            agent_id="react_agent",
            sinks=self._progress_sinks,
            attribution_tags=ctx.get("attribution_tags"),
        )
        await progress.emit("iterating", f"iter {i+1}/{self.max_iterations}", iteration=i+1)

    Failures in any individual sink are logged but don't fail the emit —
    progress is best-effort.
    """

    def __init__(
        self,
        run_id: str,
        agent_id: str,
        sinks: Optional[List[ProgressSinkPlugin]] = None,
        attribution_tags: Optional[Dict[str, str]] = None,
    ):
        self._run_id = run_id
        self._agent_id = agent_id
        self._sinks = list(sinks or [])
        self._attribution_tags = dict(attribution_tags) if attribution_tags else None

    @property
    def has_sinks(self) -> bool:
        return bool(self._sinks)

    async def emit(
        self,
        phase: str,
        message: str,
        *,
        iteration: Optional[int] = None,
        level: str = "info",
        tool_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._sinks:
            return
        event = ProgressEvent(
            run_id=self._run_id,
            agent_id=self._agent_id,
            phase=phase,  # type: ignore[arg-type]
            message=message,
            timestamp=time.time(),
            iteration=iteration,
            level=level,  # type: ignore[arg-type]
            tool_id=tool_id,
            attribution_tags=dict(self._attribution_tags) if self._attribution_tags else None,
            extra=extra,
        )
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception as e:
                logger.error(f"ProgressSink {getattr(sink, 'plugin_id', sink)} failed: {e}", exc_info=True)
