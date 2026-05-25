"""ConsoleProgressSinkPlugin — writes events to a logger; useful for CLI dev."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from genie_tooling.progress.abc import ProgressSinkPlugin
from genie_tooling.progress.types import ProgressEvent

logger = logging.getLogger("genie_tooling.progress")


class ConsoleProgressSinkPlugin(ProgressSinkPlugin):
    plugin_id: str = "console_progress_sink_v1"
    description: str = (
        "Logs each progress event to a logger. Useful for CLI dev and as a "
        "fallback sink alongside a webhook or Slack-thread sink."
    )

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def emit(self, event: ProgressEvent) -> None:
        prefix = f"[{event.agent_id}:{event.run_id[:8]}"
        if event.iteration is not None:
            prefix += f":i{event.iteration}"
        prefix += f":{event.phase}]"
        line = f"{prefix} {event.message}"
        if event.level == "error":
            logger.error(line)
        elif event.level == "warning":
            logger.warning(line)
        else:
            logger.info(line)

    async def teardown(self) -> None:
        pass
