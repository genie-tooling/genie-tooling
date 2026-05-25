"""Protocol for progress sinks."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import ProgressEvent


@runtime_checkable
class ProgressSinkPlugin(Plugin, Protocol):
    """Receives ProgressEvents from agents. Implementations push to whatever
    operator-facing channel makes sense: a Slack thread, an HTTP webhook,
    stdout for CLI dev work, OTel for centralized observability."""

    async def emit(self, event: ProgressEvent) -> None: ...
