"""Phase 6C.2 — agent progress streaming.

Agents can take minutes. Without progress streaming, the operator sees one
final blob; with it, they see "gathering Linear issues… reading Slack
threads… synthesizing…" updates in real time.

Two pieces:
* ``ProgressSinkPlugin`` — protocol for receiving progress events
  (Slack thread, webhook, stdout, OTel).
* The ``ProgressEmitter`` helper (used inside agents) buffers events and
  fans them out to all configured sinks.
"""
from .abc import ProgressSinkPlugin
from .emitter import ProgressEmitter
from .types import ProgressEvent

__all__ = ["ProgressSinkPlugin", "ProgressEmitter", "ProgressEvent"]
