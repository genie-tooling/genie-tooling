"""Replay fixture types."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

EntryKind = Literal["llm_chat", "llm_generate", "tool_call"]


@dataclass
class RecordingEntry:
    """One recorded boundary call.

    Records are matched on replay by a content-addressed *call_hash* computed
    from (kind, payload). For LLM calls, payload is the canonical JSON of
    (provider_id, messages_or_prompt, sampling_kwargs). For tool calls,
    payload is (tool_id, params).
    """

    kind: EntryKind
    call_hash: str
    request: Dict[str, Any]
    response: Any
    sequence: int = 0  # ordinal in the recording — supports duplicate-hash distinction


@dataclass
class Recording:
    """A serializable fixture. Round-trips through JSON."""

    version: int = 1
    run_label: Optional[str] = None
    entries: List[RecordingEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "run_label": self.run_label,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recording":
        return cls(
            version=int(data.get("version", 1)),
            run_label=data.get("run_label"),
            entries=[RecordingEntry(**e) for e in (data.get("entries") or [])],
        )


def call_hash(kind: EntryKind, payload: Dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of (kind, payload). Used as the
    replay-match key — same inputs → same hash → same recorded response.
    """
    canonical = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
