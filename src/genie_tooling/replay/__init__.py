"""Phase 6D.2 — Strict replay harness.

A *strict* replay fixture is a byte-for-byte record of an agent run's LLM
round-trips and tool I/O. On replay, recorded responses are substituted for
live calls — making CI regression tests fast, deterministic, and free of
LLM/API costs.
"""
from .player import ReplayMiss, ReplayPlayer
from .recorder import ReplayRecorder
from .types import Recording, RecordingEntry

__all__ = ["ReplayRecorder", "ReplayPlayer", "ReplayMiss", "Recording", "RecordingEntry"]
