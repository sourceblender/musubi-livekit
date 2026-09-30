"""Adapter-local configuration for the LiveKit voice adapter.

Configuration is passed at construction time. A consuming worker owns
environment loading and secret handling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveKitAdapterConfig:
    """Tunables for a single :class:`LiveKitAdapter` instance."""

    # --- Privacy gates (per spec § Privacy) -------------------------------
    capture_transcripts: bool = True
    """If False, ``on_session_end`` never uploads the VTT transcript."""

    capture_facts: bool = True
    """If False, the heuristic ``maybe_capture_fact`` never writes."""

    redact_pii: bool = False
    """If True, payloads pass through ``redact_pii`` before any write."""

    # --- Cache shape (per spec § ContextCache) ---------------------------
    cache_max_entries: int = 10
    cache_default_ttl_s: float = 120.0

    # --- Retrieval shape (per spec § Components) -------------------------
    deep_limit: int = 15
    fast_limit: int = 5
    fast_match_threshold: float = 0.5

    # --- Upload retry (per spec § Error handling) ------------------------
    upload_max_attempts: int = 3
    upload_backoff_s: float = 0.5
