"""LiveKitAdapter — glues LiveKit session events to Musubi SDK calls.

The adapter owns a :class:`SlowThinker` + :class:`FastTalker` pair sharing one
:class:`ContextCache`, plus the session-end artifact-upload pipeline
with bounded retry + a deferred-retry queue for hard failures.

The artifact upload path can be patched via ``client._upload_handler``;
the fallback captures the transcript in episodic memory.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from musubi_sdk.exceptions import MusubiError

try:
    # The old in-tree adapter accepted the in-tree SDK client. Keep its error
    # hierarchy during the import-path transition without requiring the server.
    from musubi.sdk.exceptions import MusubiError as CoreMusubiError
except ImportError:
    CoreMusubiError = MusubiError

from musubi_livekit.cache import ContextCache, RetrievalStatus
from musubi_livekit.config import LiveKitAdapterConfig
from musubi_livekit.fast_talker import FastTalker
from musubi_livekit.heuristics import detect_interesting_fact
from musubi_livekit.redaction import redact_pii
from musubi_livekit.slow_thinker import SlowThinker

log = logging.getLogger("musubi_livekit")

_EPISODE_CONTEXT_TAGS = ["kind:episode", "staleness:episodic"]


class LiveKitAdapter:
    """Per-session wiring of LiveKit events to Musubi SDK calls."""

    def __init__(
        self,
        *,
        client: Any,
        namespace: str,
        artifact_namespace: str,
        config: LiveKitAdapterConfig,
    ) -> None:
        self.client = client
        self.namespace = namespace
        self.artifact_namespace = artifact_namespace
        self.config = config
        self.cache = ContextCache(max_entries=config.cache_max_entries)
        # RET-007 — the ONE authoritative agent-facing channel; both talkers publish to it (last
        # retrieval wins = current-turn semantics), so a Slow Thinker pre-fetch failure is visible.
        self._retrieval_status = RetrievalStatus()
        self.slow_thinker = SlowThinker(
            client=client,
            namespace=namespace,
            cache=self.cache,
            deep_limit=config.deep_limit,
            cache_ttl_s=config.cache_default_ttl_s,
            status=self._retrieval_status,
        )
        self.fast_talker = FastTalker(
            client=client,
            namespace=namespace,
            cache=self.cache,
            fast_limit=config.fast_limit,
            match_threshold=config.fast_match_threshold,
            status=self._retrieval_status,
        )
        # Observability/state attached to the adapter for assertions in
        # tests and for an operator dashboard wiring later.
        self.upload_history: list[dict[str, Any]] = []
        self.failed_upload_queue: list[dict[str, Any]] = []

    @property
    def retrieval_status(self) -> list[str]:
        """RET-007 — the agent-facing degradation status for the CURRENT turn: the ONE authoritative
        channel both talkers publish to (most recent retrieval wins). Empty = healthy; bounded
        ``plane_*`` / ``sparse_embedding_failed`` / ``reranker_failed`` = partial degradation;
        ``[RETRIEVAL_UNAVAILABLE]`` = no memory this turn (a Slow Thinker pre-fetch failure surfaces
        here immediately, not only in the talker's own ``last_warnings``). This is the consumer the
        LiveKit agent reads to render a status message."""
        return list(self._retrieval_status.warnings)

    # ------------------------------------------------------------------
    # LiveKit event hooks
    # ------------------------------------------------------------------

    async def on_transcript_segment(self, transcript_so_far: str) -> None:
        """``transcript_segment_received`` event — Slow Thinker restart."""
        await self.slow_thinker.on_user_utterance_segment(transcript_so_far)

    async def on_user_turn_completed(self, full_utterance: str) -> None:
        """``on_user_turn_completed`` — final deep pre-fetch on the
        completed utterance. The Fast Talker reads this on the next
        speech tick."""
        await self.slow_thinker.on_user_utterance_segment(full_utterance)

    async def on_session_end(
        self,
        *,
        session_id: str,
        vtt_transcript: str,
    ) -> None:
        """``session_ends`` — upload the transcript as an artifact and
        send a summary thought. Both writes are gated by the privacy
        flags on :class:`LiveKitAdapterConfig`."""
        if self.config.capture_transcripts:
            await self._upload_transcript_with_retry(
                session_id=session_id, vtt_transcript=vtt_transcript
            )
            await self._send_session_thought(session_id=session_id)

    async def maybe_capture_fact(self, utterance: str) -> None:
        """Heuristic memory capture — runs if ``capture_facts`` is on
        AND the utterance matches an "interesting fact" pattern."""
        if not self.config.capture_facts:
            return
        if not detect_interesting_fact(utterance):
            return
        scrubbed = self.maybe_redact(utterance)
        try:
            await self.client.episodic.capture(
                namespace=self.namespace,
                content=scrubbed,
                tags=["livekit-voice", "heuristic-fact", *_EPISODE_CONTEXT_TAGS],
                importance=6,
            )
        except (MusubiError, CoreMusubiError):
            log.warning("livekit fact-capture failed", exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def maybe_redact(self, payload: str) -> str:
        """Pass-through unless ``redact_pii`` is on."""
        return redact_pii(payload) if self.config.redact_pii else payload

    async def _upload_transcript_with_retry(
        self,
        *,
        session_id: str,
        vtt_transcript: str,
    ) -> None:
        """Upload the VTT transcript with bounded exponential retry.
        On exhaustion, enqueue for deferred retry instead of dropping."""
        scrubbed = self.maybe_redact(vtt_transcript)
        if not scrubbed.strip():
            return
        payload = {
            "namespace": self.artifact_namespace,
            "title": f"Voice session {session_id}",
            "content_type": "text/vtt",
            "source_system": "livekit-session",
            "source_ref": session_id,
            "content": scrubbed.encode("utf-8"),
        }
        last_exc: Exception | None = None
        for attempt in range(1, self.config.upload_max_attempts + 1):
            if attempt > 1 and self.config.upload_backoff_s > 0:
                await asyncio.sleep(self.config.upload_backoff_s)
            try:
                handler = getattr(self.client, "_upload_handler", None)
                if callable(handler):
                    handler(**payload)
                else:
                    # Fall back to the SDK's canonical artifacts.upload
                    # surface when it ships; until then, route through
                    # episodic.capture so the adapter is always live.
                    await self.client.episodic.capture(
                        namespace=self.namespace,
                        content=scrubbed,
                        tags=["livekit-voice", "session-transcript", *_EPISODE_CONTEXT_TAGS],
                        importance=4,
                    )
                self.upload_history.append(payload)
                return
            except (MusubiError, CoreMusubiError) as exc:
                last_exc = exc
                continue
        # Retries exhausted — enqueue for deferred retry instead of
        # losing the session.
        log.warning("livekit transcript upload exhausted retries; queued for deferred retry")
        self.failed_upload_queue.append(
            {"session_id": session_id, "payload": payload, "last_error": last_exc}
        )

    async def _send_session_thought(self, *, session_id: str) -> None:
        """Optional summary thought emitted at session end."""
        try:
            await self.client.thoughts.send(
                namespace=self.namespace,
                from_presence="livekit-voice",
                to_presence="all",
                channel="scheduler",
                content=f"Session {session_id} captured.",
                importance=3,
            )
        except (MusubiError, CoreMusubiError):
            log.warning("livekit summary thought failed", exc_info=True)
