"""Retrieval warnings remain visible to a LiveKit worker."""

from typing import Any

from musubi_livekit.cache import ContextCache
from musubi_livekit.fast_talker import FastTalker
from musubi_livekit.slow_thinker import SlowThinker


class WarningClient:
    async def retrieve(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "results": [],
            "warnings": ["reranker_failed", "reranker_failed_request_rejected"],
        }


async def test_fast_talker_preserves_warning_and_cause() -> None:
    talker = FastTalker(client=WarningClient(), namespace="assistant/voice", cache=ContextCache())
    await talker.get_context("recent work")
    assert talker.last_warnings == ["reranker_failed", "reranker_failed_request_rejected"]


async def test_slow_thinker_preserves_warning_and_cause() -> None:
    thinker = SlowThinker(client=WarningClient(), namespace="assistant/voice", cache=ContextCache())
    await thinker._prefetch("recent work")
    assert thinker.last_warnings == ["reranker_failed", "reranker_failed_request_rejected"]
