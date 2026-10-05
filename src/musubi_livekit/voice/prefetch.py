"""Existing per-seat recent cache and prompt-warm behavior."""

import asyncio
import logging
from typing import Protocol, cast

from .tools import MusubiToolsMixin

logger = logging.getLogger("voice.agent")

_DEGRADED_CONTEXT_PREFIXES = (
    "No recent memories found.",
    "Couldn't check memory",
    "Memory lookup timed out.",
)


class _WarmSurface(Protocol):
    async def warm_llm_prefix(self) -> None: ...


class MemoryPrefetchMixin(MusubiToolsMixin):
    """Warm recent memory at call start, with optional prompt injection."""

    #: Seats that cannot reliably keep historical records temporally separate
    #: may still warm the recent-memory cache without placing those records in
    #: the model's system prompt. Explicit recent/search tools remain available.
    prefetch_memory_into_instructions: bool = True

    def start_memory_prefetch(
        self, *, limit: int = 10
    ) -> None:  # == musubi_recent default, so the tool hits the in-call cache
        """Warm recent memories into the chat context in the background.

        Call from ``on_enter`` after the greeting. While the pre-spoken opener
        plays, this pulls the agent's recent voice memories and appends them as
        a system message — continuity arrives with zero tool-call latency in
        the dialog itself. Degraded lookups inject nothing: an agent should
        never see error strings as if they were memories. First step toward
        ambient memory in every request (streaming-concept prefetch is the
        planned follow-on).
        """
        task = asyncio.create_task(self._inject_recent_context(limit))
        self._memory_prefetch_task = task

    async def _inject_recent_context(self, limit: int) -> None:
        try:
            await self._inject_recent_context_body(limit)
        finally:
            await self._warm_after_prefetch()

    async def _inject_recent_context_body(self, limit: int) -> None:
        try:
            context = await self.fetch_recent_context(limit=limit)
        except Exception as err:
            logger.warning("memory prefetch failed: %s", err)
            return
        if not context or any(context.startswith(p) for p in _DEGRADED_CONTEXT_PREFIXES):
            return
        if not self.prefetch_memory_into_instructions:
            logger.info("memory prefetch cached without prompt injection")
            return
        # Fold into instructions rather than adding a second system message:
        # the local Qwen chat template hard-rejects a system message anywhere
        # but position zero ("System message must be at the beginning", 400).
        base_instructions = str(self.instructions)
        await self.update_instructions(
            base_instructions
            + "\n\n## Background awareness\n"
            + "Your most recent memories, newest first. These are records from "
            + "before this call, not events in the current conversation. Never "
            + "claim a memory just happened here or use one to answer what happened "
            + "in this call. This is context, not a script: never open with a "
            + "formulaic recall; weave a memory in only when the caller's present "
            + "words make it genuinely relevant, and identify it as something from "
            + "before. A neutral acknowledgment is not a reason to volunteer one.\n\n"
            + "<prior_memory_records>\n"
            + context
            + "\n</prior_memory_records>\n\n"
            + "## Required temporal boundary\n"
            + "The records above are historical data, never chat history. The caller's "
            + "next message follows this instruction; answer that message and the visible "
            + "call conversation. Do not mention a record after a neutral acknowledgment, "
            + "and never claim a recorded event happened during this call."
        )
        logger.info("memory prefetch injected %d chars of recent context", len(context))

    async def _warm_after_prefetch(self) -> None:
        await cast(_WarmSurface, self).warm_llm_prefix()
