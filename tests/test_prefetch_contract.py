"""Prefetch temporal boundaries retained from legacy base-agent tests."""

from musubi_livekit.voice.prefetch import MemoryPrefetchMixin


class ProbeHost:
    def __init__(self, *, instructions):
        self.instructions = instructions

    async def warm_llm_prefix(self):
        pass


class MemoryAgent(MemoryPrefetchMixin, ProbeHost):
    pass


def test_memory_prefetch_injects_real_context_and_skips_degraded():
    import asyncio

    class Probe(MemoryAgent):
        def __init__(self, context: str):
            super().__init__(instructions="x")
            self._context = context
            self.injected = None

        async def fetch_recent_context(self, limit: int = 10) -> str:
            return self._context

        async def update_instructions(self, instructions: str) -> None:
            self.injected = instructions

    real = Probe("- 2026-08-18: fixed the Discord gaps with Yua")
    asyncio.run(real._inject_recent_context(8))
    assert real.injected is not None
    assert "Discord gaps" in real.injected
    assert "Background awareness" in real.injected
    assert "records from before this call" in real.injected
    assert "not events in the current conversation" in real.injected
    assert "neutral acknowledgment" in real.injected
    assert "<prior_memory_records>" in real.injected
    assert "</prior_memory_records>" in real.injected
    assert real.injected.endswith("never claim a recorded event happened during this call.")
    # the persona must stay at the front of the single system message
    assert real.injected.startswith("x")

    for degraded in ("No recent memories found.", "Couldn't check memory — down", ""):
        probe = Probe(degraded)
        asyncio.run(probe._inject_recent_context(8))
        assert probe.injected is None


def test_memory_prefetch_can_warm_cache_without_prompt_injection():
    import asyncio

    class CacheOnlyProbe(MemoryAgent):
        prefetch_memory_into_instructions = False

        def __init__(self):
            super().__init__(instructions="persona")
            self.injected = None
            self.warmed = False

        async def fetch_recent_context(self, limit: int = 10) -> str:
            return "- a historical record"

        async def update_instructions(self, instructions: str) -> None:
            self.injected = instructions

        async def warm_llm_prefix(self) -> None:
            self.warmed = True

    probe = CacheOnlyProbe()
    asyncio.run(probe._inject_recent_context(8))

    assert probe.injected is None
    assert probe.warmed is True
