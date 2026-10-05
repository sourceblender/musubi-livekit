"""The recent-memory scroll returns what it saw instead of timing out."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("google.genai")

from musubi_livekit.voice.identity import MemoryConfig as AgentConfig
from musubi_livekit.voice.provider import MusubiMemoryMixin

_CFG = AgentConfig(
    agent_name="kiko", memory_agent_tag="kiko-voice", musubi_v2_namespace="kiko/voice"
)


class Kiko(MusubiMemoryMixin):
    memory_config = _CFG
    greetings = ("Hey.",)


class _SlowLegacyPages:
    """Every page is tagged but legacy (no caller-quote provenance), and slow."""

    def __init__(self, page_s: float) -> None:
        self.page_s = page_s
        self.pages = 0

    async def list_episodic(self, *, namespace: str, limit: int, cursor: str | None):
        self.pages += 1
        await asyncio.sleep(self.page_s)
        return {
            "items": [
                {"tags": ["kiko-voice", "source:transcript"], "created_epoch": i}
                for i in range(limit)
            ],
            "next_cursor": f"c{self.pages}",
        }


def test_scroll_stops_inside_its_budget_and_reports_no_rows():
    """Guide 2026-09-02: five 0.8 s pages of untrusted rows overran the 3.0 s
    aggregate timeout and the seat said Musubi was unavailable. The truthful
    answer was that the newest rows carry no trusted provenance."""

    agent = Kiko(instructions="t")
    client = _SlowLegacyPages(page_s=0.05)
    agent._musubi_client = lambda: client  # type: ignore[method-assign]

    rows = asyncio.run(
        agent._scroll_episodic_recent(
            "kiko/voice/episodic", 10, required_tag="kiko-voice", budget_s=0.12
        )
    )

    assert rows == []
    assert 1 <= client.pages <= 2, client.pages


def test_fetch_recent_context_says_empty_not_unavailable_when_pages_are_legacy():
    agent = Kiko(instructions="t")
    client = _SlowLegacyPages(page_s=0.01)
    agent._musubi_client = lambda: client  # type: ignore[method-assign]

    assert asyncio.run(agent.fetch_recent_context(limit=10)) == "No recent memories found."


class _LowValueThenDurablePages:
    def __init__(self) -> None:
        self.pages = 0

    async def list_episodic(self, *, namespace: str, limit: int, cursor: str | None):
        self.pages += 1
        if self.pages == 1:
            return {
                "items": [
                    {
                        "content": "Caller said: Can you tell me what time it is?",
                        "tags": [
                            "kiko-voice",
                            "source:transcript",
                            "provenance:caller-quote",
                            "category:general",
                        ],
                        "created_epoch": 2,
                    }
                ],
                "next_cursor": "durable",
            }
        return {
            "items": [
                {
                    "content": "Caller said: My train leaves Tuesday at nine.",
                    "tags": [
                        "kiko-voice",
                        "source:transcript",
                        "provenance:caller-quote",
                        "category:planning",
                    ],
                    "created_epoch": 1,
                }
            ],
            "next_cursor": None,
        }


def test_scroll_continues_past_low_value_transcript_rows() -> None:
    agent = Kiko(instructions="t")
    client = _LowValueThenDurablePages()
    agent._musubi_client = lambda: client  # type: ignore[method-assign]

    rows = asyncio.run(
        agent._scroll_episodic_recent(
            "kiko/voice/episodic", 1, required_tag="kiko-voice", budget_s=1.0
        )
    )

    assert client.pages == 2
    assert [row["content"] for row in rows] == ["Caller said: My train leaves Tuesday at nine."]
