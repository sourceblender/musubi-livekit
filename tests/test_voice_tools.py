"""Tests for MusubiToolsMixin — canonical agent-tools surface.

Covers musubi_recent, musubi_search, musubi_remember, the retained
think_impl body (musubi_think is no longer LLM-exposed), and the
fetch_recent_context helper. Also asserts the
``MusubiToolsMixin`` deprecation alias still resolves to the canonical
class for one release.
"""

import asyncio
import logging
from typing import Any, cast

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("google.genai")

from musubi_livekit.voice.identity import UNCONFIGURED_CONFIG
from musubi_livekit.voice.identity import MemoryConfig as AgentConfig
from musubi_livekit.voice.tools import (
    MusubiToolsMixin,
    _has_trusted_provenance,
    _is_presentable_search_result,
    _is_recallable_memory,
)


def test_memory_mixin_alias_resolves_to_musubi_tools_mixin() -> None:
    """``MusubiToolsMixin`` is a one-release deprecation alias per
    Musubi ADR 0032. Until the alias is removed, importers must keep
    landing on the canonical class so behavior stays identical."""
    assert MusubiToolsMixin is MusubiToolsMixin


def _unwrap(tool: Any) -> Any:
    """LiveKit's `function_tool` wraps the underlying coroutine in a
    ``FunctionTool`` whose declared interface doesn't expose
    ``__wrapped__``, but the runtime always sets it (functools.wraps).
    Cast through ``Any`` so pyright doesn't complain on the test side
    while still asserting on real wire shape."""
    return cast(Any, tool).__wrapped__


def test_memory_mixin_has_musubi_recent():
    assert hasattr(MusubiToolsMixin, "musubi_recent")
    assert callable(MusubiToolsMixin.musubi_recent)


def test_memory_mixin_has_musubi_search():
    assert hasattr(MusubiToolsMixin, "musubi_search")
    assert callable(MusubiToolsMixin.musubi_search)


def test_memory_mixin_has_musubi_remember():
    assert hasattr(MusubiToolsMixin, "musubi_remember")
    assert callable(MusubiToolsMixin.musubi_remember)


def test_memory_mixin_exposes_fetch_recent_context_helper():
    """The plain-async helper used by on_enter must exist and be callable
    without the function_tool wrapping that musubi_recent carries."""
    assert hasattr(MusubiToolsMixin, "fetch_recent_context")
    assert callable(MusubiToolsMixin.fetch_recent_context)


def test_memory_mixin_default_config_is_unconfigured():
    """Absent an override the mixin defaults to the fail-loud sentinel — a
    forgotten config degrades memory to 'unavailable', it does not silently
    become Assistant (a per-seat default config)."""
    assert MusubiToolsMixin.memory_config is UNCONFIGURED_CONFIG
    assert MusubiToolsMixin.memory_config.musubi_v2_namespace is None


def test_memory_mixin_config_is_overridable():
    """A subclass can point config at a different AgentConfig."""
    guide_cfg = AgentConfig(
        agent_name="guide",
        memory_agent_tag="guide-voice",
    )

    class _GuideMemory(MusubiToolsMixin):
        memory_config = guide_cfg

    assert _GuideMemory.memory_config.memory_agent_tag == "guide-voice"
    # Parent class unaffected — still the fail-loud sentinel default.
    assert MusubiToolsMixin.memory_config is UNCONFIGURED_CONFIG


@pytest.mark.asyncio
async def test_musubi_remember_fails_closed_when_call_source_cannot_write(agent):
    """Synthetic/device callers cannot mutate durable household memory."""

    agent._musubi_client = lambda: pytest.fail("blocked memory write constructed a client")
    agent.set_memory_write_policy(allowed=False)

    result = await _unwrap(MusubiToolsMixin.musubi_remember)(
        agent,
        content="synthetic gate dialogue",
    )

    assert "not authorized" in result


def test_composed_agent_has_memory_tools(agent):
    """Memory tools are discoverable on a composed agent instance."""
    assert hasattr(agent, "musubi_recent")
    assert hasattr(agent, "musubi_search")
    assert hasattr(agent, "musubi_remember")
    assert hasattr(agent, "fetch_recent_context")
    # Default composed agent doesn't override, so tag is "assistant-voice".
    assert agent.memory_config.memory_agent_tag == "assistant-voice"


@pytest.mark.asyncio
async def test_fetch_recent_context_has_aggregate_timeout(agent, monkeypatch):
    async def slow_scroll(*args, **kwargs):
        await asyncio.sleep(0.05)
        return []

    agent._scroll_episodic_recent = slow_scroll
    monkeypatch.setattr("musubi_livekit.voice.tools._RECENT_CONTEXT_TIMEOUT_S", 0.001)

    result = await agent.fetch_recent_context(limit=10)

    assert "Musubi is unavailable" in result


# ---------------------------------------------------------------------------
# musubi_search behaviour — namespace shape, state_filter, mode
# ---------------------------------------------------------------------------


class _StubClient:
    """Records a single retrieve() call so tests can assert the wire shape
    without standing up a real Musubi server. Mirrors `MusubiClient.retrieve`
    keyword arguments exactly so signature drift breaks the test."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        *,
        tagless: bool = False,
        details: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._response = response or {"results": []}
        if not tagless:
            for row in self._response.get("results", []):
                row.setdefault("tags", ["assistant-voice"])
        self.details = details or {}
        self.calls: list[dict[str, Any]] = []
        self.get_calls: list[tuple[str, str]] = []

    async def get_episodic(self, *, namespace: str, object_id: str) -> dict[str, Any]:
        self.get_calls.append((namespace, object_id))
        return self.details[object_id]

    async def retrieve(
        self,
        *,
        namespace: str,
        query_text: str,
        mode: str = "fast",
        limit: int = 10,
        planes: list[str] | None = None,
        include_archived: bool = False,
        state_filter: list[str] | None = None,
        session: object | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "namespace": namespace,
                "query_text": query_text,
                "mode": mode,
                "limit": limit,
                "planes": planes,
                "include_archived": include_archived,
                "state_filter": state_filter,
            }
        )
        return self._response


@pytest.mark.asyncio
async def test_musubi_search_uses_tenant_wildcard_namespace(agent):
    """`musubi_search` must use `<tenant>/*/episodic` so cross-channel
    recall works (per Musubi ADR 0031). A regression to the agent's own
    channel breaks the multimodality contract — phone Assistant would stop
    seeing Discord-Assistant's deliberate stores."""
    stub = _StubClient(response={"results": []})
    agent._musubi_client = lambda: stub
    # Force a known 2-segment presence so the test isn't sensitive to fixture defaults.
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="prank", limit=5)

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["namespace"] == "assistant/*/episodic"


@pytest.mark.asyncio
async def test_musubi_search_passes_state_filter_for_fresh_save_recall(agent):
    """The whole point of musubi_search is recalling a deliberate
    musubi_remember BEFORE the maturation cron runs (otherwise voice-Assistant
    can't remember what Discord-Assistant just saved). Asserts state_filter
    explicitly includes `provisional` so fresh stores are visible."""
    stub = _StubClient(response={"results": []})
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="dentist", limit=5)

    call = stub.calls[0]
    assert call["state_filter"] == ["provisional", "matured", "promoted"]
    # Mode "deep" — recall waits on full hybrid + rerank for best hit.
    assert call["mode"] == "deep"


@pytest.mark.asyncio
async def test_musubi_search_returns_origin_channel_in_each_row(agent):
    """Result rows must surface their concrete stored namespace's
    presence segment so the LLM can attribute "you told me on Discord"
    vs "on the call". Without this, channel provenance is lost in
    rendering even though the API preserves it."""
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "a" * 27,
                    "score": 0.9,
                    "plane": "episodic",
                    "content": "the cocoa-pods prank",
                    "namespace": "assistant/discord/episodic",
                },
            ],
        },
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="prank")
    assert "[discord]" in rendered, rendered
    assert "cocoa-pods prank" in rendered, rendered


def test_legacy_transcript_rows_are_not_trusted_without_caller_quotes() -> None:
    assert not _has_trusted_provenance(
        {"tags": ["source:transcript", "companion-voice"], "content": "assistant fiction"}
    )
    assert _has_trusted_provenance(
        {
            "tags": ["source:transcript", "provenance:caller-quote", "companion-voice"],
            "content": "Caller said: exact caller evidence",
        }
    )
    assert _has_trusted_provenance(
        {"tags": ["companion-voice"], "content": "deliberate musubi_remember save"}
    )


def test_general_transcript_rows_are_auditable_but_not_recallable() -> None:
    row = {
        "tags": [
            "source:transcript",
            "provenance:caller-quote",
            "category:general",
            "assistant-voice",
        ],
        "content": "Caller said: Can you tell me what time it is?",
    }

    assert _has_trusted_provenance(row)
    assert not _is_recallable_memory(row)
    assert not _is_recallable_memory(
        {
            "tags": ["source:transcript", "provenance:caller-quote", "assistant-voice"],
            "content": "Caller said: A category-free legacy row.",
        }
    )
    assert not _is_recallable_memory(
        {
            "tags": [
                "source:transcript",
                "provenance:caller-quote",
                "category:made-up",
                "assistant-voice",
            ],
            "content": "Caller said: An unknown-category row.",
        }
    )
    assert not _is_recallable_memory(
        {
            "tags": [
                "source:transcript",
                "provenance:caller-quote",
                "category:personal",
                "assistant-voice",
            ],
            "content": "Caller said: What time is it?",
        }
    )


def test_specific_transcript_and_deliberate_rows_remain_recallable() -> None:
    assert _is_recallable_memory(
        {
            "tags": [
                "source:transcript",
                "provenance:caller-quote",
                "category:planning",
                "assistant-voice",
            ],
            "content": "Caller said: My train leaves Tuesday at nine.",
        }
    )
    assert _is_recallable_memory(
        {"tags": ["assistant-voice"], "content": "Deliberate user-requested save."}
    )
    assert _is_recallable_memory(
        {
            "tags": [
                "source:transcript",
                "provenance:caller-quote",
                "category:planning",
                "assistant-voice",
            ],
            "content": "Caller said: What time is it? Then: My train leaves Tuesday at nine.",
        }
    )


def test_tagless_search_results_require_verified_metadata() -> None:
    assert not _is_presentable_search_result(
        {"content": "Caller said: What time is it?", "namespace": "assistant/voice/episodic"}
    )
    assert not _is_presentable_search_result(
        {
            "content": "Caller said: My train leaves Tuesday at nine.",
            "namespace": "assistant/voice/episodic",
        }
    )
    # Content wording cannot prove whether this was an assistant transcript.
    assert not _is_presentable_search_result(
        {"content": "The picnic plan moved to Tuesday.", "namespace": "assistant/voice/episodic"}
    )


@pytest.mark.asyncio
async def test_musubi_search_filters_legacy_assistant_authored_rows(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "The assistant invented a fox story.",
                    "namespace": "companion/voice/episodic",
                    "tags": ["source:transcript", "companion-voice"],
                },
                {
                    "content": "Caller said: I finished the test suite.",
                    "namespace": "companion/voice/episodic",
                    "tags": [
                        "source:transcript",
                        "provenance:caller-quote",
                        "category:project",
                        "companion-voice",
                    ],
                },
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="companion",
        memory_agent_tag="companion-voice",
        musubi_v2_namespace="companion/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="tests")

    assert "finished the test suite" in rendered
    assert "fox story" not in rendered
    assert stub.calls[0]["limit"] == 15


@pytest.mark.asyncio
async def test_musubi_search_filters_general_transcript_chatter(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "Caller said: Can you tell me what time it is?",
                    "namespace": "assistant/voice/episodic",
                    "tags": [
                        "source:transcript",
                        "provenance:caller-quote",
                        "category:general",
                        "assistant-voice",
                    ],
                },
                {
                    "content": "Caller said: My train leaves Tuesday at nine.",
                    "namespace": "assistant/voice/episodic",
                    "tags": [
                        "source:transcript",
                        "provenance:caller-quote",
                        "category:planning",
                        "assistant-voice",
                    ],
                },
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday")

    assert "train leaves Tuesday" in rendered
    assert "what time is it" not in rendered


@pytest.mark.asyncio
async def test_musubi_search_joins_exact_metadata_when_retrieve_omits_tags(agent, caplog):
    """Retrieve omits tags; exact-ID episodic reads establish provenance."""

    caplog.set_level(logging.INFO, logger="voice.agent")

    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "general-row",
                    "content": "Caller said: What time is it?",
                    "namespace": "assistant/voice/episodic",
                },
                {
                    "object_id": "planning-row",
                    "content": "Caller said: My train leaves Tuesday at nine.",
                    "namespace": "assistant/voice/episodic",
                },
            ]
        },
        tagless=True,
        details={
            "planning-row": {
                "object_id": "planning-row",
                "namespace": "assistant/voice/episodic",
                "content": "Caller said: My train leaves Tuesday at nine.",
                "state": "provisional",
                "tags": ["source:transcript", "provenance:caller-quote", "category:planning"],
            }
        },
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday")

    assert "train leaves Tuesday" in rendered
    assert "What time is it" not in rendered
    assert stub.get_calls == [("assistant/voice/episodic", "planning-row")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail_change,expected",
    [
        ({"tags": ["source:transcript", "assistant-voice"]}, "No memories matched."),
        ({"object_id": "other-row"}, "degraded"),
        ({"namespace": "assistant/discord/episodic"}, "degraded"),
        ({"content": "different saved text"}, "degraded"),
        ({"state": "archived"}, "No memories matched."),
        ({"tags": None}, "degraded"),
    ],
)
async def test_musubi_search_rejects_untrusted_exact_metadata(agent, detail_change, expected):
    content = "Caller said: My train leaves Tuesday at nine."
    detail = {
        "object_id": "planning-row",
        "namespace": "assistant/voice/episodic",
        "content": content,
        "state": "provisional",
        "tags": ["source:transcript", "provenance:caller-quote", "category:planning"],
    }
    detail.update(detail_change)
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "planning-row",
                    "namespace": "assistant/voice/episodic",
                    "content": content,
                }
            ]
        },
        tagless=True,
        details={"planning-row": detail},
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    if expected == "degraded":
        assert "couldn't verify" in rendered.lower()
    else:
        assert rendered == expected


@pytest.mark.asyncio
async def test_musubi_search_verified_deliberate_save_survives_missing_sibling_metadata(agent):
    saved = "The Tuesday train leaves at nine."
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "lost-row",
                    "namespace": "assistant/voice/episodic",
                    "content": "Tuesday train unknown",
                },
                {
                    "object_id": "saved-row",
                    "namespace": "assistant/voice/episodic",
                    "content": saved,
                },
            ]
        },
        tagless=True,
        details={
            "saved-row": {
                "object_id": "saved-row",
                "namespace": "assistant/voice/episodic",
                "content": saved,
                "state": "matured",
                "tags": ["assistant-voice"],
            }
        },
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert saved in rendered
    assert "unknown" not in rendered


@pytest.mark.asyncio
async def test_musubi_search_rejects_cross_tenant_hit_before_detail_read(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "foreign-row",
                    "namespace": "companion/voice/episodic",
                    "content": "Tuesday train plan",
                }
            ]
        },
        tagless=True,
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert "couldn't verify" in rendered.lower()
    assert stub.get_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "namespace",
    [
        "companion/voice/episodic",
        "assistant/*/episodic",
        "assistant/voice/semantic",
        "assistant//episodic",
        "assistant/voice",
        "assistant/voice/episodic/extra",
        None,
    ],
)
async def test_musubi_search_rejects_inline_tagged_hit_outside_exact_tenant_plane(agent, namespace):
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "inline-row",
                    "namespace": namespace,
                    "content": "Tuesday train plan",
                    "tags": ["source:explicit-save"],
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert "couldn't verify" in rendered.lower()
    assert "Tuesday train plan" not in rendered
    assert stub.get_calls == []


@pytest.mark.asyncio
async def test_musubi_search_keeps_same_tenant_inline_hit_when_foreign_hit_precedes_it(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "object_id": "foreign-row",
                    "namespace": "companion/voice/episodic",
                    "content": "Tuesday train foreign",
                    "tags": ["source:explicit-save"],
                },
                {
                    "object_id": "own-row",
                    "namespace": "assistant/discord/episodic",
                    "content": "Tuesday train own",
                    "tags": ["source:explicit-save"],
                },
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert "Tuesday train own" in rendered
    assert "Tuesday train foreign" not in rendered
    assert stub.get_calls == []


@pytest.mark.asyncio
async def test_musubi_search_checks_lower_ranked_save_within_bounded_reads(agent):
    rows = [
        {
            "object_id": f"row-{index}",
            "namespace": "assistant/voice/episodic",
            "content": f"Tuesday train plan {index}",
        }
        for index in range(32)
    ]
    details = {
        row["object_id"]: {
            **row,
            "state": "matured",
            "tags": ["source:transcript", "assistant-voice"],
        }
        for row in rows
    }
    details["row-11"]["tags"] = ["assistant-voice"]
    stub = _StubClient(response={"results": rows}, tagless=True, details=details)
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert "Tuesday train plan 11" in rendered
    assert "Tuesday train plan 16" not in rendered
    assert len(stub.get_calls) == 15


@pytest.mark.asyncio
async def test_musubi_search_cancels_slow_metadata_read(agent, monkeypatch):
    cancelled = asyncio.Event()

    class SlowClient(_StubClient):
        async def get_episodic(self, *, namespace: str, object_id: str) -> dict[str, Any]:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {}

    stub = SlowClient(
        response={
            "results": [
                {
                    "object_id": "slow-row",
                    "namespace": "assistant/voice/episodic",
                    "content": "Tuesday train plan",
                }
            ]
        },
        tagless=True,
    )
    monkeypatch.setattr("musubi_livekit.voice.tools._SEARCH_METADATA_BUDGET_S", 0.001)
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train")

    assert "couldn't verify" in rendered.lower()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_musubi_search_skips_high_scored_bootstrap_row_for_supported_lower_hit(agent):
    """Relative provider rank cannot outweigh visible evidence for the caller's topic."""

    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "First memory. Assistant was minted as a personal assistant.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.78,
                    "score_kind": "ranked_combined",
                },
                {
                    "content": "Caller said: My train leaves Tuesday from platform four.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.41,
                    "score_kind": "ranked_combined",
                },
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(
        agent, query="Do you remember the Tuesday train platform?"
    )

    assert "train leaves Tuesday" in rendered
    assert "First memory" not in rendered


@pytest.mark.asyncio
async def test_musubi_search_abstains_when_ranked_results_lack_query_evidence(agent, caplog):
    """A high relative score for an absent or nonsense topic is not a match."""

    caplog.set_level(logging.INFO, logger="voice.agent")
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "First memory. Assistant was minted as a personal assistant.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.78,
                    "score_kind": "ranked_combined",
                },
                {
                    "content": "We traded presence markers in the living room.",
                    "namespace": "assistant/discord/episodic",
                    "score": 0.47,
                    "score_kind": "ranked_combined",
                },
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(
        agent, query="deliberately nonexistent synthetic routing marker"
    )

    assert rendered == "No memories matched."
    assert "without content-visible query evidence" in caplog.text


@pytest.mark.asyncio
async def test_musubi_search_keeps_single_short_topic_and_ignores_score_threshold(agent):
    """Short names remain searchable and a low relative score is not rejected by policy."""

    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "Caller said: Lee will meet us at noon.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.01,
                    "score_kind": "ranked_combined",
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Remember Lee")

    assert "Lee will meet us" in rendered


@pytest.mark.asyncio
async def test_musubi_search_abstains_when_query_has_only_recall_scaffolding(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "Caller said: The dentist appointment is Tuesday.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.99,
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Do you remember that?")

    assert rendered == "No memories matched."
    assert stub.calls == []


@pytest.mark.asyncio
async def test_musubi_search_blank_query_uses_safe_empty_contract_without_retrieval(agent):
    stub = _StubClient(response={"results": [{"content": "unrelated"}]})
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="  ")

    assert rendered == "No memories matched."
    assert stub.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "Okay Assistant, do you remember the lighthouse?",
        "Assistant, do you remember the first thing I told you?",
        "Any memories?",
        "Do you remember, like, what I said?",
        "Do you remembers what I mentioned?",
        "All right, do you remember what I told you?",
        "Yep, do you remember what I said?",
    ],
)
async def test_seat_names_fillers_and_plural_scaffolding_cannot_admit_bootstrap_row(agent, query):
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "First memory. Assistant said okay before the first call.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.99,
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query=query)

    assert rendered == "No memories matched."
    if query in {
        "Any memories?",
        "Do you remember, like, what I said?",
        "All right, do you remember what I told you?",
        "Yep, do you remember what I said?",
    }:
        assert stub.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["aliasa", "aliasb", "aliasc", "aliasd", "aliase"])
async def test_unambiguous_seat_alias_is_not_required_topic_evidence(agent, alias):
    query = f"{alias}, what did I say about coffee?"
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": "Caller said: Coffee is best before noon.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.01,
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
        search_stopwords=(alias,),
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query=query)

    assert "Coffee is best" in rendered


@pytest.mark.asyncio
async def test_title_alone_cannot_admit_content_the_agent_would_not_speak(agent):
    stub = _StubClient(
        response={
            "results": [
                {
                    "title": "Tuesday train platform",
                    "content": "First memory. Assistant was minted as a personal assistant.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.99,
                }
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(agent, query="Tuesday train platform")

    assert rendered == "No memories matched."


@pytest.mark.asyncio
async def test_musubi_search_does_not_log_supported_rows_beyond_display_limit(agent, caplog):
    """The display cap is not evidence that an otherwise supported row was rejected."""

    caplog.set_level(logging.INFO, logger="voice.agent")
    stub = _StubClient(
        response={
            "results": [
                {
                    "content": f"Caller said: Train plans number {index} leave Tuesday.",
                    "namespace": "assistant/voice/episodic",
                    "score": 0.9 - index / 10,
                }
                for index in range(3)
            ]
        }
    )
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )

    rendered = await _unwrap(MusubiToolsMixin.musubi_search)(
        agent, query="Tuesday train plan", limit=1
    )

    assert rendered.count("Train plans") == 1
    assert "without content-visible query evidence" not in caplog.text


# ---------------------------------------------------------------------------
# musubi_think behaviour — namespace shape, presence resolution, ack
# ---------------------------------------------------------------------------


def test_musubi_think_tool_is_not_exposed() -> None:
    """``musubi_think`` was un-registered 2026-07-10: the persona forbids
    claiming a handoff and the thought plane isn't consumed by the live
    webbing. The LLM-facing tool must be absent while ``think_impl`` (the
    programmatic body) is retained — see the module docstring."""
    assert not hasattr(MusubiToolsMixin, "musubi_think")


def test_memory_mixin_exposes_think_impl_helper() -> None:
    """``think_impl`` is the plain-async body, retained for programmatic use
    even though the ``@function_tool musubi_think`` wrapper was removed."""
    assert hasattr(MusubiToolsMixin, "think_impl")
    assert callable(MusubiToolsMixin.think_impl)


class _ThoughtStub:
    """Records send_thought calls so tests can assert wire shape."""

    def __init__(self, ack: dict[str, Any] | None = None) -> None:
        self._ack = ack or {"object_id": "thought-" + "0" * 20, "state": "delivered"}
        self.calls: list[dict[str, Any]] = []

    async def send_thought(
        self,
        *,
        namespace: str,
        from_presence: str,
        to_presence: str,
        content: str,
        channel: str = "default",
        importance: int = 5,
        session: object | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "namespace": namespace,
                "from_presence": from_presence,
                "to_presence": to_presence,
                "content": content,
                "channel": channel,
                "importance": importance,
            }
        )
        return self._ack


@pytest.mark.asyncio
async def test_musubi_think_uses_own_thought_namespace(agent) -> None:
    """``musubi_think`` must send from ``<agent>/<channel>/thought`` —
    ADR 0030 agent-as-tenant form. Regression to legacy ``caller/<agent>``
    breaks scope-token validation on the live server."""
    stub = _ThoughtStub()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="guide",
        memory_agent_tag="guide-voice",
        musubi_v2_namespace="guide/voice",
    )

    await MusubiToolsMixin.think_impl(agent, to_presence="assistant/voice", content="hey")

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["namespace"] == "guide/voice/thought"
    assert call["from_presence"] == "guide/voice"
    assert call["to_presence"] == "assistant/voice"


@pytest.mark.asyncio
async def test_musubi_think_resolves_bare_recipient_to_own_channel(agent) -> None:
    """A bare ``<agent>`` recipient must be resolved to ``<agent>/<own-channel>``
    so the model doesn't have to know channel topology to page a peer."""
    stub = _ThoughtStub()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="guide",
        memory_agent_tag="guide-voice",
        musubi_v2_namespace="guide/voice",
    )

    await MusubiToolsMixin.think_impl(agent, to_presence="assistant", content="ping")

    assert stub.calls[0]["to_presence"] == "assistant/voice"


@pytest.mark.asyncio
async def test_musubi_think_rejects_empty_recipient_or_content(agent) -> None:
    """Validation lives in ``think_impl`` so an empty arg degrades to a
    user-readable error instead of a 400 from the server."""
    stub = _ThoughtStub()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="guide",
        memory_agent_tag="guide-voice",
        musubi_v2_namespace="guide/voice",
    )

    empty_recipient = await MusubiToolsMixin.think_impl(agent, to_presence="", content="hi")
    empty_content = await MusubiToolsMixin.think_impl(agent, to_presence="assistant", content="")

    assert "to_presence is required" in empty_recipient
    assert "content is required" in empty_content
    assert stub.calls == []


@pytest.mark.asyncio
async def test_musubi_think_returns_object_id_in_ack(agent) -> None:
    """The ack rendering must surface the resolved recipient + the
    object_id so the LLM can confirm delivery in its reply."""
    stub = _ThoughtStub(ack={"object_id": "thought-abc123", "state": "delivered"})
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="guide",
        memory_agent_tag="guide-voice",
        musubi_v2_namespace="guide/voice",
    )

    rendered = await MusubiToolsMixin.think_impl(
        agent, to_presence="assistant/discord", content="deploy is done"
    )

    assert "assistant/discord" in rendered
    assert "thought-abc123" in rendered


# ---------------------------------------------------------------------------
# musubi_get — removed 2026-07-09
# ---------------------------------------------------------------------------


def test_musubi_get_is_gone() -> None:
    """It was registered as a tool and returned "not yet available", which is
    the same defect that took ``openclaw_delegate`` down: a prompt-visible tool
    the runtime cannot fulfil. Reserving a name is not worth teaching the model
    to reach for something that is not there."""
    assert not hasattr(MusubiToolsMixin, "musubi_get")


@pytest.mark.asyncio
async def test_musubi_recent_is_served_from_the_in_call_prefetch_cache(agent, monkeypatch):
    """Prefetch at call start and a mid-call musubi_recent want the same rows;
    the tool must not re-pay the Musubi scroll (0.4-0.8 s) inside a turn."""
    calls: list[int] = []

    async def scroll(namespace, need, *, required_tag=None):
        calls.append(need)
        return [
            {"text": f"row{i}", "created_epoch": 100 - i, "tags": [required_tag]}
            for i in range(need)
        ]

    monkeypatch.setattr(agent, "_scroll_episodic_recent", scroll)
    monkeypatch.setattr(agent, "_own_episodic_namespace", lambda: "assistant/voice/episodic")

    first = await agent.fetch_recent_context(limit=10)
    assert calls == [10]
    again = await agent.fetch_recent_context(limit=10)  # mid-call tool call
    smaller = await agent.fetch_recent_context(limit=5)
    assert calls == [10], "served from cache, no second scroll"
    assert again == first
    assert smaller.count("\n\n") == 4  # 5 rows

    await agent.fetch_recent_context(limit=20)  # larger than cached -> real fetch
    assert calls == [10, 20]

    agent._recent_rows_cache = None  # what musubi_remember does after a save
    await agent.fetch_recent_context(limit=10)
    assert calls == [10, 20, 10]


# --- 2026-09-22 independent review ------------------------------------------


class _CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def capture_memory(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append(kwargs)
        return {"object_id": f"obj-{len(self.calls)}"}


async def _remember(agent, *contents: str, call_sid: str = "CA-1") -> _CapturingClient:
    stub = _CapturingClient()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )
    agent.set_memory_write_policy(allowed=True, call_sid=call_sid)
    for content in contents:
        await _unwrap(MusubiToolsMixin.musubi_remember)(agent, content=content)
    return stub


@pytest.mark.asyncio
async def test_a_retried_save_reuses_its_idempotency_key(agent):
    """THE REGRESSION. The key was a fresh UUID per attempt, which is the one
    shape that cannot deduplicate anything: a timeout after Musubi accepted the
    POST is reported to us as failure, and the retry wrote a second row."""
    stub = await _remember(agent, "The demo is Friday.", "The demo is Friday.")

    assert stub.calls[0]["idempotency_key"] == stub.calls[1]["idempotency_key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_before_cancel", [False, True])
async def test_cancelled_save_reports_uncertain_outcome_and_reuses_key_on_retry(
    agent, caplog, accepted_before_cancel
):
    """A local cancellation cannot establish whether the server kept a POST."""

    class GatedCapture:
        def __init__(self):
            self.calls = []
            self.accept = asyncio.Event()
            self.accepted = asyncio.Event()
            self.release = asyncio.Event()
            self.objects = {}
            self.started = asyncio.Event()

        async def capture_memory(self, **kwargs):
            self.calls.append(kwargs)
            self.started.set()
            await self.accept.wait()
            key = kwargs["idempotency_key"]
            self.objects.setdefault(key, f"obj-{len(self.objects) + 1}")
            self.accepted.set()
            await self.release.wait()
            return {"object_id": self.objects[key]}

    stub = GatedCapture()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )
    agent.set_memory_write_policy(allowed=True, call_sid="CA-cancel")
    agent._recent_rows_cache = (0.0, 5, [])
    remember = _unwrap(MusubiToolsMixin.musubi_remember)
    with caplog.at_level(logging.WARNING, logger="voice.agent"):
        pending = asyncio.create_task(remember(agent, content="The demo is Friday."))
        await asyncio.wait_for(stub.started.wait(), 1)
        if accepted_before_cancel:
            stub.accept.set()
            await asyncio.wait_for(stub.accepted.wait(), 1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    key = stub.calls[0]["idempotency_key"]
    assert "save outcome is unknown" in caplog.text
    assert "save_ref=" in caplog.text
    assert key not in caplog.text
    assert "The demo is Friday." not in caplog.text
    assert agent._recent_rows_cache is None
    assert len(stub.objects) == int(accepted_before_cancel)

    stub.accept.set()
    stub.release.set()
    assert await remember(agent, content="The demo is Friday.") == "Got it, stored."
    assert stub.calls[1]["idempotency_key"] == key
    assert len(stub.objects) == 1


@pytest.mark.asyncio
async def test_cancelled_save_cannot_let_an_older_recent_lookup_recache_stale_rows(agent):
    started, release = asyncio.Event(), asyncio.Event()
    scroll_calls = 0

    async def slow_scroll(*_args, **_kwargs):
        nonlocal scroll_calls
        scroll_calls += 1
        started.set()
        if scroll_calls == 1:
            await release.wait()
        return []

    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )
    agent._scroll_episodic_recent = slow_scroll
    pending = asyncio.create_task(agent.fetch_recent_context())
    await asyncio.wait_for(started.wait(), 1)
    agent._invalidate_recent_rows_cache()  # same invalidation used by a cancelled save
    release.set()
    assert await asyncio.wait_for(pending, 1) == "No recent memories found."
    assert agent._recent_rows_cache is None
    assert await agent.fetch_recent_context() == "No recent memories found."
    assert scroll_calls == 2


@pytest.mark.asyncio
async def test_different_content_stays_a_different_memory(agent):
    stub = await _remember(agent, "The demo is Friday.", "The demo is Monday.")

    assert stub.calls[0]["idempotency_key"] != stub.calls[1]["idempotency_key"]


@pytest.mark.asyncio
async def test_the_same_sentence_on_a_later_call_is_a_new_memory(agent):
    """Scoped to the call, so dedupe cannot swallow a genuine re-save."""
    first = await _remember(agent, "The demo is Friday.", call_sid="CA-1")
    second = await _remember(agent, "The demo is Friday.", call_sid="CA-2")

    assert first.calls[0]["idempotency_key"] != second.calls[0]["idempotency_key"]


@pytest.mark.asyncio
async def test_missing_call_sid_uses_a_fresh_policy_scope_without_losing_retry_dedup(agent):
    stub = _CapturingClient()
    agent._musubi_client = lambda: stub
    agent.memory_config = AgentConfig(
        agent_name="assistant",
        memory_agent_tag="assistant-voice",
        musubi_v2_namespace="assistant/voice",
    )
    remember = _unwrap(MusubiToolsMixin.musubi_remember)
    agent.set_memory_write_policy(allowed=True, call_sid=None)
    await remember(agent, content="The demo is Friday.")
    await remember(agent, content="The demo is Friday.")
    assert stub.calls[0]["idempotency_key"] == stub.calls[1]["idempotency_key"]
    agent.set_memory_write_policy(allowed=True, call_sid=None)
    await remember(agent, content="The demo is Friday.")
    assert stub.calls[2]["idempotency_key"] != stub.calls[1]["idempotency_key"]


@pytest.mark.asyncio
async def test_an_in_call_save_is_tagged_as_model_authored(agent):
    """Post-call extraction means provenance:caller-quote -- it keeps Caller's
    exact [USER] wording. This row is whatever the model chose to write, and
    must not be indistinguishable from that at audit time."""
    stub = await _remember(agent, "The demo is Friday.")

    assert "provenance:in-call-tool" in stub.calls[0]["tags"]
    assert "provenance:caller-quote" not in stub.calls[0]["tags"]


def test_the_tool_no_longer_asks_the_model_to_save_on_a_goodbye():
    """An unsolicited write on a goodbye turn is the failure that disabled
    unrouted tool choice on every seat (Speaker live call, 2026-08-24). The tool
    text was still asking the model to do it."""
    doc = _unwrap(MusubiToolsMixin.musubi_remember).__doc__ or ""

    assert "proactively" not in doc
    assert "Do NOT invoke it because a call is ending." in doc
