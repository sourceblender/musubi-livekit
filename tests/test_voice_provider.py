from unittest.mock import AsyncMock

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("google.genai")

from musubi_livekit.voice.identity import MemoryConfig
from musubi_livekit.voice.postcall_memory import _validate_memory
from musubi_livekit.voice.provider import MusubiMemoryMixin


class Seat(MusubiMemoryMixin):
    memory_config = MemoryConfig("companion", "companion-voice", "companion/voice")
    prefetch_memory_into_instructions = False


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["No recent memories found.", "verified context", None])
async def test_prefetch_warms_without_injecting_when_seat_disables_injection(result):
    seat = Seat(instructions="persona")
    seat.fetch_recent_context = AsyncMock(return_value=result)
    seat.warm_llm_prefix = AsyncMock()
    seat.update_instructions = AsyncMock()
    await seat._inject_recent_context(10)
    seat.warm_llm_prefix.assert_awaited_once()
    seat.update_instructions.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieval_failure_still_warms_the_engine():
    seat = Seat(instructions="persona")
    seat.fetch_recent_context = AsyncMock(side_effect=RuntimeError("offline"))
    seat.warm_llm_prefix = AsyncMock()
    await seat._inject_recent_context(10)
    seat.warm_llm_prefix.assert_awaited_once()


def test_provider_wires_exactly_one_capture_and_shutdown(monkeypatch):
    monkeypatch.setenv("LIVEKIT_VOICE_LOGS", "/tmp/provider-test-unused")

    class Session:
        def __init__(self):
            self.handlers = []

        def on(self, name):
            def register(fn):
                self.handlers.append((name, fn))
                return fn

            return register

    class Context:
        def __init__(self):
            self.callbacks = []

        def add_shutdown_callback(self, fn):
            self.callbacks.append(fn)

    session, ctx = Session(), Context()
    Seat(instructions="persona").wire_call_memory(
        session, ctx, call_sid="SCL_real", capture_allowed=True
    )
    assert len(session.handlers) == 1
    assert session.handlers[0][0] == "close"
    assert len(ctx.callbacks) == 1


def test_provider_denies_capture_for_synthetic_caller(monkeypatch):
    import musubi_livekit.voice.provider as provider

    calls = []
    monkeypatch.setattr(provider, "wire_postcall_memory", lambda *args, **kw: calls.append(kw))
    monkeypatch.setattr(provider, "wire_musubi_shutdown", lambda ctx: None)
    Seat(instructions="persona").wire_call_memory(
        object(), object(), call_sid="synthetic", capture_allowed=False
    )
    assert calls == [
        {
            "call_sid": "synthetic",
            "namespace": "companion/voice/episodic",
            "speaker_tag": "companion-voice",
            "capture_allowed": False,
        }
    ]


@pytest.mark.parametrize(
    "raw,evidence,reason",
    [
        (None, None, "not_object"),
        ({"content": 123}, None, "content_not_string"),
        ({"content": " "}, None, "empty_content"),
        ({"content": "private"}, {}, "missing_evidence"),
        ({"content": "private", "evidence": [123]}, {}, "evidence_not_string"),
        ({"content": "private", "evidence": ["wrong quote"]}, {}, "evidence_not_caller_quote"),
        (
            {"content": "private", "evidence": ["What time is it?"]},
            {"What time is it?": "What time is it?"},
            "no_durable_evidence",
        ),
        (
            {"content": "private", "evidence": ["I prefer tea."], "category": "general"},
            {"I prefer tea.": "I prefer tea."},
            "general_category",
        ),
    ],
)
def test_rejections_record_only_fixed_reason_counts(raw, evidence, reason):
    counts = {}
    assert _validate_memory(raw, caller_evidence=evidence, rejection_counts=counts) is None
    assert counts == {reason: 1}
    assert "private" not in str(counts)


def test_seat_can_disable_ambient_capture_without_disabling_in_call_memory(monkeypatch):
    import musubi_livekit.voice.provider as provider

    class NoAmbient(Seat):
        postcall_capture_enabled = False

    calls = []
    monkeypatch.setattr(provider, "wire_postcall_memory", lambda *args, **kw: calls.append(kw))
    monkeypatch.setattr(provider, "wire_musubi_shutdown", lambda ctx: None)
    agent = NoAmbient(instructions="persona")
    agent.set_memory_write_policy(allowed=True, call_sid="SCL_real")
    agent.wire_call_memory(object(), object(), call_sid="SCL_real", capture_allowed=True)
    assert agent.memory_writes_allowed is True
    assert agent.memory_enabled() is True
    assert calls[0]["capture_allowed"] is False


@pytest.mark.asyncio
async def test_instruction_update_failure_still_warms_and_surfaces_error():
    seat = Seat(instructions="persona")
    seat.prefetch_memory_into_instructions = True
    seat.fetch_recent_context = AsyncMock(return_value="verified background context")
    seat.update_instructions = AsyncMock(side_effect=RuntimeError("update failed"))
    seat.warm_llm_prefix = AsyncMock()
    with pytest.raises(RuntimeError, match="update failed"):
        await seat._inject_recent_context(10)
    seat.warm_llm_prefix.assert_awaited_once()
