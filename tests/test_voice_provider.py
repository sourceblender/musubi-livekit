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


@pytest.mark.asyncio
async def test_extractor_emits_all_rejection_counts_without_candidate_content(monkeypatch, caplog):
    import json
    import logging
    from unittest.mock import MagicMock, patch

    from musubi_livekit.voice.postcall_memory import _extract_memories

    quote = "My train leaves Tuesday."
    transient = "What time is it?"
    private = "private-candidate-sentinel"
    candidates = [
        None,
        {"content": 123},
        {"content": " "},
        {"content": private},
        {"content": private, "evidence": [123]},
        {"content": private, "evidence": ["forged-private-quote"]},
        {"content": private, "evidence": [transient]},
        {"content": private, "evidence": [quote], "category": "general"},
        {"content": private, "evidence": [quote], "category": "planning"},
    ]
    client = MagicMock()
    client.models.generate_content.return_value.text = json.dumps({"memories": candidates})
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with (
        caplog.at_level(logging.INFO, logger="voice.agent"),
        patch("musubi_livekit.voice.postcall_memory.genai.Client", return_value=client),
    ):
        result = await _extract_memories(
            f"[10:00:00] [USER] {quote}\n[10:00:01] [USER] {transient}\n"
        )
    assert result.status == "extracted" and len(result.memories) == 1
    validation = [row.message for row in caplog.records if "validation raw=" in row.message]
    assert len(validation) == 1
    assert "raw=9 accepted=1 rejected=8" in validation[0]
    reasons = json.loads(validation[0].split("reasons=", 1)[1])
    assert reasons == dict.fromkeys(
        [
            "not_object",
            "content_not_string",
            "empty_content",
            "missing_evidence",
            "evidence_not_string",
            "evidence_not_caller_quote",
            "no_durable_evidence",
            "general_category",
        ],
        1,
    )
    for value in (private, "forged-private-quote", quote, transient):
        assert value not in caplog.text


@pytest.mark.asyncio
async def test_runtime_http_rejections_join_call_without_content(monkeypatch, tmp_path, caplog):
    import json
    import logging
    from unittest.mock import AsyncMock

    from aiohttp import web
    from google import genai
    from google.genai import types

    from musubi_livekit.voice import postcall_memory as memory

    quote = "My train leaves Tuesday."
    transient = "What time is it?"
    private = "private-candidate-sentinel"
    candidates = [
        None,
        {"content": 123},
        {"content": " "},
        {"content": private},
        {"content": private, "evidence": [123]},
        {"content": private, "evidence": ["forged-private-quote"]},
        {"content": private, "evidence": [transient]},
        {"content": private, "evidence": [quote], "category": "general"},
    ]
    requests = 0

    async def respond(request):
        nonlocal requests
        requests += 1
        assert request.method == "POST"
        return web.json_response(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": json.dumps({"memories": candidates})}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/{path:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = genai.Client(
        api_key="isolated-fake-key",
        http_options=types.HttpOptions(base_url=f"http://127.0.0.1:{port}"),
    )
    monkeypatch.setattr(memory.genai, "Client", lambda **kwargs: client)
    monkeypatch.setenv("GEMINI_API_KEY", "isolated-fake-key")
    monkeypatch.setenv("LIVEKIT_VOICE_LOGS", str(tmp_path))
    transcript = tmp_path / "phone-transcripts" / "isolated-rejection.txt"
    transcript.parent.mkdir()
    transcript.write_text(f"[10:00:00] [USER] {quote}\n[10:00:01] [USER] {transient}\n")
    capture = AsyncMock()
    monkeypatch.setattr(memory, "_capture_one", capture)
    try:
        with caplog.at_level(logging.INFO, logger="voice.agent"):
            count = await memory.run_extraction(
                call_sid="isolated-rejection",
                namespace="smoke/consumer/episodic",
                speaker_tag="isolated",
                client=object(),
            )
    finally:
        client.close()
        await runner.cleanup()
    assert requests == 1 and count == 0
    capture.assert_not_awaited()
    events = [
        json.loads(r.message.split("telemetry ", 1)[1])
        for r in caplog.records
        if r.message.startswith("postcall_memory: telemetry ")
    ]
    assert len(events) == 1
    assert events[0]["call_sid"] == "isolated-rejection"
    assert events[0]["status"] == "empty_extraction"
    assert events[0]["validation"] == {
        "raw": 8,
        "accepted": 0,
        "rejected": 8,
        "reasons": dict.fromkeys(
            [
                "not_object",
                "content_not_string",
                "empty_content",
                "missing_evidence",
                "evidence_not_string",
                "evidence_not_caller_quote",
                "no_durable_evidence",
                "general_category",
            ],
            1,
        ),
    }
    for value in (private, "forged-private-quote", quote, transient):
        assert value not in caplog.text


@pytest.mark.asyncio
async def test_skipped_validation_is_unknown_not_zero_rejections(monkeypatch, caplog):
    import json
    import logging

    from musubi_livekit.voice import postcall_memory as memory

    monkeypatch.setattr(memory, "_read_transcript", lambda _: None)
    with caplog.at_level(logging.INFO, logger="voice.agent"):
        assert (
            await memory.run_extraction(
                call_sid="no-transcript",
                namespace="smoke/consumer/episodic",
                speaker_tag=None,
            )
            == 0
        )
    event = next(
        json.loads(r.message.split("telemetry ", 1)[1])
        for r in caplog.records
        if r.message.startswith("postcall_memory: telemetry ")
    )
    assert event["status"] == "no_transcript"
    assert event["validation"] is None
