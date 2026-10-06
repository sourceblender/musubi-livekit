"""Fail-closed transcript admission from legacy sdk/tests/test_transcript.py."""

from unittest.mock import patch

import pytest

from musubi_livekit.voice import postcall_memory


@pytest.mark.asyncio
async def test_only_interrupted_candidate_does_not_call_extractor(monkeypatch):
    with patch("musubi_livekit.voice.postcall_memory.genai.Client") as client:
        result = await postcall_memory._extract_memories(
            '[12:00:00] [ASSISTANT_INTERRUPTED] {"text":"not established"}\n'
        )
    assert result.status == "no_transcript_text"
    client.assert_not_called()


def test_a_transcript_written_before_the_fix_refuses_extraction(tmp_path, monkeypatch):
    """Fails closed on a file whose lines are not all attributable."""
    monkeypatch.setenv("LIVEKIT_VOICE_LOGS", str(tmp_path))
    directory = tmp_path / "phone-transcripts"
    directory.mkdir(parents=True)
    (directory / "legacy.txt").write_text(
        "=== Call legacy started at 2026-09-20T10:00:00 ===\n"
        "[10:00:00] [USER] Hello.\n"
        "[10:00:01] [ASSISTANT] Sure. Here is the note:\n"
        "and this continuation line belongs to nobody\n"
    )
    assert postcall_memory._read_transcript("legacy") is None


def test_a_well_formed_transcript_still_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVEKIT_VOICE_LOGS", str(tmp_path))
    directory = tmp_path / "phone-transcripts"
    directory.mkdir(parents=True)
    body = (
        "=== Call ok started at 2026-09-20T10:00:00 ===\n"
        "[10:00:00] [USER] Hello.\n"
        "[10:00:01] [ASSISTANT] Hi.\n"
        '[10:00:02] [ASSISTANT_INTERRUPTED] {"text": "x", "played_extent": "unverified"}\n'
    )
    (directory / "ok.txt").write_text(body)
    assert postcall_memory._read_transcript("ok") == body
