import json

import pytest

pytest.importorskip("livekit.agents")
pytest.importorskip("google.genai")

from musubi_livekit.voice.policy import load_policy
from musubi_livekit.voice.postcall_memory import _evidence_backed_content, _validate_memory
from musubi_livekit.voice.tools import _is_recallable_memory


def test_application_format_remains_byte_exact_through_save_and_recall():
    quote = "I prefer green tea."
    result = _validate_memory(
        {"content": "candidate", "evidence": [quote], "category": "personal"},
        caller_evidence={quote: quote},
        caller_label="Person",
        wake_names=("buddy",),
    )
    assert result is not None
    assert result.content == "Person said: I prefer green tea."
    row = {
        "content": result.content,
        "tags": ["source:transcript", "provenance:caller-quote", "category:personal"],
    }
    assert _is_recallable_memory(row, caller_label="Person", wake_names=("buddy",))
    assert not _is_recallable_memory(row)
    assert _evidence_backed_content([quote, "I drink it before noon."], caller_label="Person") == (
        "Person said: I prefer green tea. Then: I drink it before noon."
    )


def test_application_wake_name_does_not_make_time_request_durable():
    quote = "Hey Buddy, what time is it?"
    assert (
        _validate_memory(
            {"content": "candidate", "evidence": [quote], "category": "personal"},
            caller_evidence={quote: quote},
            wake_names=("buddy",),
        )
        is None
    )


def test_private_policy_load_is_bounded_and_has_no_implicit_app_default(tmp_path):
    path = tmp_path / "policy.json"
    payload = {
        "caller_label": "Person",
        "wake_names": ["buddy"],
        "extraction_prompt": "Private app instructions.",
    }
    path.write_text(json.dumps(payload))
    assert load_policy(str(path)) == payload
    assert load_policy(None) == {}
    path.write_text(json.dumps({"unknown": "discarding this would hide a typo"}))
    with pytest.raises(ValueError):
        load_policy(str(path))
    path.write_text(" " * 65537)
    with pytest.raises(ValueError):
        load_policy(str(path))
