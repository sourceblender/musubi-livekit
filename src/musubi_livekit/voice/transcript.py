"""Consumer validation of voice.played-speech.v1 transcript records.

Runtime certification is engine-owned; this refuses uncertified serialized text.
"""

import hashlib
import json
import re


def valid_played_speech_association(plan_origin: object, association: object) -> bool:
    return (plan_origin, association) in {
        ("generated", "current_generated_speech"),
        ("say_atomic", "handle_chat_item"),
    }


_INTERRUPTED_LINE_RE = re.compile(
    r"^(?P<timestamp>\[[^\]\n]+\]) \[ASSISTANT_INTERRUPTED\] (?P<payload>.+)$"
)
_FAILED_OUTPUT_LINE_RE = re.compile(r"^\[[^\]\n]+\] \[ASSISTANT_UNVERIFIED\] ")


def memory_eligible_transcript(transcript: str) -> str:
    """Exclude interrupted SDK text whose played extent is not established.

    These diagnostic records are single-line JSON, not ordinary assistant
    speech. Do not feed their candidate text to extraction even as context.
    Older transcripts cannot be repaired retrospectively from plain text.
    """
    eligible: list[str] = []
    for line in transcript.splitlines(keepends=True):
        if _FAILED_OUTPUT_LINE_RE.match(line):
            continue
        match = _INTERRUPTED_LINE_RE.match(line.rstrip("\r\n"))
        if match is None:
            eligible.append(line)
            continue
        try:
            payload = json.loads(match.group("payload"))
        except (TypeError, ValueError):
            continue
        text = payload.get("text") if isinstance(payload, dict) else None
        if (
            isinstance(text, str)
            and text.strip()
            and payload.get("played_extent") == "local_playout_verified"
            and payload.get("schema") == "voice.played-speech.v1"
            and isinstance(payload.get("plan_id"), str)
            and bool(payload["plan_id"])
            and isinstance(payload.get("plan_origin"), str)
            and bool(payload["plan_origin"])
            and valid_played_speech_association(
                payload.get("plan_origin"), payload.get("association")
            )
            and payload.get("text_sha256")
            == hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        ):
            suffix = "\n" if line.endswith(("\n", "\r")) else ""
            eligible.append(f"{match.group('timestamp')} [ASSISTANT] {_one_line(text)}{suffix}")
    return "".join(eligible)


_LINE_BREAKING = re.compile("[\\x00-\\x1f\\x7f-\\x9f\\u2028\\u2029]+")


def _one_line(text: str) -> str:
    """Collapse anything that could open a new transcript line.

    The record format is one physical line per item, and postcall_memory's
    _USER_LINE_RE reads any line matching ``[...] [USER] ...`` as a caller
    quote -- with MULTILINE, a line start is exactly what follows a newline.
    So an assistant reply containing a newline and that prefix became a
    caller quote, and post-call extraction then stamped it
    ``provenance:caller-quote``. The interrupted-assistant record below has
    JSON-escaped its line breaks since it was added for this reason; the
    ordinary path wrote raw.
    """
    return _LINE_BREAKING.sub(" ", text).strip()
