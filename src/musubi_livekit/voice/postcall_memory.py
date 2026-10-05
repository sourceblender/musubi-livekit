"""Post-call memory extraction — turn the transcript into focused memories.

After a call ends, the on-call LLM only saved what the user explicitly
asked for. Everything else — the texture, the side-threads, the
half-formed ideas — sits in the transcript and disappears unless we
extract it.

This module reads the saved transcript file, sends it to Gemini Flash
for *faithful chunked extraction* (preserve detail, don't summarise),
and posts each extracted moment as a normal episodic memory. Maturation
handles importance/topic enrichment on its hourly tick like any other
row.

Wire it into the agent's session via :func:`wire_postcall_memory`. That
registers a ``close`` handler which **spawns a detached subprocess** to
run the extraction. Subprocess (not asyncio task) because the LiveKit
worker process tears down after the job ends, killing any in-flight
coroutines on its event loop — which silently lost extractions on short
calls. The subprocess survives parent shutdown and runs to completion
in its own process group.

Failure modes:
- ``LIVEKIT_VOICE_LOGS`` unset → no-op (no transcript path to read).
- Gemini errors / malformed JSON → log and skip; explicit saves still
  landed during the call. The user loses texture for that one call.
- Capture errors (transient Musubi outage) → per-memory; we keep going.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import google.genai as genai
from google.genai import types as genai_types
from livekit.agents import AgentSession

from .client import (
    MusubiClient,
    MusubiClientConfig,
    MusubiError,
    close_shared_sessions,
)
from .policy import DEFAULT_WAKE_NAMES, load_policy
from .trace import trace

logger = logging.getLogger("voice.agent")


# --- controlled vocabulary --------------------------------------------------

CATEGORIES: tuple[str, ...] = (
    "personal",
    "work",
    "project",
    "idea",
    "decision",
    "health",
    "household",
    "learning",
    "planning",
    "reflection",
    "general",
)
"""Closed set of category tags. Each extracted memory gets exactly one,
attached as ``category:<value>``. Closed set keeps tag bloat bounded."""


_GEMINI_MODEL = "gemini-2.5-flash-lite"
"""Structured extraction model. Flash Lite is cheap, fast, and good
enough at structured-output extraction over a 4-8k token transcript.
Bump to Pro if quality drifts."""


_EXTRACTION_PROMPT = 'Process the following call transcript into durable caller memories. Return JSON\n{"memories": [{"content": "...", "evidence": ["exact complete USER quote"],\n"summary": "...", "topics": ["..."], "category": "personal"}]}.\nUse only exact complete [USER] lines as evidence. Never store assistant statements,\nrequests for time/weather, greetings, farewells, ephemeral dialogue or invented facts.\nCategories: personal, work, project, idea, decision, health, household, learning, planning, reflection. Preserve coherent facts from this call, not recalled historical records.\nReturn an empty memories array if no durable caller evidence exists.\n\nTranscript:\n'


# --- data shapes ------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedMemory:
    """One faithful chunk pulled from a transcript."""

    content: str
    summary: str
    topics: list[str]
    category: str


# --- transcript discovery ---------------------------------------------------


def _voice_logs() -> Path | None:
    logs = os.environ.get("LIVEKIT_VOICE_LOGS")
    return Path(logs) if logs else None


def _transcript_path(call_sid: str) -> Path | None:
    base = _voice_logs()
    return base / "phone-transcripts" / f"{call_sid}.txt" if base else None


#: Every physical line a transcript writer produces: a timestamped record, the
#: per-call header, or blank.
_WELL_FORMED_LINE_RE = re.compile(r"^(?:\[[^\]\n]+\]\s+\[[A-Z_]+\]\s.*|=== Call .*===|\s*)$")


def _is_well_formed(transcript: str) -> bool:
    """Whether every line came from the transcript writer's record format.

    sdk.transcript writes one physical line per item, and _USER_LINE_RE below
    reads any line shaped like ``[...] [USER] ...`` as a caller quote that
    post-call extraction stamps ``provenance:caller-quote``. Text carrying an
    embedded newline used to be written raw, so an assistant reply could open
    a line of its own and forge that record. The writer now collapses control
    characters; this is the invariant that makes the guarantee checkable, and
    it fails closed on a file written before the fix.
    """
    return all(_WELL_FORMED_LINE_RE.match(line) for line in transcript.splitlines())


def _read_transcript(call_sid: str) -> str | None:
    """Read the per-call transcript file, or None if missing/unreadable."""
    path = _transcript_path(call_sid)
    if path is None or not path.exists():
        return None
    try:
        transcript = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("postcall_memory: transcript read failed: %s", exc)
        return None
    if not _is_well_formed(transcript):
        # Writing nothing is this module's established answer to evidence it
        # cannot trust, and an unattributable line is exactly that.
        logger.error(
            "postcall_memory: transcript %s has lines outside the record format; "
            "refusing extraction rather than risk attributing one to the caller",
            call_sid,
        )
        return None
    return transcript


# --- extraction -------------------------------------------------------------


def _gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


_USER_LINE_RE = re.compile(r"^\[[^\]\n]+\]\s+\[USER\]\s+(.+?)\s*$", re.MULTILINE)
_LEADING_CHATTER_RE = re.compile(
    r"^(?:(?:okay|ok|hey|hi|uh|um|well|so|and|last|thanks|thank\s+you|"
    r"perfect|all\s+right|alright|(?!))\b[,\s.:;!\-]*)",
    re.IGNORECASE,
)
_TRAILING_CHATTER_RE = re.compile(
    r"(?:[,\s.:;!\-]+(?:thanks|thank\s+you|(?!)))"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
_NON_DURABLE_EVIDENCE_RE = re.compile(
    r"""
    (?:okay|ok|yeah|yep|yes|no|perfect|great|go\s+on|sounds\s+good|
       thanks|thank\s+you|any\s+response)[.!?]*
    |
    (?:tell\s+me\s+)?what\s+(?:do\s+)?you\s+remember\s*[?.!]*
    |
    (?:can\s+you\s+hear\s+me|are\s+you\s+there|you\s+still\s+there)\s*[?.!]*
    |
    (?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:tell\s+me\s+)?
       (?:what\s+time\s+(?:is\s+it|it\s+is)|
          what(?:'s|\s+is)\s+the\s+(?:current\s+)?time)
       (?:\s+(?:right\s+now|now|here))?\s*[?.!]*
    |
    (?:do\s+you\s+know\s+)?what\s+time\s+(?:is\s+it|it\s+is)\s*[?.!]*
    |
    (?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?:tell\s+me\s+)?
       (?:what(?:'s|\s+is)\s+(?:today(?:'s)?|the\s+current)\s+(?:date|day)|
          what\s+(?:date|day)\s+(?:is\s+it|it\s+is|is\s+today))\s*[?.!]*
    |
    (?:(?:can|could|would)\s+you\s+)?(?:please\s+)?
       (?:(?:tell\s+me\s+)?what(?:'s|\s+is)\s+the\s+weather(?:\s+like)?|
          (?:check|give\s+me)\s+the\s+(?:weather|forecast)|
          how(?:'s|\s+is)\s+the\s+weather)
       (?:\s+(?:for\s+me|today|now|right\s+now|here))?\s*[?.!]*
    |
    (?:(?:can|could|would)\s+you\s+)?(?:get|check)\s+(?:me\s+)?(?:the\s+)?
       (?:weather|forecast|time)(?:\s+(?:or|and)\s+(?:the\s+)?
       (?:weather|forecast|time))?(?:\s+currently)?\s*[?.!]*
    |
    (?:(?:can|could|would)\s+you\s+)?(?:please\s+)?
       (?:tell|read|make\s+up)\s+(?:me\s+)?(?:a\s+)?
       (?:(?:another|nice|long|short|funny|bedtime)\s+)*(?:story|joke|one)
       (?:\s+about\s+.+|\s*,\s*(?:a\s+)?(?:little\s+)?
       (?:(?:nice|long|short|funny|bedtime)\s+)*(?:story|joke|one))?\s*[?.!]*
    |
    (?:(?:okay|ok|hey|(?!))[,\s]+)?
       (?:please\s+)?(?:stop|hold\s+on|wait)(?:\s+(?:talking|there))?\s*[?.!]*
    |
    (?:(?:okay|perfect|alright)[,\s.!]+)?
       (?:(?:(?!))[,\s.!]+)?
       (?:goodbye|good\s+bye|bye(?:\s+bye)?|good\s*night|have\s+a\s+good\s+night|
          talk\s+to\s+you\s+later)
       (?:[,\s]+(?:(?!)))?\s*[?.!]*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalized_text(text: str) -> str:
    return " ".join(text.split())


def is_durable_caller_evidence(
    text: str, *, wake_names: tuple[str, ...] = DEFAULT_WAKE_NAMES
) -> bool:
    """Return whether one exact caller quote may support ambient memory.

    Leading discourse markers and seat addresses do not turn a transient
    request into durable evidence. Mixed lines remain eligible when they begin
    with substantive caller content; the model may retain that whole exact
    quote, but this function never extracts facts from it.
    """

    names = "|".join(re.escape(x) for x in wake_names) or "(?!)"
    leading = re.compile(
        _LEADING_CHATTER_RE.pattern.replace("(?!)", names), _LEADING_CHATTER_RE.flags
    )
    trailing = re.compile(
        _TRAILING_CHATTER_RE.pattern.replace("(?!)", names), _TRAILING_CHATTER_RE.flags
    )
    non_durable = re.compile(
        _NON_DURABLE_EVIDENCE_RE.pattern.replace("(?!)", names), _NON_DURABLE_EVIDENCE_RE.flags
    )
    core = _normalized_text(text)
    while core:
        stripped = leading.sub("", core, count=1).strip()
        if stripped == core:
            break
        core = stripped
    core = trailing.sub("", core).strip()
    return bool(core) and non_durable.fullmatch(core) is None


def _caller_evidence(transcript: str) -> dict[str, str]:
    """Normalized exact caller lines keyed back to their transcript wording."""

    evidence: dict[str, str] = {}
    for match in _USER_LINE_RE.finditer(transcript):
        original = _normalized_text(match.group(1))
        if original:
            evidence.setdefault(original, original)
    return evidence


def _evidence_backed_content(quotes: list[str], *, caller_label: str = "Caller") -> str:
    """Store only caller-originated wording; Gemini may classify, never invent facts."""

    if len(quotes) == 1:
        return f"{caller_label} said: {quotes[0]}"
    return f"{caller_label} said: " + " Then: ".join(quotes)


def _validate_memory(
    raw: Any,
    *,
    caller_evidence: dict[str, str] | None = None,
    rejection_counts: dict[str, int] | None = None,
    caller_label: str = "Caller",
    wake_names: tuple[str, ...] = DEFAULT_WAKE_NAMES,
) -> ExtractedMemory | None:
    """Coerce a raw dict from Gemini into a clean :class:`ExtractedMemory`,
    or return None if the shape is bad enough to drop.

    Forgiving: missing fields fill with sensible defaults. Non-string
    values for the string fields (``content``, ``summary``, ``category``)
    are rejected explicitly so a payload like ``{"content": 123}``
    drops the row cleanly instead of raising ``AttributeError`` on
    ``.strip()`` — that crash used to propagate up through the
    validation loop and abort the whole extraction, an unclassified
    failure that bypassed the typed-status surface.
    """

    def reject(reason: str) -> None:
        if rejection_counts is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        return None

    if not isinstance(raw, dict):
        reject("not_object")
        return None
    raw_content = raw.get("content")
    if not isinstance(raw_content, str):
        reject("content_not_string")
        return None
    content = raw_content.strip()
    if not content:
        reject("empty_content")
        return None
    if caller_evidence is not None:
        raw_evidence = raw.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            reject("missing_evidence")
            return None
        quotes: list[str] = []
        for item in raw_evidence[:3]:
            if not isinstance(item, str):
                reject("evidence_not_string")
                return None
            normalized = _normalized_text(item)
            exact = caller_evidence.get(normalized)
            if exact is None:
                reject("evidence_not_caller_quote")
                return None
            if not is_durable_caller_evidence(exact, wake_names=wake_names):
                continue
            if exact not in quotes:
                quotes.append(exact)
        if not quotes:
            reject("no_durable_evidence")
            return None
        content = _evidence_backed_content(quotes, caller_label=caller_label)
    raw_summary = raw.get("summary")
    summary = (raw_summary if isinstance(raw_summary, str) else content[:80]).strip()
    raw_topics = raw.get("topics") or []
    topics: list[str] = []
    if isinstance(raw_topics, list):
        for t in raw_topics[:5]:
            if isinstance(t, str) and t.strip():
                topics.append(t.strip().lower())
    raw_category = raw.get("category")
    category = (raw_category if isinstance(raw_category, str) else "general").strip().lower()
    if category not in CATEGORIES:
        category = "general"
    # Ambient transcript extraction must name a durable category. ``general``
    # is useful as a shape-compatible fallback for programmatic callers, but
    # it is not enough authority to persist call chatter. The 2026-09-13 Assistant
    # call otherwise stored "Can you tell me what time it is?" as durable
    # memory despite the quote being truthful.
    if caller_evidence is not None and category == "general":
        reject("general_category")
        return None
    return ExtractedMemory(
        content=content,
        summary=summary,
        topics=topics,
        category=category,
    )


# Status hints emitted by `_extract_memories` so the caller's
# completion log distinguishes "Gemini auth broke" from "transcript was
# genuinely empty" — the conflation that hid the 2026-05-15 voice-path
# silent-loss for three days (see #29).
#
# `extracted` = memories list non-empty, capture step runs next.
# `empty_extraction` = valid transcript reached Gemini, Gemini returned
#   no memory objects. Genuinely uneventful call.
# `no_transcript_text` = whitespace-only after interrupted-candidate filtering (handled here;
#   `run_extraction` separately handles the "no transcript file" case
#   as `no_transcript`).
# `no_api_key` = GEMINI_API_KEY / GOOGLE_API_KEY unset.
# `auth_failed` = Gemini returned 401 / 403 / UNAUTHENTICATED. The
#   typical credential-rotation-not-propagated failure (see
#   wiki/gotchas/voice-deploy-traps §1).
# `transport_failed` = catch-all non-auth Gemini call failure (network
#   error, timeout, connection refused, unexpected SDK-side exception).
#   The provider may or may not have been reached — distinguishing them
#   requires more SDK-specific exception introspection than is worth
#   the complexity for the operator-side signal. Treat as "something
#   went wrong outside the auth check; look at the ERROR log line above
#   the completion line for specifics."
# `parse_failed` = Gemini returned non-JSON or non-conformant JSON
#   (no `memories` array, wrong shape).
ExtractionStatus = Literal[
    "extracted",
    "empty_extraction",
    "no_transcript_text",
    "no_api_key",
    "auth_failed",
    "transport_failed",
    "parse_failed",
]


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of one Gemini extraction call.

    ``memories`` is empty for every status except ``extracted``. The
    ``status`` field carries the cause when ``memories`` is empty so
    the completion-log line can distinguish auth failure from
    genuine no-extraction (the silent-loss class fixed in
    #29).
    """

    memories: list[ExtractedMemory]
    status: ExtractionStatus


def _classify_gemini_exception(exc: BaseException) -> ExtractionStatus:
    """Map a Gemini SDK exception to an `ExtractionStatus`.

    The google-genai SDK doesn't expose typed auth/transport exceptions
    we can isinstance-match cleanly. Falls back to substring matching
    on the formatted message — the same shape the production 2026-05-15
    incident surfaced (``"401 UNAUTHENTICATED"`` substring). Order
    matters: auth check before transport so a 401 reaching us via a
    transport-shaped exception still classifies correctly.
    """
    msg = str(exc)
    if "401" in msg or "403" in msg or "UNAUTHENTICATED" in msg or "PERMISSION_DENIED" in msg:
        return "auth_failed"
    return "transport_failed"


async def _extract_memories(transcript: str, *, policy_path: str | None = None) -> ExtractionResult:
    """Send the transcript to Gemini Flash and parse the JSON response.

    Returns an :class:`ExtractionResult` whose ``status`` field
    distinguishes the failure modes that used to collapse to ``[]``:
    auth failure, transport failure, parse failure, missing API key,
    or genuinely empty extraction. The caller uses the status hint
    on the completion log line.
    """
    from .transcript import memory_eligible_transcript

    policy = load_policy(policy_path)
    transcript = memory_eligible_transcript(transcript)
    if not transcript.strip():
        return ExtractionResult(memories=[], status="no_transcript_text")
    api_key = _gemini_api_key()
    if not api_key:
        logger.warning("postcall_memory: no Gemini API key, skipping extraction")
        return ExtractionResult(memories=[], status="no_api_key")

    client = genai.Client(api_key=api_key)

    def _call() -> str:
        # Synchronous call wrapped via to_thread below. The genai client
        # has both sync and async surfaces; sync is simpler here and we
        # don't care about latency in the post-call window.
        resp = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=policy.get("extraction_prompt", _EXTRACTION_PROMPT) + transcript,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return resp.text or ""

    try:
        raw_text = await asyncio.to_thread(_call)
    except Exception as exc:
        logger.error("postcall_memory: Gemini call failed: %s", exc)
        return ExtractionResult(memories=[], status=_classify_gemini_exception(exc))

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("postcall_memory: malformed JSON from Gemini: %s", exc)
        return ExtractionResult(memories=[], status="parse_failed")

    raw_memories = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(raw_memories, list):
        return ExtractionResult(memories=[], status="parse_failed")

    caller_evidence = _caller_evidence(transcript)
    out: list[ExtractedMemory] = []
    rejection_counts: dict[str, int] = {}
    for r in raw_memories:
        m = _validate_memory(
            r,
            caller_evidence=caller_evidence,
            rejection_counts=rejection_counts,
            caller_label=policy.get("caller_label", "Caller"),
            wake_names=tuple(policy.get("wake_names", DEFAULT_WAKE_NAMES)),
        )
        if m is not None:
            out.append(m)
    logger.info(
        "postcall_memory: validation raw=%d accepted=%d rejected=%d reasons=%s",
        len(raw_memories),
        len(out),
        len(raw_memories) - len(out),
        json.dumps(rejection_counts, sort_keys=True, separators=(",", ":")),
    )
    if not out:
        # Gemini reached + parsed but produced 0 valid memories. This is
        # the genuine "uneventful call" path, distinct from auth/parse
        # failures above.
        return ExtractionResult(memories=[], status="empty_extraction")
    return ExtractionResult(memories=out, status="extracted")


# --- capture ----------------------------------------------------------------


async def _capture_one(
    *,
    client: MusubiClient,
    namespace: str,
    memory: ExtractedMemory,
    speaker_tag: str | None,
    call_sid: str,
) -> bool:
    """Capture one extracted memory. Returns True on success.

    Tags are: the extracted topics, plus ``category:<value>``, plus the
    agent's speaker tag (e.g. ``assistant-voice``), plus a ``source:transcript``
    marker so we can tell extracted rows from explicit saves at audit
    time. Importance defaults to 5; maturation rescores hourly.
    """
    tags = list(memory.topics)
    tags.append(f"category:{memory.category}")
    tags.append("source:transcript")
    tags.append("provenance:caller-quote")
    if speaker_tag:
        tags.append(speaker_tag)

    # Derived, not random. A fresh UUID per attempt is the one key shape that
    # cannot deduplicate anything: a timeout after Musubi accepted the POST is
    # reported to us as failure, and the retry then wrote a second row.
    idem = "livekit-postcall:{}:{}".format(
        call_sid,
        hashlib.sha256(
            "\x1f".join([namespace, memory.category, memory.content, *sorted(tags)]).encode()
        ).hexdigest()[:32],
    )
    try:
        ack = await client.capture_memory(
            namespace=namespace,
            content=memory.content,
            tags=tags,
            importance=5,
            idempotency_key=idem,
        )
    except MusubiError as exc:
        logger.warning(
            "postcall_memory: capture failed for call_sid=%s: %s",
            call_sid,
            exc,
        )
        return False

    object_id = ack.get("object_id") or "<unknown>"
    trace(
        f"postcall_memory: captured object_id={object_id} category={memory.category} "
        f"call_sid={call_sid}"
    )
    return True


async def run_extraction(
    *,
    call_sid: str,
    namespace: str,
    speaker_tag: str | None,
    client: MusubiClient | None = None,
    policy_path: str | None = None,
) -> int:
    """Read the transcript, extract memories, capture them all.

    Returns the count of memories successfully captured. 0 means either
    no transcript, no extraction, or all captures failed. Designed to be
    called via ``asyncio.create_task`` from the close handler — never
    blocks the caller.

    If ``client`` is None, builds one from environment via
    :meth:`MusubiClientConfig.from_env`.
    """
    started = time.monotonic()

    def _complete(status: str, *, extracted: int = 0, captured: int = 0) -> int:
        """Single completion log line so audit/Rin can grep one shape.

        Status is one of:
        - ``no_transcript`` (transcript file missing / unreadable)
        - ``no_transcript_text`` (body whitespace-only after interrupted-candidate filtering)
        - ``no_api_key`` (GEMINI_API_KEY/GOOGLE_API_KEY unset)
        - ``auth_failed`` (Gemini 401/403 — most often a stale key
          per wiki/gotchas/voice-deploy-traps §1)
        - ``transport_failed`` (Gemini network/timeout error)
        - ``parse_failed`` (Gemini returned non-JSON or wrong shape)
        - ``empty_extraction`` (Gemini reached + parsed but produced
          zero memories — genuinely uneventful call)
        - ``captured`` (memories extracted AND at least one capture
          succeeded)
        - ``no_captures`` (memories extracted but every capture failed)

        Pre-#29, ``auth_failed`` / ``transport_failed`` /
        ``parse_failed`` / ``empty_extraction`` all logged as
        ``empty_extraction`` — silently hid the 2026-05-15 voice path
        breakage for three days.
        """
        total_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "postcall_memory: completed call_sid=%s status=%s extracted=%d captured=%d total_ms=%d",
            call_sid,
            status,
            extracted,
            captured,
            total_ms,
        )
        trace(
            f"postcall_memory: completed call_sid={call_sid} status={status} "
            f"extracted={extracted} captured={captured} total_ms={total_ms}"
        )
        return captured

    transcript = _read_transcript(call_sid)
    if transcript is None:
        return _complete("no_transcript")

    result = (
        await _extract_memories(transcript, policy_path=policy_path)
        if policy_path
        else await _extract_memories(transcript)
    )
    if result.status != "extracted":
        # Propagate the typed status from _extract_memories — distinguishes
        # auth/transport/parse failures from genuine empty extraction.
        return _complete(result.status)

    if client is None:
        cfg = MusubiClientConfig.from_env()
        client = MusubiClient(cfg)

    captured = 0
    for memory in result.memories:
        ok = await _capture_one(
            client=client,
            namespace=namespace,
            memory=memory,
            speaker_tag=speaker_tag,
            call_sid=call_sid,
        )
        if ok:
            captured += 1

    status = "captured" if captured > 0 else "no_captures"
    return _complete(status, extracted=len(result.memories), captured=captured)


# --- wiring -----------------------------------------------------------------


def wire_postcall_memory(
    session: AgentSession[Any],
    *,
    call_sid: str | None,
    namespace: str | None,
    speaker_tag: str | None,
    capture_allowed: bool,
    policy_path: str | None = None,
) -> None:
    """Register a ``close`` handler that runs transcript extraction.

    Call this AFTER ``wire_postcall_review`` in the agent entrypoint.
    Both can register on the same session — livekit allows multiple
    listeners on the ``close`` event.

    ``capture_allowed`` must come from an authoritative caller-origin check.
    Phone workers pass ``caller.source == "sip"`` so synthetic and unresolved
    dispatches fail closed. The call-ID check remains defense in depth.

    Also no-ops if any of ``call_sid``, ``namespace``, or
    ``LIVEKIT_VOICE_LOGS`` is missing.
    """
    if not capture_allowed:
        trace(f"postcall_memory: capture not allowed call_sid={call_sid}, not wiring")
        return
    if not call_sid:
        trace("postcall_memory: no call_sid, not wiring")
        return
    if "synthetic" in call_sid.lower().replace("_", "-").split("-"):
        trace(f"postcall_memory: synthetic call_sid={call_sid}, not wiring")
        return
    if not namespace:
        trace("postcall_memory: no namespace, not wiring")
        return
    if _voice_logs() is None:
        trace("postcall_memory: LIVEKIT_VOICE_LOGS unset, not wiring")
        return

    @session.on("close")
    def _on_close(ev: Any) -> None:
        _spawn_extraction_subprocess(
            call_sid=call_sid,
            namespace=namespace,
            speaker_tag=speaker_tag,
            **({"policy_path": policy_path} if policy_path else {}),
        )

    trace(f"postcall_memory wired call_sid={call_sid} namespace={namespace}")


# --- subprocess spawn -------------------------------------------------------


def _postcall_logfile() -> Path | None:
    """Where extraction-subprocess stdout/stderr land. One shared file
    across all calls — the per-line ``call_sid=`` makes ``grep`` cheap."""
    base = _voice_logs()
    return base / "postcall-memory.log" if base else None


def _spawn_extraction_subprocess(
    *,
    call_sid: str,
    namespace: str,
    speaker_tag: str | None,
    policy_path: str | None = None,
) -> None:
    """Spawn a detached Python subprocess to run :func:`run_extraction`.

    Uses ``sys.executable -m musubi_livekit.voice.postcall_memory`` so the subprocess
    runs inside the same venv as the parent agent. The child inherits
    the parent's environment (Gemini key, Musubi base URL + token,
    voice-logs dir) so no extra wiring is needed.

    Failures to spawn (FileNotFoundError, PermissionError, OSError)
    log at ERROR but do not raise — Path B is best-effort.
    """
    logfile = _postcall_logfile()

    args = [
        sys.executable,
        "-m",
        "musubi_livekit.voice.postcall_memory",
        "--call-sid",
        call_sid,
        "--namespace",
        namespace,
    ]
    if speaker_tag:
        args.extend(["--speaker-tag", speaker_tag])
    if policy_path:
        load_policy(policy_path)
        args.extend(["--policy-file", policy_path])

    try:
        if logfile is not None:
            # Parent opens, passes fd to child via Popen's dup, closes its
            # own copy. Child keeps its dup until it exits naturally.
            with logfile.open("a", encoding="utf-8") as fp:
                subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=fp,
                    stderr=fp,
                    start_new_session=True,
                    close_fds=True,
                )
        else:
            subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        trace(f"postcall_memory: spawned subprocess call_sid={call_sid}")
        logger.info("postcall_memory: spawned subprocess call_sid=%s", call_sid)
    except Exception as exc:
        logger.error("postcall_memory: failed to spawn subprocess: %s", exc)
        trace(f"postcall_memory: spawn failed call_sid={call_sid}: {exc}")


# --- CLI entry --------------------------------------------------------------


def _cli_main() -> int:
    """``python -m musubi_livekit.voice.postcall_memory --call-sid X --namespace Y [--speaker-tag Z]``

    Subprocess entry. Logs to whatever stdout/stderr was inherited by the
    spawn — typically ``$LIVEKIT_VOICE_LOGS/postcall-memory.log`` per
    :func:`_spawn_extraction_subprocess`. Exits 0 on completion (incl.
    no_transcript / empty_extraction); 2 on argparse errors.
    """
    parser = argparse.ArgumentParser(prog="musubi_livekit.voice.postcall_memory")
    parser.add_argument("--call-sid", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--speaker-tag", default=None)
    parser.add_argument("--policy-file", default=None)
    args = parser.parse_args()

    # Subprocess inherits no logging handlers; configure a minimal one so
    # logger.info / logger.error lines actually reach the inherited stderr.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _run_and_close() -> None:
        try:
            await run_extraction(
                call_sid=args.call_sid,
                namespace=args.namespace,
                speaker_tag=args.speaker_tag,
                policy_path=args.policy_file,
            )
        finally:
            # The detached subprocess owns its event loop and exits after one
            # extraction, so its shared keep-alive session must be closed here.
            await close_shared_sessions()

    asyncio.run(_run_and_close())
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
