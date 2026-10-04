"""MusubiToolsMixin — the agent-tools surface for voice agents.

Three LLM-exposed tools: ``musubi_recent``, ``musubi_search``,
``musubi_remember``. Identical names + parameter shapes across every Musubi
adapter (this mixin, the browser plugin, the Python MCP adapter) so Caller
or any model gets the same surface regardless of modality.

``musubi_think`` (presence-to-presence send) is **retained in code but not
exposed to the LLM** as of 2026-07-10. The live comms webbing reaches peers
via agent-bridge (TUI inject), not a Musubi thought-inbox scroll — no
presence subscribes to ``/v1/thoughts/stream`` for these four agents — so a
phone-side send would land in a plane nobody checks, and a persona that says
"never say you passed something along" cannot truthfully offer it. ``think_impl``
and ``MusubiClient.send_thought`` stay ready; re-add the ``@function_tool``
wrapper if a real consumer is wired. See the deleted ``musubi_think`` tool below.

``musubi_get`` was removed 2026-07-09. It was registered as a tool but
returned a "not yet available" message, because the Python MusubiClient
never gained per-plane ``get(object_id)`` accessors. A prompt-visible tool
the runtime cannot fulfil is a fabrication generator — the same defect that
took ``openclaw_delegate`` down. Reserving a name is not worth teaching the
model to reach for something that isn't there. Use ``musubi_search``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import math
import re
import secrets
import time
import uuid
from typing import Any

from livekit.agents import Agent, function_tool

from .client import (
    MusubiAuthError,
    MusubiClient,
    MusubiClientConfig,
    MusubiClientError,
    MusubiError,
    MusubiServerError,
    MusubiTimeoutError,
)
from .identity import UNCONFIGURED_CONFIG, MemoryIdentity
from .postcall_memory import CATEGORIES, is_durable_caller_evidence
from .trace import trace

logger = logging.getLogger("voice.agent")
_SAVE_LOG_KEY = secrets.token_bytes(32)


def _save_log_ref(idempotency_key: str) -> str:
    """Correlate cancels within one worker without exposing a guessable save hash."""
    return hmac.new(_SAVE_LOG_KEY, idempotency_key.encode(), hashlib.sha256).hexdigest()[:16]


_DEGRADED_LOOKUP = "Couldn't check memory — Musubi is unavailable right now."
_UNVERIFIED_LOOKUP = "Couldn't verify those memories right now."
_DEGRADED_STORE = "Memory didn't save — Musubi is unavailable right now."
MEMORY_STORED_LINE = "Got it, stored."
#: Marks an in-call save as authored by the model during the conversation,
#: distinct from provenance:caller-quote, which post-call extraction attaches
#: only to Caller's exact transcript wording.
_TOOL_PROVENANCE = "provenance:in-call-tool"
_MAX_RECENT_LIMIT = 20
_MAX_SEARCH_LIMIT = 10
_SEARCH_METADATA_MAX = 30
_SEARCH_METADATA_CONCURRENCY = 4
_SEARCH_METADATA_BUDGET_S = 1.2
# Over-fetch factor for the tag filter. Musubi's GET /v1/episodic is ~40 ms
# per ROW (measured 2026-08-18: limit=50 -> 1.96 s, right at the 2.0 s client
# timeout — the cause of the "Musubi is acting up" call failure), so page size
# is the latency knob. Nearly every row carries the agent tag in practice
# (94/96 sampled), so 2x is ample headroom; 5x made the greeting prefetch
# time out against a healthy server.
_SCROLL_MULTIPLIER = 2
_DEFAULT_IMPORTANCE = 7
# Pages of GET /v1/episodic to walk while gathering callout-worthy rows.
# Bounded so a token-mismatch on this agent's tag doesn't spiral into
# scrolling the whole namespace.
_MAX_RECENT_PAGES = 5
_RECENT_CONTEXT_TIMEOUT_S = 3.0
# In-call cache for recent rows. The start-of-call prefetch and a mid-call
# "what have you been up to?" musubi_recent ask for the SAME rows (nothing new
# lands in the agent's own stream until the call's post-call capture), yet the
# tool re-paid 0.4-0.8 s of Musubi scroll on top of the LLM's tool round-trip
# (2026-08-20 harness: the tool turn was the slowest on every call). Serve the
# tool from the prefetch when it is fresh and large enough; a musubi_remember
# during the call invalidates it so a just-saved memory is visible.
_RECENT_CACHE_TTL_S = 900.0

# Search-side state filter. Default Musubi retrieve hides `provisional` so
# unscored ambient captures don't pollute results, but a deliberate
# `musubi_remember` from another channel sits as `provisional` until the
# hourly maturation cron runs. For explicit recall we want fresh
# deliberate stores visible immediately — opt into provisional alongside
# the default `(matured, promoted)`. Per Musubi v1.2.0 / state_filter API.
_SEARCH_STATE_FILTER = ["provisional", "matured", "promoted"]
_SEARCH_QUERY_SCAFFOLD = frozenset(
    {
        "about",
        "actually",
        "ago",
        "all",
        "alright",
        "and",
        "any",
        "anything",
        "are",
        "been",
        "before",
        "but",
        "can",
        "could",
        "decide",
        "decided",
        "did",
        "didn",
        "discuss",
        "discussed",
        "does",
        "don",
        "day",
        "earlier",
        "caller",
        "first",
        "for",
        "from",
        "had",
        "has",
        "have",
        "her",
        "him",
        "his",
        "how",
        "hmm",
        "hey",
        "its",
        "know",
        "lately",
        "later",
        "like",
        "just",
        "last",
        "me",
        "memory",
        "mean",
        "mention",
        "mentioned",
        "now",
        "okay",
        "other",
        "our",
        "please",
        "recall",
        "really",
        "recently",
        "right",
        "remember",
        "remembered",
        "remind",
        "reminded",
        "said",
        "say",
        "she",
        "something",
        "talk",
        "talked",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "thing",
        "then",
        "those",
        "told",
        "today",
        "uhh",
        "umm",
        "want",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "yesterday",
        "yeah",
        "yes",
        "yep",
        "you",
        "your",
        "yup",
    }
)
_DURABLE_TRANSCRIPT_CATEGORY_TAGS = frozenset(
    f"category:{category}" for category in CATEGORIES if category != "general"
)


def _has_trusted_provenance(row: dict[str, Any]) -> bool:
    """Exclude legacy model-authored transcript rows from spoken recall.

    Before 2026-08-31, post-call extraction could persist assistant narration
    without caller evidence. Those rows remain stored for operator audit, but
    only caller-quote transcript rows and deliberate non-transcript saves may
    return to the conversational model.
    """

    tags = {tag for tag in (row.get("tags") or []) if isinstance(tag, str)}
    return "source:transcript" not in tags or "provenance:caller-quote" in tags


def _is_recallable_memory(
    row: dict[str, Any],
    *,
    caller_label: str = "Caller",
    wake_names: tuple[str, ...] = (
        "assistant",
        "guide",
        "helper",
        "companion",
        "listener",
        "speaker",
    ),
) -> bool:
    """Apply the conservative transcript-memory presentation boundary.

    Deliberate saves do not carry ``source:transcript`` and remain visible.
    Ambient transcript rows need caller-quote provenance and a specific durable
    category. Historical ``category:general`` rows remain stored for audit but
    do not enter a future model context or spoken recall.
    """

    if not _has_trusted_provenance(row):
        return False
    tags = {tag for tag in (row.get("tags") or []) if isinstance(tag, str)}
    if "source:transcript" not in tags:
        return True
    if not (tags & _DURABLE_TRANSCRIPT_CATEGORY_TAGS) or "category:general" in tags:
        return False
    content = (row.get("content") or "").strip()
    if not content.startswith(f"{caller_label} said:"):
        return False
    quotes = content.removeprefix(f"{caller_label} said:").split(" Then: ")
    return any(is_durable_caller_evidence(quote, wake_names=wake_names) for quote in quotes)


def _is_presentable_search_result(
    row: dict[str, Any],
    *,
    caller_label: str = "Caller",
    wake_names: tuple[str, ...] = (
        "assistant",
        "guide",
        "helper",
        "companion",
        "listener",
        "speaker",
    ),
) -> bool:
    """Search rows need trusted tags, either inline or from an exact-id read."""

    return isinstance(row.get("tags"), list) and _is_recallable_memory(
        row, caller_label=caller_label, wake_names=wake_names
    )


def _search_namespace_admitted(namespace: object, tenant: str) -> bool:
    """Require an exact episodic namespace inside the requesting tenant."""

    if not isinstance(namespace, str):
        return False
    parts = namespace.split("/")
    return (
        len(parts) == 3
        and parts[0] == tenant
        and bool(parts[1])
        and "*" not in namespace
        and parts[2] == "episodic"
    )


async def _resolve_search_provenance(
    client: MusubiClient, rows: list[dict[str, Any]], *, limit: int, tenant: str
) -> tuple[list[dict[str, Any]], int]:
    """Join tagless hits to exact stored rows within one bounded tool deadline.

    Unresolved rows never become spoken recall. A partial metadata outage can
    still return independently verified results in provider order.
    """

    resolved: list[dict[str, Any] | None] = [None] * len(rows)
    lookup_cap = min(_SEARCH_METADATA_MAX, limit * 3)
    lookup_count = 0
    semaphore = asyncio.Semaphore(_SEARCH_METADATA_CONCURRENCY)

    async def lookup(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any] | None, bool]:
        object_id = row.get("object_id")
        namespace = row.get("namespace")
        content = row.get("content")
        if not isinstance(object_id, str) or not isinstance(namespace, str):
            return index, None, True
        if not isinstance(content, str) or not content.strip():
            return index, None, True
        if not _search_namespace_admitted(namespace, tenant):
            return index, None, True
        try:
            async with semaphore:
                detail = await client.get_episodic(namespace=namespace, object_id=object_id)
        except Exception as exc:
            logger.warning("musubi_search: exact metadata lookup failed (%s)", type(exc).__name__)
            return index, None, True
        if type(detail) is not dict:
            return index, None, True
        if (
            detail.get("object_id") != object_id
            or detail.get("namespace") != namespace
            or not isinstance(detail.get("tags"), list)
        ):
            return index, None, True
        if detail.get("state") not in _SEARCH_STATE_FILTER:
            return index, None, False
        stored = detail.get("content")
        if not isinstance(stored, str):
            return index, None, True
        if row.get("content_truncated") is True:
            if not stored.startswith(content):
                return index, None, True
        elif stored != content:
            return index, None, True
        return index, {**row, "tags": detail["tags"]}, False

    tasks = []
    uncertain = sum(
        not isinstance(row.get("tags"), list)
        or not _search_namespace_admitted(row.get("namespace"), tenant)
        for row in rows
    )
    for index, row in enumerate(rows):
        if isinstance(row.get("tags"), list):
            if _search_namespace_admitted(row.get("namespace"), tenant):
                resolved[index] = row
        elif lookup_count < lookup_cap:
            tasks.append(asyncio.create_task(lookup(index, row)))
            lookup_count += 1
    if tasks:
        try:
            done, _pending = await asyncio.wait(tasks, timeout=_SEARCH_METADATA_BUDGET_S)
            for task in done:
                index, result, unknown = task.result()
                resolved[index] = result
                if not unknown:
                    uncertain -= 1
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return [row for row in resolved if row is not None], uncertain


def _search_term(token: str) -> str:
    """Small deterministic normalizer for query-to-result evidence.

    This is deliberately not a semantic score. It only smooths common English
    suffixes so ``plan/plans`` and ``leave/leaves`` can support the same query.
    """

    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _search_terms(
    text: str, *, query: bool, extra_stopwords: tuple[str, ...] = ()
) -> frozenset[str]:
    scaffold = _SEARCH_QUERY_SCAFFOLD | frozenset(x.casefold() for x in extra_stopwords)
    raw = re.findall(r"[a-z0-9]+", text.casefold())
    terms: set[str] = set()
    for token in raw:
        if not token.isdigit() and len(token) < 3:
            continue
        normalized = _search_term(token)
        if query and (token in scaffold or normalized in scaffold):
            continue
        terms.add(normalized)
    return frozenset(terms)


def _required_search_matches(query_terms: frozenset[str]) -> int:
    return 1 if len(query_terms) == 1 else max(2, math.ceil(len(query_terms) / 2))


def _search_result_supports_query(row: dict[str, Any], query_terms: frozenset[str]) -> bool:
    """Require content-visible topic anchors before a search hit may be spoken.

    Musubi's current ``ranked_combined`` score is relative within a result set:
    live probes placed the same bootstrap row near 0.77 for a known memory,
    nonsense, and a deliberately absent marker. Treating that number as an
    absolute relevance probability would be a false contract. This conservative
    gate instead requires at least half of the caller's distinct topic anchors,
    with two anchors required whenever the query supplies two or more.
    """

    if not query_terms:
        return False
    evidence = _search_terms(str(row.get("content") or ""), query=False)
    return len(query_terms & evidence) >= _required_search_matches(query_terms)


class MusubiToolsMixin(Agent):
    """Provides the canonical agent-tools surface.

    Three LLM-exposed tools: ``musubi_recent``, ``musubi_search``,
    ``musubi_remember``. ``musubi_think`` is retained as ``think_impl`` but
    not registered as a tool (see the module docstring).

    Per-agent scope: each agent reads/writes its own
    ``<agent>/<channel>/episodic`` per ADR 0030. Cross-channel reads
    (``musubi_search``) fan via ``<agent>/*/episodic``.

    Reads:
      - ``self.memory_config.musubi_v2_namespace`` — the ``<agent>/<channel>``
        prefix; the plane segment (``/episodic`` etc.) is appended per
        call. Unset/malformed degrades to "memory unavailable" — there is
        no ``caller/<agent>`` fabrication fallback (see ``_namespace_prefix``).
      - ``MUSUBI_V2_BASE_URL`` / ``MUSUBI_V2_TOKEN`` env.
    """

    memory_config: MemoryIdentity = UNCONFIGURED_CONFIG
    # Per-call authority, narrowed by the entrypoint after caller resolution.
    # A synthetic/device/unresolved participant may exercise conversational
    # behavior, but must not mutate the durable household memory plane.
    memory_writes_allowed: bool = True

    #: Scopes this call's derived idempotency keys, so a retry inside one call
    #: deduplicates while the same words next week are a new memory.
    memory_call_sid: str | None = None
    _memory_no_sid_scope: str | None = None
    _recent_rows_cache: tuple[float, int, list[Any]] | None = None

    def set_memory_write_policy(self, *, allowed: bool, call_sid: str | None = None) -> None:
        """Set durable-memory mutation authority for this call."""

        self.memory_writes_allowed = allowed
        self.memory_call_sid = call_sid
        self._memory_no_sid_scope = None

    def _memory_write_call_scope(self) -> str:
        if self.memory_call_sid:
            return self.memory_call_sid
        scope = getattr(self, "_memory_no_sid_scope", None)
        if scope is None:
            scope = uuid.uuid4().hex
            self._memory_no_sid_scope = scope
        return scope

    def _musubi_client(self) -> MusubiClient:
        """One place to construct the client so tests can monkeypatch."""
        return MusubiClient(config=MusubiClientConfig.from_env())

    def _namespace_prefix(self) -> str | None:
        """The validated ``<agent>/<channel>`` prefix, or ``None`` to degrade.

        Single source for every namespace the mixin builds. Returns ``None``
        — degrading the memory op to "unavailable" — when the agent has no
        ``musubi_v2_namespace`` or it isn't the canonical 2-segment
        agent-as-tenant form (ADR 0030). There is deliberately no
        ``caller/<agent>`` fabrication fallback: an unconfigured agent must not
        silently write into a real tenant — that was the misattribution bug.
        """
        prefix = self.memory_config.musubi_v2_namespace
        if not prefix:
            logger.warning(
                "musubi_v2_namespace unset (agent=%r); memory degrades to unavailable",
                self.memory_config.agent_name,
            )
            return None
        if len(prefix.split("/")) != 2:
            logger.warning(
                "musubi_v2_namespace %r is not <agent>/<channel>; memory degrades",
                prefix,
            )
            return None
        return prefix

    def _own_episodic_namespace(self) -> str | None:
        """This agent's own ``<agent>/<channel>/episodic`` namespace, or None."""
        prefix = self._namespace_prefix()
        return f"{prefix}/episodic" if prefix else None

    def _own_thought_namespace(self) -> str | None:
        """This agent's own ``<agent>/<channel>/thought`` namespace, or None."""
        prefix = self._namespace_prefix()
        return f"{prefix}/thought" if prefix else None

    def _own_presence(self) -> str | None:
        """This agent's ``<agent>/<channel>`` presence for ``from_presence``, or None."""
        return self._namespace_prefix()

    def _tenant_wildcard_episodic_namespace(self) -> str | None:
        """Tenant-wide ``<tenant>/*/episodic`` for cross-channel search (ADR 0031).

        Fans an episodic retrieve across every channel the tenant captured
        into; each result row still carries its concrete stored namespace, so
        provenance survives. ``None`` degrades when the prefix is unset or
        malformed.
        """
        prefix = self._namespace_prefix()
        return f"{prefix.split('/')[0]}/*/episodic" if prefix else None

    async def _scroll_episodic_recent(
        self,
        namespace: str,
        need: int,
        *,
        required_tag: str | None = None,
        max_pages: int = _MAX_RECENT_PAGES,
        budget_s: float = _RECENT_CONTEXT_TIMEOUT_S,
    ) -> list[dict[str, Any]]:
        """Paginate ``GET /v1/episodic`` until we have ``need`` recent
        rows or pages/cursors are exhausted. Recency-ordered (newest
        first); no time-window filter — "the last N memories" matters
        more than "memories from the last N hours" for greeting context.

        ``required_tag`` filters to rows whose ``tags`` list contains the
        value. Both deliberate saves and accepted post-call memories carry
        the seat's ``memory_agent_tag``; ambient or operational injections
        that lack it stay out. Transcript rows receive the additional durable
        provenance/category check below.
        """
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        client = self._musubi_client()
        page_size = min(need * _SCROLL_MULTIPLIER, 500)

        # Budget-aware: a namespace whose newest pages are all legacy
        # transcript rows (no trusted provenance) rejects every row, and five
        # pages at ~0.8 s overran the 3.0 s aggregate timeout on Guide's seat
        # (2026-09-02). The timeout turned a truthful "no trusted rows in the
        # newest N" into "Musubi is unavailable". Stop before a page that
        # cannot finish inside the budget and return what was actually seen.
        started = time.monotonic()
        last_page_s = 0.0
        for _ in range(max_pages):
            if (time.monotonic() - started) + last_page_s >= budget_s:
                break
            page_started = time.monotonic()
            page = await client.list_episodic(
                namespace=namespace,
                limit=page_size,
                cursor=cursor,
            )
            items = page.get("items") or []
            for r in items:
                if required_tag is not None and required_tag not in (r.get("tags") or []):
                    continue
                if not _is_recallable_memory(
                    r,
                    caller_label=self.memory_config.caller_label,
                    wake_names=self.memory_config.wake_names,
                ):
                    continue
                rows.append(r)
            last_page_s = time.monotonic() - page_started
            cursor = page.get("next_cursor")
            if not cursor or len(rows) >= need * _SCROLL_MULTIPLIER:
                break

        rows.sort(key=lambda r: r.get("created_epoch") or 0, reverse=True)
        return rows[:need]

    async def fetch_recent_context(self, limit: int = 10) -> str:
        """Plain-async fetch of recent voice memories for this agent.

        Exposed without ``@function_tool`` so ``on_enter`` can prefetch
        deterministically before the LLM gets a chance to skip the tool.

        Recency-based — returns the last ``limit`` rows tagged by this agent's
        voice mixin (``memory_agent_tag``), regardless of when they were
        captured. Filtering by tag keeps unrelated operational injections out;
        the recallability filter separately admits only deliberate saves or
        caller-backed transcript rows with a durable category.
        """
        trace(f"fetch_recent_context limit={limit}")
        limit = max(1, min(limit, _MAX_RECENT_LIMIT))
        cached = getattr(self, "_recent_rows_cache", None)
        if cached is not None:
            cached_at, cached_limit, cached_rows = cached
            if time.monotonic() - cached_at < _RECENT_CACHE_TTL_S and cached_limit >= limit:
                trace(f"fetch_recent_context: served {limit} rows from in-call cache")
                rows = cached_rows[:limit]
                if not rows:
                    return "No recent memories found."
                return "\n\n".join(_format_row(r) for r in rows)
        namespace = self._own_episodic_namespace()
        if namespace is None:
            return _DEGRADED_LOOKUP

        voice_tag = self.memory_config.memory_agent_tag
        cache_epoch = getattr(self, "_recent_rows_cache_epoch", 0)

        try:
            rows = await asyncio.wait_for(
                self._scroll_episodic_recent(namespace, limit, required_tag=voice_tag),
                timeout=_RECENT_CONTEXT_TIMEOUT_S,
            )
        except TimeoutError:
            logger.warning(
                "fetch_recent_context: aggregate timeout after %.1fs",
                _RECENT_CONTEXT_TIMEOUT_S,
            )
            return _DEGRADED_LOOKUP
        except (MusubiTimeoutError, MusubiServerError) as err:
            logger.warning("fetch_recent_context: transient %s", err)
            return _DEGRADED_LOOKUP
        except MusubiAuthError as err:
            logger.error("fetch_recent_context: auth failure: %s", err)
            return _DEGRADED_LOOKUP
        except MusubiClientError as err:
            logger.error("fetch_recent_context: bad request: %s", err)
            return _DEGRADED_LOOKUP
        except MusubiError as err:
            logger.warning("fetch_recent_context: %s", err)
            return _DEGRADED_LOOKUP

        if cache_epoch == getattr(self, "_recent_rows_cache_epoch", 0):
            self._recent_rows_cache = (time.monotonic(), limit, list(rows))
        if not rows:
            return "No recent memories found."

        return "\n\n".join(_format_row(r) for r in rows)

    def _invalidate_recent_rows_cache(self) -> None:
        self._recent_rows_cache_epoch = getattr(self, "_recent_rows_cache_epoch", 0) + 1
        self._recent_rows_cache = None

    @function_tool
    async def musubi_recent(self, limit: int = 10) -> str:
        """Fetch recent memories from your own episodic stream.

        Invocation Condition: Invoke this tool whenever the user asks
        about your own recent activity, what you talked about before,
        or what's been going on with you. Examples: "What did we talk
        about yesterday?", "What have you been up to?" You MUST call
        this tool before making any claims about past conversations.

        Recent rows may already be prefetched into an in-call cache by the
        runtime, so this tool can return them without another network lookup.

        Returns the most recent ``limit`` rows tagged by this voice
        agent. Operational / ambient writes that lack the agent tag are
        filtered out — recency, not a time window.

        Args:
            limit: Maximum number of memories to return (default 10, max 20).
        """
        return await self.fetch_recent_context(limit=limit)

    @function_tool
    async def musubi_search(self, query: str, limit: int = 5) -> str:
        """Semantic-search your memory across every channel you've spoken on.

        Invocation Condition: Invoke this tool when the user asks about
        a SPECIFIC topic, fact, or event you might know about — anything
        that isn't just "what did we talk about recently". Examples:
        "Do you remember the prank we discussed?", "What do you know
        about the dentist appointment?", "Did I tell you about the
        Stable Diffusion update?", "What's the deploy plan you saved?"

        Unlike musubi_recent (which scrolls your VOICE channel only, by
        recency — not a time window), this searches every surface you
        exist on. If you were told something on another surface, THIS is
        the tool that finds it on a phone call. Each result row carries its origin namespace so you can
        say which surface a memory came from.

        You MUST call this tool when answering recall questions. Saying
        "I remember…" without calling this tool is hallucination.

        Args:
            query: What you're searching for. Plain English; the server
                runs hybrid + rerank.
            limit: Max rows to return (default 5).
        """
        trace(f"tool=musubi_search query={query[:60]!r} limit={limit}")
        query_terms = _search_terms(
            query,
            query=True,
            extra_stopwords=(
                *self.memory_config.search_stopwords,
                *self.memory_config.wake_names,
                self.memory_config.caller_label,
            ),
        )
        if not query_terms:
            logger.info("musubi_search: abstained before retrieval; no topic anchors")
            return "No memories matched."
        limit = max(1, min(limit, _MAX_SEARCH_LIMIT))

        namespace = self._tenant_wildcard_episodic_namespace()
        if namespace is None:
            return _DEGRADED_LOOKUP

        client = self._musubi_client()
        try:
            response = await client.retrieve(
                namespace=namespace,
                query_text=query,
                mode="deep",
                limit=min(limit * 3, 30),
                state_filter=_SEARCH_STATE_FILTER,
            )
        except (MusubiTimeoutError, MusubiServerError) as err:
            logger.warning("musubi_search: transient %s", err)
            return _DEGRADED_LOOKUP
        except MusubiAuthError as err:
            logger.error("musubi_search: auth failure: %s", err)
            return _DEGRADED_LOOKUP
        except MusubiClientError as err:
            logger.error("musubi_search: bad request: %s", err)
            return _DEGRADED_LOOKUP
        except MusubiError as err:
            logger.warning("musubi_search: %s", err)
            return _DEGRADED_LOOKUP

        raw_rows = [row for row in (response.get("results") or []) if isinstance(row, dict)]
        supported_rows = [
            row for row in raw_rows if _search_result_supports_query(row, query_terms)
        ]
        verified_rows, unresolved = await _resolve_search_provenance(
            client, supported_rows, limit=limit, tenant=namespace.split("/")[0]
        )
        rows = [
            row
            for row in verified_rows
            if _is_presentable_search_result(
                row,
                caller_label=self.memory_config.caller_label,
                wake_names=self.memory_config.wake_names,
            )
        ][:limit]
        unsupported = len(raw_rows) - len(supported_rows)
        if unsupported:
            logger.info(
                "musubi_search: withheld %d result(s) without content-visible query evidence "
                "(query_anchors=%d required=%d)",
                unsupported,
                len(query_terms),
                _required_search_matches(query_terms),
            )
        if unresolved:
            logger.warning(
                "musubi_search: %d query-supported result(s) lacked verified provenance",
                unresolved,
            )
        if not rows:
            return _UNVERIFIED_LOOKUP if unresolved else "No memories matched."
        return "\n\n".join(_format_search_row(r) for r in rows)

    @function_tool
    async def musubi_remember(
        self,
        content: str,
        topics: list[str] | None = None,
        importance: int = _DEFAULT_IMPORTANCE,
    ) -> str:
        """Store a memory to Musubi for future recall.

        Invocation Condition: Invoke this tool when the user asks you to
        remember something, save something for later, or make a note.
        Examples: "Remember I have a dentist appointment Tuesday", "Save
        that for later", "Don't forget about the deploy". You MUST call
        this tool to store the memory. Saying you'll remember it without
        calling this tool means the memory is lost.

        Do NOT invoke it because a call is ending. Everything worth keeping
        from the call is extracted from the transcript afterwards, against
        Caller's exact words. An unprompted save on a goodbye turn is the
        failure that disabled unrouted tool choice on every seat
        (Speaker live call, 2026-08-24); this instruction used to ask for it.

        Tool name + parameter shape match the browser plugin's
        ``musubi_remember`` so saves on either surface look the same in
        traces and to the model.

        Args:
            content: What to remember. Write it the way you'd want to
                read it next time — natural language, not raw data.
            topics: Optional keywords for retrieval (e.g. ['joke', 'caller',
                'deploy']). Keep them short and relevant.
            importance: 1-10. Default 7. Bump higher for things you don't
                want demoted; lower for ambient context.
        """
        trace(f"tool=musubi_remember content={content[:60]!r} topics={topics!r}")
        if not self.memory_writes_allowed:
            trace("tool=musubi_remember BLOCKED call_source_not_authorized")
            logger.warning("musubi_remember blocked: call source is not authorized to write memory")
            return "Memory didn't save — this call source is not authorized for memory."
        if not content.strip():
            return "Error: content is required."

        topic_list = list(topics or [])
        # The agent's ``memory_agent_tag`` (e.g. ``assistant-voice``) goes in
        # alongside the caller's topics — that's the signal
        # :func:`fetch_recent_context` keys on to filter the greeting
        # hook to deliberate-save rows only.
        speaker_tag = self.memory_config.memory_agent_tag
        if speaker_tag and speaker_tag not in topic_list:
            topic_list.append(speaker_tag)
        # Post-call extraction stamps provenance:caller-quote and means it --
        # it keeps Caller's exact [USER] wording. This row is whatever the model
        # chose to write, so it must not be indistinguishable from that at
        # audit time.
        if _TOOL_PROVENANCE not in topic_list:
            topic_list.append(_TOOL_PROVENANCE)

        importance = max(1, min(int(importance), 10))

        namespace = self._own_episodic_namespace()
        if namespace is None:
            return _DEGRADED_STORE
        # Derived, not random. A fresh UUID per attempt is the one key shape
        # that cannot deduplicate anything: a timeout after Musubi accepted
        # the POST is reported to us as failure, and the model retrying the
        # tool then wrote a second row. Scoped to the call so the same
        # sentence next week is still a new memory.
        idem = "livekit-musubi-remember:{}:{}".format(
            self._memory_write_call_scope(),
            hashlib.sha256(
                "\x1f".join([namespace, content.strip(), *sorted(topic_list)]).encode()
            ).hexdigest()[:32],
        )

        try:
            ack = await self._musubi_client().capture_memory(
                namespace=namespace,
                content=content,
                tags=topic_list,
                importance=importance,
                idempotency_key=idem,
            )
        except asyncio.CancelledError:
            # Cancellation of the waiter cannot prove whether Musubi accepted
            # the POST. A retry uses the same key; a later recent lookup must
            # not be served from a cache populated before a possible save.
            self._invalidate_recent_rows_cache()
            logger.warning(
                "musubi_remember: cancelled while save outcome is unknown; save_ref=%s",
                _save_log_ref(idem),
            )
            raise
        except (MusubiTimeoutError, MusubiServerError) as err:
            logger.warning("musubi_remember: transient %s", err)
            return _DEGRADED_STORE
        except MusubiAuthError as err:
            logger.error("musubi_remember: auth failure: %s", err)
            return "Memory didn't save — auth failed."
        except MusubiClientError as err:
            logger.error("musubi_remember: bad request: %s", err)
            return "Memory didn't save — request rejected."
        except MusubiError as err:
            logger.warning("musubi_remember: %s", err)
            return "Memory didn't save — unknown error."

        object_id = ack.get("object_id") or "<unknown>"
        trace(f"tool=musubi_remember DONE id={object_id}")
        self._invalidate_recent_rows_cache()  # a just-saved memory must be visible to recent
        return MEMORY_STORED_LINE

    # ------------------------------------------------------------------
    # musubi_think — presence-to-presence message
    # ------------------------------------------------------------------

    async def think_impl(
        self,
        to_presence: str,
        content: str,
        channel: str = "default",
        importance: int = 5,
    ) -> str:
        """Presence-to-presence thought send. **Retained but not LLM-exposed**
        (2026-07-10) — the ``@function_tool musubi_think`` wrapper was removed
        because the live webbing does not consume the thought plane for these
        agents (see module docstring). Kept callable for programmatic use and
        so re-enabling the tool is a one-line wrapper away."""
        trace(f"tool=musubi_think to={to_presence!r} content={content[:60]!r} channel={channel!r}")
        prefix = self._namespace_prefix()
        if prefix is None:
            logger.debug("musubi_think: no namespace configured; degrading")
            return _DEGRADED_LOOKUP
        if not to_presence.strip():
            return "Error: to_presence is required."
        if not content.strip():
            return "Error: content is required."

        namespace = f"{prefix}/thought"
        from_presence = prefix
        # Bare ``<agent>`` aliases resolve to ``<agent>/<this-channel>``. The
        # prefix is a validated 2-segment ``<agent>/<channel>`` (agent-as-tenant,
        # ADR 0030), so the channel is always the second segment.
        own_channel = prefix.split("/", 1)[1]
        resolved_to = to_presence if "/" in to_presence else f"{to_presence}/{own_channel}"

        try:
            ack = await self._musubi_client().send_thought(
                namespace=namespace,
                from_presence=from_presence,
                to_presence=resolved_to,
                content=content,
                channel=channel,
                importance=importance,
            )
        except (MusubiTimeoutError, MusubiServerError) as err:
            logger.warning("musubi_think: transient %s", err)
            return "Thought didn't deliver — Musubi is unavailable."
        except MusubiAuthError as err:
            logger.error("musubi_think: auth failure: %s", err)
            return "Thought didn't deliver — auth failed."
        except MusubiClientError as err:
            logger.error("musubi_think: bad request: %s", err)
            return "Thought didn't deliver — request rejected."
        except MusubiError as err:
            logger.warning("musubi_think: %s", err)
            return "Thought didn't deliver — unknown error."

        object_id = ack.get("object_id") or "<unknown>"
        return f"Sent to {resolved_to}. (id={object_id})"

    # NOTE: the ``@function_tool musubi_think`` wrapper was removed 2026-07-10.
    # It told the model "you MUST call this to deliver a note to another agent,"
    # which contradicts every persona's "there is no delegation route from the
    # phone / never say you passed something along" — and the thought plane it
    # wrote to is not consumed by the live comms webbing. ``think_impl`` above
    # stays for programmatic use; re-add a thin ``@function_tool`` wrapper here
    # if a real consumer (SSE subscriber / inbox scroll) is wired for these agents.


def _format_row(row: dict[str, Any]) -> str:
    """One-line render for a scrolled episodic row.

    Used by ``fetch_recent_context``. Kept simple — an LLM reads this,
    not a human in a terminal.
    """
    tags = row.get("tags") or []
    agent_tag = next(
        (t for t in tags if isinstance(t, str) and t.endswith("-voice")),
        None,
    )
    speaker = agent_tag.removesuffix("-voice") if agent_tag else (row.get("namespace") or "?")
    content = (row.get("content") or "").strip()
    provenance = " [caller-quote]" if "provenance:caller-quote" in tags else ""
    return f"[{speaker}]{provenance} {content}"


def _format_search_row(row: dict[str, Any]) -> str:
    """One-line render for a retrieve hit. Surfaces the row's origin
    channel (the ``presence`` segment of the stored namespace) so the
    LLM can attribute which surface a memory came from. Falls back to the raw namespace if the row's namespace
    isn't 3-segment for any reason."""
    ns = row.get("namespace") or ""
    parts = ns.split("/")
    channel = parts[1] if len(parts) >= 2 else ns or "?"
    content = (row.get("content") or "").strip()
    tags = row.get("tags") or []
    provenance = " [caller-quote]" if "provenance:caller-quote" in tags else ""
    return f"[{channel}]{provenance} {content}"


__all__ = ["MusubiToolsMixin"]
