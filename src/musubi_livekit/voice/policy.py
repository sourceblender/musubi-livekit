"""Application-supplied private extraction policy, never logged."""

import json
from pathlib import Path
from typing import Any

DEFAULT_WAKE_NAMES = ("assistant", "guide", "helper", "companion", "listener", "speaker")


def load_policy(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = Path(path).read_bytes()
    if len(raw) > 65536:
        raise ValueError("memory policy is too large")
    data = json.loads(raw)
    allowed = {"caller_label", "wake_names", "search_stopwords", "extraction_prompt"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("invalid memory policy fields")
    for key in ("caller_label", "extraction_prompt"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            raise ValueError("memory policy strings must be non-empty")
    for key in ("wake_names", "search_stopwords"):
        if key in data and (
            not isinstance(data[key], list)
            or len(data[key]) > 128
            or any(not isinstance(x, str) or len(x) > 128 for x in data[key])
        ):
            raise ValueError("invalid memory policy names")
    if len(data.get("caller_label", "")) > 128:
        raise ValueError("caller label is too long")
    return data
