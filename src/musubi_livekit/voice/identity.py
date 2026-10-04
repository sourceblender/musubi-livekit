"""Structural memory identity: callers may supply their own immutable config."""

from dataclasses import dataclass
from typing import Protocol


class MemoryIdentity(Protocol):
    @property
    def agent_name(self) -> str: ...
    @property
    def memory_agent_tag(self) -> str: ...
    @property
    def musubi_v2_namespace(self) -> str | None: ...
    @property
    def caller_label(self) -> str: ...
    @property
    def wake_names(self) -> tuple[str, ...]: ...
    @property
    def search_stopwords(self) -> tuple[str, ...]: ...
    @property
    def postcall_policy_path(self) -> str | None: ...


@dataclass(frozen=True)
class MemoryConfig:
    agent_name: str
    memory_agent_tag: str
    musubi_v2_namespace: str | None = None
    caller_label: str = "Caller"
    wake_names: tuple[str, ...] = (
        "assistant",
        "guide",
        "helper",
        "companion",
        "listener",
        "speaker",
    )
    search_stopwords: tuple[str, ...] = ()
    postcall_policy_path: str | None = None


UNCONFIGURED_CONFIG = MemoryConfig("__unconfigured__", "", None)
