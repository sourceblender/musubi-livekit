"""A single per-call memory owner composed into the engine agent by the seat."""

from typing import Any, ClassVar

from .client import wire_musubi_shutdown
from .postcall_memory import wire_postcall_memory
from .prefetch import MemoryPrefetchMixin


class MusubiMemoryMixin(MemoryPrefetchMixin):
    memory_tool_names: ClassVar[frozenset[str]] = frozenset({"musubi_recent", "musubi_search"})

    postcall_capture_enabled: bool = True

    def memory_enabled(self) -> bool:
        return self._namespace_prefix() is not None

    def wire_call_memory(
        self, session: Any, ctx: Any, *, call_sid: str, capture_allowed: bool
    ) -> None:
        prefix = self._namespace_prefix()
        wire_postcall_memory(
            session,
            call_sid=call_sid,
            namespace=f"{prefix}/episodic" if prefix else None,
            speaker_tag=self.memory_config.memory_agent_tag,
            capture_allowed=capture_allowed and self.postcall_capture_enabled,
            **(
                {"policy_path": self.memory_config.postcall_policy_path}
                if self.memory_config.postcall_policy_path
                else {}
            ),
        )
        wire_musubi_shutdown(ctx)
