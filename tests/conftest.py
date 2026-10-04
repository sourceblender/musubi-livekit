import pytest


@pytest.fixture
def agent():
    pytest.importorskip("livekit.agents")
    pytest.importorskip("google.genai")
    from musubi_livekit.voice.identity import MemoryConfig
    from musubi_livekit.voice.provider import MusubiMemoryMixin

    class ComposedAgent(MusubiMemoryMixin):
        memory_config = MemoryConfig("assistant", "assistant-voice", "assistant/voice")

        def __init__(self):
            super().__init__(instructions="test persona")
            self._caller_from = "+15551234567"

    return ComposedAgent()
