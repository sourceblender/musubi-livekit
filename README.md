# musubi-livekit

`musubi-livekit` connects a LiveKit voice session's callbacks to the Musubi
memory API. It provides a `LiveKitAdapter`, a slow prefetcher, a fast context
reader, a bounded cache, optional fact capture, and transcript handling. The
package is a Python library: your LiveKit worker still owns the session,
event subscriptions, credentials, and user-facing responses.

## Install

Requires Python 3.12 or later. The adapter depends on the standalone
[`musubi-sdk`](https://pypi.org/project/musubi-sdk/) client, not the Musubi server.
Both packages are published on PyPI. Install the released adapter and its SDK
dependency with:

```bash
pip install musubi-livekit
```

## Use in a worker

Construct one adapter per voice session. Feed it the transcript and session
events from your own LiveKit worker. These are callbacks to wire into the
worker, not an automatically installed LiveKit plugin.

```python
from musubi_sdk import AsyncMusubiClient
from musubi_livekit import LiveKitAdapter, LiveKitAdapterConfig

client = AsyncMusubiClient(base_url=api_url, token=token)
adapter = LiveKitAdapter(
    client=client,
    namespace="assistant/voice/episodic",
    artifact_namespace="assistant/voice/artifact",
    config=LiveKitAdapterConfig(
        capture_transcripts=False,
        capture_facts=False,
    ),
)

# Call from the matching LiveKit worker events:
await adapter.on_transcript_segment(partial_text)
await adapter.on_user_turn_completed(final_text)
context = await adapter.fast_talker.get_context(final_text)
warnings = adapter.retrieval_status
await adapter.on_session_end(session_id=session_id, vtt_transcript=vtt_text)
```

Fact and transcript capture are enabled in the default config. Set the privacy
flags deliberately for your application before handling a real session.
`retrieval_status` exposes degradation warnings so the worker can tell the
user when memory is unavailable. A successful callback call is not a guarantee
that a transcript reached artifact storage: until the SDK has `artifacts.upload`,
the adapter falls back to episodic capture. See the code and tests before
promising artifact storage or delivery guarantees.

## Development

```bash
uv sync --locked --python 3.12
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/musubi_livekit
```

The unit suite exercises the extracted adapter and retrieval degradation
channel. Three historical tests remain skipped because they require a running
Musubi stack and a LiveKit session simulator; they are not release evidence.

Five integration tests exercise the adapter through the real Musubi SDK and API.
They write memories and thoughts, so point them only at a disposable test stack
with an operator-scoped test token:

```bash
MUSUBI_TEST_API_URL=http://127.0.0.1:8100/v1 \
MUSUBI_TEST_TOKEN=<disposable-test-token> \
uv run pytest -q -m integration tests/integration/test_livekit_e2e.py
```

The default test command excludes these integration tests. Their fixture
skips when either variable is missing. Linux CI also runs them against
Musubi's disposable Docker integration stack; the pinned TEI test image has
no ARM64 manifest, so that stack does not boot natively on Apple Silicon.

## Contributing and security

Discuss substantial changes in an issue first. Follow the
[Sourceblender contributing guide](https://github.com/sourceblender/.github/blob/main/CONTRIBUTING.md).
Report security issues privately through the repository Security tab or the
[organization policy](https://github.com/sourceblender/.github/blob/main/SECURITY.md).

Licensed under Apache-2.0. See [LICENSE](LICENSE).
