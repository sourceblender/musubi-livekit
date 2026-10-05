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
uv sync --locked --extra voice --python 3.12
uv run --extra voice pytest -q
uv run --extra voice ruff check src tests
uv run --extra voice ruff format --check src tests
uv run --extra voice mypy src/musubi_livekit
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
uv run --extra voice pytest -q -m integration tests/integration/test_livekit_e2e.py
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

## Household voice provider

Install the optional integration with `pip install 'musubi-livekit[voice]'`.
`musubi_livekit.voice.provider.MusubiMemoryMixin` implements the structural
call-memory contract consumed by Duet. A seat composes that mixin into its agent
and supplies `memory_config = MemoryConfig(agent_name, memory_agent_tag,
musubi_v2_namespace)`. This identity is separate from its conversation-engine
configuration. The provider retains the current recent/search/remember tool
names and result strings, per-call cache, write authority and post-call
caller-evidence validation.

For these seats, memory runs only inside the Duet process. Do not also wire the
callback adapter or memory capture into the outer LiveKit audio bridge. The
engine passes caller-origin write policy before tools can run, then invokes
prefetch and `wire_call_memory` in its lifecycle. Synthetic callers cannot
write household memory. A successful extraction subprocess does not prove a
Musubi save; verify positive writes by reading the saved object from Musubi.

Source provenance and the inherited MIT license are in `docs/VOICE-SOURCE.json`
and `docs/VOICE-MIT-LICENSE`. The existing adapter is unchanged. Two changes to
the extracted path are deliberate: a failed prefetch still warms the speaking
model, and validation logs fixed rejection-reason counts without caller text.

Run `uv sync --extra voice`, then `uv run --extra voice pytest -q` to exercise
the extracted integration alongside the adapter suite. The three existing
live-stack skips remain unproven integration coverage.

The household provider's extraction prompt and caller naming policy are
application configuration, not library defaults. Supply a private policy JSON
file through `MemoryConfig.postcall_policy_path`, plus `caller_label`,
`wake_names` and `search_stopwords` for recall. The application owns and packages
that file; the library never logs its contents. The generic default uses
`Caller said:`. An application retaining older stored prefixes must explicitly
supply that original caller label.
