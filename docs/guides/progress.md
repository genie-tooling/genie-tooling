# Progress Streaming (Phase 6C.2)

Long-running agents shouldn't only produce a single final blob. The
progress framework lets agents emit `ProgressEvent`s at iteration
boundaries; configured sinks ship them to operator-facing surfaces
(Slack thread, webhook, console, ...) in real time.

## Quick setup

```python
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

cfg = MiddlewareConfig(
    features=FeatureSettings(
        progress_sinks=["console_progress_sink", "webhook_progress_sink"],
    ),
    progress_sink_configurations={
        "webhook_progress_sink_v1": {
            "url": "https://hooks.example.com/agent-progress",
            "fire_and_forget": True,
        },
    },
)
genie = await Genie.create(config=cfg)
```

That's it. Every `ReActAgent` and `PlanAndExecuteAgent` run under this
Genie instance will fan progress events to all configured sinks.

## Bundled sinks

| Plugin ID | Description |
|---|---|
| `console_progress_sink_v1` | Logs events via the `genie_tooling.progress` logger. CLI dev / debugging. |
| `webhook_progress_sink_v1` | POSTs each event as JSON to a configured URL. Fire-and-forget by default so a slow webhook doesn't block the agent. |
| `slack_thread_progress_sink_v1` | Pushes events as threaded Slack messages under a parent message. Pair with `slack_post_message_v1` to start the thread. |

## ProgressEvent shape

```python
@dataclass
class ProgressEvent:
    run_id: str
    agent_id: str            # "react_agent" / "plan_and_execute_agent"
    phase: str               # "started" / "iterating" / "tool_call" / "tool_result" / "thinking" / "completed" / "failed"
    message: str
    timestamp: float
    iteration: Optional[int]
    level: str               # "info" / "warning" / "error"
    tool_id: Optional[str]
    attribution_tags: Optional[Dict[str, str]]
    extra: Optional[Dict[str, Any]]
```

## Sink resolution order

When `ReActAgent` builds its emitter, sinks come from (in order, all
stack):

1. **`MiddlewareConfig.default_progress_sink_ids`** — auto-loaded by
   `Genie.create()`. Survive teardown.
2. **`agent_config['progress_sinks']`** — passed to the agent
   constructor. Caller-owned.
3. **`input_context['progress_sinks']`** — passed per-call to `agent.run(...)`.
   Caller-owned.

## Writing a custom sink

```python
from genie_tooling.progress.abc import ProgressSinkPlugin
from genie_tooling.progress.types import ProgressEvent

class JiraCommentProgressSink(ProgressSinkPlugin):
    plugin_id = "jira_comment_progress_sink_v1"
    description = "Appends progress events as JIRA issue comments."

    async def setup(self, config=None):
        self._issue_key = config["issue_key"]
        # ... auth setup ...

    async def emit(self, event: ProgressEvent) -> None:
        # JIRA REST call to add comment
        ...

    async def teardown(self):
        ...
```

Register via `entry_points` in `pyproject.toml`:

```toml
[tool.poetry.plugins."genie_tooling.plugins"]
"jira_comment_progress_sink_v1" = "yourpkg.module:JiraCommentProgressSink"
```

## Best-effort semantics

A failing sink logs but doesn't propagate the exception. Progress is
**not** a hard dependency of the agent loop; a flaky Slack endpoint
should never break an SRE investigation. The webhook sink defaults to
fire-and-forget for the same reason — the POST is dispatched as a
background task and the agent moves on.
