# agent_common public API

This is the supported surface for another repository that mounts
`agent-common` as a submodule.  The import package is `agent_common`; the
project directory is `agent-common` (hyphenated).

```python
import sys
sys.path.insert(0, "/path/to/checkout/agent-common")

from agent_common import chat, dsh_events, model_runtime, llm_retry, agent_cli
```

Every public function and method has a docstring with at least one `Example`
and one `Counterexample`; `test_public_api_docs.py` enforces that contract.
Names beginning with `_` are private and may change without notice.

## `model_runtime` — model/harness/provider resolution

Use this module when you need to turn operator options into a concrete
provider route, generate a harness patch, or record a secret-free selection.

Typical flow:

```python
selection = model_runtime.resolve_agent_selection(
    harness="dsh", model="dots3-note-preview"
)
patch_yaml = model_runtime.dsh_patch_text(selection)
env = model_runtime.selection_env(selection)
record = model_runtime.effective_agent_model(selection)
```

Key entry points:

| Function | Purpose |
| --- | --- |
| `resolve_agent_selection` | Validate `(harness, model, api_key, ...)` and return an `AgentSelection`. |
| `pair_from_options` | Fill a missing harness/model axis from the other axis or the defaults. |
| `dsh_patch_text` | Generate the DSH `--patch` overlay for the selection. |
| `codex_provider_args` | Generate `codex exec -c` provider overrides. |
| `claude_environment` / `claude_model_arg` | Point Claude Code at the selected provider. |
| `selection_env` | Child-process credential environment (never persisted). |
| `effective_agent_model` / `AgentSelection.to_record` | Secret-free run record. |
| `default_reasoning_level` | The catalog default level for a named model (`None` when the model keeps its own default). |
| `model_fact` | One catalog fact (context window, caps, default level) for a named model; the way a consumer reads a shared setting. |
| `reasoning_effort_override` | The operator's `--reasoning-effort` value (from `PAPER_AGENT_REASONING_EFFORT`), or `None`. |
| `normalize_reasoning_effort` | The generic level translated to the level a model's wire accepts. |
| `endpoint_base` / `endpoint_query_model` / `endpoint_credentials` | Parse and validate custom endpoints. |
| `redact_endpoint_url` / `redact_prep_object` | Remove credentials from recorded artifacts. |
| `start_prep_proxy` / `stop_prep_proxy` | Merge a prep object into outgoing requests. |
| `start_compaction_proxy` / `stop_compaction_proxy` | Retrying proxy for the Dots compaction route. |

Counterexample: do not feed `AgentSelection.to_record()["model"]` back into a
live request for a custom endpoint — it is redacted.  Keep the live selection.

## `model_runtime` reasoning defaults

`default_reasoning_level(model)` returns the level the shared catalog records
as the deployment's default: `high` for `deepseek-flash`, `high` for
`dots3-note-preview` and `medium` for `qwen38-27b`.  It returns `None` for
GPT and Claude (their harness configuration sends no effort, so the model's own
default applies) and for the `default` sentinel or a custom endpoint.

`model_fact(model, field, default=None)` exposes the rest of the catalog the
same way — `context_window`, `max_tokens`, `chat_max_tokens` — and
`default_reasoning_level` is a typed shortcut for the reasoning field.
`dsh_patch_text` and `codex_provider_args` already apply the effort, so a
consumer that only needs the number (for an adapter default, a log line or a
documented table) should call these helpers instead of hard-coding a literal.
`PAPER_DEEPSEEK_REASONING_EFFORT`, `PAPER_DOTS_REASONING_EFFORT` and
`PAPER_QWEN_REASONING_EFFORT` remain operator overrides.

`reasoning_effort_override()` returns the shared
`PAPER_AGENT_REASONING_EFFORT` value (the channel a consumer's
`--reasoning-effort` flag exports; `GDP_AGENT_REASONING_EFFORT` is an alias),
and `normalize_reasoning_effort(model, level)` translates the generic
`off|minimal|low|medium|high|xhigh|max` vocabulary to the level the model's
wire accepts.  A consumer should set the environment variable (or pass the
level through) rather than read a family-specific variable.

Counterexample: `default_reasoning_level("gpt-sol") is None` is not an error;
substituting `"high"` for it would override the model's own default.

Policy flags (set before the first real request):

- `REJECT_ENDPOINT_CREDENTIALS` — reject URLs with userinfo/credential query
  parameters.  GDP opts out because it strips them for the generated patch.
- `MISSING_CATALOG_IS_FATAL` — make a missing model catalog fatal.  Consumers
  that change it must call `reload_model_catalog()` so the policy also applies
  to the already-loaded table.

## `llm_retry` — retry schedule, policy and proxy

```python
policy = llm_retry.load_policy()                 # defaults on missing/malformed
schedule = llm_retry.RetrySchedule(policy)
delays = schedule.delays()                       # 10, 20, ..., 3600
proxy, base_url = llm_retry.start_retry_proxy("https://api.example/v1")
...
llm_retry.stop_retry_proxy(proxy)
```

`retry_policy_yaml`, `dots_compaction`, `compaction_basic_yaml` and
`dots_dsh_overlay` generate DSH configuration text/data.

Counterexample: `RetryProxy` is a loopback HTTP proxy, not an HTTP client; use
`urllib`/your client and point it at the returned base URL.

## `chat` — token budgeting and response validation

```python
budget = chat.completion_budget(context_window=150_000, output_cap=16_384,
                                prompt_chars=len(prompt))
text = chat.message_text(choice["message"]["content"])
problem = chat.finish_reason_error(choice.get("finish_reason"), budget)
```

Counterexample: a non-empty `reasoning_content` channel is not the requested
answer; `message_text` returns only the visible content, and the caller must
report the empty-answer error.

## `dsh_events` — parse `dsh --json` streams

```python
event = dsh_events.parse_event(line)
failed = dsh_events.turn_end_failed(event)       # True/False/None
message = dsh_events.turn_end_error(event.get("reason"))
```

Counterexample: `turn_end_failed(...) is False` does not mean the line was a
turn-end event; test `is not None` when that distinction matters.

## `agent_cli` — one-shot `dsh` argv

```python
argv = agent_cli.build_dsh_argv(patch_path="/tmp/model.yml",
                                extra_args=["--foo", "bar"])
```

Counterexample: application flags belong in `extra_args`, not in
`patch_path`; the launcher treats the patch argument as a file name.

## Compatibility

The public names above are the supported surface.  Adding parameters is
backward-compatible; removing/renaming a public name or changing a documented
return shape is not.  Internal helpers (`_name`) and generated YAML text are
not covered by that promise.
