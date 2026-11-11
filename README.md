# agent-common — the single source of truth for the shared runtime

This directory is an **independent Git repository** named `agent-common`, mounted
as the `agent-common/` git submodule of the three checkouts
(`generic-debug-pipeline`, `paper-refine` and `paper-refine-eval`).

It owns the files the checkouts used to copy into themselves:

```
agent_common/                 the import package (underscores)
  __init__.py
  model_runtime.py            shared model/harness/provider runtime
  model_catalog.json          canonical model-capacity catalog
  llm_policy.json             canonical per-LLM-call retry / dual-key policy
  llm_retry.py                retry schedule, DSH overlay and loopback proxy
  dsh_events.py               dsh --json turn-end/error parsing (GDP + paper-refine)
  chat.py                     chat token budgeting/response validation
                              (paper-refine + paper-refine-eval)
  agent_cli.py                dsh argv construction (GDP + paper-refine)
check_*.py                    cross-repo consistency checks
test_llm_retry.py             retry-policy tests
```

## Why the repository is hyphenated and the package is not

Python project/distribution names may contain hyphens, but **import names must
be valid identifiers**, so the package directory uses an underscore.  This is
the standard split used by projects such as `scikit-learn` (imports as
`sklearn`) and `python-dateutil` (imports as `dateutil`): the repository is
`agent-common`, the import package is `agent_common`.

Consumers therefore add the **`agent-common/` directory** to `sys.path` and
import the inner package:

```python
from agent_common import model_runtime
from agent_common.llm_retry import RetrySchedule
```

No import statement ever contains a hyphen.  Do not rename the package to match
the repository; that would produce a `SyntaxError`/`ModuleNotFoundError`.

## Staging rule — read this before `git add`

**Never stage, copy or commit the files inside `agent-common/` in a host
checkout.**  They are not host-repository source.  They are owned here and are
changed by revising this repository; a vendored copy in a checkout would drift
exactly the way the old per-checkout copies did.

The only things a host checkout may commit are:

* `.gitmodules`;
* the `agent-common` gitlink (the pinned commit id).

In particular, do **not** run `git add agent-common/agent_common/model_runtime.py`
(or any other file under `agent-common/`) in a checkout, and do not copy these
files back into `gdp/`, `pr_eval/` or the paper-refine root.

Only the **`main`** branch of each checkout stages the submodule pointer for a
release, because `main` is the release version.  The **`dev`** branch carries
the same submodule so the code is importable while developing, but a `dev`
pointer is never a release pin: advance it freely and do not treat its files as
host-owned source.  Its submodule files remain versioned *here* in
`agent-common`, never in the host repository.

## Updating the shared runtime

```bash
# 1. Edit and commit the canonical files in this repository.
cd repos/agent-common
$EDITOR agent_common/model_runtime.py   # etc.
git add -A && git commit -m "..."

# 2. Point each checkout at the new commit.
cd repos/generic-debug-pipeline
git -C agent-common fetch origin             # origin is the path in .gitmodules
git -C agent-common checkout <sha>
git add .gitmodules agent-common             # main/release only
git commit -m "Update agent-common"
```

`sync_llm_policy.sh` and `update_model_catalog.sh` are kept as convenience
wrappers that check out the pinned submodule commit in all three checkouts.

## Verification

```bash
python3 repos/agent-common/check_model_catalog.py
python3 repos/agent-common/check_llm_policy.py
python3 repos/agent-common/check_cross_repo_consistency.py
python3 -m unittest discover -s repos/agent-common -p 'test_*.py'
```

The three checks read the `agent-common/agent_common/` package inside each
sibling checkout's submodule, so they fail if a checkout is not initialized or
if its submodule was modified locally.

## Shared logic modules

* `dsh_events.py` — one definition of the two `dsh --json` turn-end shapes,
  the success allowlist and turn-end error extraction.  GDP's runner and the
  paper-refine adapters both use it.
* `chat.py` — prompt-token estimation, completion budgeting and
  `finish_reason`/message validation.  The paper-refine adapters and
  paper-refine-eval's `pr_eval.llm` both use it.
* `agent_cli.py` — `build_dsh_argv` for a one-shot headless run.  GDP's runner
  and the paper-refine adapters both use it.
