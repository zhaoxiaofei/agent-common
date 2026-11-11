# agent_common — the single source of truth for the shared runtime

This directory is an **independent Git repository**.  It owns the files that the
three checkouts (`generic-debug-pipeline`, `paper-refine` and
`paper-refine-eval`) used to copy into themselves:

```
model_runtime.py      shared model/harness/provider runtime
model_catalog.json    canonical model-capacity catalog
llm_policy.json       canonical per-LLM-call retry / dual-key policy
llm_retry.py          retry schedule, DSH overlay and loopback retry proxy
check_*.py            cross-repo consistency checks
```

Each checkout mounts this repository as the **`agent_common/` git submodule**.
The host repository records only `.gitmodules` and the submodule's commit id;
the files themselves stay in this repository.

## Staging rule — read this before `git add`

**Never stage, copy or commit the files inside `agent_common/` in a host
checkout.**  They are not host-repository source.  They are owned here and are
changed by revising this repository; a vendored copy in a checkout would drift
exactly the way the old per-checkout copies did.

The only things a host checkout may commit are:

* `.gitmodules`;
* the `agent_common` gitlink (the pinned commit id).

In particular, do **not** run `git add agent_common/model_runtime.py` (or any
other file under `agent_common/`) in a checkout, and do not copy these files
back into `gdp/`, `pr_eval/` or the paper-refine root.

Only the **`main`** branch of each checkout stages the submodule pointer for a
release, because `main` is the release version.  The **`dev`** branch carries
the same submodule so the code is importable while developing, but a `dev`
pointer is never a release pin: advance it freely and do not treat its files as
host-owned source.  More generally, its submodule files remain versioned *here*
in `agent_common`, never in the host repository.

## Updating the shared runtime

```bash
# 1. Edit and commit the canonical files in this repository.
cd repos/agent_common
$EDITOR model_runtime.py              # etc.
git add -A && git commit -m "..."

# 2. Point each checkout at the new commit.
cd repos/generic-debug-pipeline
git -C agent_common fetch origin             # origin is the path in .gitmodules
git -C agent_common checkout <sha>
git add .gitmodules agent_common             # main/release only
git commit -m "Update agent_common"
```

`sync_llm_policy.sh` and `update_model_catalog.sh` are kept as convenience
wrappers that check out the pinned submodule commit in all three checkouts.

## Verification

```bash
python3 repos/agent_common/check_model_catalog.py
python3 repos/agent_common/check_llm_policy.py
python3 repos/agent_common/check_cross_repo_consistency.py
python3 -m unittest discover -s repos/agent_common -p 'test_*.py'
```

The three checks read the submodule working trees in the sibling checkouts, so
they fail if a checkout is not initialized or if its submodule was modified
locally.
