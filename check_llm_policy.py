#!/usr/bin/env python3
"""Verify the shared LLM retry/compaction policy in every checkout.

Canonical files: ``repos/agent-common/agent_common/llm_policy.json`` and
``repos/agent-common/agent_common/llm_retry.py``.  Each checkout mounts this
repository as the ``agent-common/`` submodule and imports the inner
``agent_common`` package.  This script also checks the derived
behaviour that matters:

* the schedule is 10s, 20s, ... capped at 3600s (ten retries);
* the compaction route names a second Dots key;
* the paper-refine adapters that cannot import a sibling module carry the same
  constants.

    python3 repos/agent-common/check_llm_policy.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parent
CANONICAL_DIR = COMMON / "agent_common"
CANONICAL_JSON = CANONICAL_DIR / "llm_policy.json"
CANONICAL_MODULE = CANONICAL_DIR / "llm_retry.py"
COPIES = (
    ROOT / "generic-debug-pipeline" / "agent-common" / "agent_common",
    ROOT / "paper-refine" / "agent-common" / "agent_common",
    ROOT / "paper-refine-eval" / "agent-common" / "agent_common",
)
# ``$HOME/generic_debug_pipeline_working_copy`` is a disposable checkout that the
# next ``git pull && git checkout .`` overwrites, so it is deliberately not
# checked here (and not treated as a maintained copy by sync_llm_policy.sh).


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    if not CANONICAL_JSON.is_file():
        print("missing canonical policy: %s" % CANONICAL_JSON)
        return 1
    if not CANONICAL_MODULE.is_file():
        print("missing canonical helper: %s" % CANONICAL_MODULE)
        return 1

    document = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
    retry = document.get("retry") or {}
    expected = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3600]
    initial = float(retry.get("initial_delay_seconds", 0))
    multiplier = float(retry.get("multiplier", 0))
    maximum = float(retry.get("max_delay_seconds", 0))
    max_retries = int(retry.get("max_retries", 0))
    schedule = [
        min(initial * multiplier ** index, maximum)
        for index in range(max_retries)
    ]
    if schedule != expected:
        problems.append("retry schedule is %r, expected %r" % (schedule, expected))
    if "RATE_LIMIT" not in (retry.get("retryable_error_codes") or []):
        problems.append("RATE_LIMIT is not retryable")
    if 429 not in (retry.get("retryable_http_statuses") or []):
        problems.append("HTTP 429 is not retryable")

    dots = document.get("dots") or {}
    if dots.get("regular_api_key_env") == dots.get("compaction_api_key_env"):
        problems.append("regular and compaction Dots keys are the same variable")
    if not dots.get("compaction_provider"):
        problems.append("no dedicated compaction provider is declared")
    compaction = document.get("compaction") or {}
    if int(compaction.get("summarization_max_tokens") or 0) <= 0:
        problems.append("summarization_max_tokens is not a positive bound")

    for directory in COPIES:
        for name in ("llm_policy.json", "llm_retry.py"):
            copy = directory / name
            if not copy.is_file():
                problems.append("missing copy: %s" % copy)
            elif digest(copy) != digest(CANONICAL_DIR / name):
                problems.append("copy differs from canonical: %s" % copy)

    # The paper-refine adapters now consume the shared runtime rather than
    # carrying a second copy of the retry constants/provider YAML.
    for name in ("paper_dots_agent.py", "paper_qwen_agent.py",
                 "paper_model_agent.py"):
        adapter = ROOT / "paper-refine" / name
        if not adapter.is_file():
            # An initialized paper-refine checkout must carry the adapter; a
            # missing file used to be silently skipped, so the final OK claimed
            # the shared-runtime wiring had been inspected when it had not.
            if (ROOT / "paper-refine").is_dir():
                problems.append("missing adapter: %s" % adapter)
            continue
        text = adapter.read_text(encoding="utf-8")
        if "import model_runtime as rt" not in text:
            problems.append("%s does not import the shared model runtime" % adapter)
        if name in ("paper_dots_agent.py", "paper_qwen_agent.py"):
            if "rt.RETRY_SCHEDULE" not in text:
                problems.append("%s does not use the shared retry schedule" % adapter)
            if "rt.dsh_patch_text" not in text:
                problems.append("%s does not generate its DSH patch via common" % adapter)

    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        return 1
    print(
        "OK schedule %r; canonical and %d submodule checkouts share sha256 %s"
        % (expected, len(COPIES), digest(CANONICAL_JSON)[:16])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
