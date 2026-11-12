#!/usr/bin/env python3
"""Verify that the three checkouts expose the same model capacities and defaults.

Each checkout is imported in its own subprocess (with its own interpreter), so
the check exercises the loader that actually ships in that repository rather
than re-reading the JSON.  A probe that cannot run, omits an expected model or
omits an expected field is a failure, not an empty (consistent-looking) result.
Byte-identity of the shared package in all three submodule checkouts is checked
first by ``check_model_catalog.py`` and ``check_shared_package.py``.

    python3 repos/agent-common/check_cross_repo_consistency.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parent
CATALOG = COMMON / "agent_common" / "model_catalog.json"
CATALOGUE = {m["id"]: m for m in json.loads(CATALOG.read_text(encoding="utf-8"))["models"]}

GDP = ROOT / "generic-debug-pipeline"
PRE = ROOT / "paper-refine-eval"
PR = ROOT / "paper-refine"

GDP_PROBE = """
import json
from gdp.agent_models import (
    AGENT_MODELS, DOTS_DEFAULT_BASE_URL, QWEN_DEFAULT_BASE_URL,
)
out = {}
for entry in AGENT_MODELS.values():
    item = {
        "context_window": entry["context_window"],
        "max_tokens": entry["max_tokens"],
        "chat_max_tokens": entry["chat_max_tokens"],
    }
    base_url = entry.get("base_url")
    if not base_url:
        if entry["id"] == "qwen3.8-27b":
            base_url = QWEN_DEFAULT_BASE_URL
        elif entry["id"] == "dots3-note-prev":
            base_url = DOTS_DEFAULT_BASE_URL
    if base_url:
        item["base_url"] = base_url
    if entry.get("default_reasoning_level"):
        item["default_reasoning_level"] = entry["default_reasoning_level"]
    out[entry["id"]] = item
print(json.dumps(out))
"""

#: The probe reports every model the shared table knows, including the ones the
#: adapter-specific module does not re-export itself.  Values it does re-export
#: (``llm.DEFAULT_*``) still override the shared-module read, so a dropped or
#: stale adapter constant is caught as well.
PRE_PROBE = """
import json
import pr_eval.llm as llm
rt = llm._model_runtime
by_id = {entry["id"]: entry for entry in rt.AGENT_MODELS.values()}

def item(model_id):
    entry = by_id[model_id]
    result = {
        "context_window": entry["context_window"],
        "max_tokens": entry["max_tokens"],
        "chat_max_tokens": entry["chat_max_tokens"],
    }
    base_url = entry.get("base_url")
    if model_id == "qwen3.8-27b":
        base_url = getattr(llm, "DEFAULT_QWEN_BASE_URL", base_url)
    elif model_id == "dots3-note-prev":
        base_url = getattr(llm, "DEFAULT_DOTS_BASE_URL", base_url)
    if base_url:
        result["base_url"] = base_url
    level = entry.get("default_reasoning_level")
    if level:
        result["default_reasoning_level"] = level
    return result

out = {model_id: item(model_id) for model_id in by_id}
out["qwen3.8-27b"].update({
    "context_window": llm.DEFAULT_MAX_CONTEXT,
    "chat_max_tokens": llm.DEFAULT_CHAT_MAX_TOKENS,
    "base_url": llm.DEFAULT_QWEN_BASE_URL,
    "default_reasoning_level": llm.DEFAULT_QWEN_REASONING_EFFORT,
})
out["dots3-note-prev"].update({
    "context_window": llm.DEFAULT_DOTS_MAX_CONTEXT,
    "base_url": llm.DEFAULT_DOTS_BASE_URL,
    "default_reasoning_level": llm.MODEL_CAPACITIES["dots3-note-prev"]["reasoning_effort"],
})
out["deepseek-flash"].update({
    "context_window": llm.DEFAULT_DEEPSEEK_MAX_CONTEXT,
    "default_reasoning_level": llm.DEFAULT_DEEPSEEK_REASONING_EFFORT,
})
print(json.dumps(out))
"""

PR_PROBE = """
import json
import paper_qwen_agent as qwen
import paper_dots_agent as dots
rt = qwen.rt
by_id = {entry["id"]: entry for entry in rt.AGENT_MODELS.values()}

def item(model_id):
    entry = by_id[model_id]
    result = {
        "context_window": entry["context_window"],
        "max_tokens": entry["max_tokens"],
        "chat_max_tokens": entry["chat_max_tokens"],
    }
    base_url = entry.get("base_url")
    if model_id == "qwen3.8-27b":
        base_url = qwen.DEFAULT_BASE_URL
    elif model_id == "dots3-note-prev":
        base_url = dots.DEFAULT_BASE_URL
    if base_url:
        result["base_url"] = base_url
    level = entry.get("default_reasoning_level")
    if level:
        result["default_reasoning_level"] = level
    return result

out = {model_id: item(model_id) for model_id in by_id}
out["qwen3.8-27b"].update({
    "context_window": qwen.DEFAULT_CONTEXT_WINDOW,
    "max_tokens": qwen.DEFAULT_MAX_TOKENS,
    "chat_max_tokens": qwen.DEFAULT_CHAT_MAX_TOKENS,
    "base_url": qwen.DEFAULT_BASE_URL,
    "default_reasoning_level": qwen.DEFAULT_EFFORT,
})
out["dots3-note-prev"].update({
    "context_window": dots.DEFAULT_CONTEXT_WINDOW,
    "max_tokens": dots.DEFAULT_MAX_TOKENS,
    "chat_max_tokens": dots.DEFAULT_CHAT_MAX_TOKENS,
    "base_url": dots.DEFAULT_BASE_URL,
    "default_reasoning_level": dots.DEFAULT_EFFORT,
})
out["deepseek-flash"].update({
    "context_window": rt.DEEPSEEK_CONTEXT_WINDOW,
    "max_tokens": rt.DEEPSEEK_MAX_TOKENS,
    "chat_max_tokens": rt.DEEPSEEK_CHAT_MAX_TOKENS,
    "default_reasoning_level": rt.DEEPSEEK_DEFAULT_EFFORT,
})
print(json.dumps(out))
"""


def interpreter(repo: Path) -> str:
    venv = repo / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def probe(label: str, repo: Path, code: str) -> Optional[Dict[str, Any]]:
    """The probe's parsed output, or ``None`` after reporting the failure."""
    try:
        result = subprocess.run(
            [interpreter(repo), "-c", code],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print("FAIL %s probe could not run: %s" % (label, exc))
        return None
    if result.returncode != 0:
        print("FAIL %s probe failed:\n%s" % (label, result.stderr.strip()))
        return None
    try:
        observed = json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        print("FAIL %s probe emitted invalid JSON: %s" % (label, exc))
        return None
    if not isinstance(observed, dict):
        print("FAIL %s probe did not emit a JSON object" % label)
        return None
    return observed


def expected_fields(model_id: str) -> set:
    """Every fact the catalog records for one model (all are required)."""
    entry = CATALOGUE[model_id]
    fields = {"context_window", "max_tokens", "chat_max_tokens"}
    if entry.get("base_url"):
        fields.add("base_url")
    if entry.get("default_reasoning_level"):
        fields.add("default_reasoning_level")
    return fields


def main() -> int:
    checks = subprocess.run(
        [sys.executable, str(COMMON / "check_model_catalog.py")],
        capture_output=True,
        text=True,
    )
    print(checks.stdout.strip())
    if checks.returncode != 0:
        return 1

    problems = []
    for label, repo, code in (
        ("gdp", GDP, GDP_PROBE),
        ("paper-refine-eval", PRE, PRE_PROBE),
        ("paper-refine", PR, PR_PROBE),
    ):
        observed = probe(label, repo, code)
        if observed is None:
            # A probe that could not run is a failed check, not an empty
            # observation that happens to be consistent.
            problems.append("%s probe failed" % label)
            continue
        for model_id in sorted(set(CATALOGUE) - set(observed)):
            problems.append("%s: missing model %s" % (label, model_id))
        for model_id in sorted(set(observed) - set(CATALOGUE)):
            problems.append("%s: unexpected model %s" % (label, model_id))
        for model_id in sorted(set(observed) & set(CATALOGUE)):
            fields = expected_fields(model_id)
            for field in sorted(fields - set(observed[model_id])):
                problems.append("%s: %s is missing %s" % (label, model_id, field))
            for field in sorted(fields & set(observed[model_id])):
                value = observed[model_id][field]
                expected = CATALOGUE[model_id].get(field)
                if expected != value:
                    problems.append(
                        "%s: %s.%s is %r, catalog says %r"
                        % (label, model_id, field, value, expected)
                    )
    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        return 1
    print(
        "OK gdp, paper-refine-eval and paper-refine expose every catalog model, "
        "field and default reasoning level"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
