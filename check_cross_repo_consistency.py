#!/usr/bin/env python3
"""Verify that the three checkouts expose the same model capacities and defaults.

Each checkout is imported in its own subprocess (with its own interpreter), so
the check exercises the loader that actually ships in that repository rather
than re-reading the JSON.  Byte-identity of the catalog in all three submodule
checkouts is checked first by ``check_model_catalog.py``.

    python3 repos/agent-common/check_cross_repo_consistency.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
for name, base_url in (
    ("deepseek-flash", ""),
    ("qwen38-27b", QWEN_DEFAULT_BASE_URL),
    ("dots3-note-preview", DOTS_DEFAULT_BASE_URL),
):
    entry = AGENT_MODELS[name]
    out[entry["id"]] = {
        "context_window": entry["context_window"],
        "max_tokens": entry["max_tokens"],
        "chat_max_tokens": entry["chat_max_tokens"],
    }
    if base_url:
        out[entry["id"]]["base_url"] = base_url
    if entry.get("default_reasoning_level"):
        out[entry["id"]]["default_reasoning_level"] = entry["default_reasoning_level"]
print(json.dumps(out))
"""

PRE_PROBE = """
import json, sys
sys.path.insert(0, ".")
from pr_eval import llm
print(json.dumps({
    "qwen3.8-27b": {
        "context_window": llm.DEFAULT_MAX_CONTEXT,
        "chat_max_tokens": llm.DEFAULT_CHAT_MAX_TOKENS,
        "base_url": llm.DEFAULT_QWEN_BASE_URL,
        "default_reasoning_level": llm.DEFAULT_QWEN_REASONING_EFFORT,
    },
    "dots3-note-prev": {
        "context_window": llm.DEFAULT_DOTS_MAX_CONTEXT,
        "base_url": llm.DEFAULT_DOTS_BASE_URL,
        "default_reasoning_level": llm.MODEL_CAPACITIES["dots3-note-prev"]["reasoning_effort"],
    },
    "deepseek-flash": {
        "context_window": llm.DEFAULT_DEEPSEEK_MAX_CONTEXT,
        "default_reasoning_level": llm.DEFAULT_DEEPSEEK_REASONING_EFFORT,
    },
}))
"""

PR_PROBE = """
import json
import paper_qwen_agent as qwen
import paper_dots_agent as dots
print(json.dumps({
    "qwen3.8-27b": {
        "context_window": qwen.DEFAULT_CONTEXT_WINDOW,
        "max_tokens": qwen.DEFAULT_MAX_TOKENS,
        "chat_max_tokens": qwen.DEFAULT_CHAT_MAX_TOKENS,
        "base_url": qwen.DEFAULT_BASE_URL,
        "default_reasoning_level": qwen.DEFAULT_EFFORT,
    },
    "dots3-note-prev": {
        "context_window": dots.DEFAULT_CONTEXT_WINDOW,
        "max_tokens": dots.DEFAULT_MAX_TOKENS,
        "chat_max_tokens": dots.DEFAULT_CHAT_MAX_TOKENS,
        "base_url": dots.DEFAULT_BASE_URL,
        "default_reasoning_level": dots.DEFAULT_EFFORT,
    },
}))
"""


def interpreter(repo: Path) -> str:
    venv = repo / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def probe(label: str, repo: Path, code: str) -> dict:
    result = subprocess.run(
        [interpreter(repo), "-c", code],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("FAIL %s probe failed:\n%s" % (label, result.stderr.strip()))
        return {}
    return json.loads(result.stdout)


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
        for model_id, fields in observed.items():
            for field, value in fields.items():
                expected = CATALOGUE.get(model_id, {}).get(field)
                if expected != value:
                    problems.append(
                        "%s: %s.%s is %r, catalog says %r"
                        % (label, model_id, field, value, expected)
                    )
    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        return 1
    print("OK gdp, paper-refine-eval and paper-refine expose the catalog values and default reasoning level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
