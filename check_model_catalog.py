#!/usr/bin/env python3
"""Verify the model catalog in every checkout's agent_common submodule.

Canonical file: ``repos/agent-common/agent_common/model_catalog.json``.  Each
checkout mounts this repository as ``agent-common/``; run this after editing the catalog
and committing it here.  It fails on a byte difference (a checkout's submodule
drifted), an invalid capacity pair, a missing or unexpected model id, or a
`default_reasoning_level` that is not the documented policy (DeepSeek and Dots
high, Qwen medium; GPT and Claude carry no level so their own default applies).

    python3 repos/agent-common/check_model_catalog.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parent
CANONICAL = COMMON / "agent_common" / "model_catalog.json"
COPIES = [
    ROOT / "generic-debug-pipeline" / "agent-common" / "agent_common" / "model_catalog.json",
    ROOT / "paper-refine-eval" / "agent-common" / "agent_common" / "model_catalog.json",
    ROOT / "paper-refine" / "agent-common" / "agent_common" / "model_catalog.json",
]

#: The model ids the shared runtime's ``AGENT_MODELS`` table names.  A catalog
#: that is missing one of these (or carries an extra one) is not the canonical
#: catalog even when every entry it does contain is internally valid.
EXPECTED_MODEL_IDS = frozenset(
    {
        "deepseek-flash",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "qwen3.8-27b",
        "dots3-note-prev",
    }
)

#: The reasoning level each deployment serves when the operator sets no
#: override.  This is the policy the runtime and every harness read from the
#: catalog; a catalog edit that changes one of these is a policy change, not a
#: capacity refresh, so the check pins it here.
REASONING_DEFAULTS = {
    "deepseek-flash": "high",
    "dots3-note-prev": "high",
    "qwen3.8-27b": "medium",
}
#: GPT and Claude keep the model's own default: the generated configuration
#: names no reasoning effort for them, so their entries must carry no level.
MODEL_DEFAULT_IDS = (
    "gpt-6.1-sol",
    "gpt-6.1-terra",
    "claude-opus-5-5",
    "claude-sonnet-5-5",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    if not CANONICAL.is_file():
        print("missing canonical catalog: %s" % CANONICAL)
        return 1
    expected = digest(CANONICAL)
    document = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        problems.append("unsupported schema_version %r" % document.get("schema_version"))
    seen = set()
    for entry in document.get("models", []):
        model_id = entry.get("id")
        if not model_id or model_id in seen:
            problems.append("missing or duplicate model id %r" % (model_id,))
            continue
        seen.add(model_id)
        context = entry.get("context_window")
        output = entry.get("max_tokens")
        if not isinstance(context, int) or context <= 0:
            problems.append("%s: bad context_window %r" % (model_id, context))
        if not isinstance(output, int) or output <= 0:
            problems.append("%s: bad max_tokens %r" % (model_id, output))
        elif isinstance(context, int) and output >= context:
            problems.append(
                "%s: max_tokens %d does not fit context_window %d"
                % (model_id, output, context)
            )
        sources = entry.get("source")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(item, str) and item.strip() for item in sources)
        ):
            problems.append("%s: missing source provenance" % model_id)
        if not entry.get("verified_at"):
            problems.append("%s: missing verified_at" % model_id)
        chat = entry.get("chat_max_tokens")
        if not isinstance(chat, int) or chat <= 0:
            problems.append("%s: bad chat_max_tokens %r" % (model_id, chat))
        elif isinstance(context, int) and chat >= context:
            problems.append(
                "%s: chat_max_tokens %d does not fit context_window %d"
                % (model_id, chat, context)
            )
        elif isinstance(output, int) and chat < output:
            problems.append(
                "%s: chat_max_tokens %d is below max_tokens %d"
                % (model_id, chat, output)
            )
        level = entry.get("default_reasoning_level")
        if level is not None and level not in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            problems.append(
                "%s: bad default_reasoning_level %r" % (model_id, level)
            )
        expected_level = REASONING_DEFAULTS.get(model_id)
        if expected_level is not None and level != expected_level:
            problems.append(
                "%s: default_reasoning_level is %r, the policy requires %r"
                % (model_id, level, expected_level)
            )
        if model_id in MODEL_DEFAULT_IDS and level is not None:
            problems.append(
                "%s: GPT and Claude keep the model's own default, so the entry "
                "must not carry default_reasoning_level %r"
                % (model_id, level)
            )
    for path in COPIES:
        if not path.is_file():
            problems.append("missing copy: %s" % path)
        elif digest(path) != expected:
            problems.append("copy differs from canonical: %s" % path)
    for model_id in sorted(EXPECTED_MODEL_IDS - seen):
        problems.append("missing expected model id %s" % model_id)
    for model_id in sorted(seen - EXPECTED_MODEL_IDS):
        problems.append("unexpected model id %s" % model_id)
    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        return 1
    print(
        "OK %d models; default reasoning levels %s; canonical and %d submodule "
        "checkouts share sha256 %s"
        % (
            len(seen),
            ", ".join(
                "%s=%s" % pair for pair in sorted(REASONING_DEFAULTS.items())
            ),
            len(COPIES),
            expected[:16],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
