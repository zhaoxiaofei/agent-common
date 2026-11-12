#!/usr/bin/env python3
"""Verify that every shared package file is identical in all three checkouts.

``check_model_catalog.py`` compares only ``model_catalog.json`` and
``check_llm_policy.py`` only ``llm_policy.json``/``llm_retry.py``.  That left
``model_runtime.py``, ``chat.py``, ``dsh_events.py``, ``agent_cli.py`` and
``__init__.py`` unverified: a checkout that edited one of them locally still
passed the three checks, contradicting the README's promise that a locally
modified submodule is detected.

    python3 repos/agent-common/check_shared_package.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parent
CANONICAL_DIR = COMMON / "agent_common"
COPIES = (
    ROOT / "generic-debug-pipeline" / "agent-common" / "agent_common",
    ROOT / "paper-refine" / "agent-common" / "agent_common",
    ROOT / "paper-refine-eval" / "agent-common" / "agent_common",
)


def package_files(directory: Path) -> dict:
    """Every tracked-looking file in one shared package, keyed by relative path."""
    files = {}
    if not directory.is_dir():
        return files
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files[relative.as_posix()] = path
    return files


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    canonical = package_files(CANONICAL_DIR)
    if not canonical:
        print("missing canonical package files: %s" % CANONICAL_DIR)
        return 1
    for directory in COPIES:
        copy = package_files(directory)
        for name in sorted(canonical):
            if name not in copy:
                problems.append("missing copy: %s" % (directory / name))
            elif digest(copy[name]) != digest(canonical[name]):
                problems.append("copy differs from canonical: %s" % (directory / name))
        for name in sorted(set(copy) - set(canonical)):
            problems.append("unexpected file in copy: %s" % (directory / name))
    if problems:
        for problem in problems:
            print("FAIL %s" % problem)
        return 1
    print(
        "OK %d shared package files; canonical and %d submodule checkouts are "
        "byte-identical"
        % (len(canonical), len(COPIES))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
