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
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

COMMON = Path(__file__).resolve().parent
ROOT = COMMON.parent
CATALOG = COMMON / "agent_common" / "model_catalog.json"


def _read_catalogue(path: Optional[Path] = None) -> tuple:
    """The catalog by model id, or ``({}, reason)`` for a malformed document.

    Loading this at module import used to raise before ``main()`` could report
    the catalog corruption the script exists to catch.
    """
    path = CATALOG if path is None else path
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        # RecursionError is not a ValueError; a deeply nested document used to
        # escape this guard and crash at import before main could report it.
        return {}, "cannot read catalog %s: %s" % (path, exc)
    if not isinstance(document, dict):
        return {}, "catalog %s is not a JSON object" % path
    models = document.get("models")
    if not isinstance(models, list):
        return {}, "catalog %s has no models list" % path
    catalogue = {}
    for entry in models:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            return {}, "catalog %s has an invalid model entry" % path
        catalogue[entry["id"]] = entry
    return catalogue, ""


CATALOGUE, _CATALOGUE_ERROR = _read_catalogue()

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
        "chat_max_tokens": entry.get("chat_max_tokens"),
    }
    if entry.get("dsh_provider"):
        item["provider"] = entry["dsh_provider"]
    if entry.get("api_key_env"):
        item["api_key_env"] = entry["api_key_env"]
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
        "chat_max_tokens": entry.get("chat_max_tokens"),
    }
    if entry.get("dsh_provider"):
        result["provider"] = entry["dsh_provider"]
    if entry.get("api_key_env"):
        result["api_key_env"] = entry["api_key_env"]
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
        "chat_max_tokens": entry.get("chat_max_tokens"),
    }
    if entry.get("dsh_provider"):
        result["provider"] = entry["dsh_provider"]
    if entry.get("api_key_env"):
        result["api_key_env"] = entry["api_key_env"]
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
            text=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("FAIL %s probe timed out" % label)
        return None
    except OSError as exc:
        print("FAIL %s probe could not run: %s" % (label, exc))
        return None
    stdout = result.stdout.decode("utf-8", "replace")
    stderr = result.stderr.decode("utf-8", "replace")
    if result.returncode != 0:
        print("FAIL %s probe failed:\n%s" % (label, _redacted_text(stderr.strip())))
        return None
    try:
        observed = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        print("FAIL %s probe emitted invalid JSON: %s" % (label, exc))
        return None
    if not isinstance(observed, dict):
        print("FAIL %s probe did not emit a JSON object" % label)
        return None
    return observed


def expected_fields(
    model_id: str, catalogue: Optional[Mapping[str, Any]] = None
) -> set:
    """Every fact the catalog records for one model (all are required)."""
    entry = (CATALOGUE if catalogue is None else catalogue)[model_id]
    fields = {"context_window", "max_tokens"}
    # The runtime treats chat_max_tokens as optional.  A probe may still report
    # its adapter's default for it, so main() ignores the key when the catalog
    # omits it instead of calling the observation an unexpected field.
    if entry.get("chat_max_tokens") is not None:
        fields.add("chat_max_tokens")
    for field in ("base_url", "default_reasoning_level", "provider", "api_key_env"):
        if entry.get(field):
            fields.add(field)
    return fields


def _redacted(value: Any) -> Any:
    """A URL-valued field with credentials removed, for the failure output.

    The non-secret query is kept: dropping the whole query made
    ``?api-version=2023`` and ``?api-version=2024`` print identically, hiding
    the actual drift.
    """
    if not isinstance(value, str):
        return value
    try:
        from agent_common.model_runtime import _credential_key, redact_endpoint_url

        safe = redact_endpoint_url(value)
        parsed = urllib.parse.urlsplit(value)
        kept = [
            (key, item)
            for key, item in urllib.parse.parse_qsl(
                parsed.query.replace(";", "&"), keep_blank_values=True
            )
            if not _credential_key(key)
        ]
        if kept:
            safe += "?" + urllib.parse.urlencode(kept, doseq=True)
        return safe
    except Exception:  # noqa: BLE001 - the probe's own failure is reported elsewhere
        # Never print a credential just because the shared redactor was
        # unavailable; keep only the scheme/host/path.
        text = value.split("#", 1)[0].split("?", 1)[0]
        if "://" in text:
            scheme, _, rest = text.partition("://")
            rest = rest.lstrip("/")
            authority, separator, tail = rest.partition("/")
            authority = authority.rsplit("@", 1)[-1]
            return scheme + "://" + authority + separator + tail
        return text.rsplit("@", 1)[-1] if "@" in text else text


_URL_PATTERN = re.compile(r"https?://[^\s'\"]+")


def _redacted_text(text: str) -> str:
    """Redact credential-bearing URLs embedded in free-form probe output."""
    return _URL_PATTERN.sub(lambda match: str(_redacted(match.group(0))), text)


def main() -> int:
    catalogue, error = _read_catalogue()
    if error:
        print("FAIL %s" % error)
        return 1
    try:
        checks = subprocess.run(
            [sys.executable, str(COMMON / "check_model_catalog.py")],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("FAIL check_model_catalog.py timed out")
        return 1
    print(checks.stdout.strip())
    if checks.returncode != 0:
        # A child crash (for example a checker bug) writes its diagnostic to
        # stderr and leaves stdout empty; discarding stderr hid the root cause.
        if checks.stderr.strip():
            print(checks.stderr.strip())
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
        for model_id in sorted(set(catalogue) - set(observed)):
            problems.append("%s: missing model %s" % (label, model_id))
        for model_id in sorted(set(observed) - set(catalogue)):
            problems.append("%s: unexpected model %s" % (label, model_id))
        for model_id in sorted(set(observed) & set(catalogue)):
            observation = observed[model_id]
            if not isinstance(observation, Mapping):
                problems.append(
                    "%s: %s observation is not an object: %r"
                    % (label, model_id, observation)
                )
                continue
            fields = expected_fields(model_id, catalogue)
            for field in sorted(fields - set(observation)):
                problems.append("%s: %s is missing %s" % (label, model_id, field))
            for field in sorted(set(observation) - fields):
                # F44 made the single-shot chat cap optional in the catalog.
                # A sibling probe can still expose its own adapter default
                # when the catalog omits the value; that is not a fact the
                # canonical catalog owns, so it is not an unexpected field.
                if (
                    field == "chat_max_tokens"
                    and catalogue[model_id].get(field) is None
                ):
                    continue
                problems.append(
                    "%s: %s has unexpected field %s" % (label, model_id, field)
                )
            for field in sorted(fields & set(observation)):
                observed_value = observation[field]
                expected = catalogue[model_id].get(field)
                if expected != observed_value:
                    # A drifted checkout can report a credential-bearing
                    # base_url; the checker used to print it verbatim three
                    # times.  model_runtime's own recorder redacts, so the
                    # mismatch output must too.
                    problems.append(
                        "%s: %s.%s is %r, catalog says %r"
                        % (
                            label,
                            model_id,
                            field,
                            _redacted(observed_value),
                            _redacted(expected),
                        )
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
