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

import datetime
import hashlib
import json
import sys
import urllib.parse
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


def _redacted_url(value: object) -> object:
    """A URL safe to print in a diagnostic (never its credential).

    A non-secret query parameter is kept: dropping the whole query made
    ``?api-version=2023`` and ``?api-version=2024`` print identically, hiding
    the drift this checker reports.
    """
    if not isinstance(value, str):
        return value
    try:
        from agent_common.model_runtime import _credential_key, redact_endpoint_url

        safe = redact_endpoint_url(value)
        try:
            query = urllib.parse.urlsplit(value).query
        except ValueError:
            query = ""
        kept = [
            (key, item)
            for key, item in urllib.parse.parse_qsl(
                query.replace(";", "&"), keep_blank_values=True
            )
            if not _credential_key(key)
        ]
        if kept:
            safe += "?" + urllib.parse.urlencode(kept, doseq=True)
        return safe
    except Exception:  # noqa: BLE001 - a checker reports, never crashes
        # Last-resort textual scrub: keep the scheme/host/path and drop
        # userinfo/query, including the malformed triple-slash spelling.
        text = value.split("#", 1)[0].split("?", 1)[0]
        if "://" in text:
            scheme, _, rest = text.partition("://")
            rest = rest.lstrip("/")
            authority, separator, tail = rest.partition("/")
            authority = authority.rsplit("@", 1)[-1]
            return scheme + "://" + authority + separator + tail
        if "@" in text:
            return text.rsplit("@", 1)[-1]
        return text


def _iso_date(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def main() -> int:
    problems = []
    if not CANONICAL.is_file():
        print("missing canonical catalog: %s" % CANONICAL)
        return 1
    try:
        expected = digest(CANONICAL)
        document = json.loads(CANONICAL.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        # RecursionError is not a ValueError; a deeply nested document used to
        # escape this guard and crash the checker before it could print FAIL.
        print("FAIL cannot read canonical catalog %s: %s" % (CANONICAL, exc))
        return 1
    if not isinstance(document, dict):
        print("FAIL canonical catalog is not a JSON object")
        return 1
    updated_at = document.get("updated_at")
    if not _iso_date(updated_at):
        problems.append("bad updated_at %r" % (updated_at,))
    models = document.get("models", [])
    if not isinstance(models, list):
        print("FAIL canonical catalog models is not a list")
        return 1
    schema_version = document.get("schema_version")
    # ``True == 1`` in Python, so an equality test alone accepted a boolean
    # schema_version as the integer 1.
    if type(schema_version) is not int or schema_version != 1:
        problems.append("unsupported schema_version %r" % (schema_version,))
    seen = set()
    for entry in models:
        if not isinstance(entry, dict):
            problems.append("catalog entry is not an object: %r" % (entry,))
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            problems.append("bad model id %r" % (model_id,))
            continue
        if model_id in seen:
            problems.append("duplicate model id %r" % (model_id,))
            continue
        seen.add(model_id)
        context = entry.get("context_window")
        output = entry.get("max_tokens")
        if type(context) is not int or context <= 0:
            problems.append("%s: bad context_window %r" % (model_id, context))
        if type(output) is not int or output <= 0:
            problems.append("%s: bad max_tokens %r" % (model_id, output))
        elif type(context) is int and output >= context:
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
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            problems.append("%s: bad name %r" % (model_id, name))
        verified_at = entry.get("verified_at")
        if not verified_at:
            problems.append("%s: missing verified_at" % model_id)
        elif not _iso_date(verified_at):
            problems.append("%s: bad verified_at %r" % (model_id, verified_at))
        # The runtime treats a missing chat_max_tokens as optional (and removes
        # it from the table); the checker used to require it, so a catalog the
        # runtime accepted was rejected here.
        chat = entry.get("chat_max_tokens")
        if chat is not None:
            if type(chat) is not int or chat <= 0:
                problems.append("%s: bad chat_max_tokens %r" % (model_id, chat))
            elif type(context) is int and chat >= context:
                problems.append(
                    "%s: chat_max_tokens %d does not fit context_window %d"
                    % (model_id, chat, context)
                )
            elif type(output) is int and chat < output:
                problems.append(
                    "%s: chat_max_tokens %d is below max_tokens %d"
                    % (model_id, chat, output)
                )
        # The runtime rejects these at import; the checker used to report OK
        # and exit 0 for a catalog the runtime could not load.
        base_url = entry.get("base_url")
        if base_url is not None:
            if not isinstance(base_url, str) or not base_url.strip():
                problems.append("%s: bad base_url %r" % (model_id, base_url))
            else:
                try:
                    parsed = urllib.parse.urlsplit(base_url)
                    host = parsed.hostname
                    port = parsed.port
                except ValueError as exc:
                    problems.append(
                        "%s: bad base_url %r (%s)"
                        % (model_id, _redacted_url(base_url), exc)
                    )
                else:
                    if not host:
                        problems.append(
                            "%s: base_url has no host: %r"
                            % (model_id, _redacted_url(base_url))
                        )
                    elif port is not None and not 0 < port <= 65535:
                        problems.append(
                            "%s: base_url has an invalid port: %r"
                            % (model_id, _redacted_url(base_url))
                        )
        for field in ("api_key_env", "provider"):
            value = entry.get(field)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                problems.append("%s: bad %s %r" % (model_id, field, value))
        level = entry.get("default_reasoning_level")
        # Check the type before set membership: a list/dict level is
        # unhashable and used to abort the checker with a TypeError.
        if level is not None and (
            not isinstance(level, str)
            or level not in {
                "minimal",
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }
        ):
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
            continue
        try:
            copy_digest = digest(path)
        except OSError as exc:
            # An unreadable copy is a specific FAIL, not a traceback that
            # aborts every other check.
            problems.append("cannot read copy %s: %s" % (path, exc))
            continue
        if copy_digest != expected:
            problems.append("copy differs from canonical: %s" % path)
    for model_id in sorted(EXPECTED_MODEL_IDS - seen):
        problems.append("missing expected model id %s" % model_id)
    for model_id in sorted(seen - EXPECTED_MODEL_IDS):
        problems.append("unexpected model id %s" % model_id)
    # Finally, exercise the loader this checker protects rather than only the
    # JSON shape: a catalog the runtime rejects (or interprets differently) must
    # not print OK.
    if not problems:
        try:
            from agent_common import model_runtime as runtime

            models_table = {
                name: dict(entry) for name, entry in runtime.AGENT_MODELS.items()
            }
            old_path = runtime._MODEL_CATALOG_PATH
            old_secrets = dict(runtime._CATALOG_BASE_SECRETS)
            try:
                runtime._MODEL_CATALOG_PATH = CANONICAL
                runtime._apply_model_catalog(models_table)
            finally:
                runtime._MODEL_CATALOG_PATH = old_path
                runtime._CATALOG_BASE_SECRETS = old_secrets
        except Exception as exc:  # noqa: BLE001 - a checker reports, never crashes
            problems.append(
                "the shared runtime does not accept this catalog: %s: %s"
                % (type(exc).__name__, exc)
            )
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
