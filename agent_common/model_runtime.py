"""Shared model, harness and provider runtime for all three repositories.

The canonical copy lives in ``repos/agent-common``; each checkout mounts it
as the ``agent-common/`` git submodule (``update_model_catalog.sh`` checks the
pinned submodule out).  A checkout must not edit the submodule's files.

The two axes are:

* ``--agent-harness`` -- the CLI that runs a session: ``dsh`` (DeepSeek
  Harness), ``gpt-codex`` (OpenAI Codex), or ``claude-code`` (Claude Code);
* ``--agent-model`` -- a known model, the ``default`` sentinel (pass no model
  and let the harness profile/CLI own it), or an arbitrary OpenAI-compatible
  ``/chat/completions`` URL.

The module owns the model catalog, the wire-protocol support matrix, the DSH
``--patch`` overlays, the Codex ``-c`` overrides, the Claude Code environment,
and the request-preparation proxy.  It is a leaf module: standard library plus
the sibling ``llm_retry`` module.
"""

from __future__ import annotations

import contextlib
import http.client
import http.server
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__:
    from .llm_retry import (
        RetrySchedule,
        dots_compaction,
        dots_dsh_overlay,
        load_policy as _load_llm_policy,
        retry_policy_yaml,
        start_retry_proxy,
        stop_retry_proxy,
    )
else:  # pragma: no cover - a paper-refine frozen root has no package
    from llm_retry import (  # type: ignore
        RetrySchedule,
        dots_compaction,
        dots_dsh_overlay,
        load_policy as _load_llm_policy,
        retry_policy_yaml,
        start_retry_proxy,
        stop_retry_proxy,
    )


class ConfigError(Exception):
    """A model/harness configuration error.

    The host repository replaces this class at import time (GDP aliases its
    ``gdp.errors.ConfigError``) so callers keep their own error hierarchy.
    """

# ---------------------------------------------------------------------------
# Harnesses
# ---------------------------------------------------------------------------
AGENT_HARNESS_DSH = "dsh"
AGENT_HARNESS_GPT_CODEX = "gpt-codex"
AGENT_HARNESS_CLAUDE_CODE = "claude-code"

AGENT_HARNESSES: Dict[str, Dict[str, str]] = {
    AGENT_HARNESS_DSH: {
        "label": "DeepSeek Harness",
        "binary": "dsh",
        "default_model": "deepseek-flash",
    },
    AGENT_HARNESS_GPT_CODEX: {
        "label": "GPT Codex",
        "binary": "codex",
        "default_model": "gpt-sol",
    },
    AGENT_HARNESS_CLAUDE_CODE: {
        "label": "Claude Code",
        "binary": "claude",
        "default_model": "claude-sonnet",
    },
}
AGENT_HARNESS_CHOICES: Tuple[str, ...] = tuple(AGENT_HARNESSES)
DEFAULT_AGENT_HARNESS = AGENT_HARNESS_DSH
DEFAULT_AGENT_MODEL = "deepseek-flash"
#: A sentinel model choice: pass no model to the harness and let its own
#: profile/CLI configuration select one (for example
#: ``agent-default-model`` in the DSH headless profile).
MODEL_DEFAULT = "default"

#: Whether ``resolve_agent_selection``/``agent_model_choice`` reject an endpoint
#: URL that carries a credential.  The shared runtime is conservative by default
#: (paper-refine); GDP opts out because its adapter strips userinfo from the
#: generated patch and hands the secret to the live request instead.
REJECT_ENDPOINT_CREDENTIALS = True

#: Whether a missing catalog file is fatal.  paper-refine keeps the documented
#: default capacities so setup can report the missing asset itself; GDP and
#: paper-refine-eval fail closed at import time.
MISSING_CATALOG_IS_FATAL = False

# The legacy spellings the old ``--agent`` option accepted and still accepts.
LEGACY_AGENTS: Tuple[str, ...] = ("codex", "dsh")
LEGACY_DEFAULT_AGENT = "codex"

#: What the legacy shortcuts mean as a (harness, model) pair.  ``None`` on the
#: model side means the operator's CLI configuration owns the model; the legacy
#: command is kept exactly as before either way.
LEGACY_AGENT_PAIRS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "codex": (AGENT_HARNESS_GPT_CODEX, None),
    "dsh": (AGENT_HARNESS_DSH, "deepseek-flash"),
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
#: ``dots3-note-prev`` advertises 512K (524,288), but the endpoint rejects a
#: request whose prompt plus ``max_tokens`` crosses 393,000 with a generic HTTP
#: 400.  Declaring the advertised window disables DSH's compaction until ~419K,
#: past the provider's real limit.  Every capacity below is a placeholder: the
#: authoritative numbers live in ``model_catalog.json`` beside this module (the
#: canonical copy is ``repos/agent-common/agent_common/model_catalog.json``).

# "latest" model ids were read from the installed harness/catalog on 2026-11-10
# (the same ids the paper-refine reference implementation records):
# Codex knows gpt-6.1-sol / gpt-6.1-terra; the DSH catalog ships deepseek-flash,
# gpt-6.1-sol/terra, claude-opus-5-5 and claude-sonnet-5-5; Claude Code resolves
# the `opus`/`sonnet` aliases to the latest Anthropic models.  `qwen38-27b` is
# the local vLLM checkpoint the qwen adapter serves; `dots3-note-preview` is the
# operator-facing name for the Dots API model `dots3-note-prev`.
AGENT_MODELS: Dict[str, Dict[str, Any]] = {
    "deepseek-flash": {
        "label": "DeepSeek Flash",
        "id": "deepseek-flash",
        "family": "deepseek",
        "default_harness": AGENT_HARNESS_DSH,
        "harnesses": (
            AGENT_HARNESS_DSH,
            AGENT_HARNESS_GPT_CODEX,
            AGENT_HARNESS_CLAUDE_CODE,
        ),
        "dsh_provider": "deepseek-official",
        "context_window": 1048576,
        "max_tokens": 384000,
    },
    "gpt-sol": {
        "label": "GPT Sol",
        "id": "gpt-6.1-sol",
        "family": "gpt",
        "default_harness": AGENT_HARNESS_GPT_CODEX,
        "harnesses": (AGENT_HARNESS_DSH, AGENT_HARNESS_GPT_CODEX),
        "dsh_provider": "openai",
        "context_window": 1050000,
        "max_tokens": 128000,
    },
    "gpt-terra": {
        "label": "GPT Terra",
        "id": "gpt-6.1-terra",
        "family": "gpt",
        "default_harness": AGENT_HARNESS_GPT_CODEX,
        "harnesses": (AGENT_HARNESS_DSH, AGENT_HARNESS_GPT_CODEX),
        "dsh_provider": "openai",
        "context_window": 1050000,
        "max_tokens": 128000,
    },
    "claude-opus": {
        "label": "Claude Opus",
        "id": "claude-opus-5-5",
        "alias": "opus",
        "family": "claude",
        "default_harness": AGENT_HARNESS_CLAUDE_CODE,
        "harnesses": (AGENT_HARNESS_DSH, AGENT_HARNESS_CLAUDE_CODE),
        "dsh_provider": "anthropic",
        "context_window": 1000000,
        "max_tokens": 128000,
    },
    "claude-sonnet": {
        "label": "Claude Sonnet",
        "id": "claude-sonnet-5-5",
        "alias": "sonnet",
        "family": "claude",
        "default_harness": AGENT_HARNESS_CLAUDE_CODE,
        "harnesses": (AGENT_HARNESS_DSH, AGENT_HARNESS_CLAUDE_CODE),
        "dsh_provider": "anthropic",
        "context_window": 1000000,
        "max_tokens": 128000,
    },
    "qwen38-27b": {
        "label": "Qwen3.8-27B",
        "id": "qwen3.8-27b",
        "family": "qwen",
        "default_harness": AGENT_HARNESS_DSH,
        "harnesses": (
            AGENT_HARNESS_DSH,
            AGENT_HARNESS_GPT_CODEX,
            AGENT_HARNESS_CLAUDE_CODE,
        ),
        "dsh_provider": "qwen",
        "context_window": 150000,
        "max_tokens": 32768,
    },
    "dots3-note-preview": {
        "label": "Dots3 Note Preview",
        "id": "dots3-note-prev",
        "family": "dots",
        "default_harness": AGENT_HARNESS_DSH,
        "harnesses": (
            AGENT_HARNESS_DSH,
            AGENT_HARNESS_GPT_CODEX,
            AGENT_HARNESS_CLAUDE_CODE,
        ),
        "dsh_provider": "dots",
        "context_window": 393000,
        "max_tokens": 65536,
    },
}

#: The shared model catalog.  Its canonical copy lives in
#: ``repos/common/model_catalog.json`` and every checkout carries an identical
#: copy beside this module; only the two capacity numbers are read from it.
#: Labels, aliases and the harness matrix stay in ``AGENT_MODELS``.
_MODEL_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")


def _apply_model_catalog(models: Dict[str, Dict[str, Any]]) -> None:
    """Overwrite each model's capacity from the shared catalog."""
    try:
        with _MODEL_CATALOG_PATH.open(encoding="utf-8") as handle:
            document = json.load(handle)
        by_id = {entry["id"]: entry for entry in document["models"]}
        for name, entry in models.items():
            facts = by_id.get(entry["id"])
            if facts is None:
                raise ConfigError(
                    "model %r (%s) is missing from %s"
                    % (name, entry["id"], _MODEL_CATALOG_PATH)
                )
            context_window = int(facts["context_window"])
            max_tokens = int(facts["max_tokens"])
            if max_tokens <= 0 or context_window <= max_tokens:
                raise ConfigError(
                    "model %r: max_tokens %d must be positive and below "
                    "context_window %d in %s"
                    % (name, max_tokens, context_window, _MODEL_CATALOG_PATH)
                )
            entry["context_window"] = context_window
            entry["max_tokens"] = max_tokens
            default_level = facts.get("default_reasoning_level")
            if default_level:
                entry["default_reasoning_level"] = str(default_level)
    except FileNotFoundError as exc:
        # A checkout refreshed from upstream may not carry the catalog yet (the
        # adapter fallbacks below already document that case). Keep the table's
        # documented default capacities instead of failing the import, so setup
        # can still report a genuinely missing asset rather than dying in a
        # raw traceback before its own manifest check runs.
        if MISSING_CATALOG_IS_FATAL:
            raise ConfigError(
                "cannot read the shared model catalog %s: %s (copy it beside "
                "model_runtime.py)" % (_MODEL_CATALOG_PATH, exc)
            ) from exc
        return
    except OSError as exc:
        raise ConfigError(
            "cannot read the shared model catalog %s: %s (copy it beside "
            "model_runtime.py)" % (_MODEL_CATALOG_PATH, exc)
        ) from exc
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        raise ConfigError(
            "the shared model catalog %s is malformed: %s"
            % (_MODEL_CATALOG_PATH, exc)
        ) from exc


#: ``context_window``/``max_tokens`` in the table above are placeholders: the
#: catalog overlay replaces them at import, so edit the catalog, not those
#: numbers.
_apply_model_catalog(AGENT_MODELS)


def reload_model_catalog() -> None:
    """Re-read and apply the catalog with the current policy flags.

    Import-time validation uses the shared defaults.  A checkout whose policy
    differs (GDP and paper-refine-eval fail closed on a missing catalog) sets
    the flag and calls this immediately, so the policy is not silently lost.
    """
    _apply_model_catalog(AGENT_MODELS)


AGENT_MODEL_CHOICES: Tuple[str, ...] = tuple(AGENT_MODELS) + (MODEL_DEFAULT,)

#: Capacity numbers resolved from the shared catalog (authoritative).
QWEN_CONTEXT_WINDOW = int(AGENT_MODELS["qwen38-27b"]["context_window"])
QWEN_MAX_TOKENS = int(AGENT_MODELS["qwen38-27b"]["max_tokens"])
DOTS_CONTEXT_WINDOW = int(AGENT_MODELS["dots3-note-preview"]["context_window"])
DOTS_MAX_TOKENS = int(AGENT_MODELS["dots3-note-preview"]["max_tokens"])
DEEPSEEK_CONTEXT_WINDOW = int(AGENT_MODELS["deepseek-flash"]["context_window"])
DEEPSEEK_MAX_TOKENS = int(AGENT_MODELS["deepseek-flash"]["max_tokens"])

#: Per-request retry and compaction-key policy.  The canonical file lives in
#: ``repos/agent-common/agent_common/llm_policy.json``; the submodule copy beside this module
#: is checked out by ``repos/agent-common/sync_llm_policy.sh``.  A malformed copy
#: falls back to the
#: documented defaults, so a run never fails over a tuning file.
_LLM_POLICY_PATH = Path(__file__).with_name("llm_policy.json")
LLM_POLICY = _load_llm_policy(_LLM_POLICY_PATH)
RETRY_SCHEDULE = RetrySchedule(LLM_POLICY)


#: Accepted spellings of the model names above.  The operator-facing names use
#: hyphens, but "gpt sol", "qwen3.8-27b" and the provider ids are ordinary
#: ways to spell the same choice.  This is the one table: ``agent_pair`` and
#: every sibling checkout derive their choice parser from it, so a provider id
#: accepted here cannot be rejected there.
_MODEL_ALIASES = {
    "gpt-6.1-sol": "gpt-sol",
    "gpt6-1-sol": "gpt-sol",
    "gpt-sol-latest": "gpt-sol",
    "gpt-6.1-terra": "gpt-terra",
    "gpt6-1-terra": "gpt-terra",
    "gpt-terra-latest": "gpt-terra",
    "claude-opus-5-5": "claude-opus",
    "claude-sonnet-5-5": "claude-sonnet",
    "opus": "claude-opus",
    "sonnet": "claude-sonnet",
    "claude-opus-latest": "claude-opus",
    "claude-sonnet-latest": "claude-sonnet",
    "deepseek": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "qwen": "qwen38-27b",
    "qwen3.8-27b": "qwen38-27b",
    "qwen-3.8-27b": "qwen38-27b",
    "qwen3-8-27b": "qwen38-27b",
    "dots": "dots3-note-preview",
    "dots3": "dots3-note-preview",
    "dots3-note-prev": "dots3-note-preview",
    "dots-3-note-prev": "dots3-note-preview",
    "dots3-note-preview": "dots3-note-preview",
}

#: The harness shortcuts an operator may type for a canonical harness name.
_HARNESS_ALIASES = {
    "deepseek": AGENT_HARNESS_DSH,
    "deepseek-harness": AGENT_HARNESS_DSH,
    "codex": AGENT_HARNESS_GPT_CODEX,
    "claude": AGENT_HARNESS_CLAUDE_CODE,
}

#: Provider qwen serves its OpenAI-compatible route on by default.
QWEN_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
QWEN_DEFAULT_MODEL = "qwen3.8-27b"
QWEN_DEFAULT_API_KEY = "dummy"
#: qwen3.8-27b's default reasoning level is a model fact stored with the
#: shared catalog (``default_reasoning_level``), so every checkout reads the
#: same level.  An operator override (``PAPER_QWEN_REASONING_EFFORT``) still
#: wins, and the deployment also accepts low and xhigh.  The literal is the
#: fallback for a catalog that predates the field.
QWEN_DEFAULT_EFFORT = str(
    AGENT_MODELS["qwen38-27b"].get("default_reasoning_level") or "medium"
)
DOTS_DEFAULT_BASE_URL = "https://note3-prev-api.askdiandian.com/v1"
DOTS_DEFAULT_MODEL = "dots3-note-prev"
DOTS_DEFAULT_EFFORT = "medium"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "recursive/flint-v1.0:free"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_XAI_MODEL = "grok-4.5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

#: Every direct (non-harness) model API the three repos know about.  A
#: ``key_env`` of ``None`` means the endpoint is local and keyless.
API_PROVIDER_DEFAULTS: Dict[str, Dict[str, Optional[str]]] = {
    "qwen": {
        "base_url": QWEN_DEFAULT_BASE_URL,
        "model": QWEN_DEFAULT_MODEL,
        "key_env": None,
        "api_key": QWEN_DEFAULT_API_KEY,
    },
    "dots": {
        "base_url": DOTS_DEFAULT_BASE_URL,
        "model": DOTS_DEFAULT_MODEL,
        "key_env": "DOTS_API_KEY",
        "api_key": None,
    },
    "openrouter": {
        "base_url": DEFAULT_OPENROUTER_BASE_URL,
        "model": DEFAULT_OPENROUTER_MODEL,
        "key_env": OPENROUTER_KEY_ENV,
        "api_key": None,
    },
    "xai": {
        "base_url": DEFAULT_XAI_BASE_URL,
        "model": DEFAULT_XAI_MODEL,
        "key_env": "XAI_API_KEY",
        "api_key": None,
    },
    "openai": {
        "base_url": DEFAULT_OPENAI_BASE_URL,
        "model": DEFAULT_OPENAI_MODEL,
        "key_env": "OPENAI_API_KEY",
        "api_key": None,
    },
}

#: Environment variable the generated provider config names for a custom
#: endpoint's key.  The reference implementation uses the PAPER_ spelling; the
#: GDP_ spelling is accepted as an alias so the pipeline's own variables work
#: too.  The key itself is only ever exported to the child, never written into
#: the provider patch, the run record or the log.
AGENT_MODEL_API_KEY_ENV = "PAPER_AGENT_MODEL_API_KEY"
AGENT_MODEL_API_KEY_ENV_ALIASES: Tuple[str, ...] = (
    "PAPER_AGENT_MODEL_API_KEY",
    "GDP_AGENT_MODEL_API_KEY",
)
AGENT_MODEL_NAME_ENV = "PAPER_AGENT_MODEL_NAME"
AGENT_MODEL_NAME_ENV_ALIASES: Tuple[str, ...] = (
    "PAPER_AGENT_MODEL_NAME",
    "GDP_AGENT_MODEL_NAME",
)
AGENT_MODEL_PREP_JSON_ENV = "PAPER_AGENT_MODEL_PREP_JSON"
AGENT_MODEL_PREP_JSON_ENV_ALIASES: Tuple[str, ...] = (
    "PAPER_AGENT_MODEL_PREP_JSON",
    "GDP_AGENT_MODEL_PREP_JSON",
)
AGENT_HARNESS_ENV_ALIASES: Tuple[str, ...] = (
    "PAPER_AGENT_HARNESS",
    "GDP_AGENT_HARNESS",
)
AGENT_MODEL_ENV_ALIASES: Tuple[str, ...] = (
    "PAPER_AGENT_MODEL",
    "GDP_AGENT_MODEL",
)

DEFAULT_ENDPOINT_CONTEXT = 32768
DEFAULT_ENDPOINT_MAX_TOKENS = 8192


def _first_env(names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def agent_choice(value: Any) -> str:
    """Normalize an operator spelling for a harness or model name.

    ``gpt codex``, ``gpt-codex`` and ``GPT_CODEX`` all become ``gpt-codex``.
    """
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def agent_model_is_endpoint(value: Any) -> bool:
    """True when ``--agent-model`` is an arbitrary chat-completions endpoint URL."""
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def normalize_agent_model(value: Any) -> str:
    """Canonical model name, or the URL unchanged when it is a custom endpoint."""
    raw = str(value).strip()
    if agent_model_is_endpoint(raw):
        return raw
    slug = agent_choice(raw)
    if slug == MODEL_DEFAULT:
        return MODEL_DEFAULT
    return _MODEL_ALIASES.get(slug, slug)


def agent_model_choice(value: Any) -> str:
    """``--agent-model`` accepts a known model name or an http(s) endpoint URL."""
    raw = str(value).strip()
    if agent_model_is_endpoint(raw):
        try:
            endpoint_base(raw)      # validate the URL shape
        except ConfigError as exc:
            raise ValueError(str(exc)) from None
        credential = endpoint_credentials(raw)
        if credential and REJECT_ENDPOINT_CREDENTIALS:
            raise ValueError(
                "--agent-model endpoint %r carries a credential in its %s; "
                "pass the key with --agent-model-api-key (or "
                "PAPER_AGENT_MODEL_API_KEY) instead"
                % (redact_endpoint_url(raw), credential)
            )
        return raw
    slug = normalize_agent_model(raw)
    if slug == MODEL_DEFAULT:
        return MODEL_DEFAULT
    if slug not in AGENT_MODELS:
        raise ValueError(
            "unknown model %r; use one of %s, or an http(s) chat-completions "
            "endpoint URL" % (raw, ", ".join(AGENT_MODEL_CHOICES))
        )
    return slug


def agent_harness_choice(value: Any) -> str:
    name = agent_choice(value)
    name = _HARNESS_ALIASES.get(name, name)
    if name not in AGENT_HARNESSES:
        raise ValueError(
            "unknown harness %r; use one of %s"
            % (value, ", ".join(AGENT_HARNESS_CHOICES))
        )
    return name


def agent_model_label(model: Optional[str]) -> str:
    """A short label for a model choice (an endpoint URL is not repeated)."""
    if not model:
        return "config-owned"
    if model == MODEL_DEFAULT:
        return "harness-default"
    return "custom-endpoint" if agent_model_is_endpoint(model) else str(model)


def agent_pair_supported(harness: str, model: Optional[str]) -> bool:
    """Can ``harness`` reach ``model`` over a protocol it speaks?"""
    if model == MODEL_DEFAULT:
        return True
    if model is None:
        # A legacy harness with no model choice leaves the model to the CLI's
        # own configuration; every harness can run that way.
        return True
    if agent_model_is_endpoint(model):
        # An arbitrary endpoint is the OpenAI Chat Completions protocol; dsh's
        # llm-pi-ai route is the harness that speaks it directly.  Codex can
        # only use a Responses custom provider, and Claude Code only Anthropic
        # Messages, so neither can be guessed from a chat-completions URL.
        return harness == AGENT_HARNESS_DSH
    info = AGENT_MODELS.get(model) or {}
    return harness in (info.get("harnesses") or ())


def agent_model_for_harness(harness: str) -> str:
    """The model a harness uses when only the harness was chosen."""
    return (AGENT_HARNESSES.get(harness) or {}).get(
        "default_model", DEFAULT_AGENT_MODEL
    )


def agent_harness_for_model(model: str) -> str:
    """The harness a model uses when only the model was chosen."""
    if model == MODEL_DEFAULT:
        return DEFAULT_AGENT_HARNESS
    if agent_model_is_endpoint(model):
        return AGENT_HARNESS_DSH
    return (AGENT_MODELS.get(model) or {}).get(
        "default_harness", DEFAULT_AGENT_HARNESS
    )


def agent_selection_label(harness: Optional[str], model: Optional[str]) -> str:
    if harness and model:
        return "%s/%s" % (harness, agent_model_label(model))
    return harness or (agent_model_label(model) if model else "config-owned")


def pair_from_options(
    harness: Optional[str],
    model: Optional[str],
    fallback_harness: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the two independent options, filling one from the other/default.

    A missing side is not guessed across the wire-protocol boundary: the
    model's default harness (GPT -> gpt-codex, Claude -> claude-code,
    DeepSeek/Qwen/Dots/URL -> dsh) or the harness's default model (dsh ->
    deepseek-flash, gpt-codex -> gpt-sol, claude-code -> claude-sonnet) wins.
    """
    if not harness and not model:
        resolved_harness = fallback_harness or DEFAULT_AGENT_HARNESS
        resolved_model = fallback_model or agent_model_for_harness(resolved_harness)
        return resolved_harness, resolved_model
    if harness and not model:
        return harness, agent_model_for_harness(harness)
    if model and not harness:
        return agent_harness_for_model(model), model
    return harness, model


# ---------------------------------------------------------------------------
# The resolved selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentSelection:
    """One resolved (harness, model) choice plus its request-side settings."""

    harness: str
    model: Optional[str] = None
    legacy: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    prep_json: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_endpoint(self) -> bool:
        return agent_model_is_endpoint(self.model)

    @property
    def is_default(self) -> bool:
        return self.model == MODEL_DEFAULT

    @property
    def is_legacy(self) -> bool:
        return bool(self.legacy)

    @property
    def label(self) -> str:
        return agent_selection_label(self.harness, self.model)

    def model_info(self) -> Dict[str, Any]:
        if self.model is None or self.is_endpoint or self.is_default:
            return {}
        return dict(AGENT_MODELS.get(self.model) or {})

    def effective_model_id(self, raw_model: Optional[str] = None) -> Optional[str]:
        """The provider model id: the raw ``--model`` wins, then the model name.

        ``--model`` remains the low-level override it always was.  For a custom
        endpoint it names the provider model when ``--agent-model-name`` did
        not; for a native model it replaces the catalog id.
        """
        if raw_model:
            return str(raw_model)
        if self.is_endpoint:
            return self.model_name or endpoint_query_model(str(self.model or "")) or None
        if self.model is None or self.is_default:
            return None
        return str((AGENT_MODELS.get(self.model) or {}).get("id") or self.model)

    def to_record(self, raw_model: Optional[str] = None) -> Dict[str, Any]:
        """Secret-free shape for ``command.json`` / ``status.json``."""
        record_model = (
            redact_endpoint_url(str(self.model))
            if self.is_endpoint and self.model is not None
            else self.model
        )
        return {
            "harness": self.harness,
            "model": record_model,
            "id": self.effective_model_id(raw_model),
            "legacy": self.legacy,
            # A caller may put a credential in the endpoint's query string. This record is
            # written to command.json and status.json, which every agent can read, so the
            # persisted endpoint keeps only the scheme/host/path. The live AgentSelection
            # still carries the raw URL for the actual request.
            "endpoint": str(record_model)
            if self.is_endpoint and record_model is not None
            else None,
            "model_name": self.model_name if self.is_endpoint else None,
            "api_key_set": bool(
                self.api_key
                or (
                    self.is_endpoint
                    and _first_env(AGENT_MODEL_API_KEY_ENV_ALIASES)
                )
            ),
            "prep_keys": sorted(str(key) for key in self.prep_json),
        }


def resolve_agent_selection(
    *,
    legacy_agent: Optional[str] = None,
    harness: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    prep_json: Optional[Mapping[str, Any]] = None,
) -> AgentSelection:
    """Validate and resolve one producer/judge selection.

    Raises :class:`gdp.errors.ConfigError` for a missing endpoint, an unknown
    choice, an unsupported wire-protocol pair or request options that do not
    belong to a named model.
    """
    try:
        prep = dict(prep_json or {})
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "the model request preparation must be a JSON object: %s" % exc
        ) from None
    if legacy_agent:
        if legacy_agent not in LEGACY_AGENT_PAIRS:
            raise ConfigError(
                "unknown legacy agent %r (expected one of %s)"
                % (legacy_agent, ", ".join(LEGACY_AGENT_PAIRS))
            )
        legacy_harness, legacy_model = LEGACY_AGENT_PAIRS[legacy_agent]
        if legacy_harness is None:
            return AgentSelection(harness="", model=None, legacy=legacy_agent)
        if harness or model or api_key or model_name or prep:
            raise ConfigError(
                "--agent is the legacy shortcut and cannot be combined with "
                "--agent-harness/--agent-model/--agent-model-api-key/"
                "--agent-model-name/--agent-model-prep-json; use one form or the other"
            )
        return AgentSelection(
            harness=legacy_harness,
            model=legacy_model,
            legacy=legacy_agent,
        )

    try:
        if harness is not None:
            harness = agent_harness_choice(harness)
        if model is not None:
            model = agent_model_choice(model)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    resolved_harness, resolved_model = pair_from_options(harness, model)

    if (api_key or model_name or prep) and not agent_model_is_endpoint(resolved_model):
        raise ConfigError(
            "--agent-model-api-key / --agent-model-name / --agent-model-prep-json "
            "apply only to a custom endpoint URL passed as --agent-model, not to "
            "the named model %r" % resolved_model
        )
    if agent_model_is_endpoint(resolved_model):
        if resolved_harness != AGENT_HARNESS_DSH:
            raise ConfigError(
                "--agent-harness %r cannot drive a custom API endpoint "
                "(--agent-model %r): a chat-completions URL is the OpenAI Chat "
                "Completions protocol, which the dsh harness speaks directly. "
                "Use --agent-harness dsh (the default) for a custom endpoint."
                % (resolved_harness, resolved_model)
            )
        endpoint_base(str(resolved_model))
        credential = endpoint_credentials(str(resolved_model))
        if credential and REJECT_ENDPOINT_CREDENTIALS:
            raise ConfigError(
                "--agent-model endpoint %r carries a credential in its %s. The "
                "generated provider patch and the run record must stay "
                "credential-free; pass the key with --agent-model-api-key (or "
                "PAPER_AGENT_MODEL_API_KEY) instead."
                % (redact_endpoint_url(str(resolved_model)), credential)
            )
    elif not agent_pair_supported(resolved_harness, resolved_model):
        allowed = ", ".join(
            (AGENT_MODELS.get(resolved_model) or {}).get("harnesses") or ()
        ) or "none"
        raise ConfigError(
            "--agent-harness %r cannot drive --agent-model %r (supported "
            "harnesses: %s). The two axes are independent only where the wire "
            "protocol supports it; choose a supported pair."
            % (resolved_harness, resolved_model, allowed)
        )
    return AgentSelection(
        harness=resolved_harness,
        model=resolved_model,
        api_key=(str(api_key).strip() if isinstance(api_key, str) and api_key.strip() else None),
        model_name=(str(model_name).strip() if isinstance(model_name, str)
                    and model_name.strip() else None),
        prep_json=prep,
    )


# ---------------------------------------------------------------------------
# Endpoint URL handling
# ---------------------------------------------------------------------------
def redact_endpoint_url(url: str) -> str:
    """The endpoint URL safe to persist: no userinfo, query or fragment.

    ``--agent-model`` accepts an arbitrary OpenAI-compatible URL, and a caller may put an API
    key in its query string. Persisting that value verbatim contradicted ``to_record``'s
    "secret-free shape" contract and the CLI's promise that endpoint credentials are not
    written into the run record. The live selection keeps the raw URL; only the recorded copy
    is stripped.
    """
    text = str(url or "").strip()
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return text
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def split_endpoint_credentials(url: str) -> Tuple[str, str]:
    """Split a credential-bearing endpoint into (safe URL, secret).

    ``--agent-model`` accepts a URL whose userinfo or query string carries a
    gateway credential.  A child process's argv is world-readable, so the
    credential must not stay there.  The returned URL keeps the scheme, host,
    port, path, fragment and every non-secret query parameter (notably
    ``model=``); the first userinfo/query credential is returned for the
    documented ``PAPER_AGENT_MODEL_API_KEY`` channel.  A URL with no
    credential is returned unchanged with an empty secret.
    """
    text = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return text, ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return text, ""
    secret = ""
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        if ":" in userinfo:
            secret = userinfo.rsplit(":", 1)[1]
        else:
            secret = userinfo
        netloc = host
    kept: list[Tuple[str, str]] = []
    if parsed.query:
        try:
            pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        except ValueError:
            pairs = []
        for key, value in pairs:
            if _secret_query_key(key):
                if not secret:
                    secret = value
                continue
            kept.append((key, value))
    query = urllib.parse.urlencode(kept) if kept else ""
    return (
        urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, query, parsed.fragment)
        ),
        secret,
    )


def _secret_query_key(name: str) -> bool:
    """Whether a query parameter name looks like a credential."""
    upper = str(name).upper().replace("-", "_")
    return any(
        upper == marker or upper.endswith("_" + marker)
        for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    )


def endpoint_base(url: str) -> str:
    """The provider base URL for a chat-completions endpoint.

    ``https://host/v1/chat/completions`` -> ``https://host/v1``; DSH appends
    ``/chat/completions`` itself.  Query/fragment are dropped, but a ``model=``
    query parameter is preserved as the model name by :func:`endpoint_query_model`.

    URL userinfo is stripped: the base URL is written verbatim into every stage's
    ``dsh_model_patch.yml``, so keeping ``user:password@`` there persisted a credential in an
    agent-readable artifact.  Endpoint credentials belong in ``--agent-model-api-key`` (or the
    provider environment variable), which the generated patch references by name.
    """
    text = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
        host = parsed.hostname
        parsed.port          # a non-numeric or out-of-range port is malformed too
    except ValueError:
        raise ConfigError(
            "--agent-model endpoint %r is not an http(s) URL with a host" % text
        ) from None
    if parsed.scheme not in ("http", "https") or not host:
        raise ConfigError(
            "--agent-model endpoint %r is not an http(s) URL with a host" % text
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(
            "--agent-model endpoint %r has an invalid port: %s" % (text, exc)
        ) from None
    if port is not None and not 0 < port <= 65535:
        raise ConfigError(
            "--agent-model endpoint %r has an invalid port: %d" % (text, port)
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))


#: Query keys that carry a credential.  The endpoint's key belongs in
#: ``--agent-model-api-key`` / the key environment variable, not in the URL:
#: the URL is written to the generated patch and to the run's own records.
_CREDENTIAL_QUERY_KEYS = {
    "api_key",
    "apikey",
    "api-key",
    "key",
    "token",
    "access_token",
    "auth",
    "authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
}


def endpoint_credentials(url: str) -> str:
    """A short description of the credential a URL carries, or "".

    Userinfo (``user:secret@host``) and a credential-named query parameter are
    both rejected before a run: neither can be honoured by the generated DSH
    patch (which carries no query string) and both would be written to the run
    record.  A non-secret query parameter such as ``?model=`` stays allowed.
    """
    try:
        parsed = urllib.parse.urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return "userinfo"
    try:
        pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    except ValueError:
        return ""
    for key, _value in pairs:
        lowered = str(key).strip().lower()
        if lowered in _CREDENTIAL_QUERY_KEYS or lowered.endswith(
                ("_key", "_token", "_secret", "_password")
        ):
            return "query parameter %r" % key
    return ""


def endpoint_query_model(url: str) -> str:
    try:
        values = urllib.parse.parse_qs(urllib.parse.urlsplit(str(url)).query)
    except ValueError:
        return ""
    return str((values.get("model") or [""])[0]).strip()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects for the key-bearing ``/models`` probe.

    urllib's default redirect handler reuses the original request headers, including the
    ``Authorization`` header built from ``--agent-model-api-key``.  A redirected probe would
    therefore hand the key to whatever host the endpoint names.  Returning ``None`` makes
    urllib raise ``HTTPError`` (caught below) instead of following the 3xx.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _endpoint_catalog(base_url: str, api_key: str = "", timeout: float = 5.0) -> List[Dict[str, Any]]:
    """The endpoint's ``/models`` entries (an empty list when it exposes none)."""
    if not base_url:
        return []
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    request = urllib.request.Request(
        "%s/models" % base_url.rstrip("/"), headers=headers, method="GET"
    )
    try:
        opener = urllib.request.build_opener(_NoRedirectHandler)
        with opener.open(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        TypeError,
        http.client.HTTPException,
    ):
        # BadStatusLine/IncompleteRead are HTTPException, not OSError; a malformed upstream
        # response must yield an empty catalog, not escape as an internal error.
        return []
    data = body.get("data") if isinstance(body, dict) else None
    return [item for item in (data or []) if isinstance(item, dict) and item.get("id")]


def _endpoint_context(entries: Sequence[Mapping[str, Any]], model: str) -> Optional[int]:
    for entry in entries:
        if str(entry.get("id")) != str(model):
            continue
        for key in (
            "context_length",
            "max_model_len",
            "context_window",
            "max_context_tokens",
            "max_input_tokens",
        ):
            try:
                value = int(entry.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return None


def resolve_endpoint_model(
    selection: AgentSelection, *, probe: bool = True, api_key: Optional[str] = None
) -> Tuple[str, Optional[int], str]:
    """``(model_name, context_window, error)`` for a custom endpoint.

    ``--agent-model-name`` wins; a ``?model=`` query parameter is next; then a
    single model listed by ``/v1/models``; otherwise ``default`` (a keyless
    local proxy commonly serves only one model but exposes no listing).
    """
    if selection.model_name:
        if not probe:
            return selection.model_name, None, ""
        key = api_key or selection.api_key or _first_env(AGENT_MODEL_API_KEY_ENV_ALIASES)
        entries = _endpoint_catalog(endpoint_base(str(selection.model or "")), key)
        return selection.model_name, _endpoint_context(entries, selection.model_name), ""
    query_model = endpoint_query_model(str(selection.model or ""))
    if query_model:
        return query_model, None, ""
    if not probe:
        return "default", None, ""
    key = api_key or selection.api_key or _first_env(AGENT_MODEL_API_KEY_ENV_ALIASES)
    entries = _endpoint_catalog(endpoint_base(str(selection.model or "")), key)
    models = [str(entry.get("id")) for entry in entries]
    if len(models) == 1:
        return models[0], _endpoint_context(entries, models[0]), ""
    if len(models) > 1:
        return "", None, (
            "the endpoint lists %d models (%s%s); pass --agent-model-name to "
            "choose one" % (
                len(models),
                ", ".join(models[:8]),
                "..." if len(models) > 8 else "",
            )
        )
    return "default", None, ""


# ---------------------------------------------------------------------------
# Request preparation (--agent-model-prep-json)
# ---------------------------------------------------------------------------
def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` in place and return ``base``.

    A missing key in ``override`` leaves the default untouched; a nested object
    is merged key by key; any other value (including a list or a scalar)
    replaces the default.  This is the ``--agent-model-prep-json`` contract: an
    incomplete object is valid because it only overrides the keys it carries.
    """
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_prep_json(value: Any) -> Dict[str, Any]:
    """A JSON object for ``--agent-model-prep-json`` (an empty value is ``{}``)."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (ValueError, RecursionError) as exc:
        raise ConfigError("--agent-model-prep-json is not valid JSON: %s" % exc) from None
    if not isinstance(parsed, dict):
        raise ConfigError("--agent-model-prep-json must be a JSON object")
    return parsed


class _PrepProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward DSH's OpenAI-compatible requests, merging the prep object."""

    protocol_version = "HTTP/1.0"

    def log_message(self, *args: Any) -> None:        # keep the adapter's stderr readable
        pass

    def _target(self) -> str:
        base = str(self.server.target_base).rstrip("/")  # type: ignore[attr-defined]
        path = self.path or "/"
        if path == "/v1" or path.startswith("/v1/"):
            path = path[3:]
        return base + path

    def _forward(self, method: str, body: Optional[bytes] = None) -> None:
        headers = {}
        for key, value in self.headers.items():
            if key.lower() in (
                "host",
                "content-length",
                "connection",
                "transfer-encoding",
                "accept-encoding",
            ):
                continue
            headers[key] = value
        if body is not None:
            headers["Content-Length"] = str(len(body))
        request = urllib.request.Request(
            self._target(), data=body, headers=headers, method=method
        )
        try:
            response = urllib.request.urlopen(request, timeout=600.0)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            self.send_response(exc.code)
            ctype = exc.headers.get("Content-Type") if exc.headers else None
            self.send_header("Content-Type", ctype or "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(raw)
            return
        except (urllib.error.URLError, OSError, TimeoutError, http.client.HTTPException) as exc:
            # ``BadStatusLine``/``IncompleteRead`` are ``http.client.HTTPException``, not
            # ``OSError``; without them a malformed upstream response escaped ``do_POST`` and
            # the harness saw a closed connection instead of this structured 502 body.
            raw = json.dumps(
                {"error": {"message": "custom endpoint proxy failed: %s" % exc}}
            ).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(raw)
            return
        self.send_response(getattr(response, "status", 200))
        for key, value in response.headers.items():
            if key.lower() in (
                "transfer-encoding",
                "connection",
                "content-length",
                "content-encoding",
            ):
                continue
            self.send_header(key, value)
        self.end_headers()
        try:
            read_chunk = getattr(response, "read1", response.read)
            while True:
                chunk = read_chunk(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ValueError, http.client.HTTPException):
            # The response headers are already on the wire, so a structured 502 cannot replace
            # them; close the truncated response instead of letting the handler thread die.
            pass
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def do_GET(self) -> None:
        self._forward("GET")

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        body = self.rfile.read(length) if length else b""
        prep = getattr(self.server, "prep", None)  # type: ignore[attr-defined]
        if prep and body:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                deep_merge(payload, prep)
                body = json.dumps(payload).encode("utf-8")
        self._forward("POST", body)


#: Proxies started by this process.  ``stop_prep_proxy`` accepts one server so
#: concurrent verifier passes can stop only their own proxy.
_ACTIVE_PROXIES: set = set()
_PROXY_BY_BASE: Dict[str, Any] = {}
_PROXY_LOCK = threading.Lock()


def start_prep_proxy(target_base: str, prep: Mapping[str, Any]) -> Tuple[str, str]:
    """``(base_url, error)`` for a local proxy that merges ``prep`` into POSTs.

    The returned base URL replaces the endpoint in the generated DSH patch; the
    proxy is otherwise transparent.  Only created when a prep object is present.
    """
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PrepProxyHandler)
    except OSError as exc:
        return "", "%s: %s" % (type(exc).__name__, exc)
    server.target_base = target_base.rstrip("/")          # type: ignore[attr-defined]
    server.prep = dict(prep)                              # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._gdp_thread = thread                           # type: ignore[attr-defined]
    base = "http://127.0.0.1:%s/v1" % server.server_address[1]
    with _PROXY_LOCK:
        _ACTIVE_PROXIES.add(server)
        _PROXY_BY_BASE[base] = server
    return base, ""


def stop_prep_proxy(server: Any = None) -> None:
    """Stop one proxy, or every proxy this process started when omitted.

    ``server`` may be the base URL returned by :func:`start_prep_proxy`; parallel
    verifier passes each stop only their own proxy.
    """
    with _PROXY_LOCK:
        if isinstance(server, str):
            server = _PROXY_BY_BASE.pop(server, None)
        targets = [server] if server is not None else list(_ACTIVE_PROXIES)
        for item in targets:
            _ACTIVE_PROXIES.discard(item)
            for base, known in list(_PROXY_BY_BASE.items()):
                if known is item:
                    _PROXY_BY_BASE.pop(base, None)
    for item in targets:
        if item is None:
            continue
        with contextlib.suppress(OSError):
            item.shutdown()
        with contextlib.suppress(OSError):
            item.server_close()


#: The DSH compaction summarizer calls the LLM adapter directly, outside
#: ``dsh-llm-retry``; the shared proxy gives that one call site the same
#: 10s-to-3600s per-call backoff.
def compaction_proxy_target(
    selection: Optional[AgentSelection], *, base_url_override: Optional[str] = None
) -> str:
    """The upstream base URL whose compaction calls need the retrying proxy.

    ``base_url_override`` is the caller's resolved provider base (an adapter's
    ``--base-url`` override); the compaction route must share the regular
    provider's upstream rather than fall back to the environment-derived default
    while the regular provider uses the override.
    """
    if selection is None or selection.model is None:
        return ""
    try:
        settings = _provider_family_settings(selection)
    except (ConfigError, KeyError, TypeError, ValueError):
        return ""
    if settings.get("family") != "dots":
        return ""
    return str(base_url_override or settings.get("base_url") or "")


def start_compaction_proxy(
    target_base: str, *, log_callback: Optional[Any] = None
) -> Tuple[Optional[Any], str]:
    """Start the retrying proxy for the dots compaction route."""
    return start_retry_proxy(
        target_base,
        RETRY_SCHEDULE,
        name="gdp-compaction",
        log=log_callback,
    )


def stop_compaction_proxy(proxy: Any = None) -> None:
    """Stop one compaction proxy, or every one this process started."""
    stop_retry_proxy(proxy)


# ---------------------------------------------------------------------------
# Model-family configuration
# ---------------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except (TypeError, ValueError):
        return default


def _normalized_family(selection: AgentSelection) -> str:
    if selection.is_endpoint:
        return "endpoint"
    return str((AGENT_MODELS.get(selection.model or "") or {}).get("family") or "")


def _provider_family_settings(selection: AgentSelection) -> Dict[str, Any]:
    """The endpoint, key and limits a provider family needs at request time."""
    if selection.is_default:
        return {
            "family": "default",
            "base_url": "",
            "api_key": None,
            "api_key_env": None,
            "model": None,
            "context_window": None,
            "max_tokens": None,
            "effort": None,
        }
    family = _normalized_family(selection)
    if family == "endpoint":
        base = endpoint_base(str(selection.model or ""))
        return {
            "family": family,
            "base_url": base,
            "api_key": selection.api_key or _first_env(AGENT_MODEL_API_KEY_ENV_ALIASES),
            "api_key_env": AGENT_MODEL_API_KEY_ENV,
            "model": selection.effective_model_id() or "default",
            "context_window": _int_env(
                "PAPER_AGENT_CONTEXT_WINDOW", DEFAULT_ENDPOINT_CONTEXT
            ),
            "max_tokens": _int_env(
                "PAPER_AGENT_MAX_TOKENS", DEFAULT_ENDPOINT_MAX_TOKENS
            ),
            "effort": None,
        }
    if family == "qwen":
        return {
            "family": family,
            "base_url": _env("PAPER_QWEN_BASE_URL", QWEN_DEFAULT_BASE_URL).rstrip("/"),
            "api_key": _env("PAPER_QWEN_API_KEY", QWEN_DEFAULT_API_KEY),
            "api_key_env": "PAPER_QWEN_API_KEY",
            "model": _env("PAPER_QWEN_MODEL", QWEN_DEFAULT_MODEL),
            "context_window": _int_env(
                "PAPER_QWEN_CONTEXT_WINDOW", QWEN_CONTEXT_WINDOW
            ),
            "max_tokens": _int_env("PAPER_QWEN_MAX_TOKENS", QWEN_MAX_TOKENS),
            "effort": _env("PAPER_QWEN_REASONING_EFFORT", QWEN_DEFAULT_EFFORT).lower(),
        }
    if family == "dots":
        compaction = dots_compaction(LLM_POLICY)
        regular_env = compaction.get("regular_api_key_env", "DOTS_API_KEY")
        compaction_key = _env(compaction["compaction_api_key_env"])
        for alias in compaction.get("compaction_api_key_aliases") or ():
            if not compaction_key:
                compaction_key = _env(str(alias))
        return {
            "family": family,
            "base_url": _env("PAPER_DOTS_BASE_URL", DOTS_DEFAULT_BASE_URL).rstrip("/"),
            "api_key": _env("PAPER_DOTS_API_KEY") or _env(regular_env),
            "api_key_env": regular_env,
            "compaction_api_key": compaction_key,
            "compaction_api_key_env": compaction["compaction_api_key_env"],
            "model": _env("PAPER_DOTS_MODEL", DOTS_DEFAULT_MODEL),
            "context_window": _int_env(
                "PAPER_DOTS_CONTEXT_WINDOW", DOTS_CONTEXT_WINDOW
            ),
            "max_tokens": _int_env("PAPER_DOTS_MAX_TOKENS", DOTS_MAX_TOKENS),
            "effort": _env("PAPER_DOTS_REASONING_EFFORT", DOTS_DEFAULT_EFFORT).lower(),
        }
    if family == "deepseek":
        info = selection.model_info()
        return {
            "family": family,
            "base_url": "",
            "api_key": _env("DEEPSEEK_API_KEY") or _env("PAPER_DEEPSEEK_API_KEY"),
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": selection.effective_model_id() or "deepseek-flash",
            # The shared catalog overlays AGENT_MODELS; read the capacity there
            # instead of duplicating it, so a catalog update cannot leave the
            # DeepSeek family on the old window.
            "context_window": info.get("context_window", DEEPSEEK_CONTEXT_WINDOW),
            "max_tokens": info.get("max_tokens", DEEPSEEK_MAX_TOKENS),
            "effort": _env("PAPER_DEEPSEEK_REASONING_EFFORT", "high").lower(),
        }
    info = selection.model_info()
    return {
        "family": family,
        "base_url": "",
        "api_key": None,
        "api_key_env": None,
        "model": selection.effective_model_id() or (selection.model or ""),
        "context_window": info.get("context_window"),
        "max_tokens": info.get("max_tokens"),
        "effort": None,
    }


def selection_env(selection: Optional[AgentSelection]) -> Dict[str, str]:
    """Credential environment needed by the selected provider.

    Values are injected into the child process only; they are never written to
    ``command.json`` or the run log.  dsh uses its own credential store for the
    native DeepSeek route, so only the adapter-backed families are listed.
    """
    if selection is None or selection.model is None:
        return {}
    settings = _provider_family_settings(selection)
    env: Dict[str, str] = {}
    family = settings["family"]
    if family == "endpoint":
        key = settings["api_key"] or "dummy"
        env[AGENT_MODEL_API_KEY_ENV] = key
        env["GDP_AGENT_MODEL_API_KEY"] = key
    elif family == "deepseek":
        if settings["api_key"]:
            env.setdefault("DEEPSEEK_API_KEY", str(settings["api_key"]))
    elif family == "qwen":
        env["PAPER_QWEN_API_KEY"] = str(settings["api_key"] or QWEN_DEFAULT_API_KEY)
    elif family == "dots":
        if settings["api_key"]:
            env[str(settings["api_key_env"])] = str(settings["api_key"])
            env["PAPER_DOTS_API_KEY"] = str(settings["api_key"])
        # The compaction summarizer reads a second credential so its calls do
        # not compete with the agent's calls for one TPM budget.  When no
        # second key is exported, fall back to the agent key so the route
        # still starts (the dual-key isolation is then simply absent).
        compaction_key = settings.get("compaction_api_key") or settings["api_key"]
        if compaction_key:
            env[str(settings.get("compaction_api_key_env") or "DOTS_API_KEY_1474")] = str(
                compaction_key
            )
    return env


def selection_problem(
    selection: Optional[AgentSelection], *, dry_run: bool = False
) -> str:
    """A configuration problem that would fail the first real request ('' = none)."""
    if selection is None or selection.model is None:
        return ""
    settings = _provider_family_settings(selection)
    family = settings["family"]
    if dry_run:
        return ""
    if family == "deepseek" and selection.harness in (
        AGENT_HARNESS_GPT_CODEX,
        AGENT_HARNESS_CLAUDE_CODE,
    ):
        if not settings["api_key"]:
            return (
                "DEEPSEEK_API_KEY is not set; %s/deepseek-flash needs the "
                "DeepSeek Responses/Anthropic route credential. Export "
                "DEEPSEEK_API_KEY (or PAPER_DEEPSEEK_API_KEY)."
                % selection.harness
            )
    if family == "dots" and not settings["api_key"]:
        return (
            "DOTS_API_KEY is not set; dots3-note-preview needs the Dots API "
            "key. Export DOTS_API_KEY (or PAPER_DOTS_API_KEY)."
        )
    return ""


def dsh_patch_text(
    selection: AgentSelection,
    *,
    raw_model: Optional[str] = None,
    base_url_override: Optional[str] = None,
    model_name_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    context_window_override: Optional[int] = None,
    max_tokens_override: Optional[int] = None,
    effort_override: Optional[str] = None,
    compaction_base_url: Optional[str] = None,
    endpoint_api_key_env: Optional[str] = None,
) -> str:
    """The ``dsh --patch`` overlay that selects or configures the provider.

    An empty string means "no patch": the harness profile's own
    ``agent-default-model`` selects the model.  A low-level ``--model`` still has to reach the
    harness with the ``default`` sentinel; only the bare sentinel owns no model at all.  A
    provider-only override has no profile model id available here, and a patch layer replaces
    the whole entry config, so it must not pin the sentinel as the provider's model either.
    """
    if selection.is_default and not raw_model:
        return ""
    settings = _provider_family_settings(selection)
    family = settings["family"]
    if family == "endpoint":
        model_name = (
            raw_model
            or model_name_override
            or selection.model_name
            or endpoint_query_model(str(selection.model or ""))
            or "default"
        )
        # The patch is a credential-free artifact: userinfo and a query string
        # belong to the live request, never to a file an agent can read.
        base_url = redact_endpoint_url(base_url_override or settings["base_url"])
        return (
            (
                "# Generated by generic_debug_pipeline for this stage: --agent-model.\n"
                "- id: llm-pi-ai\n"
                "  config:\n"
                "    providers:\n"
                "      endpoint:\n"
            '        displayName: "Custom API endpoint"\n'
            "        apiKeyEnv: %s\n"
            "        api: openai-completions\n"
            "        baseURL: %s\n"
            "        models:\n"
            "          - id: %s\n"
            "            name: %s\n"
            "            contextWindow: %s\n"
            "            maxTokens: %s\n"
            "            reasoningEfforts: false\n"
            "        compat:\n"
            "          maxTokensField: max_tokens\n"
            "          supportsStore: false\n"
            + retry_policy_yaml(LLM_POLICY, "        ")
            + "- id: agent-default-model\n"
            "  config:\n"
            "    provider: endpoint\n"
            "    model: %s\n"
            )
            % (
                endpoint_api_key_env or AGENT_MODEL_API_KEY_ENV,
                json.dumps(base_url),
                json.dumps(model_name),
                json.dumps(model_name),
                int(
                    context_window_override
                    or settings["context_window"]
                    or DEFAULT_ENDPOINT_CONTEXT
                ),
                int(
                    max_tokens_override
                    or settings["max_tokens"]
                    or DEFAULT_ENDPOINT_MAX_TOKENS
                ),
                json.dumps(model_name),
            )
        )
    if family == "qwen":
        model = raw_model or settings["model"]
        return (
            (
                "# Generated by generic_debug_pipeline for this stage: --agent-model qwen38-27b.\n"
            "- id: llm-pi-ai\n"
            "  config:\n"
            "    providers:\n"
            "      qwen:\n"
            "        displayName: %s\n"
            "        apiKeyEnv: PAPER_QWEN_API_KEY\n"
            "        api: openai-completions\n"
            "        baseURL: %s\n"
            "        models:\n"
            "          - id: %s\n"
            '            name: "Qwen3.8-27B"\n'
            "            contextWindow: %s\n"
            "            maxTokens: %s\n"
            "            reasoningEfforts:\n"
            "              low: low\n"
            "              medium: medium\n"
            "              xhigh: xhigh\n"
            + retry_policy_yaml(LLM_POLICY, "        ")
            + "- id: agent-default-model\n"
            "  config:\n"
            "    provider: qwen\n"
            "    model: %s\n"
            "    reasoningEffort: %s\n"
            )
            % (
                json.dumps("Qwen vLLM (%s)" % model),
                json.dumps(base_url_override or settings["base_url"]),
                json.dumps(model),
                int(
                    context_window_override
                    or settings["context_window"]
                    or QWEN_CONTEXT_WINDOW
                ),
                int(max_tokens_override or settings["max_tokens"] or QWEN_MAX_TOKENS),
                json.dumps(model),
                json.dumps(effort_override or settings["effort"] or QWEN_DEFAULT_EFFORT),
            )
        )
    if family == "dots":
        model = raw_model or settings["model"]
        return dots_dsh_overlay(
            LLM_POLICY,
            model=model,
            base_url=base_url_override or settings["base_url"],
            context_window=int(
                context_window_override
                or settings["context_window"]
                or DOTS_CONTEXT_WINDOW
            ),
            max_tokens=int(
                max_tokens_override or settings["max_tokens"] or DOTS_MAX_TOKENS
            ),
            compaction_base_url=compaction_base_url,
        )
    info = selection.model_info()
    model = raw_model or selection.effective_model_id() or selection.model or ""
    provider = (
        provider_override
        or info.get("dsh_provider")
        or ("openai" if family == "gpt" else "deepseek-official")
    )
    return (
        "# Generated by generic_debug_pipeline for this stage: --agent-model.\n"
        "# A patch layer replaces the whole entry config, so the provider route is repeated.\n"
        "- id: agent-default-model\n"
        "  config:\n"
        "    provider: %s\n"
        "    model: %s\n"
        % (json.dumps(provider), json.dumps(model))
    )


def codex_provider_args(
    selection: AgentSelection, *, raw_model: Optional[str] = None
) -> List[str]:
    """The ``-c`` overrides that select the provider for ``codex exec``."""
    if selection.model is None or selection.is_default:
        return []
    settings = _provider_family_settings(selection)
    family = settings["family"]
    model = raw_model or selection.effective_model_id() or settings["model"]
    if family == "endpoint":
        raise ConfigError(
            "a custom --agent-model endpoint cannot run on gpt-codex; use dsh"
        )
    if family == "gpt":
        # The model choice owns the provider too: a Codex config repointed at
        # another provider must not redirect a GPT model.
        return ["-c", _codex_config("model_provider", "openai")]
    if family == "deepseek":
        values = [
            _codex_config("model_provider", "deepseek"),
            _codex_config("model", model),
            _codex_config(
                "model_providers.deepseek.name", "DeepSeek"
            ),
            _codex_config(
                "model_providers.deepseek.base_url", "https://api.deepseek.com"
            ),
            _codex_config(
                "model_providers.deepseek.env_key", "DEEPSEEK_API_KEY"
            ),
            _codex_config("model_providers.deepseek.wire_api", "responses"),
            _codex_config("web_search", "disabled"),
            _codex_config(
                "model_context_window",
                int(settings["context_window"] or DEEPSEEK_CONTEXT_WINDOW),
            ),
            _codex_config(
                "model_reasoning_effort", settings["effort"] or "high"
            ),
        ]
        return [item for pair in (["-c", value] for value in values) for item in pair]
    if family == "qwen":
        values = [
            _codex_config("model_provider", "qwen"),
            _codex_config("model", model),
            _codex_config(
                "model_providers.qwen.name", "Qwen vLLM (%s)" % model
            ),
            _codex_config(
                "model_providers.qwen.base_url", settings["base_url"]
            ),
            _codex_config("model_providers.qwen.env_key", "PAPER_QWEN_API_KEY"),
            _codex_config("model_providers.qwen.wire_api", "responses"),
            _codex_config("web_search", "disabled"),
            _codex_config(
                "model_context_window",
                int(settings["context_window"] or QWEN_CONTEXT_WINDOW),
            ),
            _codex_config(
                "model_reasoning_effort", settings["effort"] or QWEN_DEFAULT_EFFORT
            ),
        ]
        return [item for pair in (["-c", value] for value in values) for item in pair]
    if family == "dots":
        values = [
            _codex_config("model_provider", "dots"),
            _codex_config("model", model),
            _codex_config("model_providers.dots.name", "Dots API"),
            _codex_config(
                "model_providers.dots.base_url", settings["base_url"]
            ),
            _codex_config("model_providers.dots.env_key", "DOTS_API_KEY"),
            _codex_config("model_providers.dots.wire_api", "responses"),
            _codex_config("web_search", "disabled"),
            _codex_config(
                "model_context_window",
                int(settings["context_window"] or DOTS_CONTEXT_WINDOW),
            ),
            _codex_config(
                "model_reasoning_effort", settings["effort"] or DOTS_DEFAULT_EFFORT
            ),
        ]
        return [item for pair in (["-c", value] for value in values) for item in pair]
    return []


def _codex_config(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "%s=%s" % (name, "true" if value else "false")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "%s=%s" % (name, value)
    return "%s=%s" % (name, json.dumps(str(value)))


def claude_model_arg(
    selection: AgentSelection, *, raw_model: Optional[str] = None
) -> Optional[str]:
    """The ``--model`` value for Claude Code (``None`` = its configured model)."""
    if selection.model is None or selection.is_default:
        return None
    family = _normalized_family(selection)
    if family == "claude":
        info = selection.model_info()
        return str(raw_model or info.get("alias") or info.get("id") or selection.model)
    # The DeepSeek/Qwen/Dots routes select the model through the Anthropic
    # environment (ANTHROPIC_MODEL); passing the provider id as `--model` would
    # override that environment with an id Claude Code may not recognise.
    return str(raw_model) if raw_model else None


def claude_environment(
    selection: Optional[AgentSelection], *, raw_model: Optional[str] = None
) -> Dict[str, str]:
    """Environment overrides that point Claude Code at the selected provider."""
    if selection is None or selection.model is None or selection.is_default:
        return {}
    settings = _provider_family_settings(selection)
    family = settings["family"]
    if family in ("claude", "gpt", "endpoint"):
        return {}
    root = str(settings["base_url"] or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    key = str(settings["api_key"] or "dummy")
    model = raw_model or str(settings["model"])
    if family == "deepseek":
        primary = raw_model or "deepseek-flash[1m]"
        small = raw_model or "deepseek-flash"
        return {
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            "ANTHROPIC_AUTH_TOKEN": key,
            "ANTHROPIC_API_KEY": key,
            "ANTHROPIC_MODEL": primary,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": primary,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": primary,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": small,
            "CLAUDE_CODE_SUBAGENT_MODEL": small,
            "CLAUDE_CODE_EFFORT_LEVEL": "max",
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "786432",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    return {
        "ANTHROPIC_BASE_URL": root,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_API_KEY": key,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


_PREP_SECRET_KEYS = frozenset(
    {
        "key",
        "api_key",
        "api-key",
        "apikey",
        "access_key",
        "access-token",
        "access_token",
        "token",
        "auth_token",
        "auth-token",
        "api_token",
        "api-token",
        "secret",
        "secret_key",
        "secret-key",
        "private_key",
        "private-key",
        "client_secret",
        "client-secret",
        "credential",
        "credentials",
        "password",
        "authorization",
    }
)
_PREP_SECRET_SUFFIXES = (
    "_api_key",
    "-api-key",
    "_access_key",
    "_access_token",
    "_auth_token",
    "_api_token",
    "_secret_key",
    "-secret-key",
    "_private_key",
    "-private-key",
    "_client_secret",
    "_password",
    "_secret",
)


def effective_agent_model(cmd: Sequence[Any]) -> Dict[str, Any]:
    """A best-effort, secret-free label for the model a command will use.

    This is a diagnostic helper for callers that only have the resolved argv
    (for example a run record or a test); the pipeline itself records the
    selection directly.  It understands the native provider-adapter spelling
    (``--harness/--model`` and ``--endpoint/--model-name``), ``codex exec -m``
    and ``claude --model``.
    """
    argv = [str(item) for item in (cmd or [])]
    exe = os.path.basename(argv[0]).lower() if argv else ""

    def _arg(name: str, default: str = "") -> str:
        try:
            index = argv.index(name)
        except ValueError:
            return default
        return argv[index + 1] if index + 1 < len(argv) else default

    if "--endpoint" in argv:
        return {
            "name": _arg("--model-name") or "default",
            "endpoint": _arg("--endpoint"),
            "harness": _arg("--harness", "dsh"),
            "source": "--agent-model(custom endpoint)/--agent-model-api-key",
        }
    if "--harness" in argv and "--model" in argv:
        model = _arg("--model")
        canonical = normalize_agent_model(model)
        info = AGENT_MODELS.get(canonical) or {}
        return {
            "name": model,
            "id": info.get("id") or model,
            "harness": _arg("--harness"),
            "source": "--agent-model/--agent-harness",
        }
    if "codex" in exe:
        return {
            "name": _arg("-m") or None,
            "harness": AGENT_HARNESS_GPT_CODEX,
            "source": "codex config or -m",
        }
    if "claude" in exe:
        return {
            "name": _arg("--model") or None,
            "harness": AGENT_HARNESS_CLAUDE_CODE,
            "source": "claude config or --model",
        }
    if "dsh" in exe:
        return {
            "name": None,
            "harness": AGENT_HARNESS_DSH,
            "source": "dsh profile",
        }
    return {
        "name": None,
        "harness": None,
        "source": "external CLI config (not recorded; the CLI owns the model)",
    }


def redact_prep_object(value: Any) -> Any:
    """Recursively mask credential-looking *values* in a prep-json object.

    Request defaults such as ``max_tokens`` or ``max_completion_tokens`` are
    ordinary generation parameters, not credentials, so they stay readable.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            text = str(key).lower()
            is_secret = text in _PREP_SECRET_KEYS or text.endswith(_PREP_SECRET_SUFFIXES)
            if is_secret:
                # A secret-named key masks its whole value, whatever its type: a
                # list/dict under `api_key` used to lose the key context when the
                # recursion descended into it and returned the credential unchanged.
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact_prep_object(item)
        return result
    if isinstance(value, list):
        return [redact_prep_object(item) for item in value]
    return value
