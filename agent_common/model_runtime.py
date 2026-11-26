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
import math
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__:
    from .llm_retry import (
        DOTS_WIRE_EFFORT,
        RetrySchedule,
        dots_compaction,
        _PROXY_REGISTRY as _SHARED_PROXY_REGISTRY,
        _forwardable_request_headers,
        _forwardable_response_headers,
        _read_request_body,
        _relay_response_body,
        _request_is_chunked,
        _yaml_scalar,
        dots_dsh_overlay,
        load_policy as _load_llm_policy,
        retry_policy_yaml,
        start_retry_proxy,
        stop_retry_proxy,
    )
else:  # pragma: no cover - a paper-refine frozen root has no package
    from llm_retry import (  # type: ignore
        DOTS_WIRE_EFFORT,
        RetrySchedule,
        dots_compaction,
        _PROXY_REGISTRY as _SHARED_PROXY_REGISTRY,
        _forwardable_request_headers,
        _forwardable_response_headers,
        _read_request_body,
        _relay_response_body,
        _request_is_chunked,
        _yaml_scalar,
        dots_dsh_overlay,
        load_policy as _load_llm_policy,
        retry_policy_yaml,
        start_retry_proxy,
        stop_retry_proxy,
    )


class ConfigError(Exception):
    """A model/harness configuration error.

    The host repository may replace this class at import time (for example GDP
    rebinds it to its own ``errors.ConfigError``) so callers keep their error
    hierarchy.

    Example:
        ``resolve_agent_selection(model="unknown")`` raises this error (or the
        host's rebound subclass).

    Counterexample:
        Do not catch broad ``Exception`` around a resolution call; that also
        hides transport and programming errors the caller must see.
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
        "chat_max_tokens": 384000,
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
        "chat_max_tokens": 128000,
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
        "chat_max_tokens": 128000,
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
        "chat_max_tokens": 128000,
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
        "chat_max_tokens": 128000,
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
        "chat_max_tokens": 80000,
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
        "chat_max_tokens": 160000,
    },
}

#: The shared model catalog.  Its canonical copy lives in
#: ``repos/agent-common/agent_common/model_catalog.json`` and every checkout
#: carries an identical copy beside this module; the capacity numbers and the
#: default reasoning level are read from it.  Labels, aliases and the harness
#: matrix stay in ``AGENT_MODELS``.
_MODEL_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")

#: Facts owned by the shared catalog.  Every reload drops them before applying
#: the new document, so a value the new catalog no longer supplies cannot
#: survive as a stale overlay.
_CATALOG_OWNED_MODEL_KEYS = (
    "context_window",
    "max_tokens",
    "chat_max_tokens",
    "default_reasoning_level",
    "dsh_provider",
    "base_url",
    "api_key_env",
)

#: Credentials split out of a catalog-provided base URL.  The public
#: ``AGENT_MODELS`` table and the exported ``QWEN_*``/``DOTS_*`` constants
#: stay credential-free; the live route reads the secret from here.
_CATALOG_BASE_SECRETS: Dict[str, str] = {}


def _textual_authority(text: str) -> str:
    """The authority-like component of an unparsable URL-shaped string.

    A normal value puts the authority directly after the scheme separator, but
    a malformed form may carry it after one extra slash (``http:/user@host/v1``)
    or after the empty authority of ``http:///user@host/v1``.  The userinfo is
    still the first ``@``-bearing component before the path, so look one
    segment further when the first one has no ``@``.
    """
    head = str(text or "").split("#", 1)[0].split("?", 1)[0]
    rest = head
    if "://" in rest:
        rest = rest.split("://", 1)[1]
    elif rest.startswith("//"):
        rest = rest[2:]
    rest = rest.lstrip("/")
    authority = rest.split("/", 1)[0]
    if "@" not in authority and "/" in rest:
        tail = rest.split("/", 1)[1]
        segment = tail.split("/", 1)[0]
        if "@" in segment:
            return segment
    return authority


def _safe_endpoint_text(text: str) -> str:
    """Strip a URL-shaped string's authority userinfo, query and fragment.

    Used when the value cannot be parsed as an absolute HTTP(S) URL (a malformed
    or scheme-relative ``PAPER_*_BASE_URL``).  The generated patch/argv/
    environment are persisted artifacts, so the credential-bearing part must be
    removed textually rather than returned verbatim.  The textual authority
    finder also covers malformed ``http:///user:pw@host/v1`` and
    ``http:/user:pw@host/v1`` forms that ``urlsplit`` cannot represent.
    """
    safe = str(text or "").split("#", 1)[0].split("?", 1)[0]
    prefix = ""
    rest = safe
    if "://" in rest:
        scheme, _, rest = rest.partition("://")
        prefix = scheme + "://"
        # An extra slash is only part of the malformed credential spelling
        # (``http:///user:pw@host/v1``); ``https:///v1`` with no userinfo must
        # keep its shape.
        stripped = rest.lstrip("/")
        if "@" in stripped.split("/", 1)[0]:
            rest = stripped
    elif rest.startswith("//"):
        prefix = "//"
        rest = rest[2:]
    authority, separator, tail = rest.partition("/")
    if "@" in authority:
        authority = authority.rsplit("@", 1)[-1]
    elif "@" in tail.split("/", 1)[0]:
        segment, next_separator, remainder = tail.partition("/")
        tail = segment.rsplit("@", 1)[-1] + next_separator + remainder
    return prefix + authority + separator + tail


def _query_pairs(query: str) -> list:
    """Parse a query string, also treating ``;`` as a separator.

    ``parse_qsl`` only splits on ``&``, so a legacy ``x=1;api_key=SECRET`` query
    kept the credential inside one value and every credential predicate missed
    it.  A malformed query yields no pairs: it is not safe to forward blindly.
    """
    try:
        return urllib.parse.parse_qsl(
            str(query or "").replace(";", "&"), keep_blank_values=True
        )
    except ValueError:
        return []


def _carries_unparsed_credential(text: str) -> bool:
    """Detect userinfo/credential query textually when ``urlsplit`` fails."""
    without_fragment = str(text or "").split("#", 1)[0]
    _before_query, _, query = without_fragment.partition("?")
    if "@" in _textual_authority(without_fragment):
        return True
    if query:
        if any(
            _credential_key(key) for key, _value in _query_pairs(query)
        ):
            return True
    return False


def _usable_textual_authority(authority: str) -> bool:
    """Whether a scheme-less authority names a host a client could use."""
    try:
        parsed = urllib.parse.urlsplit("//" + authority)
        host = parsed.hostname
        # ``port`` raises for a malformed port; it is not a secret carrier.
        parsed.port
    except ValueError:
        return False
    return bool(host)


def _textual_endpoint_secret(text: str) -> str:
    """The userinfo/query secret of an unparsable endpoint, or ``""``.

    Used by ``split_endpoint_credentials`` for a scheme-less form such as
    ``user:pw@host/v1`` that ``urlsplit`` reads as scheme ``user`` with no
    netloc, and for the malformed ``http:///user:pw@host/v1`` spelling.  A
    value whose authority is not usable (for example the unparsable host
    ``[bad``) keeps the documented empty-secret result: there is no endpoint to
    carry the credential to.  A bare username (``TOKEN@host``) is the secret
    too: the previous split silently dropped it and sent an unauthenticated
    request.
    """
    before_query = str(text or "").split("#", 1)[0]
    _head, _, query = before_query.partition("?")
    authority = _textual_authority(before_query)
    if "@" in authority:
        if not _usable_textual_authority(authority):
            return ""
        userinfo = authority.rsplit("@", 1)[0]
        _user, separator, password = userinfo.partition(":")
        if separator:
            return password
        # A username-only userinfo is the credential (for example a gateway
        # bearer token written as ``https://TOKEN@host/v1``).
        return userinfo
    if query:
        for key, value in _query_pairs(query):
            if _secret_query_key(key):
                return value
    return ""


def _redacted_endpoint_text(value: Optional[str]) -> Optional[str]:
    """Redact a URL-shaped or otherwise credential-bearing value.

    A plain model id is returned unchanged.
    """
    if value is None:
        return None
    text = str(value)
    # A scheme is not required for a credential: a bare ``user:pw@host/v1``
    # carries userinfo just as a URL-shaped value does, and a host-less value
    # can still carry a credential query parameter.
    if (
        agent_model_is_endpoint(text)
        or endpoint_credentials(text)
        or _carries_unparsed_credential(text)
    ):
        return redact_endpoint_url(text)
    return value


def split_endpoint_credentials(url: str) -> Tuple[str, str]:
    """Split a credential-bearing endpoint into (safe URL, secret).

    ``--agent-model`` accepts a URL whose userinfo or query string carries a
    gateway credential.  A child process's argv is world-readable, so the
    credential must not stay there.  The returned URL keeps the scheme, host,
    port, path, fragment and every non-secret query parameter (notably
    ``model=``); the first userinfo/query credential is returned for the
    documented ``PAPER_AGENT_MODEL_API_KEY`` channel.  A URL with no
    credential is returned unchanged with an empty secret.

    Example:
        ``split_endpoint_credentials("https://user:pw@host/v1?model=m")`` returns
        the safe URL plus ``"pw"``.

    Counterexample:
        A URL without a credential returns the URL unchanged and an empty secret;
        an empty secret is not an error.
    """
    text = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        # The URL is malformed (for example ``[bad``), but the caller's artifact
        # must still never receive the credential-bearing part.  Strip it
        # textually, mirroring endpoint_base's fallback.
        return _safe_endpoint_text(text), ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        # A scheme-relative (``//user:pw@host/v1``) or non-http(s) form cannot be
        # used as an endpoint, but it must not leak a credential or query secret
        # into the generated patch/argv/environment either.  A bare
        # ``user:pw@host/v1`` parses with scheme "user" and no netloc, so the
        # old ``endpoint_credentials`` check missed it and returned the raw
        # credential; detect it textually and carry the secret separately.
        secret = _textual_endpoint_secret(text)
        if secret or endpoint_credentials(text) or _carries_unparsed_credential(text):
            return _safe_endpoint_text(text), secret
        return text, ""
    secret = ""
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        # Split at the first colon: a legal password may itself contain a
        # colon, and ``rsplit`` returned only its tail.  A bare username is
        # itself the credential (a gateway bearer token written as
        # ``https://TOKEN@host/v1``); dropping it made the live request
        # unauthenticated while ``endpoint_credentials`` still reported a
        # credential.
        _user, separator, password = userinfo.partition(":")
        secret = password if separator else userinfo
        netloc = host
    kept: list[Tuple[str, str]] = []
    if parsed.query:
        for key, value in _query_pairs(parsed.query):
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


#: Credential-looking names, compared after lowercasing and dropping every
#: non-alphanumeric character.  This is the one predicate shared by the query
#: splitter, the endpoint validator and the prep-object redactor, so a spelling
#: cannot be a credential in one path and an ordinary parameter in another.
_CREDENTIAL_KEY_NAMES = frozenset(
    {
        "key",
        "apikey",
        "apikeys",
        "xapikey",
        "token",
        "tokens",
        "accesstoken",
        "accesstokens",
        "authtoken",
        "authtokens",
        "apitoken",
        "apitokens",
        "secret",
        "secrets",
        "secretkey",
        "secretkeys",
        "privatekey",
        "privatekeys",
        "clientsecret",
        "clientsecrets",
        "password",
        "passwords",
        "passwd",
        "passwds",
        "credential",
        "credentials",
        "authorization",
        "authorizations",
        "auth",
        "bearer",
        "accesskey",
        "accesskeys",
        "accesskeyid",
        "subscriptionkey",
        # Signed-URL / browser-session spellings.  AWS presigned URLs carry
        # ``X-Amz-Signature`` and signed gateway URLs commonly carry
        # ``signature``/``sig``; neither is an ordinary generation parameter.
        "signature",
        "signatures",
        "sig",
        "sigs",
        "cookie",
        "cookies",
        "refreshtoken",
        "refreshtokens",
    }
)
_CREDENTIAL_KEY_SUFFIXES = (
    "_key",
    "_token",
    "_secret",
    "_password",
    "_passwd",
    "_credential",
    "_credentials",
    "_authorization",
    "_auth",
    "_apikey",
    "_signature",
    "_sig",
    "_cookie",
)


def _credential_key(name: Any) -> bool:
    """Whether a query/prep key looks like it carries a credential."""
    text = str(name or "").strip().lower()
    compact = "".join(character for character in text if character.isalnum())
    if compact in _CREDENTIAL_KEY_NAMES:
        return True
    normalized = text.replace("-", "_").replace(" ", "_")
    return normalized.endswith(_CREDENTIAL_KEY_SUFFIXES)


def _secret_query_key(name: str) -> bool:
    """Backward-compatible spelling of :func:`_credential_key` for query names."""
    return _credential_key(name)


#: Query keys that carry a credential are recognised by the shared
#: :func:`_credential_key` predicate: the endpoint's key belongs in
#: ``--agent-model-api-key`` / the key environment variable, not in the URL,
#: because the URL is written to the generated patch and the run's own records.


def endpoint_credentials(url: str) -> str:
    """A short description of the credential a URL carries, or "".

    Userinfo (``user:secret@host``) and a credential-named query parameter are
    both rejected before a run: neither can be carried by the generated DSH
    patch (which has no query string), so the credential must be passed
    separately through ``--agent-model-api-key`` / the key environment.  Run
    records are redacted by ``to_record``; the rejection is about the live
    request, not about persisting the value.  A non-secret query parameter such
    as ``?model=`` stays allowed.

    Example:
        ``endpoint_credentials("https://host/v1?api_key=x")`` returns
        ``"query parameter 'api_key'"``.

    Counterexample:
        ``?model=x`` is not a credential and returns ``""``.
    """
    try:
        parsed = urllib.parse.urlsplit(str(url or "").strip())
    except ValueError:
        return ""
    if parsed.username is not None or parsed.password is not None:
        return "userinfo"
    # A malformed authority (``http:///user:pw@host/v1`` or
    # ``http:/user:pw@host/v1``) is not represented by ``urlsplit``, but it
    # still carries a credential that must be rejected/redacted.  Only consult
    # the textual fallback when there is no parsed authority: for a well-formed
    # URL an ``@`` in the first path segment is a path character, not userinfo.
    if not parsed.netloc and "@" in _textual_authority(str(url or "")):
        return "userinfo"
    for key, _value in _query_pairs(parsed.query):
        if _credential_key(key):
            return "query parameter %r" % key
    return ""


#: Levels a catalog may name as a model's default.  ``off`` is a per-run
#: override, not a deployment default (``check_model_catalog`` pins the same
#: vocabulary).  Defined before the import-time catalog load below, which
#: cannot yet use ``REASONING_EFFORT_LEVELS`` declared later in the module.
_CATALOG_DEFAULT_LEVELS = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "max"}
)


def _stored_endpoint_problem(url: str) -> str:
    """A short reason a catalog ``base_url`` is unusable, or ``""``.

    Defined before the import-time catalog load; the public
    :func:`endpoint_base` is declared later in this module and cannot be called
    there yet.  Non-http and scheme-relative values remain importable: a
    checkout may carry a legacy catalog value that the credential scrub must
    still handle (see ``test_a_non_http_catalog_base_url_imports_credential_free``),
    so only a value with no host or an invalid port is rejected here.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        return str(exc)
    if not host:
        return "not a URL with a host"
    if port is not None and not 0 < port <= 65535:
        return "invalid port %d" % port
    return ""


def _apply_model_catalog(models: Dict[str, Dict[str, Any]]) -> None:
    """Overwrite each model's facts from the shared catalog, atomically.

    Nothing is written until every model has been validated, so a later invalid
    entry cannot leave an earlier model half-updated.
    """
    global _CATALOG_BASE_SECRETS
    try:
        with _MODEL_CATALOG_PATH.open(encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            # ``document.get`` on a list/string/null raised a raw AttributeError,
            # which is not in the malformed-catalog tuple below and escaped the
            # documented ConfigError contract.
            raise ConfigError(
                "the shared model catalog %s must be a JSON object, not %s"
                % (_MODEL_CATALOG_PATH, type(document).__name__)
            )
        # The checker rejects an unsupported schema_version, but the runtime
        # used to ignore it: a future document could load with silently
        # different semantics while check_model_catalog passed on another
        # file.  Require the same exact integer contract here.
        if type(document.get("schema_version")) is not int or document.get(
            "schema_version"
        ) != 1:
            raise ConfigError(
                "unsupported schema_version %r in %s"
                % (document.get("schema_version"), _MODEL_CATALOG_PATH)
            )
        by_id: Dict[str, Dict[str, Any]] = {}
        base_secrets: Dict[str, str] = {}
        for entry in document["models"]:
            model_id = entry["id"]
            if model_id in by_id:
                # check_model_catalog rejects duplicates, but the runtime used
                # to keep whichever entry came last.
                raise ConfigError(
                    "duplicate model id %r in %s"
                    % (model_id, _MODEL_CATALOG_PATH)
                )
            by_id[model_id] = entry
        pending: Dict[str, Dict[str, Any]] = {}
        stale: Dict[str, Tuple[str, ...]] = {}
        for name, entry in models.items():
            facts = by_id.get(entry["id"])
            if facts is None:
                raise ConfigError(
                    "model %r (%s) is missing from %s"
                    % (name, entry["id"], _MODEL_CATALOG_PATH)
                )
            # ``int(...)`` used to accept 32768.9 and "150000", silently
            # changing the cap while check_model_catalog rejected the same
            # document.  Require the exact JSON integer type both places accept.
            for capacity in ("context_window", "max_tokens"):
                if type(facts[capacity]) is not int:
                    raise ConfigError(
                        "model %r: %s %r must be an integer in %s"
                        % (
                            name,
                            capacity,
                            facts[capacity],
                            _MODEL_CATALOG_PATH,
                        )
                    )
            context_window = facts["context_window"]
            max_tokens = facts["max_tokens"]
            if max_tokens <= 0 or context_window <= max_tokens:
                raise ConfigError(
                    "model %r: max_tokens %d must be positive and below "
                    "context_window %d in %s"
                    % (name, max_tokens, context_window, _MODEL_CATALOG_PATH)
                )
            update: Dict[str, Any] = {
                "context_window": context_window,
                "max_tokens": max_tokens,
            }
            chat_max_tokens = facts.get("chat_max_tokens")
            if chat_max_tokens is not None:
                if type(chat_max_tokens) is not int:
                    raise ConfigError(
                        "model %r: chat_max_tokens %r must be an integer in %s"
                        % (
                            name,
                            chat_max_tokens,
                            _MODEL_CATALOG_PATH,
                        )
                    )
                if (
                    chat_max_tokens <= 0
                    or chat_max_tokens < max_tokens
                    or chat_max_tokens >= context_window
                ):
                    raise ConfigError(
                        "model %r: chat_max_tokens %d must be positive, at "
                        "least max_tokens %d and below context_window %d in %s"
                        % (
                            name,
                            chat_max_tokens,
                            max_tokens,
                            context_window,
                            _MODEL_CATALOG_PATH,
                        )
                    )
                update["chat_max_tokens"] = chat_max_tokens
            default_level = facts.get("default_reasoning_level")
            # Only a missing/None field is absent.  A present but falsy value
            # (``[]``, ``0``, ``False``, ``{}``) was silently dropped here while
            # check_model_catalog rejected it, leaving the model on a hard-coded
            # fallback instead of failing the atomic reload.
            if default_level is not None:
                if (
                    not isinstance(default_level, str)
                    or default_level not in _CATALOG_DEFAULT_LEVELS
                ):
                    raise ConfigError(
                        "model %r: default_reasoning_level %r is not one of %s "
                        "in %s"
                        % (
                            name,
                            default_level,
                            ", ".join(sorted(_CATALOG_DEFAULT_LEVELS)),
                            _MODEL_CATALOG_PATH,
                        )
                    )
                update["default_reasoning_level"] = default_level
            # The catalog is also the one source for the provider route: the
            # runtime used to keep separate QWEN_/DOTS_ base-URL and key-env
            # literals that a catalog edit could not reach.
            # Only a missing/None field is absent.  A present but falsy value
            # (``[]``, ``0``, ``""``) was silently dropped by a truthiness
            # guard, leaving the model on a hard-coded fallback provider.
            if "provider" in facts and facts["provider"] is not None:
                provider = facts["provider"]
                # A list/dict provider used to become its repr and reach the
                # generated patch as a bogus provider id.  The catalog is the
                # routing source of truth, so reject it on reload.
                if not isinstance(provider, str) or not provider.strip():
                    raise ConfigError(
                        "model %r: provider %r must be a non-empty string in %s"
                        % (name, provider, _MODEL_CATALOG_PATH)
                    )
                update["dsh_provider"] = provider
            if "base_url" in facts and facts["base_url"] is not None:
                # Only absent/None means absent.  A present falsy value
                # (``[]``, ``0``, ``False``, ``""``) was dropped by the old
                # truthiness guard, so the import kept the built-in localhost
                # fallback instead of rejecting the malformed catalog.
                raw_base = facts["base_url"]
                if not isinstance(raw_base, str) or not raw_base.strip():
                    raise ConfigError(
                        "model %r: base_url %r must be a non-empty string in %s"
                        % (name, raw_base, _MODEL_CATALOG_PATH)
                    )
                # The catalog is a shared, agent-readable artifact.  Keep the
                # public table and the exported constants credential-free and
                # carry the live secret separately.
                safe_base, base_secret = split_endpoint_credentials(raw_base)
                try:
                    # Validate the URL shape at load time; the DSH route fails
                    # on a bad base only when a run is configured, while the
                    # Claude route silently exported it to the child.
                    problem = _stored_endpoint_problem(safe_base)
                    if problem:
                        raise ConfigError(problem)
                except ConfigError as exc:
                    raise ConfigError(
                        "model %r: base_url in %s is not a usable endpoint: %s"
                        % (name, _MODEL_CATALOG_PATH, exc)
                    ) from None
                update["base_url"] = safe_base
                if base_secret:
                    base_secrets[name] = base_secret
            if "api_key_env" in facts and facts["api_key_env"] is not None:
                key_env = facts["api_key_env"]
                # 123 used to become the environment variable name "123";
                # a non-string key env cannot name a usable credential source.
                if not isinstance(key_env, str) or not key_env.strip():
                    raise ConfigError(
                        "model %r: api_key_env %r must be a non-empty string in %s"
                        % (name, key_env, _MODEL_CATALOG_PATH)
                    )
                update["api_key_env"] = key_env
            pending[name] = update
            # A field the previous catalog supplied must disappear when the new
            # catalog omits it; an overlay would otherwise leave it stale and
            # _refresh_catalog_constants would re-export the old value.
            stale[name] = tuple(
                key for key in _CATALOG_OWNED_MODEL_KEYS if key not in update
            )
        for name, update in pending.items():
            for key in stale[name]:
                models[name].pop(key, None)
            models[name].update(update)
        _CATALOG_BASE_SECRETS = dict(base_secrets)
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
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
        raise ConfigError(
            "the shared model catalog %s is malformed: %s"
            % (_MODEL_CATALOG_PATH, exc)
        ) from exc


#: ``context_window``/``max_tokens``/``chat_max_tokens`` in the table above are
#: placeholders: the catalog overlay replaces them at import, so edit the
#: catalog, not those numbers.
_apply_model_catalog(AGENT_MODELS)


def reload_model_catalog() -> None:
    """Re-read and apply the catalog with the current policy flags.

    Import-time validation uses the shared defaults.  A checkout whose policy
    differs (GDP and paper-refine-eval fail closed on a missing catalog) sets
    the flag and calls this immediately, so the policy is not silently lost.

    Example:
        ``reload_model_catalog()`` re-reads the catalog after a consumer changes
        ``MISSING_CATALOG_IS_FATAL``.

    Counterexample:
        It does not reset ``AGENT_MODELS`` to factory defaults; it overlays the
        current catalog, so deliberate table edits survive.
    """
    _apply_model_catalog(AGENT_MODELS)
    # The call sites read these module constants at request time; refreshing
    # them here makes a successful reload visible instead of leaving the
    # process on the import-time values.
    _refresh_catalog_constants()


AGENT_MODEL_CHOICES: Tuple[str, ...] = tuple(AGENT_MODELS) + (MODEL_DEFAULT,)

#: Capacity numbers resolved from the shared catalog (authoritative).
QWEN_CONTEXT_WINDOW = int(AGENT_MODELS["qwen38-27b"]["context_window"])
QWEN_MAX_TOKENS = int(AGENT_MODELS["qwen38-27b"]["max_tokens"])
QWEN_CHAT_MAX_TOKENS = int(
    AGENT_MODELS["qwen38-27b"].get("chat_max_tokens") or QWEN_MAX_TOKENS
)
DOTS_CONTEXT_WINDOW = int(AGENT_MODELS["dots3-note-preview"]["context_window"])
DOTS_MAX_TOKENS = int(AGENT_MODELS["dots3-note-preview"]["max_tokens"])
DOTS_CHAT_MAX_TOKENS = int(
    AGENT_MODELS["dots3-note-preview"].get("chat_max_tokens") or DOTS_MAX_TOKENS
)
DEEPSEEK_CONTEXT_WINDOW = int(AGENT_MODELS["deepseek-flash"]["context_window"])
DEEPSEEK_MAX_TOKENS = int(AGENT_MODELS["deepseek-flash"]["max_tokens"])
DEEPSEEK_CHAT_MAX_TOKENS = int(
    AGENT_MODELS["deepseek-flash"].get("chat_max_tokens") or DEEPSEEK_MAX_TOKENS
)

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

#: Provider qwen serves its OpenAI-compatible route on by default.  The catalog
#: owns the base URL/model/key-env fields; these names stay for callers that
#: imported the old constants.
QWEN_DEFAULT_BASE_URL = str(
    AGENT_MODELS["qwen38-27b"].get("base_url") or "http://127.0.0.1:8000/v1"
)
QWEN_DEFAULT_MODEL = str(AGENT_MODELS["qwen38-27b"].get("id") or "qwen3.8-27b")
QWEN_DEFAULT_API_KEY = "dummy"
#: The reasoning level each provider family serves when the operator sets no
#: override.  The shared catalog's ``default_reasoning_level`` is authoritative
#: (the catalog overlay above copies it into ``AGENT_MODELS``); the literals are
#: the fallbacks for a catalog that predates the field.  GPT and Claude carry no
#: default there on purpose: their harness configuration names no reasoning
#: effort, so the model's own default applies.  An operator override
#: (``PAPER_AGENT_REASONING_EFFORT``) still wins.
QWEN_DEFAULT_EFFORT = str(
    AGENT_MODELS["qwen38-27b"].get("default_reasoning_level") or "medium"
)
DOTS_DEFAULT_EFFORT = str(
    AGENT_MODELS["dots3-note-preview"].get("default_reasoning_level") or "high"
)
DEEPSEEK_DEFAULT_EFFORT = str(
    AGENT_MODELS["deepseek-flash"].get("default_reasoning_level") or "high"
)
DOTS_DEFAULT_BASE_URL = str(
    AGENT_MODELS["dots3-note-preview"].get("base_url")
    or "https://note3-prev-api.askdiandian.com/v1"
)
DOTS_DEFAULT_MODEL = str(
    AGENT_MODELS["dots3-note-preview"].get("id") or "dots3-note-prev"
)
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


def _refresh_catalog_constants() -> None:
    """Recompute the exported catalog-derived constants after a reload."""
    global QWEN_CONTEXT_WINDOW, QWEN_MAX_TOKENS, QWEN_CHAT_MAX_TOKENS
    global DOTS_CONTEXT_WINDOW, DOTS_MAX_TOKENS, DOTS_CHAT_MAX_TOKENS
    global DEEPSEEK_CONTEXT_WINDOW, DEEPSEEK_MAX_TOKENS, DEEPSEEK_CHAT_MAX_TOKENS
    global QWEN_DEFAULT_BASE_URL, QWEN_DEFAULT_MODEL
    global DOTS_DEFAULT_BASE_URL, DOTS_DEFAULT_MODEL
    global QWEN_DEFAULT_EFFORT, DOTS_DEFAULT_EFFORT, DEEPSEEK_DEFAULT_EFFORT

    QWEN_CONTEXT_WINDOW = int(AGENT_MODELS["qwen38-27b"]["context_window"])
    QWEN_MAX_TOKENS = int(AGENT_MODELS["qwen38-27b"]["max_tokens"])
    QWEN_CHAT_MAX_TOKENS = int(
        AGENT_MODELS["qwen38-27b"].get("chat_max_tokens") or QWEN_MAX_TOKENS
    )
    DOTS_CONTEXT_WINDOW = int(AGENT_MODELS["dots3-note-preview"]["context_window"])
    DOTS_MAX_TOKENS = int(AGENT_MODELS["dots3-note-preview"]["max_tokens"])
    DOTS_CHAT_MAX_TOKENS = int(
        AGENT_MODELS["dots3-note-preview"].get("chat_max_tokens") or DOTS_MAX_TOKENS
    )
    DEEPSEEK_CONTEXT_WINDOW = int(AGENT_MODELS["deepseek-flash"]["context_window"])
    DEEPSEEK_MAX_TOKENS = int(AGENT_MODELS["deepseek-flash"]["max_tokens"])
    DEEPSEEK_CHAT_MAX_TOKENS = int(
        AGENT_MODELS["deepseek-flash"].get("chat_max_tokens") or DEEPSEEK_MAX_TOKENS
    )
    QWEN_DEFAULT_BASE_URL = str(
        AGENT_MODELS["qwen38-27b"].get("base_url") or "http://127.0.0.1:8000/v1"
    )
    QWEN_DEFAULT_MODEL = str(
        AGENT_MODELS["qwen38-27b"].get("id") or "qwen3.8-27b"
    )
    DOTS_DEFAULT_BASE_URL = str(
        AGENT_MODELS["dots3-note-preview"].get("base_url")
        or "https://note3-prev-api.askdiandian.com/v1"
    )
    DOTS_DEFAULT_MODEL = str(
        AGENT_MODELS["dots3-note-preview"].get("id") or "dots3-note-prev"
    )
    QWEN_DEFAULT_EFFORT = str(
        AGENT_MODELS["qwen38-27b"].get("default_reasoning_level") or "medium"
    )
    DOTS_DEFAULT_EFFORT = str(
        AGENT_MODELS["dots3-note-preview"].get("default_reasoning_level") or "high"
    )
    DEEPSEEK_DEFAULT_EFFORT = str(
        AGENT_MODELS["deepseek-flash"].get("default_reasoning_level") or "high"
    )
    API_PROVIDER_DEFAULTS["qwen"]["base_url"] = QWEN_DEFAULT_BASE_URL
    API_PROVIDER_DEFAULTS["qwen"]["model"] = QWEN_DEFAULT_MODEL
    API_PROVIDER_DEFAULTS["dots"]["base_url"] = DOTS_DEFAULT_BASE_URL
    API_PROVIDER_DEFAULTS["dots"]["model"] = DOTS_DEFAULT_MODEL

#: One name per shared agent knob: the key the generated provider config names
#: for a custom endpoint, the model/name/prep request options, and the harness
#: selection.  The historical PAPER_/GDP_ aliases are gone: a consumer's CLI
#: flag exports the single name below, and the runtime reads only that name.
#: The key itself is only ever exported to the child, never written into the
#: provider patch, the run record or the log.
AGENT_MODEL_API_KEY_ENV = "PAPER_AGENT_MODEL_API_KEY"
AGENT_MODEL_NAME_ENV = "PAPER_AGENT_MODEL_NAME"
AGENT_MODEL_PREP_JSON_ENV = "PAPER_AGENT_MODEL_PREP_JSON"
AGENT_HARNESS_ENV = "PAPER_AGENT_HARNESS"
AGENT_MODEL_ENV = "PAPER_AGENT_MODEL"

#: Operator override for the reasoning level of whichever model a selection
#: names.  It wins over the catalog default, and it is the only override that
#: also reaches GPT and Claude, whose default is the model's own.  A consumer's
#: ``--reasoning-effort`` flag exports this; ``none`` is accepted as a spelling
#: of ``off``.  The vocabulary is the union the harnesses understand; a family
#: that lacks a level uses its nearest documented one (see
#: :func:`normalize_reasoning_effort`).
AGENT_REASONING_EFFORT_ENV = "PAPER_AGENT_REASONING_EFFORT"
REASONING_EFFORT_LEVELS: Tuple[str, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

DEFAULT_ENDPOINT_CONTEXT = 32768
DEFAULT_ENDPOINT_MAX_TOKENS = 8192

#: Bound on the optional ``/models`` listing body: the probe must not read an
#: arbitrarily large response into memory.
_MAX_MODELS_BODY_BYTES = 1 << 20


def reasoning_effort_override() -> Optional[str]:
    """The operator's explicit reasoning level for the selected model, or None.

    A consumer's ``--reasoning-effort`` flag exports
    ``PAPER_AGENT_REASONING_EFFORT``.  The value wins over the catalog default,
    for every family, including GPT and Claude (whose default is the model's
    own).  ``none`` is accepted as a spelling of ``off``.

    Example:
        With ``PAPER_AGENT_REASONING_EFFORT=high`` it returns ``"high"``; with
        the variable unset it returns ``None`` and the catalog default applies.

    Counterexample:
        An unknown value raises ``ConfigError`` instead of being dropped: the
        harnesses accept different vocabularies, so a misspelling must fail
        before a run rather than produce an invalid provider config.
    """
    value = _env(AGENT_REASONING_EFFORT_ENV).strip().lower()
    if not value:
        return None
    if value == "none":
        return "off"
    if value not in REASONING_EFFORT_LEVELS:
        raise ConfigError(
            "unknown reasoning effort %r; use one of %s"
            % (value, ", ".join(REASONING_EFFORT_LEVELS))
        )
    return value


#: How each family's wire spells a generic level.  A level the family does not
#: document maps onto its nearest one, so one operator flag works everywhere:
#: DeepSeek has no ``medium`` (it maps to ``high``) and Qwen has no ``high`` or
#: ``max`` (both map to its highest level, ``xhigh``).  Dots has a single
#: thinking level and GPT/Claude are validated by the harness, so their values
#: pass through unchanged.
_EFFORT_WIRE_LEVELS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "off": "off",
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    },
    "qwen": {
        "off": "off",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
        "max": "xhigh",
    },
}


def normalize_reasoning_effort(model: Any, level: str) -> str:
    """Translate a generic reasoning level to the level a model's wire accepts.

    The generic vocabulary is ``off|minimal|low|medium|high|xhigh|max``.  A
    family that lacks a level uses its nearest documented one: DeepSeek has no
    ``medium`` (it maps to ``high``) and Qwen has no ``high``/``max`` (both map
    to its highest level, ``xhigh``).  Dots, GPT, Claude and a custom endpoint
    pass the level through; their harness validates it.

    Example:
        ``normalize_reasoning_effort("deepseek-flash", "medium") == "high"``
        and ``normalize_reasoning_effort("qwen38-27b", "max") == "xhigh"``.

    Counterexample:
        An unknown level raises ``ConfigError``; it is never silently replaced
        by the catalog default, because the operator asked for something
        specific.
    """
    text = str(level or "").strip().lower()
    if text == "none":
        text = "off"
    if text not in REASONING_EFFORT_LEVELS:
        raise ConfigError(
            "unknown reasoning effort %r; use one of %s"
            % (level, ", ".join(REASONING_EFFORT_LEVELS))
        )
    family = str((AGENT_MODELS.get(normalize_agent_model(model)) or {}).get("family") or "")
    return (_EFFORT_WIRE_LEVELS.get(family) or {}).get(text, text)


def agent_choice(value: Any) -> str:
    """Normalize an operator spelling for a harness or model name.

    ``gpt codex``, ``gpt-codex`` and ``GPT_CODEX`` all become ``gpt-codex``.

    Example:
        ``agent_choice("GPT Codex") == "gpt-codex"``.

    Counterexample:
        It does not validate the name; an unknown value is returned normalized.
        Use ``agent_harness_choice``/``agent_model_choice`` to reject it.
    """
    return str(value).strip().lower().replace("_", "-").replace(" ", "-")


def agent_model_is_endpoint(value: Any) -> bool:
    """True when ``--agent-model`` is an arbitrary chat-completions endpoint URL.

    Example:
        ``agent_model_is_endpoint("https://host/v1")`` is ``True``.

    Counterexample:
        The named model ``"http-server"`` is not an endpoint; only the http(s)
        prefix counts.
    """
    text = str(value or "").strip().lower()
    return text.startswith("http://") or text.startswith("https://")


def normalize_agent_model(value: Any) -> str:
    """Canonical model name, or the URL unchanged when it is a custom endpoint.

    Example:
        ``normalize_agent_model("GPT SOL") == "gpt-sol"``.

    Counterexample:
        A custom URL is returned unchanged, including its query string; use
        ``endpoint_base`` to build a provider base URL.
    """
    raw = str(value).strip()
    if agent_model_is_endpoint(raw):
        return raw
    slug = agent_choice(raw)
    if slug == MODEL_DEFAULT:
        return MODEL_DEFAULT
    return _MODEL_ALIASES.get(slug, slug)


def agent_model_choice(value: Any) -> str:
    """``--agent-model`` accepts a known model name or an http(s) endpoint URL.

    Example:
        ``agent_model_choice("GPT-6.1-SOL") == "gpt-sol"``.

    Counterexample:
        An unknown name raises ``ValueError``; a URL with a credential raises unless
        ``REJECT_ENDPOINT_CREDENTIALS`` is disabled.
    """
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
            "endpoint URL"
            % (_redacted_endpoint_text(raw), ", ".join(AGENT_MODEL_CHOICES))
        )
    return slug


def agent_harness_choice(value: Any) -> str:
    """Normalize and validate a harness choice.

    Accepts the canonical names plus the operator aliases ``deepseek``,
    ``deepseek-harness``, ``codex`` and ``claude``.

    Example:
        ``agent_harness_choice("codex") == "gpt-codex"``.

    Counterexample:
        ``"openai"`` is not an alias and raises ``ValueError``; a provider name
        is not automatically a harness name.
    """
    name = agent_choice(value)
    name = _HARNESS_ALIASES.get(name, name)
    if name not in AGENT_HARNESSES:
        raise ValueError(
            "unknown harness %r; use one of %s"
            % (value, ", ".join(AGENT_HARNESS_CHOICES))
        )
    return name


def agent_model_label(model: Optional[str]) -> str:
    """A short label for a model choice (an endpoint URL is not repeated).

    Example:
        ``agent_model_label("gpt-sol") == "gpt-sol"`` and
        ``agent_model_label("https://host/v1") == "custom-endpoint"``.

    Counterexample:
        It is a display label, not a model id; ``"default"`` becomes
        ``"harness-default"`` and cannot be passed back to the provider.
    """
    if not model:
        return "config-owned"
    if model == MODEL_DEFAULT:
        return "harness-default"
    if agent_model_is_endpoint(model):
        return "custom-endpoint"
    return str(_redacted_endpoint_text(model))


def model_fact(model: Any, field: str, default: Any = None) -> Any:
    """One shared-catalog fact for a named model.

    The catalog owns the numbers every consumer would otherwise duplicate: the
    context window, the harness output cap (``max_tokens``), the single-shot
    chat cap (``chat_max_tokens``) and ``default_reasoning_level``.  Read them
    here so a catalog revision needs no change outside this repository.

    Example:
        ``model_fact("qwen38-27b", "context_window") == 150000`` and
        ``model_fact("dots3-note-preview", "chat_max_tokens") == 160000``.

    Counterexample:
        An unknown model, the ``default`` sentinel or a custom endpoint returns
        ``default``; it does not raise, so validate a name with
        ``agent_model_choice`` first when a typo must be an error.
    """
    slug = normalize_agent_model(model)
    if slug == MODEL_DEFAULT:
        return default
    value = (AGENT_MODELS.get(slug) or {}).get(field)
    return default if value is None else value


def default_reasoning_level(model: Any) -> Optional[str]:
    """The reasoning level a named model serves when no operator override is set.

    The shared catalog's ``default_reasoning_level`` field is the one source of
    truth, so every harness reads the same level: ``high`` for DeepSeek,
    ``high`` for Dots and ``medium`` for Qwen.  GPT and Claude entries
    deliberately omit the field, because their deployments keep the model's own
    default and the generated configuration sends no reasoning-effort field.

    Example:
        ``default_reasoning_level("deepseek-flash") == "high"``,
        ``default_reasoning_level("dots3-note-prev") == "high"`` and
        ``default_reasoning_level("qwen3.8-27b") == "medium"``.

    Counterexample:
        ``default_reasoning_level("gpt-sol")`` and
        ``default_reasoning_level("claude-opus")`` return ``None``; do not
        substitute a literal for a model that keeps its own default, and do not
        call this for a custom endpoint URL.
    """
    level = model_fact(model, "default_reasoning_level")
    return str(level) if level else None


def _canonical_harness(value: Any) -> Any:
    """Resolve an operator harness spelling without validating/raising."""
    if not value:
        return value
    name = agent_choice(value)
    return _HARNESS_ALIASES.get(name, name)


def _canonical_model(value: Any) -> Any:
    """Resolve a model alias without validating/raising."""
    if not value:
        return value
    return normalize_agent_model(value)


def agent_pair_supported(harness: str, model: Optional[str]) -> bool:
    """Can ``harness`` reach ``model`` over a protocol it speaks?

    Example:
        ``agent_pair_supported("gpt-codex", "gpt-sol")`` is ``True``.

    Counterexample:
        ``agent_pair_supported("claude-code", "gpt-sol")`` is ``False``; the two
        axes are not freely interchangeable.
    """
    harness = _canonical_harness(harness)
    model = _canonical_model(model) if model is not None else None
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
    """The model a harness uses when only the harness was chosen.

    Example:
        ``agent_model_for_harness("claude-code") == "claude-sonnet"``.

    Counterexample:
        An unknown harness falls back to the global default model rather than
        raising; validate the harness first if that matters.
    """
    harness = _canonical_harness(harness) or harness
    return (AGENT_HARNESSES.get(harness) or {}).get(
        "default_model", DEFAULT_AGENT_MODEL
    )


def agent_harness_for_model(model: str) -> str:
    """The harness a model uses when only the model was chosen.

    Example:
        ``agent_harness_for_model("gpt-sol") == "gpt-codex"``.

    Counterexample:
        A custom endpoint always maps to ``dsh``; do not infer ``codex`` from the
        URL's model name.
    """
    model = _canonical_model(model) if model else model
    if model == MODEL_DEFAULT:
        return DEFAULT_AGENT_HARNESS
    if agent_model_is_endpoint(model):
        return AGENT_HARNESS_DSH
    return (AGENT_MODELS.get(model) or {}).get(
        "default_harness", DEFAULT_AGENT_HARNESS
    )


def agent_selection_label(harness: Optional[str], model: Optional[str]) -> str:
    """A compact ``harness/model`` label for logs and run records.

    The model side is passed through :func:`agent_model_label`, so an endpoint
    is shown as ``custom-endpoint`` rather than its credential-bearing URL.

    Example:
        ``agent_selection_label("dsh", "dots3-note-preview")`` returns
        ``"dsh/dots3-note-preview"``.

    Counterexample:
        ``agent_selection_label(None, None)`` returns ``"config-owned"``; it is
        a display label, not a parseable model id.
    """
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

    Example:
        ``pair_from_options(None, "gpt-sol")`` returns
        ``("gpt-codex", "gpt-sol")``.

    Counterexample:
        Empty inputs use the fallback/default pair; the function does not return
        ``(None, None)``.
    """
    harness = _canonical_harness(harness)
    model = _canonical_model(model) if model else model
    fallback_harness = _canonical_harness(fallback_harness)
    fallback_model = _canonical_model(fallback_model) if fallback_model else fallback_model
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
    """One resolved (harness, model) choice plus its request-side settings.

    The object is frozen: it is a value passed between the CLI, the adapter and
    the run-record writer, not mutable state.

    Example:
        ``resolve_agent_selection(harness="dsh", model="gpt-sol")`` returns an
        ``AgentSelection(harness="dsh", model="gpt-sol", ...)``.

    Counterexample:
        Do not assign ``selection.model = ...`` to retarget a run; create a new
        selection through ``resolve_agent_selection`` so validation still runs.
    """

    harness: str
    model: Optional[str] = None
    legacy: Optional[str] = None
    # Never print the live key from the auto-generated repr; to_record() is the
    # one explicit serialiser and it reports only api_key_set.
    api_key: Optional[str] = field(default=None, repr=False)
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

    def __repr__(self) -> str:
        """The auto-generated repr, with an endpoint's credentials redacted.

        ``api_key`` was already ``repr=False``, but an endpoint selection keeps
        the raw URL (including userinfo/query) for the live request; printing
        the selection must not write that credential to a log.
        """
        model = self.model
        if model is not None:
            try:
                model = _redacted_endpoint_text(model)
            except (TypeError, ValueError):
                model = "<redacted endpoint>"
        model_name = _redacted_endpoint_text(self.model_name)
        prep = redact_prep_object(dict(self.prep_json or {}))
        return (
            "AgentSelection(harness=%r, model=%r, legacy=%r, model_name=%r, "
            "prep_json=%r)"
            % (self.harness, model, self.legacy, model_name, prep)
        )

    @property
    def label(self) -> str:
        return agent_selection_label(self.harness, self.model)

    def model_info(self) -> Dict[str, Any]:
        """The catalog entry for this named model, or an empty dict.

        Example:
            A ``qwen38-27b`` selection returns the catalog entry with its
            context window and default reasoning level.

        Counterexample:
            A custom endpoint or the ``default`` sentinel returns ``{}``; the
            caller must not read capacities from it.
        """
        if self.model is None or self.is_endpoint or self.is_default:
            return {}
        # The dataclass is public, so a directly constructed alias
        # (``AgentSelection(model="gpt-6.1-sol")``) must resolve like a
        # selected one instead of silently losing its catalog facts.
        key = normalize_agent_model(self.model)
        return dict(AGENT_MODELS.get(key) or {})

    def effective_model_id(self, raw_model: Optional[str] = None) -> Optional[str]:
        """The provider model id: the raw ``--model`` wins, then the model name.

        ``--model`` remains the low-level override it always was.  For a custom
        endpoint it names the provider model when ``--agent-model-name`` did
        not; for a native model it replaces the catalog id.

        Example:
            A named ``qwen38-27b`` selection returns the catalog id
            ``qwen3.8-27b``.

        Counterexample:
            A raw ``--model`` override always wins, even over an endpoint's
            ``--agent-model-name``; do not expect the catalog id in that case.
        """
        if raw_model:
            return str(raw_model)
        if self.is_endpoint:
            return self.model_name or endpoint_query_model(str(self.model or "")) or None
        if self.model is None or self.is_default:
            return None
        key = normalize_agent_model(self.model)
        return str((AGENT_MODELS.get(key) or {}).get("id") or self.model)

    def to_record(
        self,
        raw_model: Optional[str] = None,
        *,
        endpoint_api_key_env: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Secret-free shape for ``command.json`` / ``status.json``.

        ``endpoint_api_key_env`` is the same optional variable name a caller
        passes to :func:`selection_env` / :func:`dsh_patch_text`; it lets the
        record report a credential that lives only in that caller-named
        variable.

        Example:
            The record contains ``harness``, ``model``, ``id`` and ``api_key_set`` and
            never the key itself.

        Counterexample:
            An endpoint's query string is stripped from the recorded model; the record
            is not a way to recover the live URL.
        """
        record_model = _redacted_endpoint_text(self.model)
        # ``--agent-model-name`` is also a persisted value; a model name pasted
        # as a URL must not smuggle userinfo/query credentials into the record.
        record_model_name = _redacted_endpoint_text(self.model_name)
        record_id = _redacted_endpoint_text(self.effective_model_id(raw_model))
        endpoint_value = (
            str(record_model)
            if self.is_endpoint and record_model is not None
            else None
        )
        model_name_value = record_model_name if self.is_endpoint else None
        # A raw endpoint URL on a *named* selection is the endpoint route (the
        # patch/env builders resolve it that way), so the record must not claim
        # a named GPT model with an endpoint id and no endpoint.  The
        # sentinel/legacy shape keeps its historical raw-URL id.
        if (
            raw_model
            and agent_model_is_endpoint(str(raw_model))
            and self.model is not None
            and not self.is_default
            and not self.is_endpoint
        ):
            effective = AgentSelection(
                harness=self.harness,
                model=str(raw_model),
                legacy=self.legacy,
                api_key=self.api_key,
                model_name=self.model_name,
                prep_json=self.prep_json,
            )
            record_model = _redacted_endpoint_text(effective.model)
            record_model_name = _redacted_endpoint_text(effective.model_name)
            record_id = _redacted_endpoint_text(
                effective.effective_model_id()
            ) or "default"
            endpoint_value = (
                str(record_model)
                if effective.is_endpoint and record_model is not None
                else None
            )
            model_name_value = (
                record_model_name if effective.is_endpoint else None
            )
        # Report the credential the live request will actually use: an
        # explicit key, the endpoint's key environment (default or the
        # caller-named one), or a URL userinfo/query secret.  Checking only
        # ``api_key``/the default variable hid the URL credential.  A
        # sentinel/None selection can take its endpoint from ``--model``
        # (``dsh_patch_text`` builds the live block from exactly that shape),
        # so resolve the same effective endpoint before checking.
        effective_endpoint = None
        if self.is_endpoint:
            effective_endpoint = str(self.model)
        elif raw_model and agent_model_is_endpoint(str(raw_model)):
            effective_endpoint = str(raw_model)
        api_key_set = bool(self.api_key)
        if not api_key_set and effective_endpoint is not None:
            key_env_names = [AGENT_MODEL_API_KEY_ENV]
            if endpoint_api_key_env:
                key_env_names.append(endpoint_api_key_env)
            api_key_set = any(_env(name) for name in key_env_names)
            if not api_key_set:
                _safe_url, url_secret = split_endpoint_credentials(
                    effective_endpoint
                )
                api_key_set = bool(url_secret)
        if not api_key_set:
            # Native provider selections resolve their key from the provider
            # environment (DOTS_API_KEY, DEEPSEEK_API_KEY, the catalog's
            # api_key_env, ...); the record used to report False even though
            # selection_env exported the live credential.  The local Qwen
            # route's historical "dummy" is not a configured credential.
            try:
                settings = _provider_family_settings(
                    _effective_selection(self, raw_model)
                )
            except ConfigError:
                settings = {}
            key = settings.get("api_key")
            api_key_set = bool(key) and str(key) != "dummy"
        return {
            "harness": self.harness,
            "model": record_model,
            "id": record_id,
            "legacy": self.legacy,
            # A caller may put a credential in the endpoint's query string. This record is
            # written to command.json and status.json, which every agent can read, so the
            # persisted endpoint keeps only the scheme/host/path. The live AgentSelection
            # still carries the raw URL for the actual request.
            "endpoint": endpoint_value,
            "model_name": model_name_value,
            "api_key_set": api_key_set,
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

    Raises :class:`agent_common.model_runtime.ConfigError` (or the host's
    rebound subclass) for a missing endpoint, an unknown
    choice, an unsupported wire-protocol pair or request options that do not
    belong to a named model.

    Example:
        ``resolve_agent_selection(harness="dsh", model="dots3-note-preview")``
        returns an ``AgentSelection`` with credentials resolved from the
        environment.

    Counterexample:
        A named model combined with ``api_key``/``model_name``/``prep_json`` raises
        ``ConfigError``; those options belong to a custom endpoint only.
    """
    # Validate through the same strict JSON contract as the string spelling:
    # ``dict(...)`` accepted a list of pairs (not a JSON object), and a direct
    # NaN/Infinity mapping was accepted although the JSON contract forbids it.
    prep = parse_prep_json(prep_json)
    if legacy_agent:
        if not isinstance(legacy_agent, str):
            raise ConfigError(
                "unknown legacy agent %r; a legacy agent name must be a string"
                % (legacy_agent,)
            )
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
    for option, value in (("--agent-model-api-key", api_key),
                          ("--agent-model-name", model_name)):
        if value is not None and not isinstance(value, str):
            # The constructor only copied a str and silently dropped 123/bytes,
            # so the credential/name vanished without an error.
            raise ConfigError(
                "%s must be a string, not %s" % (option, type(value).__name__)
            )
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
        # The dataclass is frozen, so exposing a mutable mapping let one
        # consumer change the validated request preparation.  A recursive
        # read-only deep copy keeps equality/serialisation, rejects assignment
        # at any depth, and stays JSON-serialisable for the prep proxy.
        prep_json=_freeze_prep(prep),
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

    Example:
        ``redact_endpoint_url("https://user:secret@host/v1?api_key=x")`` returns
        ``"https://host/v1"``.

    Counterexample:
        It is for records, not live requests; using the redacted URL would drop the
        credential and the query.
    """
    text = str(url or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return _safe_endpoint_text(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        # A scheme-relative, non-http or host-less URL still may carry
        # userinfo/query credentials; the textual fallback removes them.
        return _safe_endpoint_text(text)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def endpoint_base(url: str) -> str:
    """The provider base URL for a chat-completions endpoint.

    ``https://host/v1/chat/completions`` -> ``https://host/v1``; DSH appends
    ``/chat/completions`` itself.  Query/fragment are dropped, but a ``model=``
    query parameter is preserved as the model name by :func:`endpoint_query_model`.

    URL userinfo is stripped: the base URL is written verbatim into every stage's
    ``dsh_model_patch.yml``, so keeping ``user:password@`` there persisted a credential in an
    agent-readable artifact.  Endpoint credentials belong in ``--agent-model-api-key`` (or the
    provider environment variable), which the generated patch references by name.

    Example:
        ``endpoint_base("https://host/v1/chat/completions")`` returns
        ``"https://host/v1"``.

    Counterexample:
        A URL with no host or a non-http scheme raises ``ConfigError``; it does not
        return the input unchanged.
    """
    text = str(url or "").strip()
    try:
        safe = redact_endpoint_url(text)
    except ValueError:
        # ``urlsplit`` cannot parse some malformed URLs (an unmatched ``[``
        # raises "Invalid IPv6 URL"), and the redactor uses it too.  Strip the
        # credential-bearing parts textually so the ConfigError below stays
        # credential-free and the documented exception type is preserved.
        safe = text.split("#", 1)[0].split("?", 1)[0]
        if "://" in safe:
            scheme, _, remainder = safe.partition("://")
            safe = scheme + "://" + remainder.rsplit("@", 1)[-1]
    try:
        parsed = urllib.parse.urlsplit(text)
        host = parsed.hostname
    except ValueError:
        raise ConfigError(
            "--agent-model endpoint %r is not an http(s) URL with a host" % safe
        ) from None
    if parsed.scheme not in ("http", "https") or not host:
        raise ConfigError(
            "--agent-model endpoint %r is not an http(s) URL with a host" % safe
        )
    try:
        # ``parsed.port`` must stay outside the host-validation try above: its
        # ValueError describes an invalid port, not a host-less URL, and used
        # to make the dedicated branch below unreachable.
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(
            "--agent-model endpoint %r has an invalid port: %s" % (safe, exc)
        ) from None
    if port is not None and not 0 < port <= 65535:
        raise ConfigError(
            "--agent-model endpoint %r has an invalid port: %d" % (safe, port)
        )
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
        # A path such as /v1//chat/completions leaves the separator slash
        # behind; a base ending in "/" makes DSH request //chat/completions.
        path = path.rstrip("/")
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))


def endpoint_query_model(url: str) -> str:
    """The ``model=`` query value of an endpoint URL, or ``""``.

    The value is a hint for the provider model id; ``--agent-model-name`` (or
    an explicit raw model) still wins elsewhere.

    Example:
        ``endpoint_query_model("https://host/v1?model=qwen")`` returns
        ``"qwen"``.

    Counterexample:
        A malformed URL or a missing ``model`` key returns ``""``; this helper
        does not raise for a bad URL.
    """
    try:
        query = urllib.parse.urlsplit(str(url)).query
    except ValueError:
        return ""
    # Every other endpoint parser treats ``;`` as a query separator; using
    # ``parse_qs`` here kept ``model=x;api_key=SECRET`` as one value and
    # exported the credential as the model id.
    for key, value in _query_pairs(query):
        if key == "model":
            return str(value).strip()
    return ""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects for the key-bearing ``/models`` probe.

    urllib's default redirect handler reuses the original request headers, including the
    ``Authorization`` header built from ``--agent-model-api-key``.  A redirected probe would
    therefore hand the key to whatever host the endpoint names.  Returning ``None`` makes
    urllib raise ``HTTPError`` (caught below) instead of following the 3xx.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


class _EndpointProbeError(Exception):
    """The endpoint's ``/models`` listing could not be read."""


def _endpoint_catalog(base_url: str, api_key: str = "", timeout: float = 5.0) -> List[Dict[str, Any]]:
    """The endpoint's ``/models`` entries (an empty list when it exposes none).

    A ``404``/``405`` response means the endpoint serves no listing route, which
    is the documented "no listing" case and returns an empty list.  A failed
    probe raises :class:`_EndpointProbeError` instead of looking like an empty
    listing: a 401/500/network failure must not be reported as "the endpoint
    lists no models" and silently fall back to ``default``.
    """
    if not base_url:
        return []
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    request = urllib.request.Request(
        "%s/models" % base_url.rstrip("/"), headers=headers, method="GET"
    )
    try:
        # An explicit empty ProxyHandler keeps urllib's default environment
        # proxy out of this key-bearing request: a corporate/attacker proxy must
        # not receive the Authorization header just because HTTP(S)_PROXY is set.
        opener = urllib.request.build_opener(
            _NoRedirectHandler, urllib.request.ProxyHandler({})
        )
        with opener.open(request, timeout=timeout) as response:
            # A malicious/broken endpoint can stream an unbounded body; the
            # timeout is per socket operation, not a byte budget.  Read one
            # byte past the bound so an oversized listing is a failed probe
            # instead of an out-of-memory.
            raw_body = response.read(_MAX_MODELS_BODY_BYTES + 1)
        if len(raw_body) > _MAX_MODELS_BODY_BYTES:
            raise _EndpointProbeError(
                "the endpoint /models listing is larger than %d bytes"
                % _MAX_MODELS_BODY_BYTES
            )
        body = json.loads(raw_body.decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        # ``HTTPError`` is itself a response object; leaving it to GC retains
        # the upstream socket on repeated probe failures.
        try:
            if exc.code in (404, 405):
                # The endpoint serves no listing route; that is the documented
                # "no listing" case, not a failed probe.  A keyless local proxy
                # commonly serves only /chat/completions.
                return []
            raise _EndpointProbeError(
                "the endpoint /models listing failed with HTTP %d; check the API "
                "key and that the endpoint serves an OpenAI-compatible /models "
                "route" % exc.code
            ) from None
        finally:
            with contextlib.suppress(OSError):
                exc.close()
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        TypeError,
        http.client.HTTPException,
    ) as exc:
        # BadStatusLine/IncompleteRead are HTTPException, not OSError.  A
        # malformed upstream response is a probe failure, not an empty listing.
        raise _EndpointProbeError(
            "the endpoint /models listing failed: %s" % exc
        ) from None
    if not isinstance(body, dict):
        raise _EndpointProbeError(
            "the endpoint /models listing returned a non-object JSON body; "
            "expected an OpenAI-compatible {\"data\": [...]} document"
        )
    data = body.get("data")
    if not isinstance(data, list):
        # A scalar/array body (or a missing/non-list ``data``) is a malformed
        # listing, not the documented 404/405 "no listing" case.  Returning []
        # made resolve_endpoint_model fall back to "default" and report a
        # successful preflight for a broken endpoint.
        raise _EndpointProbeError(
            "the endpoint /models listing has no data list; "
            "expected an OpenAI-compatible {\"data\": [...]} document"
        )
    return [item for item in data if isinstance(item, dict) and item.get("id")]


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
            except (TypeError, ValueError, OverflowError):
                # ``1e400`` parses to float('inf'), and int(inf) raises
                # OverflowError rather than ValueError.
                continue
            if value > 0:
                return value
    return None


def _endpoint_request_key(
    selection: AgentSelection,
    api_key: Optional[str] = None,
    *,
    endpoint_api_key_env: Optional[str] = None,
) -> str:
    """The credential the live endpoint request would send.

    Keep this in one place: the ``/models`` preflight must authenticate with
    exactly the key the request uses.  An explicit key wins, then the
    selection's key, then the default key environment (or the caller-named one
    the generated patch declares), then a credential carried by the URL
    (accepted when ``REJECT_ENDPOINT_CREDENTIALS`` is off).
    """
    if api_key:
        return str(api_key)
    if selection.api_key:
        return str(selection.api_key)
    env_names = [AGENT_MODEL_API_KEY_ENV]
    if endpoint_api_key_env and endpoint_api_key_env not in env_names:
        env_names.append(endpoint_api_key_env)
    for name in env_names:
        env_key = _env(name)
        if env_key:
            return env_key
    _safe_url, url_secret = split_endpoint_credentials(
        str(selection.model or "")
    )
    return url_secret


_DIAGNOSTIC_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]+")
_DIAGNOSTIC_CREDENTIAL_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password|passwd|"
    r"authorization|signature|sig)\s*[:=]\s*[^\s,;]+"
)


def _diagnostic_model_id(value: Any, limit: int = 120) -> str:
    """One safe line for an endpoint-supplied model id in an error message."""
    text = _redacted_endpoint_text(str(value)) or ""
    text = _DIAGNOSTIC_CONTROL_RE.sub(" ", text).strip()
    text = _DIAGNOSTIC_CREDENTIAL_RE.sub(
        lambda match: "%s=<redacted>" % match.group(1), text
    )
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def resolve_endpoint_model(
    selection: AgentSelection,
    *,
    probe: bool = True,
    api_key: Optional[str] = None,
    endpoint_api_key_env: Optional[str] = None,
) -> Tuple[str, Optional[int], str]:
    """``(model_name, context_window, error)`` for a custom endpoint.

    ``--agent-model-name`` wins; a ``?model=`` query parameter is next; then a
    single model listed by ``/v1/models``; otherwise ``default`` (a keyless
    local proxy commonly serves only one model but exposes no listing).

    Example:
        A URL plus ``--agent-model-name`` returns that name; a URL with ``?model=``
        falls back to the query value.

    Counterexample:
        A named (non-URL) model returns an empty resolution; do not call this for
        catalog models.  A failed ``/models`` probe returns an error rather than
        the ``default`` fallback, because a 401/500/network failure is not an
        empty listing.
    """
    if not selection.is_endpoint:
        return "", None, ""
    if selection.model_name:
        if not probe:
            return selection.model_name, None, ""
        key = _endpoint_request_key(
        selection, api_key, endpoint_api_key_env=endpoint_api_key_env
    )
        try:
            entries = _endpoint_catalog(
                endpoint_base(str(selection.model or "")), key
            )
        except _EndpointProbeError:
            # The caller named the model explicitly, so a probe failure only
            # costs the optional context window.
            return selection.model_name, None, ""
        return selection.model_name, _endpoint_context(entries, selection.model_name), ""
    query_model = endpoint_query_model(str(selection.model or ""))
    if query_model:
        return query_model, None, ""
    if not probe:
        return "default", None, ""
    key = _endpoint_request_key(
        selection, api_key, endpoint_api_key_env=endpoint_api_key_env
    )
    try:
        entries = _endpoint_catalog(endpoint_base(str(selection.model or "")), key)
    except _EndpointProbeError as exc:
        return "", None, str(exc)
    # A listing that repeats one id names one model, not an ambiguity.  Keep
    # the first-seen order so the error message (when there really are several)
    # stays deterministic.
    models = list(dict.fromkeys(str(entry.get("id")) for entry in entries))
    if len(models) == 1:
        return models[0], _endpoint_context(entries, models[0]), ""
    if len(models) > 1:
        return "", None, (
            "the endpoint lists %d models (%s%s); pass --agent-model-name to "
            "choose one" % (
                len(models),
                ", ".join(_diagnostic_model_id(model) for model in models[:8]),
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

    Example:
        ``deep_merge({"a":{"b":1}}, {"a":{"c":2}})`` yields
        ``{"a":{"b":1,"c":2}}``.

    Counterexample:
        A list or scalar override replaces the whole value; it is not appended or
        merged element-wise.  A non-mapping override or a cyclic object raises
        ``ConfigError`` instead of an ``AttributeError``/``RecursionError``.
    """
    if not isinstance(base, dict):
        raise ConfigError("the merge base must be a JSON object")
    if not isinstance(override, Mapping):
        raise ConfigError(
            "the merge override must be a JSON object, not %s"
            % type(override).__name__
        )
    return _deep_merge(base, override, set(), 0)


def _deep_merge(
    base: Dict[str, Any],
    override: Mapping[str, Any],
    active: set,
    depth: int,
) -> Dict[str, Any]:
    if depth > 64:
        raise ConfigError("the request preparation object is nested too deeply")
    marker = id(override)
    if marker in active:
        raise ConfigError("the request preparation object is cyclic")
    active.add(marker)
    try:
        for key, value in override.items():
            if isinstance(value, Mapping):
                if id(value) in active:
                    raise ConfigError("the request preparation object is cyclic")
                if isinstance(base.get(key), dict):
                    _deep_merge(base[key], value, active, depth + 1)
                else:
                    base[key] = value
            else:
                base[key] = value
    finally:
        active.discard(marker)
    return base


def _reject_json_constant(value: str) -> Any:
    """Reject the NaN/Infinity spellings the JSON decoder accepts by default."""
    raise ValueError("non-finite JSON number %s" % value)


def _finite_json_float(value: str) -> float:
    """Parse a JSON float, rejecting an exponent that overflows to infinity."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number %s" % value)
    return number


class _FrozenDict(dict):
    """A read-only ``dict`` that ``json.dumps`` can still serialise.

    ``MappingProxyType`` protects only its own level and is not a JSON type, so
    the resolved ``prep_json`` cannot use it: ``start_prep_proxy`` must be able
    to encode the validated object for the live request.  A dict subclass keeps
    the native encoding path while every mutation method raises.
    """

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("the resolved prep_json mapping is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable

    def __ior__(self, other: Any) -> "_FrozenDict":
        self._immutable()


class _FrozenList(list):
    """A read-only ``list`` that ``json.dumps`` can still serialise."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("the resolved prep_json list is read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    append = _immutable
    extend = _immutable
    insert = _immutable
    remove = _immutable
    pop = _immutable
    clear = _immutable
    sort = _immutable
    reverse = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable


def _freeze_prep(value: Any) -> Any:
    """A recursively frozen, deep-copied ``prep_json`` tree.

    The ``AgentSelection`` dataclass is frozen, so one consumer must not be
    able to change the validated request preparation through a nested value.
    The copy is also detached from the caller's object, so later mutation of
    the caller's mapping cannot alter the stored selection.
    """
    if isinstance(value, Mapping):
        return _FrozenDict(
            (key, _freeze_prep(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return _FrozenList(_freeze_prep(item) for item in value)
    return value


def parse_prep_json(value: Any) -> Dict[str, Any]:
    """A JSON object for ``--agent-model-prep-json`` (an empty value is ``{}``).

    Example:
        ``parse_prep_json('{"temperature":0}')`` returns
        ``{"temperature":0}``.

    Counterexample:
        A JSON array or invalid JSON raises ``ConfigError``; it never returns a
        non-object.
    """
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        # A caller can pass the object directly, bypassing the string parser's
        # guards.  Validate it with the same strict JSON contract (no NaN or
        # Infinity, no non-serialisable/cyclic value) before accepting it; the
        # value later goes through json.dumps for the live request.  Convert
        # any Mapping (MappingProxyType, a custom mapping, a frozen selection's
        # prep_json) to a concrete dict first, because json.dumps only knows
        # dict; a list of pairs is not a Mapping and stays rejected.
        try:
            value = dict(value)
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ConfigError(
                "--agent-model-prep-json is not valid JSON: %s" % exc
            ) from None
        return value
    try:
        # The default decoder accepts NaN/Infinity, which are not valid JSON
        # and would later be serialised into an invalid request body.
        parsed = json.loads(
            str(value),
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (ValueError, RecursionError) as exc:
        raise ConfigError("--agent-model-prep-json is not valid JSON: %s" % exc) from None
    if not isinstance(parsed, dict):
        raise ConfigError("--agent-model-prep-json must be a JSON object")
    return parsed


#: Same redirect refusal for the prep proxy; the explicit empty ProxyHandler
#: also keeps the caller's Authorization header off an environment proxy.
_PREP_OPENER = urllib.request.build_opener(
    _NoRedirectHandler, urllib.request.ProxyHandler({})
)


def _reject_dot_segments(path: str) -> None:
    """Reject ``.``/``..`` path segments before they escape the proxy prefix.

    A single ``unquote`` let a double-encoded ``%252e%252e`` through; the
    upstream may decode it once more and resolve ``..`` outside the prefix.
    Decode repeatedly until stable, then refuse any residual encoded dot,
    slash or backslash.
    """
    decoded = str(path or "")
    for _attempt in range(8):
        candidate = urllib.parse.unquote(decoded)
        if candidate == decoded:
            break
        decoded = candidate
    if any(segment in (".", "..") for segment in decoded.split("/")):
        raise ValueError(
            "request target must not contain '.' or '..' path segments"
        )
    if re.search(r"(?i)%2e|%2f|%5c", decoded):
        raise ValueError(
            "request target must not contain encoded '.' or path separators"
        )


def _append_query(target: str, extra: str) -> str:
    """Merge ``extra`` into ``target``'s query, preferring the client's value.

    Blindly appending produced ``?api-version=2023&api-version=2024``; a gateway
    may take either.  A duplicate key already present in the client request
    wins, and every other configured parameter is added.
    """
    if not extra:
        return target
    base, separator, existing = target.partition("?")
    if not separator:
        return target + "?" + extra
    pairs = _query_pairs(existing)
    client_keys = {key for key, _value in pairs}
    pairs += [
        (key, value) for key, value in _query_pairs(extra) if key not in client_keys
    ]
    return base + "?" + urllib.parse.urlencode(pairs, doseq=True)


class _PrepProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward DSH's OpenAI-compatible requests, merging the prep object."""

    protocol_version = "HTTP/1.0"

    def log_message(self, *args: Any) -> None:        # keep the adapter's stderr readable
        pass

    def _send_error(self, status: int, message: str) -> None:
        raw = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(raw)

    def _target(self) -> str:
        base = str(self.server.target_base).rstrip("/")  # type: ignore[attr-defined]
        path = self.path or "/"
        try:
            parsed_path = urllib.parse.urlsplit(path)
        except ValueError as exc:
            raise ValueError("invalid request target: %s" % exc) from None
        if parsed_path.scheme or parsed_path.netloc or path.startswith("//"):
            raise ValueError("absolute-form request targets are not supported")
        if not path.startswith("/"):
            raise ValueError("request target must start with '/'")
        # Compare only the path component: ``/v1?x=1`` is under /v1 even
        # though it does not start with ``/v1/``, and prefixing it as well
        # produced ``/v1/v1?x=1`` on the upstream.
        path_only = path.split("?", 1)[0]
        _reject_dot_segments(path_only)
        if path_only == "/v1" or path_only.startswith("/v1/"):
            path = path[3:]
        # http.client writes the request line as ASCII; a raw non-ASCII path
        # raised UnicodeEncodeError and killed the handler thread.  Percent
        # encode it, keeping existing escapes and reserved characters.
        target = base + urllib.parse.quote(path, safe="/?&=%:@!$'()*+,;~-._[]")
        query = str(getattr(self.server, "target_query", "") or "")
        return _append_query(target, query)

    def _forward(self, method: str, body: Optional[bytes] = None) -> None:
        headers = _forwardable_request_headers(
            self.headers, extra_skip=("accept-encoding",)
        )
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            target = self._target()
        except ValueError as exc:
            self._send_error(400, str(exc))
            return
        request = urllib.request.Request(
            target, data=body, headers=headers, method=method
        )
        try:
            response = _PREP_OPENER.open(request, timeout=600.0)
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                # urllib's default redirect handler would resend the request,
                # including the Authorization header, to the Location host and
                # a 302 would even turn the POST into a GET.  Refuse instead.
                raw = json.dumps(
                    {
                        "error": {
                            "message": "custom endpoint proxy refused a redirect "
                            "(HTTP %d); credentials are not forwarded to another "
                            "host" % exc.code
                        }
                    }
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    self.wfile.write(raw)
                with contextlib.suppress(OSError):
                    exc.close()
                return
            try:
                raw = exc.read()
            except (OSError, http.client.HTTPException) as read_error:
                # A truncated upstream error body makes ``read()`` raise
                # IncompleteRead.  Without this guard it escaped the handler,
                # closed the client connection and hid the upstream status.
                raw = json.dumps(
                    {
                        "error": {
                            "message": "custom endpoint proxy could not read "
                            "the upstream error body: %s" % read_error
                        }
                    }
                ).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                with contextlib.suppress(OSError):
                    self.wfile.write(raw)
                with contextlib.suppress(OSError):
                    exc.close()
                return
            self.send_response(exc.code)
            forwarded_headers = _forwardable_response_headers(
                exc.headers.items() if exc.headers else ()
            )
            for key, value in forwarded_headers:
                self.send_header(key, value)
            if not any(
                key.lower() == "content-type" for key, _value in forwarded_headers
            ):
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(raw)
            with contextlib.suppress(OSError):
                exc.close()
            return
        except ValueError as exc:
            # ``http.client`` encodes the request line as ASCII, so a non-ASCII
            # path raises UnicodeEncodeError (a ValueError).  The client must
            # get a 4xx, not an empty response and a dead handler thread.
            self._send_error(400, str(exc))
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
        response_header_items = list(response.headers.items())
        declared_length = next(
            (
                value
                for name, value in response_header_items
                if name.lower() == "content-length"
            ),
            None,
        )
        self.send_response(getattr(response, "status", 200))
        for key, value in _forwardable_response_headers(response_header_items):
            self.send_header(key, value)
        if declared_length is not None:
            # ``_forwardable_response_headers`` drops the framing header; re-emit
            # the upstream value so a truncated success body raises
            # IncompleteRead on the client instead of looking like a short 200.
            self.send_header("Content-Length", declared_length)
        self.end_headers()
        try:
            _relay_response_body(
                response,
                lambda chunk: (self.wfile.write(chunk), self.wfile.flush()),
            )
        finally:
            with contextlib.suppress(OSError):
                response.close()

    def do_GET(self) -> None:
        self._forward("GET")

    def _chunked(self) -> bool:
        return _request_is_chunked(self.headers)

    def do_POST(self) -> None:
        if self._chunked():
            # Reading only Content-Length silently forwarded a chunked POST
            # without its body.  Reject the unsupported framing (the retry
            # proxy does the same) instead of corrupting a valid request.
            raw = json.dumps(
                {
                    "error": {
                        "message": "chunked request bodies are not supported by "
                        "the custom endpoint proxy"
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(raw)
            return
        try:
            body = _read_request_body(self.rfile, self.headers) or b""
        except ValueError:
            # Do not treat an unparsable Content-Length as an empty body and
            # forward a corrupted request.
            raw = json.dumps(
                {"error": {"message": "invalid Content-Length header"}}
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(raw)
            return
        prep = getattr(self.server, "prep", None)  # type: ignore[attr-defined]
        if prep and body:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                deep_merge(payload, prep)
                try:
                    body = json.dumps(payload, allow_nan=False).encode("utf-8")
                except (TypeError, ValueError) as exc:
                    self._send_error(
                        400, "prep object is not JSON-serialisable: %s" % exc
                    )
                    return
        self._forward("POST", body)


#: The prep proxies live in the same process-wide registry as the retry
#: proxies, tagged ``prep``; ``stop_prep_proxy`` accepts one server so
#: concurrent verifier passes can stop only their own proxy.
_PROXY_BY_BASE: Dict[str, Any] = _SHARED_PROXY_REGISTRY.by_base


def start_prep_proxy(target_base: str, prep: Mapping[str, Any]) -> Tuple[str, str]:
    """``(base_url, error)`` for a local proxy that merges ``prep`` into POSTs.

    The returned base URL replaces the endpoint in the generated DSH patch; the
    proxy is otherwise transparent.  Only created when a prep object is present.

    Example:
        ``base, err = start_prep_proxy("https://host/v1", {"temperature":0})``
        returns a loopback base URL.

    Counterexample:
        An empty prep object still starts a proxy if called directly; callers
        should skip it when there is nothing to merge.  An empty or non-http(s)
        target returns ``("", error)`` and starts nothing.
    """
    target_text = str(target_base or "")
    try:
        _safe_target, target_secret = split_endpoint_credentials(target_text)
    except (TypeError, ValueError):
        _safe_target, target_secret = "", ""
    if target_secret:
        # ``endpoint_base`` strips userinfo for artifact safety; silently
        # starting an unauthenticated proxy hides the lost credential until the
        # first 401.  Reject it like RetryProxy does.
        return "", (
            "target_base must not carry a credential; pass it in the "
            "forwarded Authorization header instead"
        )
    try:
        normalized_target = endpoint_base(target_text)
    except ConfigError as exc:
        return "", str(exc)
    # ``endpoint_base`` deliberately drops the query for artifact safety.  A
    # gateway can require a non-secret query parameter (``api-version=2024``),
    # so keep the non-secret part for the forwarded request only.
    target_query = ""
    try:
        parsed_target = urllib.parse.urlsplit(target_text)
        if parsed_target.query:
            pairs = urllib.parse.parse_qsl(
                parsed_target.query, keep_blank_values=True
            )
            kept = [
                (key, value)
                for key, value in pairs
                if not _secret_query_key(key)
            ]
            target_query = urllib.parse.urlencode(kept, doseq=True)
    except ValueError:
        target_query = ""
    try:
        prep_object = dict(prep)
        # Fail at startup, not inside the first request's handler thread.
        # allow_nan=False rejects a directly supplied NaN/Infinity too.
        json.dumps(prep_object, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return "", "prep object is not JSON-serialisable: %s" % exc
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PrepProxyHandler)
    except OSError as exc:
        return "", "%s: %s" % (type(exc).__name__, exc)
    server.target_base = normalized_target                  # type: ignore[attr-defined]
    server.target_query = target_query                     # type: ignore[attr-defined]
    server.prep = prep_object                              # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server._gdp_thread = thread                           # type: ignore[attr-defined]
    base = "http://127.0.0.1:%s/v1" % server.server_address[1]
    _SHARED_PROXY_REGISTRY.add(server, kind="prep", base=base)
    return base, ""


def stop_prep_proxy(server: Any = None) -> None:
    """Stop one prep proxy, or every prep proxy this process started when omitted.

    ``server`` may be the base URL returned by :func:`start_prep_proxy`; parallel
    verifier passes each stop only their own proxy.

    Example:
        ``stop_prep_proxy(base)`` stops the proxy for that base URL;
        ``stop_prep_proxy()`` stops every prep proxy this process started.

    Counterexample:
        Stopping an unknown base is a no-op; it does not raise.
    """
    if isinstance(server, str):
        # An unknown base URL is not "all proxies": the docstring's no-op
        # counterexample.  Only an explicit None means stop all prep proxies.
        targets = _SHARED_PROXY_REGISTRY.targets(base=server, kind="prep")
    elif server is not None:
        targets = _SHARED_PROXY_REGISTRY.targets(item=server)
    else:
        targets = _SHARED_PROXY_REGISTRY.targets(kind="prep")
    for item in targets:
        if item is None:
            continue
        thread = getattr(item, "_gdp_thread", None)
        with contextlib.suppress(OSError):
            item.shutdown()
        with contextlib.suppress(OSError):
            item.server_close()
        # ``start_prep_proxy`` stores the serving thread; without the join a
        # caller cannot wait for the handler thread before reusing state.  The
        # retry proxy's ``stop`` already joins.
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)


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

    Example:
        A Dots selection returns its upstream base URL.

    Counterexample:
        A non-Dots selection returns ``""``; there is no proxy target to start.
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
    """Start the retrying proxy for the dots compaction route.

    Example:
        ``start_compaction_proxy("https://dots/v1")`` returns ``(proxy, base)`` or
        ``(None, "")`` on failure.

    Counterexample:
        It always uses the dedicated compaction name and retry policy; do not use
        it as a general-purpose proxy for other providers.
    """
    return start_retry_proxy(
        target_base,
        RETRY_SCHEDULE,
        name="gdp-compaction",
        log=log_callback,
        kind="compaction",
    )


def stop_compaction_proxy(proxy: Any = None) -> None:
    """Stop one compaction proxy, or every one this process started.

    Example:
        ``stop_compaction_proxy(proxy)`` stops one; ``stop_compaction_proxy()``
        stops all.

    Counterexample:
        A ``None`` argument means "all compaction proxies", not "none"; an
        unrelated proxy started with ``start_retry_proxy()`` is not touched.
    """
    if proxy is None:
        stop_retry_proxy(kind="compaction")
    else:
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


def _env_base_url(name: str, default: str) -> Tuple[str, str]:
    """(credential-free base URL, extracted secret) for an environment URL.

    The generated patch/argv/environment are persisted artifacts, so a
    ``PAPER_*_BASE_URL`` that carries userinfo or a credential query parameter
    must not reach them verbatim.  The secret is returned separately and folded
    into the provider key instead.
    """
    base, secret = split_endpoint_credentials(_env(name, default))
    return base.rstrip("/"), secret


def _normalized_family(selection: AgentSelection) -> str:
    if selection.is_endpoint:
        return "endpoint"
    return str((AGENT_MODELS.get(selection.model or "") or {}).get("family") or "")


def _effective_selection(
    selection: AgentSelection, raw_model: Optional[str]
) -> AgentSelection:
    """The selection a raw ``--model`` names for the env/record builders.

    ``dsh_patch_text``/``codex_provider_args``/``claude_environment`` infer a
    family from a raw model when the selection is the ``default`` sentinel, and
    treat a raw endpoint URL as an endpoint route.  ``selection_env`` and
    ``to_record`` need the same inference or they report/export no credential
    for a shape the patch configures.
    """
    if not raw_model:
        return selection
    text = str(raw_model)
    if agent_model_is_endpoint(text):
        return AgentSelection(
            harness=selection.harness,
            model=text,
            legacy=selection.legacy,
            api_key=selection.api_key,
            model_name=selection.model_name,
            prep_json=selection.prep_json,
        )
    if selection.model is not None and not selection.is_default:
        return selection
    canonical = normalize_agent_model(text)
    if canonical in AGENT_MODELS:
        return AgentSelection(
            harness=selection.harness,
            model=canonical,
            legacy=selection.legacy,
            api_key=selection.api_key,
            model_name=selection.model_name,
            prep_json=selection.prep_json,
        )
    return selection


def _provider_family_settings(selection: AgentSelection) -> Dict[str, Any]:
    """The endpoint, key and limits a provider family needs at request time."""
    override = reasoning_effort_override()
    if selection.is_default:
        return {
            "family": "default",
            "base_url": "",
            "api_key": None,
            "api_key_env": None,
            "model": None,
            "context_window": None,
            "max_tokens": None,
            "effort": override,
        }
    family = _normalized_family(selection)
    if family == "endpoint":
        safe_url, url_secret = split_endpoint_credentials(
            str(selection.model or "")
        )
        base = endpoint_base(safe_url)
        # With REJECT_ENDPOINT_CREDENTIALS disabled the caller keeps userinfo or
        # a credential query in the live URL and expects the request to use it;
        # the patch stays credential-free.  Split it out here so the key reaches
        # selection_env/the forwarded header instead of being silently dropped.
        return {
            "family": family,
            "base_url": base,
            "api_key": (
                selection.api_key
                or _env(AGENT_MODEL_API_KEY_ENV)
                or url_secret
            ),
            "api_key_env": AGENT_MODEL_API_KEY_ENV,
            "model": selection.effective_model_id() or "default",
            "context_window": _int_env(
                "PAPER_AGENT_CONTEXT_WINDOW", DEFAULT_ENDPOINT_CONTEXT
            ),
            "max_tokens": _int_env(
                "PAPER_AGENT_MAX_TOKENS", DEFAULT_ENDPOINT_MAX_TOKENS
            ),
            # A custom endpoint has no catalog default, but an explicit
            # operator override still applies; normalize_reasoning_effort()
            # passes a custom endpoint's level through unchanged.
            "effort": override,
        }
    if family == "qwen":
        base_url, base_secret = _env_base_url(
            "PAPER_QWEN_BASE_URL", QWEN_DEFAULT_BASE_URL
        )
        if not _env("PAPER_QWEN_BASE_URL"):
            base_secret = _CATALOG_BASE_SECRETS.get("qwen38-27b") or base_secret
        key_env = str(
            model_fact("qwen38-27b", "api_key_env") or "PAPER_QWEN_API_KEY"
        )
        return {
            "family": family,
            "provider": model_fact("qwen38-27b", "dsh_provider") or "qwen",
            "base_url": base_url,
            "api_key": _env(key_env) or base_secret or QWEN_DEFAULT_API_KEY,
            "api_key_env": key_env,
            "model": _env("PAPER_QWEN_MODEL", QWEN_DEFAULT_MODEL),
            "context_window": _int_env(
                "PAPER_QWEN_CONTEXT_WINDOW", QWEN_CONTEXT_WINDOW
            ),
            "max_tokens": _int_env("PAPER_QWEN_MAX_TOKENS", QWEN_MAX_TOKENS),
            "effort": normalize_reasoning_effort(
                "qwen38-27b", override or QWEN_DEFAULT_EFFORT
            ),
        }
    if family == "dots":
        compaction = dots_compaction(LLM_POLICY)
        regular_env = str(
            model_fact("dots3-note-preview", "api_key_env")
            or compaction.get("regular_api_key_env")
            or "DOTS_API_KEY"
        )
        compaction_key = _env(compaction["compaction_api_key_env"])
        for alias in compaction.get("compaction_api_key_aliases") or ():
            if not compaction_key:
                compaction_key = _env(str(alias))
        base_url, base_secret = _env_base_url(
            "PAPER_DOTS_BASE_URL", DOTS_DEFAULT_BASE_URL
        )
        if not _env("PAPER_DOTS_BASE_URL"):
            base_secret = (
                _CATALOG_BASE_SECRETS.get("dots3-note-preview") or base_secret
            )
        return {
            "family": family,
            "provider": model_fact("dots3-note-preview", "dsh_provider") or "dots",
            "base_url": base_url,
            "api_key": (
                _env("PAPER_DOTS_API_KEY") or _env(regular_env) or base_secret
            ),
            "api_key_env": regular_env,
            "compaction_api_key": compaction_key,
            "compaction_api_key_env": compaction["compaction_api_key_env"],
            "model": _env("PAPER_DOTS_MODEL", DOTS_DEFAULT_MODEL),
            "context_window": _int_env(
                "PAPER_DOTS_CONTEXT_WINDOW", DOTS_CONTEXT_WINDOW
            ),
            "max_tokens": _int_env("PAPER_DOTS_MAX_TOKENS", DOTS_MAX_TOKENS),
            "effort": normalize_reasoning_effort(
                "dots3-note-preview", override or DOTS_DEFAULT_EFFORT
            ),
        }
    if family == "deepseek":
        info = selection.model_info()
        return {
            "family": family,
            "provider": info.get("dsh_provider") or "deepseek-official",
            "base_url": "",
            "api_key": _env("DEEPSEEK_API_KEY") or _env("PAPER_DEEPSEEK_API_KEY"),
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": selection.effective_model_id() or "deepseek-flash",
            # The shared catalog overlays AGENT_MODELS; read the capacity there
            # instead of duplicating it, so a catalog update cannot leave the
            # DeepSeek family on the old window.
            "context_window": info.get("context_window", DEEPSEEK_CONTEXT_WINDOW),
            "max_tokens": info.get("max_tokens", DEEPSEEK_MAX_TOKENS),
            # The catalog's default is named explicitly rather than left to
            # the adapter's own fallback, so a catalog revision is the only
            # place this level changes.
            "effort": normalize_reasoning_effort(
                "deepseek-flash", override or DEEPSEEK_DEFAULT_EFFORT
            ),
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
        # GPT and Claude keep the model's own default unless the operator
        # explicitly overrides it; then the harness receives that level.
        "effort": override,
    }


def selection_env(
    selection: Optional[AgentSelection],
    *,
    endpoint_api_key_env: Optional[str] = None,
    raw_model: Optional[str] = None,
) -> Dict[str, str]:
    """Credential environment needed by the selected provider.

    Values are injected into the child process only; they are never written to
    ``command.json`` or the run log.  dsh uses its own credential store for the
    native DeepSeek route, so only the adapter-backed families are listed.  An
    explicit reasoning-effort override is carried too, so a nested adapter that
    is handed this environment cannot fall back to the catalog default.

    ``endpoint_api_key_env`` must be the same name a caller passed to
    :func:`dsh_patch_text`; otherwise the generated provider config would read a
    variable this function never exports.  Pass the same ``raw_model`` too: a
    ``default`` sentinel plus a raw catalog model gets its provider's key
    environment from :func:`dsh_patch_text`, and this function must export it.

    Example:
        A Dots selection returns ``DOTS_API_KEY`` and the compaction key variable
        when configured.  An endpoint selection whose patch uses
        ``endpoint_api_key_env="OTHER_KEY"`` exports ``OTHER_KEY``.

    Counterexample:
        These values are child-process environment only; never persist this dict in
        ``command.json``.
    """
    env: Dict[str, str] = {}
    override = reasoning_effort_override()
    if override:
        # A consumer that builds a clean child environment carries the same
        # explicit level the parent process applies, so a nested adapter cannot
        # silently fall back to the catalog default.
        env[AGENT_REASONING_EFFORT_ENV] = override
    if selection is None or selection.model is None:
        return env
    settings = _provider_family_settings(
        _effective_selection(selection, raw_model)
    )
    family = settings["family"]
    if family == "endpoint":
        # The caller named the variable the generated patch reads.  A real key
        # may already live only in that variable (``OTHER_KEY=K``), so read it
        # before falling back to the local-route dummy; overwriting it with
        # "dummy" silently dropped the credential.
        key = settings["api_key"]
        if not key and endpoint_api_key_env:
            key = _env(endpoint_api_key_env)
        env[endpoint_api_key_env or AGENT_MODEL_API_KEY_ENV] = key or "dummy"
    elif family == "deepseek":
        if settings["api_key"]:
            env.setdefault("DEEPSEEK_API_KEY", str(settings["api_key"]))
    elif family == "qwen":
        env[str(settings["api_key_env"])] = str(
            settings["api_key"] or QWEN_DEFAULT_API_KEY
        )
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
    selection: Optional[AgentSelection],
    *,
    dry_run: bool = False,
    raw_model: Optional[str] = None,
    endpoint_api_key_env: Optional[str] = None,
) -> str:
    """A configuration problem that would fail the first real request ('' = none).

    Example:
        A Dots selection without ``DOTS_API_KEY`` returns a human-readable problem.

    Counterexample:
        ``dry_run=True`` returns ``""`` by design; absence of a problem in dry-run
        does not mean the live request will work.  Pass the same ``raw_model``
        given to :func:`dsh_patch_text` when the selection is the ``default``
        sentinel, or the preflight inspects the wrong family.
    """
    if selection is None or selection.model is None:
        return ""
    if dry_run:
        # Do not evaluate the request-side settings at all: an invalid operator
        # override must not turn the documented dry-run no-op into an exception.
        return ""
    effective = _effective_selection(selection, raw_model)
    try:
        settings = _provider_family_settings(effective)
    except ConfigError as exc:
        # The documented contract is a diagnostic string, so an invalid
        # reasoning override must not escape as an exception.
        return str(exc)
    family = settings["family"]
    if family == "endpoint":
        # Run the same endpoint resolution the live request uses, so an
        # ambiguous listing (or a failed /models probe) is reported here
        # instead of surfacing as a plausible but wrong ``default`` model.
        _model_name, _context_window, error = resolve_endpoint_model(
            effective, endpoint_api_key_env=endpoint_api_key_env
        )
        return error
    # Validate the same route capacities/base URL the generated patch enforces;
    # selection_problem used to return "" for a value dsh_patch_text rejected.
    try:
        dsh_patch_text(effective)
    except ConfigError as exc:
        return str(exc)
    if family == "deepseek" and effective.harness in (
        AGENT_HARNESS_GPT_CODEX,
        AGENT_HARNESS_CLAUDE_CODE,
    ):
        if not settings["api_key"]:
            return (
                "DEEPSEEK_API_KEY is not set; %s/deepseek-flash needs the "
                "DeepSeek Responses/Anthropic route credential. Export "
                "DEEPSEEK_API_KEY (or PAPER_DEEPSEEK_API_KEY)."
                % effective.harness
            )
    if family == "dots" and not settings["api_key"]:
        return (
            "DOTS_API_KEY is not set; dots3-note-preview needs the Dots API "
            "key. Export DOTS_API_KEY (or PAPER_DOTS_API_KEY)."
        )
    return ""


def _resolve_token_capacities(
    settings: Mapping[str, Any],
    context_window_override: Optional[int],
    max_tokens_override: Optional[int],
    default_context: int,
    default_max_tokens: int,
) -> Tuple[int, int]:
    """Validated ``(context_window, max_tokens)`` for a generated patch.

    ``None`` means "use the catalog/default"; an explicit zero or negative
    value is rejected instead of being silently treated as unset, and a max
    that does not fit inside the context window is rejected too.
    """
    if context_window_override is not None and type(context_window_override) is not int:
        # ``int(True)`` is 1 and ``int(150000.9)`` truncates; the catalog and
        # the checker require an exact JSON integer, so the override path must
        # not silently accept a different type.
        raise ConfigError("the context window override must be an integer")
    if max_tokens_override is not None and type(max_tokens_override) is not int:
        raise ConfigError("the max tokens override must be an integer")
    context = (
        settings.get("context_window")
        if context_window_override is None
        else context_window_override
    )
    max_tokens = (
        settings.get("max_tokens")
        if max_tokens_override is None
        else max_tokens_override
    )
    if context is None:
        context = default_context
    if max_tokens is None:
        max_tokens = default_max_tokens
    try:
        context = int(context)
        max_tokens = int(max_tokens)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(
            "the context window and max tokens must be integers"
        ) from None
    if context <= 0:
        raise ConfigError(
            "the context window must be positive, not %d" % context
        )
    if max_tokens <= 0:
        raise ConfigError("max tokens must be positive, not %d" % max_tokens)
    if max_tokens >= context:
        raise ConfigError(
            "max tokens %d must be below the context window %d"
            % (max_tokens, context)
        )
    return context, max_tokens


def _normalized_patch_base_url(value: Any) -> str:
    """A credential-free, chat-completions-normalized provider base URL.

    A gateway may require a non-secret query parameter (``api-version=2024``);
    ``endpoint_base`` drops the whole query for artifact safety, so the
    non-secret part is re-attached here.  Every generated route uses this
    helper, so qwen/dots/endpoint/override URLs behave the same way.
    """
    text = str(value or "")
    safe = endpoint_base(text)
    query = _non_secret_query(text)
    return safe + ("?" + query if query else "")


def _non_secret_query(url: Any) -> str:
    """The non-credential query of ``url``, in its original spelling order."""
    try:
        parsed = urllib.parse.urlsplit(str(url or ""))
    except ValueError:
        return ""
    kept = [
        (key, value)
        for key, value in _query_pairs(parsed.query)
        if not _secret_query_key(key)
    ]
    return urllib.parse.urlencode(kept, doseq=True) if kept else ""


def _is_loopback_url(value: Any) -> bool:
    """Whether ``value`` names a loopback host (the retry-proxy shape)."""
    try:
        host = urllib.parse.urlsplit(str(value or "")).hostname
    except ValueError:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def _normalized_route_base_url(value: Any) -> str:
    """Normalize a route base URL, tolerating trailing-slash legacy forms.

    ``dsh_patch_text`` uses :func:`endpoint_base` and fails on a malformed URL.
    The Codex and Claude routes historically accepted the rstrip-only value and
    kept a ``/chat/completions`` suffix; normalize it here so all three routes
    agree.

    A value with no host (for example the credential-free ``host/v1`` left
    behind by scrubbing ``user:pw@host/v1``) is not a usable endpoint either;
    exporting it made every child request fail, so it raises the same
    ``ConfigError`` as :func:`endpoint_base` instead of returning a bare host.
    """
    text = str(value or "")
    if not text:
        # A native route (DeepSeek) has no base URL; "" is "no override", not
        # an invalid endpoint.
        return ""
    try:
        normalized = endpoint_base(text)
    except ConfigError:
        try:
            parsed = urllib.parse.urlsplit(text)
        except ValueError:
            # Unparsable legacy forms (already credential-free) stay tolerated.
            return text.rstrip("/")
        if not parsed.netloc:
            raise
        if parsed.scheme not in ("http", "https"):
            raise
        # Drop the query here: the shared tail below re-attaches the
        # non-secret query exactly once (the old text.rstrip("/") kept it and
        # the tail appended it a second time).
        normalized = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", "")
        ).rstrip("/")
    query = _non_secret_query(text)
    return normalized + ("?" + query if query else "")


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
    compaction_is_proxy: Optional[bool] = None,
) -> str:
    """The ``dsh --patch`` overlay that selects or configures the provider.

    An empty string means "no patch": the harness profile's own
    ``agent-default-model`` selects the model.  A low-level ``--model`` still has to reach the
    harness with the ``default`` sentinel; only the bare sentinel owns no model at all.  A
    provider-only override has no profile model id available here, and a patch layer replaces
    the whole entry config, so it must not pin the sentinel as the provider's model either.

    ``endpoint_api_key_env`` is the ``apiKeyEnv`` the generated endpoint patch reads; a caller
    that uses it must pass the same name to :func:`selection_env`.

    Example:
        A Dots selection returns the ``llm-pi-ai`` provider patch plus the
        compaction route.

    Counterexample:
        ``selection.is_default`` with no raw model returns ``""`` (no patch); do
        not write an empty patch file.
    """
    if (selection.model is None or selection.is_default) and not raw_model:
        # The CLI owns the model; the harness profile's own entry selects it.
        return ""
    settings = _provider_family_settings(selection)
    family = settings["family"]
    raw_info: Dict[str, Any] = {}
    raw_endpoint = bool(raw_model) and agent_model_is_endpoint(str(raw_model))
    if raw_model and (
        selection.is_default
        or selection.model is None
        or agent_model_is_endpoint(str(raw_model))
    ):
        # The sentinel/legacy shape has no catalog entry of its own, but a raw
        # model still names its family.  A raw endpoint URL always names the
        # endpoint route, even on a named selection; otherwise it was emitted
        # as that provider's model id.  Resolve the family *before* the
        # per-family builders so a raw qwen/dots/deepseek model gets the same
        # provider block (base URL, key env, limits, effort) as a named one.
        raw_info = AGENT_MODELS.get(normalize_agent_model(raw_model)) or {}
        if raw_info:
            inferred = AgentSelection(
                harness=selection.harness,
                model=normalize_agent_model(raw_model),
                api_key=selection.api_key,
                prep_json=selection.prep_json,
            )
            settings = _provider_family_settings(inferred)
            family = settings["family"]
        elif raw_endpoint:
            # An http(s) raw --model is a custom endpoint, not a provider model
            # id.  Without this the final branch pointed DeepSeek at the URL
            # and dropped the base URL entirely.
            selection = AgentSelection(
                harness=selection.harness,
                model=str(raw_model),
                legacy=selection.legacy,
                api_key=selection.api_key,
                model_name=selection.model_name,
                prep_json=selection.prep_json,
            )
            settings = _provider_family_settings(selection)
            family = settings["family"]
    if family == "endpoint":
        # When the raw model *is* the endpoint URL, the provider model id must
        # come from --agent-model-name/?model=/default, not from the URL.
        model_name = (
            (None if raw_endpoint else raw_model)
            or model_name_override
            or selection.model_name
            or endpoint_query_model(str(selection.model or ""))
            or "default"
        )
        model_name = _redacted_endpoint_text(str(model_name)) or "default"
        # The patch is a credential-free artifact: userinfo and a query string
        # belong to the live request, never to a file an agent can read.
        base_url = _normalized_patch_base_url(
            base_url_override or settings["base_url"]
        )
        if base_url_override is None:
            # A gateway can require a non-secret query parameter
            # (``api-version=2024``); endpoint_base drops the whole query for
            # artifact safety, and the prep proxy is only started when a prep
            # object exists.  Keep the non-secret part in the generated base.
            query = _non_secret_query(selection.model)
            if query and "?" not in base_url:
                base_url += "?" + query
        context_window, max_tokens = _resolve_token_capacities(
            settings,
            context_window_override,
            max_tokens_override,
            DEFAULT_ENDPOINT_CONTEXT,
            DEFAULT_ENDPOINT_MAX_TOKENS,
        )
        provider = provider_override or "endpoint"
        # A custom endpoint has no catalog default, so without an operator
        # override the patch keeps effort selection off.  An explicit override
        # (argument or environment) is passed through (the endpoint's harness
        # validates the vocabulary), matching the documented "overrides every
        # family" contract instead of carrying the level in the child
        # environment while ignoring it here.
        endpoint_effort = effort_override or settings["effort"]
        if endpoint_effort:
            # Validate the explicit override against the one documented
            # vocabulary; an endpoint's level otherwise passed straight into
            # ``reasoningEffort: "turbo"``.
            endpoint_effort = normalize_reasoning_effort("", str(endpoint_effort))
        if endpoint_effort:
            effort_declaration = (
                "            reasoningEfforts:\n"
                # Quoted: an unquoted `off` is a YAML 1.1 boolean.
                '              "off":\n'
                "              minimal: minimal\n"
                "              low: low\n"
                "              medium: medium\n"
                "              high: high\n"
                "              xhigh: xhigh\n"
                "              max: max\n"
            )
            effort_line = "    reasoningEffort: %s\n" % json.dumps(
                # A custom endpoint's level passes through unchanged (the
                # environment override was already validated by
                # reasoning_effort_override()); normalizing by model_name could
                # misclassify a catalog-shaped id as another family.
                str(endpoint_effort)
            )
        else:
            effort_declaration = "            reasoningEfforts: false\n"
            effort_line = ""
        return (
            (
                "# Generated by generic_debug_pipeline for this stage: --agent-model.\n"
                "- id: llm-pi-ai\n"
                "  config:\n"
                "    providers:\n"
                "      %s:\n"
            '        displayName: "Custom API endpoint"\n'
            "        apiKeyEnv: %s\n"
            "        api: openai-completions\n"
            "        baseURL: %s\n"
            "        models:\n"
            "          - id: %s\n"
            "            name: %s\n"
            "            contextWindow: %s\n"
            "            maxTokens: %s\n"
            "%s"
            "        compat:\n"
            "          maxTokensField: max_tokens\n"
            "          supportsStore: false\n"
            + retry_policy_yaml(LLM_POLICY, "        ")
            + "- id: agent-default-model\n"
            "  config:\n"
            "    provider: %s\n"
            "    model: %s\n"
            "%s"
            )
            % (
                _yaml_scalar(provider),
                _yaml_scalar(endpoint_api_key_env or AGENT_MODEL_API_KEY_ENV),
                json.dumps(base_url),
                json.dumps(model_name),
                json.dumps(model_name),
                context_window,
                max_tokens,
                effort_declaration,
                _yaml_scalar(provider),
                json.dumps(model_name),
                effort_line,
            )
        )
    if family == "qwen":
        model = _redacted_endpoint_text(raw_model) or _redacted_endpoint_text(
            settings["model"]
        )
        provider = provider_override or settings.get("provider") or "qwen"
        base_url = _normalized_patch_base_url(
            base_url_override or settings["base_url"]
        )
        context_window, max_tokens = _resolve_token_capacities(
            settings,
            context_window_override,
            max_tokens_override,
            QWEN_CONTEXT_WINDOW,
            QWEN_MAX_TOKENS,
        )
        return (
            (
                "# Generated by generic_debug_pipeline for this stage: --agent-model qwen38-27b.\n"
            "- id: llm-pi-ai\n"
            "  config:\n"
            "    providers:\n"
            "      %s:\n"
            "        displayName: %s\n"
            "        apiKeyEnv: %s\n"
            "        api: openai-completions\n"
            "        baseURL: %s\n"
            "        models:\n"
            "          - id: %s\n"
            '            name: "Qwen3.8-27B"\n'
            "            contextWindow: %s\n"
            "            maxTokens: %s\n"
            "            reasoningEfforts:\n"
            # Quoted: an unquoted `off` is a YAML 1.1 boolean, not the level name.
            "              \"off\":\n"
            "              low: low\n"
            "              medium: medium\n"
            "              xhigh: xhigh\n"
            + retry_policy_yaml(LLM_POLICY, "        ")
            + "- id: agent-default-model\n"
            "  config:\n"
            "    provider: %s\n"
            "    model: %s\n"
            "    reasoningEffort: %s\n"
            )
            % (
                _yaml_scalar(provider),
                json.dumps("Qwen vLLM (%s)" % model),
                _yaml_scalar(settings["api_key_env"]),
                json.dumps(base_url),
                json.dumps(model),
                context_window,
                max_tokens,
                _yaml_scalar(provider),
                json.dumps(model),
                json.dumps(
                    normalize_reasoning_effort(
                        "qwen38-27b",
                        effort_override
                        or settings["effort"]
                        or QWEN_DEFAULT_EFFORT,
                    )
                ),
            )
        )
    if family == "dots":
        model = _redacted_endpoint_text(raw_model) or _redacted_endpoint_text(
            settings["model"]
        )
        base_url = _normalized_patch_base_url(
            base_url_override or settings["base_url"]
        )
        compaction_base = (
            _normalized_patch_base_url(compaction_base_url)
            if compaction_base_url
            else None
        )
        # A direct compaction URL keeps the retry ladder; only the loopback
        # retry proxy owns the retries.  The caller may state that explicitly.
        proxy_in_path = (
            compaction_is_proxy
            if compaction_is_proxy is not None
            else _is_loopback_url(compaction_base)
        )
        context_window, max_tokens = _resolve_token_capacities(
            settings,
            context_window_override,
            max_tokens_override,
            DOTS_CONTEXT_WINDOW,
            DOTS_MAX_TOKENS,
        )
        return dots_dsh_overlay(
            LLM_POLICY,
            model=model,
            provider=provider_override or settings.get("provider") or "dots",
            base_url=base_url,
            context_window=context_window,
            max_tokens=max_tokens,
            regular_api_key_env=settings["api_key_env"],
            compaction_base_url=compaction_base,
            # The proxy owns the retries only when the caller started one; a
            # direct compaction URL keeps its own retryPolicy.
            compaction_is_proxy=bool(proxy_in_path),
            reasoning_effort=str(
                effort_override or settings["effort"] or DOTS_DEFAULT_EFFORT
            ),
        )
    if family == "deepseek":
        model = (
            _redacted_endpoint_text(raw_model)
            or selection.effective_model_id()
            or settings["model"]
        )
        effort = normalize_reasoning_effort(
            "deepseek-flash",
            effort_override or settings["effort"] or DEEPSEEK_DEFAULT_EFFORT,
        )
        # The native DeepSeek adapter serves ``high`` when the request names no
        # effort, so the catalog's default has to be written into the patch.
        return (
            "# Generated by generic_debug_pipeline for this stage: "
            "--agent-model deepseek-flash.\n"
            "# A patch layer replaces the whole entry config, so the provider route is repeated.\n"
            "- id: agent-default-model\n"
            "  config:\n"
            "    provider: %s\n"
            "    model: %s\n"
            "    reasoningEffort: %s\n"
            % (
                json.dumps(
                    provider_override
                    or settings.get("provider")
                    or "deepseek-official"
                ),
                json.dumps(model),
                json.dumps(effort),
            )
        )
    info = dict(raw_info) if raw_info else selection.model_info()
    model = _redacted_endpoint_text(
        raw_model or selection.effective_model_id() or selection.model or ""
    )
    provider = (
        provider_override
        or info.get("dsh_provider")
        or ("openai" if family == "gpt" else "deepseek-official")
    )
    # GPT and Claude keep the model's own default unless the operator named a
    # level; naming it here is the one way the shared override reaches them.
    lines = [
        "# Generated by generic_debug_pipeline for this stage: --agent-model.",
        "# A patch layer replaces the whole entry config, so the provider route is repeated.",
        "- id: agent-default-model",
        "  config:",
        "    provider: %s" % json.dumps(provider),
        "    model: %s" % json.dumps(model),
    ]
    effort = effort_override or settings["effort"]
    if effort:
        lines.append(
            "    reasoningEffort: %s"
            % json.dumps(normalize_reasoning_effort(model, str(effort)))
        )
    return "\n".join(lines) + "\n"


def codex_provider_args(
    selection: AgentSelection, *, raw_model: Optional[str] = None
) -> List[str]:
    """The ``-c`` overrides that select the provider for ``codex exec``.

    Example:
        A Qwen selection returns ``-c`` provider arguments for ``codex exec``.

    Counterexample:
        A custom endpoint raises ``ConfigError``; Codex cannot speak the
        chat-completions URL directly.
    """
    if raw_model and agent_model_is_endpoint(str(raw_model)):
        # A raw http(s) ``--model`` is an endpoint, never a provider model id,
        # even when the selection names another model.  Emitting the URL as
        # ``model="https://host/v1"`` made Codex send a nonsense model id.
        raise ConfigError(
            "a custom --agent-model endpoint cannot run on gpt-codex; use dsh"
        )
    if selection.model is None or selection.is_default:
        inferred = None
        if raw_model and agent_model_is_endpoint(str(raw_model)):
            # A custom endpoint cannot run on the Codex route.  Returning []
            # here let Codex silently keep its own configured model; the named
            # endpoint branch below already raises the same ConfigError.
            raise ConfigError(
                "a custom --agent-model endpoint cannot run on gpt-codex; use dsh"
            )
        if raw_model:
            raw_info = AGENT_MODELS.get(normalize_agent_model(raw_model)) or {}
            if raw_info:
                # A raw model names its family even when the selection is the
                # sentinel/legacy shape; build the same provider args a named
                # selection would.
                inferred = AgentSelection(
                    harness=selection.harness,
                    model=normalize_agent_model(raw_model),
                    api_key=selection.api_key,
                    prep_json=selection.prep_json,
                )
        if inferred is None:
            # The CLI owns the model, but an explicit level still reaches codex:
            # `-c model_reasoning_effort` does not depend on the model id.
            effort = reasoning_effort_override()
            return (
                ["-c", _codex_config("model_reasoning_effort", effort)]
                if effort
                else []
            )
        selection = inferred
    settings = _provider_family_settings(selection)
    family = settings["family"]
    if family == "claude":
        # Codex has no Claude builder.  Returning [] meant the operator's
        # selection never reached the child and no record showed the mismatch;
        # a directly constructed Claude selection has no raw_model either, so
        # the guard must not depend on it.
        raise ConfigError("a Claude model cannot run on gpt-codex; use claude-code")
    # The argv is world-readable, so a credential-bearing URL must never be
    # emitted verbatim; the DSH route already redacts the same raw_model.
    model = (
        _redacted_endpoint_text(raw_model)
        or selection.effective_model_id()
        or settings["model"]
    )
    if family == "endpoint":
        raise ConfigError(
            "a custom --agent-model endpoint cannot run on gpt-codex; use dsh"
        )
    if family == "gpt":
        # The model choice owns the provider too: a Codex config repointed at
        # another provider must not redirect a GPT model.
        values = [_codex_config("model_provider", "openai")]
        if model:
            values.append(_codex_config("model", model))
        if settings["effort"]:
            values.append(
                _codex_config("model_reasoning_effort", settings["effort"])
            )
        return [item for pair in (["-c", value] for value in values) for item in pair]
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
                "model_providers.deepseek.env_key", settings["api_key_env"]
            ),
            _codex_config("model_providers.deepseek.wire_api", "responses"),
            _codex_config("web_search", "disabled"),
            _codex_config(
                "model_context_window",
                int(settings["context_window"] or DEEPSEEK_CONTEXT_WINDOW),
            ),
            _codex_config(
                "model_reasoning_effort",
                settings["effort"] or DEEPSEEK_DEFAULT_EFFORT,
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
                "model_providers.qwen.base_url",
                _normalized_route_base_url(settings["base_url"]),
            ),
            _codex_config(
                "model_providers.qwen.env_key", settings["api_key_env"]
            ),
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
                "model_providers.dots.base_url",
                _normalized_route_base_url(settings["base_url"]),
            ),
            _codex_config("model_providers.dots.env_key", settings["api_key_env"]),
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
    """The ``--model`` value for Claude Code (``None`` = its configured model).

    Example:
        A Claude selection returns its alias/id; a DeepSeek/Qwen/Dots selection
        returns the raw override or ``None``.

    Counterexample:
        ``None`` means "let Claude Code use its configured model"; it is not an
        error.  A raw http(s) endpoint URL raises ``ConfigError``: a URL is not
        a Claude model id.
    """
    if raw_model and agent_model_is_endpoint(str(raw_model)):
        raise ConfigError(
            "a custom --agent-model endpoint cannot run on claude-code; use dsh"
        )
    # A raw model may be a credentialed non-URL value; the child argv is
    # world-readable, so keep only the credential-free shape.
    raw = _redacted_endpoint_text(raw_model)
    if selection.model is None or selection.is_default:
        # A config-owned selection still honours an explicit --model override;
        # returning early here silently dropped it on the Claude route while
        # dsh/codex already honoured it.
        return str(raw) if raw else None
    family = _normalized_family(selection)
    if family == "claude":
        info = selection.model_info()
        return str(raw or info.get("alias") or info.get("id") or selection.model)
    # The DeepSeek/Qwen/Dots routes select the model through the Anthropic
    # environment (ANTHROPIC_MODEL); passing the provider id as `--model` would
    # override that environment with an id Claude Code may not recognise.
    return str(raw) if raw else None


def _claude_code_effort(level: Any) -> str:
    """Translate a generic level to the vocabulary Claude Code accepts.

    Claude Code accepts ``low``/``medium``/``high``/``max`` and silently ignores
    anything else.  A Qwen ``max`` is normalized to the provider wire's
    ``xhigh``, which Claude Code would ignore, so this route maps the *generic*
    level instead.  ``off`` has no Claude Code spelling and is omitted.
    """
    text = str(level or "").strip().lower()
    if text in ("", "off"):
        return ""
    if text == "minimal":
        return "low"
    if text == "xhigh":
        return "max"
    if text in ("low", "medium", "high", "max"):
        return text
    return ""


def claude_environment(
    selection: Optional[AgentSelection], *, raw_model: Optional[str] = None
) -> Dict[str, str]:
    """Environment overrides that point Claude Code at the selected provider.

    Example:
        A Dots selection returns ``ANTHROPIC_BASE_URL`` and the model aliases for
        Claude Code.

    Counterexample:
        A GPT/Claude/endpoint selection returns ``{}``; the function does not
        invent an Anthropic route for an unsupported family.  A raw http(s)
        ``--model`` raises ``ConfigError`` because it is an endpoint, not a
        Claude model id.
    """
    if raw_model and agent_model_is_endpoint(str(raw_model)):
        raise ConfigError(
            "a custom --agent-model endpoint cannot run on claude-code; use dsh"
        )
    if selection is None:
        # Preserve the documented "carry an explicit override even with no
        # selection" behaviour.
        effort = _claude_code_effort(reasoning_effort_override())
        return {"CLAUDE_CODE_EFFORT_LEVEL": effort} if effort else {}
    if selection.model is None or selection.is_default:
        # The sentinel/legacy shape still honours a raw catalog model: build
        # the same inferred selection dsh_patch_text/codex_provider_args use,
        # so a raw qwen/dots model gets its Anthropic route.  Without a catalog
        # entry the CLI owns the model, but an explicit level still reaches
        # Claude Code.
        inferred = None
        if raw_model:
            raw_info = AGENT_MODELS.get(normalize_agent_model(raw_model)) or {}
            if raw_info:
                inferred = AgentSelection(
                    harness=selection.harness,
                    model=normalize_agent_model(raw_model),
                    api_key=selection.api_key,
                    prep_json=selection.prep_json,
                )
        if inferred is None:
            effort = _claude_code_effort(reasoning_effort_override())
            return {"CLAUDE_CODE_EFFORT_LEVEL": effort} if effort else {}
        selection = inferred
    settings = _provider_family_settings(selection)
    family = settings["family"]
    if family in ("claude", "gpt"):
        # Native Claude (and GPT over the harness) keep the model default
        # unless the operator named a level, and that level must use Claude
        # Code's spelling rather than the generic one.
        effort = _claude_code_effort(settings["effort"])
        return {"CLAUDE_CODE_EFFORT_LEVEL": effort} if effort else {}
    if family == "endpoint":
        return {}
    root = _normalized_route_base_url(settings["base_url"])
    # Keep a non-secret query when trimming the Anthropic ``/v1`` suffix.
    root_path, query_separator, root_query = root.partition("?")
    if root_path.endswith("/v1"):
        root_path = root_path[:-3]
    root = root_path + (
        query_separator + root_query if query_separator else ""
    )
    # Dots and DeepSeek require a real key; only the local Qwen route may run
    # with the historical dummy value.  A missing key must not be turned into a
    # bogus bearer token that hides the documented selection_problem.
    key = str(settings["api_key"] or "")
    if family == "qwen":
        key = key or "dummy"
    model = str(
        _redacted_endpoint_text(raw_model)
        or _redacted_endpoint_text(settings["model"])
    )
    if family == "deepseek":
        primary = str(_redacted_endpoint_text(raw_model) or "deepseek-flash[1m]")
        small = str(_redacted_endpoint_text(raw_model) or "deepseek-flash")
        env = {
            "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
            "ANTHROPIC_MODEL": primary,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": primary,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": primary,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": small,
            "CLAUDE_CODE_SUBAGENT_MODEL": small,
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "786432",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if key:
            env["ANTHROPIC_AUTH_TOKEN"] = key
            env["ANTHROPIC_API_KEY"] = key
        # Same catalog default as the DSH and Codex routes; an operator
        # override of the DeepSeek effort applies here too.  Translate the
        # generic level so ``off`` emits no ignored value and a level Claude
        # Code does not accept never reaches it.
        effort = _claude_code_effort(
            settings["effort"] or DEEPSEEK_DEFAULT_EFFORT
        )
        if effort:
            env["CLAUDE_CODE_EFFORT_LEVEL"] = effort
        return env
    env = {
        "ANTHROPIC_BASE_URL": root,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    if key:
        env["ANTHROPIC_AUTH_TOKEN"] = key
        env["ANTHROPIC_API_KEY"] = key
    # Only an explicit operator override adds the effort: these
    # Anthropic-compatible endpoints do not document the field, so the model's
    # own default stays in place otherwise.  Translate the generic level, not
    # the provider wire level (qwen ``max`` is wire ``xhigh``, which Claude Code
    # ignores).
    override = reasoning_effort_override()
    if override:
        claude_effort = _claude_code_effort(override)
        if claude_effort:
            env["CLAUDE_CODE_EFFORT_LEVEL"] = claude_effort
    return env


def effective_agent_model(cmd: Sequence[Any]) -> Dict[str, Any]:
    """A best-effort, secret-free label for the model a command will use.

    This is a diagnostic helper for callers that only have the resolved argv
    (for example a run record or a test); the pipeline itself records the
    selection directly.  It understands the native provider-adapter spelling
    (``--harness/--model`` and ``--endpoint/--model-name``), ``codex exec -m``
    and ``claude --model``.

    Example:
        A named selection returns a secret-free record with the provider id.

    Counterexample:
        An endpoint record contains only the redacted URL and an ``api_key_set``
        boolean; the key itself is never present.
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
        model_name = _arg("--model-name") or "default"
        return {
            "name": redact_endpoint_url(model_name),
            "endpoint": redact_endpoint_url(_arg("--endpoint")),
            "harness": _arg("--harness", "dsh"),
            # The record is secret-free: report only whether a key reached the
            # command, never its value.
            "api_key_set": bool(
                _arg("--api-key")
                or _arg("--agent-model-api-key")
                or _env(AGENT_MODEL_API_KEY_ENV)
            ),
            "source": "--agent-model(custom endpoint)/--agent-model-api-key",
        }
    if "--harness" in argv and "--model" in argv:
        model = _arg("--model")
        # A raw endpoint URL may carry userinfo/query credentials; the record
        # must keep only the scheme/host/path, exactly like to_record().
        record_model = _redacted_endpoint_text(model)
        canonical = normalize_agent_model(model)
        info = AGENT_MODELS.get(canonical) or {}
        return {
            "name": record_model,
            "id": info.get("id") or record_model,
            "harness": _arg("--harness"),
            "api_key_set": bool(
                _arg("--api-key")
                or _arg("--agent-model-api-key")
                or _env(AGENT_MODEL_API_KEY_ENV)
            ),
            "source": "--agent-model/--agent-harness",
        }
    if "codex" in exe:
        return {
            # A -m value may be a credentialed endpoint URL; this record is
            # documented as secret-free, so redact it like the other branches.
            "name": _redacted_endpoint_text(_arg("-m")) or None,
            "harness": AGENT_HARNESS_GPT_CODEX,
            "source": "codex config or -m",
        }
    if "claude" in exe:
        return {
            "name": _redacted_endpoint_text(_arg("--model")) or None,
            "harness": AGENT_HARNESS_CLAUDE_CODE,
            "source": "claude config or --model",
        }
    if "dsh" in exe:
        # The dsh branch used to always report "profile", losing an explicit
        # ``--model``/``-m`` passed after the profile flags.
        model = _arg("--model") or _arg("-m")
        if model:
            return {
                "name": _redacted_endpoint_text(model),
                "harness": AGENT_HARNESS_DSH,
                "source": "dsh argv --model",
            }
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


def redact_prep_object(
    value: Any,
    *,
    _seen: Optional[set] = None,
    _depth: int = 0,
) -> Any:
    """Recursively mask credential-looking *values* in a prep-json object.

    Request defaults such as ``max_tokens`` or ``max_completion_tokens`` are
    ordinary generation parameters, not credentials, so they stay readable.

    Example:
        ``redact_prep_object({"api_key":{"value":"sk-live"}})`` returns
        ``{"api_key":"[REDACTED]"}``.

    Counterexample:
        A non-secret key such as ``max_tokens`` stays readable; do not redact whole
        request objects.  A cyclic container is reported as ``"[CYCLE]"`` and a
        container deeper than the guard as ``"[TRUNCATED]"`` instead of raising
        ``RecursionError``.
    """
    # A prep JSON object is caller-supplied and may be adversarial: a
    # self-reference used to turn ``repr(selection)`` (a log/error path) into a
    # RecursionError, and a ~2000-deep object does the same.  Track the current
    # path so a shared (DAG) sub-object is not mistaken for a cycle.
    if _depth > 64:
        return "[TRUNCATED]"
    seen = _seen if _seen is not None else set()
    if isinstance(value, (dict, list, tuple)):
        marker = id(value)
        if marker in seen:
            return "[CYCLE]"
        seen.add(marker)
        try:
            if isinstance(value, dict):
                # A header pair names its credential in the sibling ``name``
                # field (``{"name": "Authorization", "value": "Bearer ..."}``);
                # the ``value`` key itself does not look like a credential, so
                # it used to be printed verbatim.
                header_name = value.get("name")
                if "value" in value and header_name is not None and _credential_key(
                    header_name
                ):
                    pair = {}
                    for key, item in value.items():
                        if str(key).lower() == "value":
                            pair[str(key)] = "[REDACTED]"
                        else:
                            pair[str(key)] = redact_prep_object(
                                item, _seen=seen, _depth=_depth + 1
                            )
                    return pair
                result = {}
                for key, item in value.items():
                    if _credential_key(key):
                        # A secret-named key masks its whole value, whatever its
                        # type: a list/dict under `api_key` used to lose the key
                        # context when the recursion descended into it and
                        # returned the credential unchanged.
                        result[str(key)] = "[REDACTED]"
                    else:
                        result[str(key)] = redact_prep_object(
                            item, _seen=seen, _depth=_depth + 1
                        )
                return result
            if isinstance(value, list):
                return [
                    redact_prep_object(item, _seen=seen, _depth=_depth + 1)
                    for item in value
                ]
            return tuple(
                redact_prep_object(item, _seen=seen, _depth=_depth + 1)
                for item in value
            )
        finally:
            seen.discard(marker)
    if isinstance(value, str):
        # A URL-shaped or credential-bearing string value is a secret even
        # under an ordinary key such as ``endpoint``/``base_url``; mask it.
        return _redacted_endpoint_text(value)
    return value
