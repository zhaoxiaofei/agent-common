"""Shared severity tiers, recovery records and the ``--fail-tier`` failure policy.

This module is the single source of truth for the diagnostic contract used by
``generic-debug-pipeline`` and ``paper-refine``:

* a fixed, ordered five-tier severity taxonomy (``DEBUG`` .. ``CRITICAL``);
* stable diagnostic codes rendered as ``TIER-<NAME>-<CODE>: <message>``;
* recovery records that state what failed, what recovery did, and what the
  output now contains (or omits);
* an incremental JSONL journal whose fallback keeps records when the primary
  audit destination is unavailable;
* the strict ``diagnostic_rank > configured_fail_tier_rank`` policy.  The
  highest tier (``CRITICAL``) is the default, so no diagnostic is above the
  threshold and every diagnostic follows recovery instead of terminating.

Severity describes impact, never recoverability.  Every tier has a recovery
routine; no tier is inherently unrecoverable.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import re
import sys
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Ordered severity taxonomy
# ---------------------------------------------------------------------------

TIER_DEBUG = 0
TIER_INFO = 1
TIER_WARNING = 2
TIER_ERROR = 3
TIER_CRITICAL = 4

TIER_NAMES = {
    TIER_DEBUG: "DEBUG",
    TIER_INFO: "INFO",
    TIER_WARNING: "WARNING",
    TIER_ERROR: "ERROR",
    TIER_CRITICAL: "CRITICAL",
}
TIER_RANKS = {name: rank for rank, name in TIER_NAMES.items()}

#: The default threshold: nothing exceeds CRITICAL, so the default run is
#: ``always-continue`` and only recovery routines act on diagnostics.
DEFAULT_FAIL_TIER = TIER_CRITICAL

#: Exit status used when a diagnostic is strictly above the configured fail
#: tier.  It is distinct from usage (2), internal/unexpected (3) and SIGINT
#: (130) so an operator (and a test) can tell a deliberate threshold stop from
#: an ordinary failure.
THRESHOLD_EXIT_CODE = 6

_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")

#: Credential-looking names, normalised by lowercasing and dropping every
#: non-alphanumeric character.  ``recovery`` deliberately stays dependency-free,
#: so this mirrors ``model_runtime._credential_key`` instead of importing it.
#: Generation counters (``max_tokens``, ``prompt_tokens``, ...) are *not*
#: credentials and stay readable: only the singular ``token`` spelling and the
#: ``_token`` suffix are secrets.
_CREDENTIAL_KEY_NAMES = frozenset(
    {
        "key",
        "apikey",
        "apikeys",
        "xapikey",
        "token",
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
    """Whether a mapping/assignment key looks like it carries a credential."""
    text = str(name or "").strip().lower()
    compact = "".join(character for character in text if character.isalnum())
    if compact in _CREDENTIAL_KEY_NAMES:
        return True
    normalized = text.replace("-", "_").replace(" ", "_")
    return normalized.endswith(_CREDENTIAL_KEY_SUFFIXES)


#: The key/separator of a credential assignment in free text (``token=...``,
#: ``"api_key": "..."`` or ``Authorization: Bearer ...``).  Only one key (one
#: token, or one quoted phrase) and the separator are consumed: the caller
#: keeps scanning the value, so an outer non-credential assignment such as
#: ``body=api_key=SECRET`` cannot swallow the nested credential, and a
#: preceding word cannot become the key.
_ASSIGNMENT_KEY_RE = re.compile(
    r"(?i)(?<![\w.-])"
    r"(?:(?P<quote>[\"'])(?P<quoted>[^\"'\n]{1,80})(?P=quote)"
    r"|(?P<key>[A-Za-z0-9_.-]{1,80}))"
    r"(?P<sep>\s*[:=]\s*)"
)
#: The scalar value that follows a credential assignment key.
_ASSIGNMENT_VALUE_RE = re.compile(
    r"(?i)(?:(?:bearer|basic)\s+)?"
    r"(?:\"[^\"\n]*\"|'[^'\n]*'|[^\s,;}{]+)"
)
_URL_RE = re.compile(
    r"(?i)(?:"
    r"(?:[A-Za-z][A-Za-z0-9+.\-]*://"
    r"|(?<![\w:/])//"
    r"|(?<![\w:/])[A-Za-z][A-Za-z0-9+.\-]*:/)"
    r"[^\s'\"<>]+"
    # A bare-userinfo endpoint in free text (``user:pw@host/v1``) still
    # carries a password.  It needs a path, so an e-mail address and a
    # ``mailto:user@host`` value are not matched.
    r"|(?<![\w:/@])[A-Za-z0-9._~%+\-]+:[^\s:@/]+@[A-Za-z0-9.\-]+"
    r"(?::\d+)?/[^\s'\"<>]*"
    r")"
)
_BEARER_RE = re.compile(
    r"(?i)\b(bearer|basic)\s+([A-Za-z0-9._~+/=-]{6,})"
)


def _query_pairs(query: str) -> List[Tuple[str, str]]:
    try:
        return urllib.parse.parse_qsl(
            str(query or "").replace(";", "&"), keep_blank_values=True
        )
    except ValueError:
        return []


def _redact_query_component(component: str) -> str:
    """Mask credential-named values in a query or fragment string."""
    if not component:
        return component
    if not any(separator in component for separator in ("=", "&", ";")):
        return component
    pairs = _query_pairs(component)
    if not pairs:
        return component
    return urllib.parse.urlencode(
        [
            (key, "<redacted>" if _credential_key(key) else value)
            for key, value in pairs
        ],
        doseq=True,
    )


def _credential_query_or_fragment(text: str) -> bool:
    """Whether a scheme-less value carries a credential-named query/fragment."""
    query = text.partition("?")[2].partition("#")[0]
    fragment = text.partition("#")[2]
    return any(
        "=" in component
        and any(
            _credential_key(key) for key, _value in _query_pairs(component)
        )
        for component in (query, fragment)
    )


def _looks_url_shaped(text: str) -> bool:
    """Whether masking userinfo on ``text`` cannot mangle ordinary prose.

    An e-mail address (``user@host``) has no path and stays untouched; a
    scheme-relative, one-slash or bare-userinfo endpoint does not.
    """
    value = str(text)
    if "://" in value or value.startswith("//"):
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:/", value):
        return True
    if "@" in value:
        _userinfo, _separator, rest = value.rpartition("@")
        # A bare-userinfo endpoint carries a path after the host; an e-mail
        # address or a ``mailto:user@host`` value does not, so leave it alone.
        if "/" in rest:
            return True
    # A scheme-less value with a credential in its query or fragment
    # (``foo?api_key=SECRET``) is still a credential-bearing URL.
    if _credential_query_or_fragment(value):
        return True
    return False


def _redact_textual_userinfo(base: str) -> str:
    """Mask the userinfo of an authority ``urlsplit`` could not represent."""
    match = re.match(
        r"^(?P<prefix>(?:[A-Za-z][A-Za-z0-9+.\-]*:)?/{0,2})(?P<rest>.*)$",
        base,
    )
    if match is None:
        return base
    prefix, rest = match.group("prefix"), match.group("rest")
    authority, separator, tail = rest.partition("/")
    if "@" in authority:
        userinfo, _at, host = authority.rpartition("@")
        user, colon, _password = userinfo.partition(":")
        authority = (
            user + ":<redacted>@" + host
            if colon
            else "<redacted>@" + host
        )
    return prefix + authority + separator + tail


def _redact_url_textually(raw: str) -> str:
    """Scrub a URL ``urlsplit`` did not expose as scheme + authority.

    Covers scheme-relative (``//user:pw@host/v1``), one-slash
    (``http:/user:pw@host/v1``) and bare-userinfo (``user:pw@host/v1``) forms.
    """
    head, fragment_separator, fragment = raw.partition("#")
    base, query_separator, query = head.partition("?")
    query = _redact_query_component(query)
    fragment = _redact_query_component(fragment)
    base = _redact_textual_userinfo(base)
    return (
        base
        + (query_separator + query if query_separator else "")
        + (fragment_separator + fragment if fragment_separator else "")
    )


def _redact_url(text: str) -> str:
    """Mask URL userinfo and credential-named query/fragment values.

    A value without ``://`` is not automatically safe: a scheme-relative
    (``//user:pw@host/v1``), one-slash (``http:/user:pw@host/v1``) or bare
    userinfo (``user:pw@host/v1``) form still carries a password.  The old
    early return leaked all three into the durable journal, and even a parsed
    URL kept its fragment verbatim.
    """
    raw = str(text)
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return (
            _redact_url_textually(raw) if _looks_url_shaped(raw) else raw
        )
    if parsed.scheme and parsed.netloc:
        netloc = parsed.netloc
        if "@" in netloc:
            netloc = "<redacted>@" + netloc.rsplit("@", 1)[-1]
        query = _redact_query_component(parsed.query)
        fragment = _redact_query_component(parsed.fragment)
        return urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, query, fragment)
        )
    if not _looks_url_shaped(raw):
        return raw
    return _redact_url_textually(raw)


def _redact_assignment_value(value: str) -> str:
    """Mask a scalar assignment value, preserving surrounding quotes."""
    if value[:1] in ("'", '"') and value[-1:] == value[:1]:
        return "%s<redacted>%s" % (value[0], value[0])
    return "<redacted>"


def _redact_assignments(text: str) -> str:
    """Mask credential assignments, including ones nested inside a value.

    Example: ``body=api_key=SECRET`` becomes ``body=api_key=<redacted>``.

    Counterexample: ``max_tokens=512`` is a generation counter, not a
    credential, and stays readable.
    """
    pieces = []
    position = 0
    while True:
        match = _ASSIGNMENT_KEY_RE.search(text, position)
        if match is None:
            pieces.append(text[position:])
            return "".join(pieces)
        pieces.append(text[position : match.end()])
        key = match.group("quoted") or match.group("key")
        if _credential_key(key):
            value = _ASSIGNMENT_VALUE_RE.match(text, match.end())
            if value is not None:
                pieces.append(_redact_assignment_value(value.group(0)))
                position = value.end()
                continue
        position = match.end()


def _redact_bearer(match: "re.Match[str]") -> str:
    """Mask a credential-shaped Bearer/Basic token, not ordinary prose.

    A real header value (``Authorization: Bearer <token>``) is handled by the
    key-aware assignment pass, so this pass only needs to recognise tokens that
    have a credential shape: a digit/symbol, or a long mixed-case run.  Plain
    words such as "configuration", "authentication" and "token" stay readable.
    """
    token = match.group(2)
    prose_word = token.isalpha() and token.islower()
    if match.group(1).lower() == "bearer" and not prose_word:
        # "Bearer" is a credential scheme far more often than an English
        # adjective, so a non-lowercase word after it is treated as a token.
        return "%s <redacted>" % match.group(1)
    credential_shaped = (
        any(not character.isalnum() for character in token)
        or any(character.isdigit() for character in token)
        or (_credential_key(token) and not token.islower())
        or (len(token) >= 16 and not (token.islower() or token.isupper()))
    )
    if not credential_shaped:
        return match.group(0)
    return "%s <redacted>" % match.group(1)

#: Outcomes a recovery record may state.  They are deliberately distinct so a
#: reader can tell "continued unchanged" from "degraded with omissions".
OUTCOMES = (
    "recorded",      # informational diagnostic; recorded and continued
    "retried",       # a bounded attempt failed; another attempt follows
    "recovered",     # recovery succeeded and the operation continued
    "continued",     # operation continued; diagnostic noted
    "degraded",      # fallback/substitution used; output is partial
    "skipped",       # affected item was skipped; output is incomplete
    "paused",        # no progress possible; waiting for intervention
    "exhausted",     # every attempt failed; no fallback applied
    "failed",        # recovery itself failed; work is incomplete
    "terminated",    # diagnostic above the configured fail tier
)


class RecoveryPolicyError(ValueError):
    """A diagnostic, record or fail-tier value violated the shared contract.

    Example: ``parse_fail_tier("SEVERE")`` raises this because ``SEVERE`` is
    not a tier name or a rank from 0 to 4.

    Counterexample: catching this error and substituting ``CRITICAL`` silently
    would hide a typo and change the configured policy; report a usage error.
    """


class FailTierError(RecoveryPolicyError):
    """A ``--fail-tier`` value is not a supported name or integer rank.

    Example: ``parse_fail_tier("error")`` returns rank 3.

    Counterexample: ``parse_fail_tier("5")`` raises instead of clamping to
    CRITICAL, because silently substituting another threshold is forbidden.
    """


class ThresholdExceeded(Exception):
    """A diagnostic was strictly above the configured fail tier.

    ``emit`` raises this only after the diagnostic and recovery record have
    been persisted, so a caller can perform cleanup in a ``finally`` block and
    then terminate with :data:`THRESHOLD_EXIT_CODE`.

    Example: with ``--fail-tier WARNING`` an ``ERROR`` diagnostic raises this
    exception and identifies the code, severity, threshold and operation.

    Counterexample: an ``ERROR`` diagnostic under the default ``CRITICAL``
    threshold does not raise; it must follow its recovery routine.
    """

    def __init__(self, diagnostic: "Diagnostic", threshold: int, exit_code: int = THRESHOLD_EXIT_CODE):
        self.diagnostic = diagnostic
        self.threshold = threshold
        self.exit_code = exit_code
        super().__init__(threshold_failure_message(diagnostic, threshold))

    @property
    def code(self) -> str:
        return self.diagnostic.code

    @property
    def tier(self) -> int:
        return self.diagnostic.tier


class ReportWriteError(OSError):
    """Neither the primary nor the fallback recovery report could be written.

    Example: ``MarkdownReport.write`` raises this when the report path and its
    fallback path are both unwritable.

    Counterexample: a successful fallback write does not raise; it returns the
    fallback path so the caller can report where the records went.
    """


# ---------------------------------------------------------------------------
# Tier and code parsing
# ---------------------------------------------------------------------------


def tier_rank(value: Any) -> int:
    """Return the integer rank (0..4) for a tier name or rank.

    Example: ``tier_rank("warning")`` and ``tier_rank(2)`` both return 2.

    Counterexample: ``tier_rank("WARN")`` raises; the documented spelling is
    ``WARNING``, and an alias would make the policy ambiguous.
    """
    if isinstance(value, bool):
        raise FailTierError("a boolean is not a severity tier: %r" % (value,))
    if isinstance(value, int):
        if value in TIER_NAMES:
            return value
        raise FailTierError("tier rank must be 0..4, got %r" % (value,))
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            number = int(text)
            if number in TIER_NAMES:
                return number
            raise FailTierError("tier rank must be 0..4, got %r" % (value,))
        name = text.upper()
        if name.startswith("TIER-"):
            name = name[len("TIER-"):]
        if name in TIER_RANKS:
            return TIER_RANKS[name]
    raise FailTierError(
        "unknown severity tier %r; use one of %s or a rank from 0 to 4"
        % (value, ", ".join(TIER_NAMES[rank] for rank in sorted(TIER_NAMES)))
    )


def tier_name(value: Any) -> str:
    """Return the canonical upper-case name for a tier name or rank.

    Example: ``tier_name(4)`` returns ``"CRITICAL"``.

    Counterexample: ``tier_name(None)`` raises; callers that permit a default
    must use :func:`parse_fail_tier`, which documents the default explicitly.
    """
    return TIER_NAMES[tier_rank(value)]


def parse_fail_tier(value: Any, *, default: int = DEFAULT_FAIL_TIER) -> int:
    """Validate a ``--fail-tier`` value and return its integer rank.

    ``None`` selects ``default`` (CRITICAL).  Names are case-insensitive and
    integer ranks 0 through 4 are accepted.  Every other value raises, so the
    CLI can print a usage error instead of substituting a different threshold.

    Example: ``parse_fail_tier("error") == 3`` and ``parse_fail_tier("0") == 0``.

    Counterexample: ``parse_fail_tier("-1")`` and ``parse_fail_tier("")``
    raise; an empty or out-of-range explicit flag is never treated as unset.
    """
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        raise FailTierError("--fail-tier requires a tier name or a rank from 0 to 4")
    return tier_rank(value)


def fail_tier_help() -> str:
    """The one-line help text shared by every entry point's ``--fail-tier``.

    Example: ``parser.add_argument("--fail-tier", type=..., help=fail_tier_help())``.

    Counterexample: reusing this text for another option would claim a policy
    the option does not implement.
    """
    return (
        "severity threshold: diagnostics strictly above it terminate the run; "
        "at or below it recovery continues. Names (debug/info/warning/error/"
        "critical) or ranks 0-4, case-insensitive. Default: CRITICAL, which is "
        "the always-continue configuration (no diagnostic exceeds it)."
    )


def diagnostic_code(code: Any) -> str:
    """Validate and return a stable diagnostic code such as ``INPUT_UNREADABLE``.

    Example: ``diagnostic_code("GIT_COMMIT_FAILED")`` returns the same string.

    Counterexample: ``diagnostic_code("git commit failed")`` raises; a code
    must be upper-case words joined by single underscores so it is stable in
    records and greppable in logs.
    """
    if not isinstance(code, str) or not _CODE_RE.match(code):
        raise RecoveryPolicyError(
            "diagnostic code must look like UPPER_WORDS, got %r" % (code,)
        )
    return code


def format_diagnostic(tier: Any, code: Any, message: str) -> str:
    """Render the canonical ``TIER-<NAME>-<CODE>: <message>`` line.

    Example: ``format_diagnostic("error", "INPUT_UNREADABLE", "Could not read
    the input.")`` returns ``TIER-ERROR-INPUT_UNREADABLE: Could not read the
    input.``

    Counterexample: ``format_diagnostic(9, "X", "m")`` raises rather than
    inventing a severity for an unknown rank.
    """
    return "TIER-%s-%s: %s" % (
        tier_name(tier), diagnostic_code(code), _one_line(message)
    )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def redact_value(value: Any) -> Any:
    """Return a copy of ``value`` with credential-looking fields masked.

    Example: ``redact_value({"api_key": "sk-1"})`` returns
    ``{"api_key": "<redacted>"}``.

    Counterexample: a generation counter such as ``max_tokens`` is not a
    credential and stays readable; a credential-bearing URL under an ordinary
    key (``endpoint``) is still reduced to its credential-free shell.
    """
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if isinstance(key, str) and _credential_key(key):
                result[key] = "<redacted>"
            else:
                result[key] = redact_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_url(value)
    return value


_LINE_BREAK_RE = re.compile(r"[\x00-\x1f\x7f\u2028\u2029]+")
_BACKTICK_RUN_RE = re.compile(r"`+")


def _one_line(text: Any) -> str:
    """Collapse control characters and line breaks into single spaces."""
    return _LINE_BREAK_RE.sub(" ", str(text)).strip()


def _inline_code(text: Any) -> str:
    """Render ``text`` as a CommonMark code span that cannot close early.

    Example: ``_inline_code("a`b")`` uses a two-backtick fence because the
    value contains a one-backtick run.

    Counterexample: ``_inline_code("plain")`` uses the usual single fence.
    """
    value = str(text)
    longest = max(
        (len(run.group(0)) for run in _BACKTICK_RUN_RE.finditer(value)),
        default=0,
    )
    fence = "`" * (longest + 1)
    # A fence abutting a value that starts/ends with a backtick would be read
    # as one longer run; one padding space keeps the fence separate (CommonMark
    # strips the single padding space again inside the code span).
    pad = " " if value.startswith("`") or value.endswith("`") else ""
    return "%s%s%s%s%s" % (fence, pad, value, pad, fence)


def redact_text(text: str) -> str:
    """Mask credentials in a free-text diagnostic message.

    URL userinfo, credential-named query values, ``Bearer``/``Basic`` tokens,
    JSON/quoted assignments and plural or spaced credential names are masked.
    Ordinary prose ("the token expired") and generation counters such as
    ``max_tokens=512`` are left alone.

    Example: ``redact_text("token=abc123 failed")`` returns
    ``"token=<redacted> failed"``.

    Counterexample: prose such as "the token expired" is unchanged; there is
    no assignment value to mask.
    """
    def _url(match: "re.Match[str]") -> str:
        return _redact_url(match.group(0))

    redacted = _URL_RE.sub(_url, str(text))
    redacted = _BEARER_RE.sub(_redact_bearer, redacted)
    return _redact_assignments(redacted)


# ---------------------------------------------------------------------------
# Diagnostic and recovery records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    """One application diagnostic: tier, stable code, operation and message.

    Example: ``Diagnostic(TIER_ERROR, "INPUT_UNREADABLE", "find-bugs",
    "Could not read the requested input.").formatted`` renders the canonical
    line.

    Counterexample: a message that contains a credential must be redacted
    before it reaches this object; the object records context safely but does
    not guess at arbitrary prose.
    """

    tier: int
    code: str
    operation: str
    message: str
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        rank = tier_rank(self.tier)
        object.__setattr__(self, "tier", rank)
        object.__setattr__(self, "code", diagnostic_code(self.code))
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise RecoveryPolicyError("a diagnostic needs an affected operation")
        if not isinstance(self.message, str) or not self.message.strip():
            raise RecoveryPolicyError("a diagnostic needs a message")
        # Agent/prose text can contain line breaks or terminal controls; a
        # diagnostic line and a commit-message bullet must stay one printable
        # line so embedded text cannot forge another diagnostic or section.
        object.__setattr__(
            self, "operation", redact_text(_one_line(self.operation))
        )
        object.__setattr__(self, "message", _one_line(self.message))

    @property
    def rank(self) -> int:
        """The numeric rank of this diagnostic's severity.

        Example: a ``CRITICAL`` diagnostic has ``rank == 4``.

        Counterexample: comparing a diagnostic to a tier *name* directly is not
        supported; compare ``rank`` with ``parse_fail_tier(...)``.
        """
        return self.tier

    @property
    def tier_name(self) -> str:
        """The canonical name of this diagnostic's tier.

        Example: ``Diagnostic(TIER_WARNING, ...).tier_name`` is ``"WARNING"``.

        Counterexample: rendering the rank instead of the name in a user-facing
        line would break the documented ``TIER-<NAME>-<CODE>`` format.
        """
        return TIER_NAMES[self.tier]

    @property
    def formatted(self) -> str:
        """The canonical ``TIER-<NAME>-<CODE>: <message>`` diagnostic line.

        Example: ``Diagnostic(3, "GIT_COMMIT_FAILED", "commit", "x").formatted``
        starts with ``TIER-ERROR-GIT_COMMIT_FAILED:``.

        Counterexample: writing only the message loses the tier and code a
        human reviewer needs to look up the recovery routine.
        """
        return format_diagnostic(self.tier, self.code, redact_text(self.message))

    def record_context(self) -> Dict[str, Any]:
        """The redacted context persisted with this diagnostic.

        Example: ``Diagnostic(..., context={"path": "a", "token": "t"})
        .record_context()`` masks only ``token``.

        Counterexample: the live credential-bearing mapping itself is never
        returned; callers cannot accidentally persist it.
        """
        return redact_value(dict(self.context or {}))


@dataclasses.dataclass
class RecoveryRecord:
    """The auditable account of one diagnostic and its recovery attempt.

    Every field the project policy requires is present.  ``output_effects`` is
    prose (or a list joined by semicolons) explaining what the output now
    contains, omits, substituted or leaves unresolved.

    Example: a record for ``AGENT_FAILED`` states ``attempt=2/2``,
    ``outcome="degraded"`` and ``omissions=["stage verify_fixes did not run"]``.

    Counterexample: a record whose outcome is ``recovered`` must not also
    carry an omission; honest reporting requires ``degraded`` instead.
    """

    run_id: str
    event_id: str
    timestamp: str
    sequence: int
    code: str
    tier: int
    operation: str
    failure: str
    recovery_action: str
    attempt: int = 1
    max_attempts: int = 1
    outcome: str = "recorded"
    fallback: str = ""
    omissions: Sequence[str] = dataclasses.field(default_factory=list)
    substitutions: Sequence[str] = dataclasses.field(default_factory=list)
    affected_outputs: Sequence[str] = dataclasses.field(default_factory=list)
    validation: str = ""
    remaining_uncertainty: str = ""
    human_review: bool = False
    output_effects: str = ""
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tier = tier_rank(self.tier)
        diagnostic_code(self.code)
        if self.outcome not in OUTCOMES:
            raise RecoveryPolicyError(
                "unknown recovery outcome %r; expected one of %s"
                % (self.outcome, ", ".join(OUTCOMES))
            )
        if self.attempt < 1 or self.max_attempts < 1:
            raise RecoveryPolicyError("recovery attempt counts must be >= 1")
        if self.attempt > self.max_attempts:
            raise RecoveryPolicyError("recovery attempt cannot exceed max_attempts")
        for name in ("omissions", "substitutions", "affected_outputs"):
            value = getattr(self, name)
            if isinstance(value, str):
                setattr(self, name, [_one_line(value)] if value else [])
            else:
                setattr(self, name, [_one_line(item) for item in (value or ())])
        # ``operation`` reaches to_dict(), to_markdown() and the threshold-stop
        # message, so it is redacted like every other free-text field.  The old
        # ``_one_line``-only treatment persisted ``POST https://u:pw@...``
        # verbatim in the durable journal.
        self.operation = redact_text(_one_line(self.operation))
        for name in (
            "failure",
            "recovery_action",
            "fallback",
            "validation",
            "remaining_uncertainty",
            "output_effects",
        ):
            setattr(
                self,
                name,
                redact_text(_one_line(getattr(self, name) or "")),
            )
        for name in ("omissions", "substitutions", "affected_outputs"):
            setattr(
                self,
                name,
                [redact_text(item) for item in (getattr(self, name) or ())],
            )
        self.context = redact_value(dict(self.context or {}))

    @property
    def tier_name(self) -> str:
        """The canonical severity name for this record.

        Example: a record at rank 3 reports ``tier_name == "ERROR"``.

        Counterexample: a reader must not infer recovery success from the tier;
        only ``outcome`` states what happened.
        """
        return TIER_NAMES[self.tier]

    @property
    def needs_review(self) -> bool:
        """Whether this record requires a human to review the result.

        Example: an ``exhausted`` record always needs review.

        Counterexample: a successful, fully validated retry under the same
        tier does not automatically need review.
        """
        return bool(self.human_review or self.outcome in ("exhausted", "failed", "paused", "terminated", "degraded", "skipped"))

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-ready record, including the canonical formatted line.

        Example: ``json.dumps(record.to_dict())`` is stable enough to diff
        across runs.

        Counterexample: returning the dataclass itself would leak tuples and
        make an incremental journal non-JSON-serializable.
        """
        payload = {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "code": self.code,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "operation": self.operation,
            "failure": self.failure,
            "recovery_action": self.recovery_action,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "outcome": self.outcome,
            "fallback": self.fallback,
            "omissions": list(self.omissions),
            "substitutions": list(self.substitutions),
            "affected_outputs": list(self.affected_outputs),
            "validation": self.validation,
            "remaining_uncertainty": self.remaining_uncertainty,
            "human_review": bool(self.human_review),
            "output_effects": self.output_effects,
            "context": dict(self.context),
        }
        payload["formatted"] = format_diagnostic(
            self.tier, self.code, self.failure or self.operation
        )
        return payload

    def to_markdown(self, *, index: int = 1) -> str:
        """One markdown bullet block describing this recovery event.

        Example: ``record.to_markdown(index=1)`` starts with
        ``1. **TIER-ERROR-...**`` and continues with the consequences.

        Counterexample: do not render only ``recovery_action``; the omitted
        work and affected outputs are what a reviewer needs.
        """
        lines = [
            "%d. **%s** — `%s` (operation: %s; attempt %d/%d; outcome: `%s`)"
            % (
                index,
                format_diagnostic(self.tier, self.code, self.operation),
                self.timestamp,
                self.operation,
                self.attempt,
                self.max_attempts,
                self.outcome,
            ),
            "   - Failed: %s" % (self.failure or "(not recorded)"),
            "   - Recovery: %s" % (self.recovery_action or "(not recorded)"),
        ]
        if self.fallback:
            lines.append("   - Fallback: %s" % self.fallback)
        if self.substitutions:
            lines.append("   - Substitutions: %s" % "; ".join(self.substitutions))
        if self.omissions:
            lines.append("   - Omissions / skipped work: %s" % "; ".join(self.omissions))
        if self.affected_outputs:
            lines.append("   - Affected outputs: %s" % ", ".join(self.affected_outputs))
        lines.append("   - Effect on outputs: %s" % (self.output_effects or "(not recorded)"))
        lines.append("   - Validation performed: %s" % (self.validation or "none recorded"))
        lines.append(
            "   - Remaining uncertainty: %s"
            % (self.remaining_uncertainty or "none recorded")
        )
        lines.append(
            "   - Human review: %s" % ("required" if self.needs_review else "not required")
        )
        return "\n".join(lines)


def _timestamp(clock: Optional[Callable[[], _dt.datetime]] = None) -> str:
    now = clock() if clock else _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id(prefix: str = "run") -> str:
    """A sortable, collision-resistant run identifier.

    Example: ``new_run_id("gdp")`` returns something like
    ``gdp-20261127T030000Z-1a2b3c4d``.

    Counterexample: using a bare timestamp lets two runs in the same second
    share a journal directory and overwrite each other's records.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "%s-%s-%s" % (prefix, stamp, uuid.uuid4().hex[:8])


def _single_path_component(value: Any, what: str) -> str:
    """Reject a run id/filename that would escape its journal directory."""
    text = str(value or "")
    if (
        not text
        or text in (".", "..")
        or "\x00" in text
        or os.path.isabs(text)
        or os.sep in text
        or (os.altsep is not None and os.altsep in text)
    ):
        raise RecoveryPolicyError(
            "%s must be a single path component, got %r" % (what, value)
        )
    return text


# ---------------------------------------------------------------------------
# Incremental journal
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary sibling and ``os.replace``.

    Example: a reader never observes a half-written recovery report.

    Counterexample: it creates missing parent directories, so a caller that
    must not create a directory (a mistyped output root) has to check the
    parent itself first; ``MarkdownReport`` does exactly that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp-%s" % (path.name, uuid.uuid4().hex[:8]))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def record_sequence(item: Any) -> int:
    """Return a record's numeric sequence, tolerating a malformed journal line.

    Example: ``record_sequence({"sequence": 3})`` returns ``3``.

    Counterexample: a hand-edited ``{"sequence": "x"}`` returns ``0`` rather
    than crashing the recovery report or the commit-message renderer.
    """
    try:
        return int(item.get("sequence") or 0) if isinstance(item, Mapping) else 0
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any, _seen: Optional[set] = None) -> Any:
    """Convert an arbitrary value to something ``json.dumps`` can serialize.

    A diagnostic context may carry a ``Path``, a set or a caller object.  The
    journal promises a failed write never raises, and a raw ``TypeError`` from
    ``json.dumps`` violated that promise; non-JSON values are converted instead
    of aborting the incident path.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    seen = _seen if _seen is not None else set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            return "<cycle>"
        seen.add(marker)
        try:
            return {
                str(key): _json_safe(item, seen) for key, item in value.items()
            }
        finally:
            seen.discard(marker)
    if isinstance(value, (list, tuple, set, frozenset)):
        marker = id(value)
        if marker in seen:
            return "<cycle>"
        seen.add(marker)
        try:
            return [_json_safe(item, seen) for item in value]
        finally:
            seen.discard(marker)
    if isinstance(value, Path):
        return str(value)
    return _one_line(repr(value))


class RecoveryJournal:
    """Append-only JSONL recovery journal with a documented local fallback.

    The journal writes each event immediately and fsyncs it, so an interrupted
    run keeps everything recorded before the interruption.  ``run.json``
    records ``status = "in_progress"`` until :meth:`finalize` writes the final
    status.  If the primary directory cannot be written, the event is appended
    to the fallback directory and retained in memory for a later flush.

    Example: ``journal = RecoveryJournal(run_dir / "recovery", run_id);
    journal.start(); ...; journal.finalize("completed")``.

    Counterexample: constructing a journal does not create directories and a
    failed write does not raise; callers inspect ``audit_errors`` or call
    :meth:`drain_alerts` so the audit failure has its own recovery.
    """

    def __init__(
        self,
        directory: Any,
        run_id: str,
        *,
        fallback_directory: Any = None,
        filename: str = "recovery.jsonl",
        clock: Optional[Callable[[], _dt.datetime]] = None,
    ) -> None:
        self.directory = Path(directory)
        self.run_id = _single_path_component(run_id, "run_id")
        self.filename = _single_path_component(filename, "filename")
        self.fallback_directory = (
            Path(fallback_directory) if fallback_directory is not None else None
        )
        self.clock = clock
        self.events_path = self.directory / self.filename
        self.run_path = self.directory / "run.json"
        self._lock = threading.RLock()
        self._sequence = 0
        self._sequence_seeded = False
        self._event_ids: set = set()
        self._started_at: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.status = "not_started"
        self.audit_errors: List[Dict[str, Any]] = []
        self.pending: List[Dict[str, Any]] = []
        self._warned = False

    def start(self, metadata: Optional[Mapping[str, Any]] = None) -> bool:
        """Create the journal directory and write the in-progress manifest.

        Example: ``start({"command": "find-bugs"})`` writes ``run.json`` with
        ``status: "in_progress"``.

        Counterexample: if the directory cannot be created this returns
        ``False`` and records an audit error; it does not raise and does not
        pretend the journal was started.
        """
        self.metadata = redact_value(dict(metadata or {}))
        started_at = _timestamp(self.clock)
        self._started_at = started_at
        manifest = {
            "run_id": self.run_id,
            "status": "in_progress",
            "started_at": started_at,
            "finished_at": None,
            "metadata": self.metadata,
            "events_file": str(self.events_path),
            "fallback_directory": (
                str(self.fallback_directory) if self.fallback_directory else None
            ),
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                self.run_path,
                json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
            )
            self.status = "in_progress"
            return True
        except (OSError, TypeError, ValueError) as exc:
            self._record_audit_error("JOURNAL_START_FAILED", exc)
            self._fallback_manifest(manifest)
            self.status = "in_progress"
            return False

    def _seed_sequence(self) -> None:
        """Start numbering after the records an earlier process already wrote."""
        if self._sequence_seeded:
            return
        self._sequence_seeded = True
        highest = 0
        known = set()
        paths = [self.events_path]
        if self.fallback_directory is not None:
            paths.append(self.fallback_directory / self.filename)
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    entry = json.loads(line)
                except (ValueError, RecursionError):
                    # A deeply nested line is malformed journal input, not a
                    # reason to crash the reader; the documented contract is
                    # to skip such lines.
                    continue
                if not isinstance(entry, dict):
                    continue
                highest = max(highest, record_sequence(entry))
                event_id = str(entry.get("event_id") or "")
                if event_id:
                    known.add(event_id)
                    match = re.search(r"-(\d+)$", event_id)
                    if match:
                        highest = max(highest, int(match.group(1)))
        self._event_ids = known
        self._sequence = max(self._sequence, highest)

    def next_event_id(self) -> str:
        """Allocate the next stable event identifier for this run.

        Example: the first event is ``<run_id>-0001``.

        Counterexample: deriving the identifier from a timestamp alone can
        collide when two threads record events in the same instant.
        """
        with self._lock:
            self._seed_sequence()
            self._sequence += 1
            return "%s-%04d" % (self.run_id, self._sequence)

    def append(self, record: Mapping[str, Any]) -> bool:
        """Append one recovery record, falling back when the primary path fails.

        Example: on success the line is in ``recovery.jsonl`` and readable by a
        later process that renders the commit message.

        Counterexample: a failed write does not raise; it is retained in
        ``pending``, copied to the fallback, and reported via ``audit_errors``.
        """
        payload = dict(record)
        with self._lock:
            self._seed_sequence()
            self._sequence = max(self._sequence, record_sequence(payload))
            event_id = str(payload.get("event_id") or "")
            if event_id and event_id in self._event_ids:
                # Reopening a run must not overwrite the earlier event under
                # the same id (``records`` keeps only the first occurrence).
                replacement = self.next_event_id()
                self._record_audit_error(
                    "JOURNAL_EVENT_ID_REUSED",
                    ValueError(
                        "event id %r already exists; reassigned %r"
                        % (event_id, replacement)
                    ),
                )
                payload["event_id"] = replacement
            if payload.get("event_id"):
                self._event_ids.add(str(payload["event_id"]))
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._append_line(self.events_path, payload)
                return True
            except (OSError, TypeError, ValueError) as exc:
                self.pending.append(payload)
                self._record_audit_error("JOURNAL_WRITE_FAILED", exc)
                return self._write_fallback(payload)

    def _append_line(self, path: Path, payload: Mapping[str, Any]) -> None:
        line = json.dumps(
            _json_safe(dict(payload)), sort_keys=True, ensure_ascii=False
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_fallback(self, payload: Mapping[str, Any]) -> bool:
        if self.fallback_directory is None:
            return False
        try:
            self.fallback_directory.mkdir(parents=True, exist_ok=True)
            self._append_line(self.fallback_directory / self.filename, payload)
            return True
        except (OSError, TypeError, ValueError) as exc:
            self._record_audit_error("JOURNAL_FALLBACK_FAILED", exc)
            return False

    def _fallback_manifest(self, manifest: Mapping[str, Any]) -> None:
        if self.fallback_directory is None:
            return
        try:
            self.fallback_directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                self.fallback_directory / ("%s-run.json" % self.run_id),
                json.dumps(
                    _json_safe(dict(manifest)), indent=2, sort_keys=True
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._record_audit_error("JOURNAL_MANIFEST_FALLBACK_FAILED", exc)

    def _record_audit_error(self, code: str, exc: BaseException) -> None:
        with self._lock:
            if len(self.audit_errors) < 20:
                self.audit_errors.append(
                    {
                        "code": code,
                        "message": redact_text(str(exc)),
                        "timestamp": _timestamp(self.clock),
                        "primary": str(self.events_path),
                        "fallback": (
                            str(self.fallback_directory / self.filename)
                            if self.fallback_directory
                            else None
                        ),
                    }
                )

    def drain_alerts(self) -> List[Dict[str, Any]]:
        """Return and clear the audit errors recorded since the last drain.

        Example: a CLI drains alerts after a run and reports the fallback path
        once, without recursing through the same broken journal.

        Counterexample: treating an audit error as a reason to raise would make
        a read-only or full disk defeat an ``always-continue`` run.
        """
        with self._lock:
            alerts = list(self.audit_errors)
            self.audit_errors = []
            return alerts

    def flush_pending(self) -> int:
        """Retry persisting records retained after a primary-write failure.

        Example: after the disk recovers, ``flush_pending()`` returns the
        number of records now written to the primary journal.

        Counterexample: this never retries forever; each call makes one attempt
        per pending record and leaves the rest pending.
        """
        with self._lock:
            if not self.pending:
                return 0
            written: List[Dict[str, Any]] = []
            for payload in list(self.pending):
                try:
                    self.directory.mkdir(parents=True, exist_ok=True)
                    self._append_line(self.events_path, payload)
                    written.append(payload)
                except OSError as exc:
                    self._record_audit_error("JOURNAL_FLUSH_FAILED", exc)
                    break
            self.pending = [item for item in self.pending if item not in written]
            return len(written)

    def finalize(
        self,
        status: str,
        *,
        summary: Optional[Mapping[str, Any]] = None,
        exit_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Close the run manifest with its final status and summary.

        Example: ``finalize("completed", summary={"events": 3})`` marks the
        journal finalized so a reader can distinguish it from an interrupted
        run.

        Counterexample: a finalize that cannot write still returns the intended
        manifest and records an audit error; it does not raise in the middle of
        cleanup.
        """
        self.flush_pending()
        manifest = {
            "run_id": self.run_id,
            "status": status,
            "started_at": self._started_at
            or (self.metadata or {}).get("started_at")
            or None,
            "finished_at": _timestamp(self.clock),
            "exit_code": exit_code,
            "metadata": self.metadata,
            "summary": redact_value(dict(summary or {})),
            "events_file": str(self.events_path),
            "fallback_directory": (
                str(self.fallback_directory) if self.fallback_directory else None
            ),
            "pending_events": len(self.pending),
        }
        if not manifest["started_at"]:
            for path in (
                self.run_path,
                (
                    self.fallback_directory / ("%s-run.json" % self.run_id)
                    if self.fallback_directory is not None
                    else None
                ),
            ):
                if path is None:
                    continue
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(existing, Mapping):
                    manifest["started_at"] = existing.get("started_at")
                if manifest["started_at"]:
                    break
        self.status = status
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                self.run_path,
                json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
            )
        except (OSError, TypeError, ValueError) as exc:
            self._record_audit_error("JOURNAL_FINALIZE_FAILED", exc)
            self._fallback_manifest(manifest)
        return manifest

    def records(self) -> List[Dict[str, Any]]:
        """Read every persisted record, primary first, then fallback.

        Example: a commit stage in a later process renders the earlier failed
        commit attempts from this list.

        Counterexample: malformed lines are skipped, not treated as an empty
        history; the caller can compare with ``pending`` and ``audit_errors``.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        paths = [self.events_path]
        if self.fallback_directory is not None:
            paths.append(self.fallback_directory / self.filename)
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                try:
                    entry = json.loads(line)
                except (ValueError, RecursionError):
                    # A deeply nested line is malformed journal input, not a
                    # reason to crash the reader; the documented contract is
                    # to skip such lines.
                    continue
                if isinstance(entry, dict):
                    event_id = str(entry.get("event_id") or "")
                    if event_id:
                        seen.setdefault(event_id, entry)
                    else:
                        seen[line] = entry
        return sorted(
            seen.values(),
            key=lambda item: (record_sequence(item), str(item.get("event_id") or "")),
        )


# ---------------------------------------------------------------------------
# Markdown recovery report (paper-refine's RECOVERY.md)
# ---------------------------------------------------------------------------

_SECTION_START = "<!-- recovery-run:%s -->"
_SECTION_END = "<!-- /recovery-run:%s -->"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _neutralize_markers(text: str) -> str:
    """Escape managed-section markers embedded in user/agent-authored text."""
    return (
        str(text)
        .replace("<!-- /recovery-run:", "&lt;!-- /recovery-run:")
        .replace("<!-- recovery-run:", "&lt;!-- recovery-run:")
    )


class MarkdownReport:
    """A markdown report that preserves earlier runs and delimits each run.

    Sections are delimited by HTML comments keyed by run id, so a later run can
    replace exactly its own section, keep every earlier section, and mark a
    previous ``IN PROGRESS`` section as interrupted.

    Example: ``report.upsert_section(run_id, body)`` writes
    ``RECOVERY.md`` with every prior run intact.

    Counterexample: truncating and rewriting the file would erase the history
    the policy requires when an output root is reused.
    """

    def __init__(
        self,
        path: Any,
        *,
        fallback_path: Any = None,
        title: str = "Recovery report",
    ) -> None:
        self.path = Path(path)
        self.fallback_path = Path(fallback_path) if fallback_path is not None else None
        self.title = title

    def read(self) -> str:
        """Return the current report text (empty when it does not exist).

        Example: ``report.read()`` after two runs contains both run sections.

        Counterexample: an unreadable file is not treated as empty silently by
        writers; :meth:`upsert_section` surfaces the error through
        :class:`ReportWriteError` rather than discarding history.
        """
        try:
            return self.path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise ReportWriteError("cannot read recovery report %s: %s" % (self.path, exc)) from exc

    def upsert_section(self, run_id: str, body: str) -> Path:
        """Insert or replace one run's delimited section and write the report.

        Example: finalizing run ``r2`` replaces its in-progress section while
        run ``r1`` remains byte-for-byte intact.

        Counterexample: a body without the managed markers is still wrapped;
        callers must not supply their own section markers.
        """
        text = self.read() or "# %s\n" % self.title
        if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
            raise RecoveryPolicyError(
                "run id %r is not a safe report section identifier" % (run_id,)
            )
        # Record prose is agent-authored.  An embedded managed-section marker
        # would split the next upsert and leave attacker-chosen text outside
        # the section, so display it as escaped text instead.
        body = _neutralize_markers(str(body))
        start = _SECTION_START % run_id
        end = _SECTION_END % run_id
        section = "%s\n%s\n%s" % (start, body.rstrip(), end)
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _old, tail = rest.split(end, 1)
            text = head + section + tail
        else:
            text = text.rstrip() + "\n\n" + section + "\n"
        return self._write(text)

    def mark_interrupted(self, active_run_id: str) -> int:
        """Mark other still-``IN PROGRESS`` sections as interrupted.

        Example: a crashed earlier run is shown as ``INTERRUPTED`` when the next
        run starts.

        Counterexample: finalized sections are never rewritten; their recorded
        status is history.
        """
        try:
            text = self.read()
        except ReportWriteError:
            return 0
        if not text:
            return 0
        marker = "Status: **IN PROGRESS**"
        replacement = "Status: **INTERRUPTED** (the process ended before finalizing this run)"
        changed = 0
        for run_id in re.findall(r"<!-- recovery-run:([^ ]+) -->", text):
            if run_id == active_run_id:
                continue
            start = _SECTION_START % run_id
            end = _SECTION_END % run_id
            if start not in text or end not in text:
                continue
            head, rest = text.split(start, 1)
            section, tail = rest.split(end, 1)
            if marker in section:
                text = head + start + section.replace(marker, replacement) + end + tail
                changed += 1
        if changed:
            self._write(text)
        return changed

    def _write(self, text: str) -> Path:
        try:
            # A report may only be written where its directory already exists.
            # Creating the parent here would turn a mistyped --root into an
            # empty pipeline root, so a missing parent is a write failure that
            # falls back to the documented location.
            if not self.path.parent.is_dir():
                raise FileNotFoundError(
                    "recovery report directory does not exist: %s" % self.path.parent
                )
            _atomic_write_text(self.path, text)
            return self.path
        except OSError as exc:
            if self.fallback_path is None:
                raise ReportWriteError(
                    "cannot write recovery report %s: %s" % (self.path, exc)
                ) from exc
            try:
                _atomic_write_text(self.fallback_path, text)
                return self.fallback_path
            except OSError as fallback_exc:
                raise ReportWriteError(
                    "cannot write recovery report %s (%s) nor fallback %s (%s)"
                    % (self.path, exc, self.fallback_path, fallback_exc)
                ) from fallback_exc


# ---------------------------------------------------------------------------
# Failure policy and controller
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FailurePolicy:
    """The strict ``diagnostic_rank > fail_tier_rank`` termination policy.

    Example: ``FailurePolicy(TIER_WARNING)`` lets a WARNING recover and makes
    an ERROR terminate.

    Counterexample: a lower-threshold comparison (``>=``) would terminate on a
    diagnostic exactly equal to the threshold, contradicting the documented
    equality rule.
    """

    fail_tier: int = DEFAULT_FAIL_TIER

    def __post_init__(self) -> None:
        object.__setattr__(self, "fail_tier", parse_fail_tier(self.fail_tier))

    def exceeds(self, diagnostic: Diagnostic) -> bool:
        """Whether ``diagnostic`` is strictly above this policy's threshold.

        Example: an ERROR diagnosis under ``FailurePolicy(TIER_WARNING)``
        returns ``True``.

        Counterexample: an ERROR diagnosis under ``FailurePolicy(TIER_ERROR)``
        returns ``False``; equality recovers and continues.
        """
        return diagnostic.rank > self.fail_tier

    def describe(self, diagnostic: Diagnostic) -> str:
        """A one-line policy statement naming code, tier, threshold, operation.

        Example: ``FailurePolicy(2).describe(error_diag)`` names both
        ``ERROR`` and ``WARNING`` and the affected operation.

        Counterexample: the text never claims the diagnostic was unrecoverable;
        the threshold is a policy choice, not a property of the tier.
        """
        return (
            "%s (%s rank %d) exceeds --fail-tier %s (rank %d) for operation %s; "
            "controlled termination requested"
            % (
                diagnostic.code,
                diagnostic.tier_name,
                diagnostic.rank,
                TIER_NAMES[self.fail_tier],
                self.fail_tier,
                diagnostic.operation,
            )
        )


def threshold_failure_message(diagnostic: Diagnostic, threshold: int) -> str:
    """The user-facing message for a threshold-triggered stop.

    Example: the message names the diagnostic code, severity, configured
    threshold and affected operation, as the CLI contract requires.

    Counterexample: printing only "failed" would leave an operator unable to
    know which threshold stopped the run or which operation was affected.
    """
    return (
        "TIER-%s-%s: %s (severity %s rank %d; configured --fail-tier %s rank %d; "
        "affected operation: %s). The diagnostic and its recovery record were "
        "preserved before the stop; a threshold at or above this severity would "
        "continue with recovery."
        % (
            diagnostic.tier_name,
            diagnostic.code,
            redact_text(diagnostic.message),
            diagnostic.tier_name,
            diagnostic.rank,
            TIER_NAMES.get(threshold, "?"),
            threshold,
            diagnostic.operation,
        )
    )


@dataclasses.dataclass
class RecoveryResult:
    """The outcome of a bounded recovery attempt.

    Example: ``result.ok`` is ``False`` and ``result.error`` holds the last
    exception when every attempt and the fallback failed.

    Counterexample: ``degraded`` does not mean the requested work was fully
    validated; callers must propagate it to their status output.
    """

    ok: bool
    value: Any = None
    degraded: bool = False
    error: Optional[BaseException] = None
    attempts: int = 0
    record: Optional[RecoveryRecord] = None


class RecoveryController:
    """Emit diagnostics, apply the fail-tier policy and journal every recovery.

    The controller is deliberately dependency-free (no pipeline imports), so
    both repositories exercise exactly the same semantics.

    Example: configure it with a :class:`RecoveryJournal`, call
    :meth:`run_with_recovery` around a flaky operation, and pass its summary to
    the repository-specific report renderer.

    Counterexample: catching :class:`ThresholdExceeded` inside a stage and
    continuing would defeat the configured threshold; only cleanup belongs in
    that handler.
    """

    def __init__(
        self,
        *,
        run_id: str,
        policy: Optional[FailurePolicy] = None,
        journal: Optional[RecoveryJournal] = None,
        report_callback: Optional[Callable[["RecoveryController"], None]] = None,
        logger: Optional[Callable[[str], None]] = None,
        clock: Optional[Callable[[], _dt.datetime]] = None,
        exit_code: int = THRESHOLD_EXIT_CODE,
    ) -> None:
        self.run_id = run_id
        self.policy = policy or FailurePolicy()
        self.journal = journal
        self.report_callback = report_callback
        self.logger = logger or (lambda line: print(line, file=sys.stderr))
        self.clock = clock
        self.exit_code = exit_code
        self._sequence = 0
        self._lock = threading.RLock()
        self.records: List[RecoveryRecord] = []

    def emit(
        self,
        diagnostic: Diagnostic,
        *,
        recovery_action: str,
        outcome: str = "recorded",
        attempt: int = 1,
        max_attempts: int = 1,
        fallback: str = "",
        omissions: Sequence[str] = (),
        substitutions: Sequence[str] = (),
        affected_outputs: Sequence[str] = (),
        validation: str = "",
        remaining_uncertainty: str = "",
        human_review: bool = False,
        output_effects: str = "",
        failure: Optional[str] = None,
        journal: bool = True,
    ) -> RecoveryRecord:
        """Record one diagnostic and enforce the fail-tier policy.

        A diagnostic strictly above the threshold is recorded first and then
        raises :class:`ThresholdExceeded`; at or below it, the record is
        returned and the caller continues with recovery.

        Example: ``emit(Diagnostic(TIER_ERROR, ...), recovery_action="retry")``
        records, logs, and continues under the default CRITICAL threshold.

        Counterexample: emitting a record for a successful check would inflate
        the audit; only incidents and their recoveries belong here.
        """
        with self._lock:
            self._sequence += 1
            event_id = (
                self.journal.next_event_id()
                if self.journal is not None
                else "%s-%04d" % (self.run_id, self._sequence)
            )
            record = RecoveryRecord(
                run_id=self.run_id,
                event_id=event_id,
                timestamp=_timestamp(self.clock),
                sequence=self._sequence,
                code=diagnostic.code,
                tier=diagnostic.tier,
                operation=diagnostic.operation,
                failure=redact_text(failure or diagnostic.message),
                recovery_action=recovery_action,
                attempt=attempt,
                max_attempts=max_attempts,
                outcome=outcome,
                fallback=fallback,
                omissions=list(omissions),
                substitutions=list(substitutions),
                affected_outputs=list(affected_outputs),
                validation=validation,
                remaining_uncertainty=remaining_uncertainty,
                human_review=human_review,
                output_effects=output_effects,
                context=diagnostic.record_context(),
            )
            self.records.append(record)
            persisted = True
            if journal and self.journal is not None:
                persisted = self.journal.append(record.to_dict())
            if not persisted:
                # The threshold stop is still enforced, but a reader must not
                # believe the record reached the durable journal.
                self._log_text(
                    "TIER-WARNING-AUDIT_WRITE_FAILED: %s record retained only "
                    "in memory/pending; the configured fail-tier stop still "
                    "applies" % record.code
                )
            self._log(record)
        if self.report_callback is not None:
            try:
                self.report_callback(self)
            except Exception:  # report failures are themselves handled by the caller
                pass
        if self.policy.exceeds(diagnostic):
            raise ThresholdExceeded(diagnostic, self.policy.fail_tier, self.exit_code)
        return record

    def info(
        self,
        code: str,
        operation: str,
        message: str,
        **kwargs: Any,
    ) -> RecoveryRecord:
        """Record-and-continue for an INFORMATION diagnostic.

        Example: a skipped optional capability is recorded at INFO with
        ``outcome="continued"``.

        Counterexample: using INFO for a failed requested operation would hide
        an omission that a reviewer must see.
        """
        kwargs.setdefault("outcome", "continued")
        kwargs.setdefault("recovery_action", "recorded and continued")
        return self.emit(Diagnostic(TIER_INFO, code, operation, message), **kwargs)

    def warn(
        self,
        code: str,
        operation: str,
        message: str,
        **kwargs: Any,
    ) -> RecoveryRecord:
        """Record a WARNING and the recovery that keeps the run going.

        Example: a transient subprocess failure with a successful retry is
        recorded at WARNING and ``outcome="recovered"``.

        Counterexample: a WARNING whose retries were exhausted is not reported
        as ``recovered``; use ``outcome="exhausted"`` or ``"degraded"``.
        """
        return self.emit(Diagnostic(TIER_WARNING, code, operation, message), **kwargs)

    def error(
        self,
        code: str,
        operation: str,
        message: str,
        **kwargs: Any,
    ) -> RecoveryRecord:
        """Record an ERROR and its explicit recovery action.

        Example: an unreadable input is retried through an alternative path;
        the record states the retry and its effect on the output.

        Counterexample: suppressing the exception without a recovery action is
        not an ERROR recovery and must not be recorded as one.
        """
        return self.emit(Diagnostic(TIER_ERROR, code, operation, message), **kwargs)

    def critical(
        self,
        code: str,
        operation: str,
        message: str,
        **kwargs: Any,
    ) -> RecoveryRecord:
        """Record a CRITICAL incident and the recovery that preserves work.

        Example: an unexpected exception at a work boundary is recorded with a
        fallback that marks the affected item incomplete and continues others.

        Counterexample: a CRITICAL diagnostic does not imply termination; under
        the default threshold it follows its recovery routine.
        """
        return self.emit(Diagnostic(TIER_CRITICAL, code, operation, message), **kwargs)

    def run_with_recovery(
        self,
        diagnostic: Diagnostic,
        action: str,
        function: Callable[[int], Any],
        *,
        max_attempts: int = 2,
        fallback: Optional[Callable[[BaseException], Any]] = None,
        fallback_description: str = "",
        omissions: Sequence[str] = (),
        affected_outputs: Sequence[str] = (),
        validation: str = "",
        remaining_uncertainty: str = "",
        human_review: bool = False,
    ) -> RecoveryResult:
        """Run ``function`` with bounded retries, records and an optional fallback.

        Every failed attempt emits a diagnostic *before* the next attempt, so
        the fail-tier policy is evaluated when the incident occurs rather than
        only after retries are exhausted.  Exhaustion without a fallback
        returns a failed :class:`RecoveryResult`; it never raises the work
        exception and never terminates an ``always-continue`` run.

        Example: ``run_with_recovery(diag, "retry read", read, max_attempts=2,
        fallback=use_default)`` returns a degraded result when both reads fail.

        Counterexample: ``max_attempts=0`` is rejected; an unbounded loop would
        turn a transient incident into a hang.
        """
        if max_attempts < 1:
            raise RecoveryPolicyError("max_attempts must be >= 1")
        last_error: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                value = function(attempt)
                if attempt == 1:
                    # Nothing failed: recording "recovered" would inflate the
                    # audit and make the fail-tier policy stop a successful
                    # operation.
                    return RecoveryResult(True, value, False, None, attempt, None)
                record = self.emit(
                    Diagnostic(
                        diagnostic.tier,
                        diagnostic.code,
                        diagnostic.operation,
                        "Recovered on attempt %d of %d: %s"
                        % (attempt, max_attempts, diagnostic.message),
                    ),
                    recovery_action=action,
                    outcome="recovered",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    affected_outputs=affected_outputs,
                    validation=validation,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=human_review,
                    output_effects="the requested operation completed after recovery; "
                    "no omission was introduced by this recovery",
                )
                return RecoveryResult(True, value, False, None, attempt, record)
            except ThresholdExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - the boundary owns the category
                last_error = exc
                exhausted = attempt >= max_attempts
                self.emit(
                    Diagnostic(
                        diagnostic.tier,
                        diagnostic.code,
                        diagnostic.operation,
                        "%s (attempt %d of %d failed: %s)"
                        % (diagnostic.message, attempt, max_attempts, redact_text(str(exc))),
                    ),
                    recovery_action=action,
                    # The final failed attempt is always "exhausted"; a successful
                    # fallback then adds one "degraded" record.  Labelling the last
                    # attempt "degraded" as well double-counted the same incident.
                    outcome="retried" if not exhausted else "exhausted",
                    attempt=attempt,
                    max_attempts=max_attempts,
                    fallback=fallback_description if fallback is not None else "",
                    affected_outputs=affected_outputs,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=human_review or (exhausted and fallback is None),
                    output_effects=(
                        "the attempt failed; another bounded attempt follows"
                        if not exhausted
                        else (
                            "all attempts failed; the fallback described by this record "
                            "determines the output's completeness"
                            if fallback is not None
                            else "all attempts failed; the affected work is incomplete and "
                            "the output must be treated as partial"
                        )
                    ),
                    failure=redact_text(str(exc)),
                )
        assert last_error is not None
        if fallback is not None:
            try:
                value = fallback(last_error)
                record = self.emit(
                    Diagnostic(
                        diagnostic.tier,
                        diagnostic.code,
                        diagnostic.operation,
                        "%s (fallback used after %d attempt(s))"
                        % (diagnostic.message, max_attempts),
                    ),
                    recovery_action=action,
                    outcome="degraded",
                    attempt=max_attempts,
                    max_attempts=max_attempts,
                    fallback=fallback_description,
                    omissions=omissions,
                    affected_outputs=affected_outputs,
                    validation=validation,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=True,
                    output_effects="a fallback result replaced the requested operation; the "
                    "output is degraded and the omissions listed here apply",
                )
                return RecoveryResult(True, value, True, last_error, max_attempts, record)
            except ThresholdExceeded:
                raise
            except Exception as fallback_exc:  # noqa: BLE001
                record = self.emit(
                    Diagnostic(
                        diagnostic.tier,
                        diagnostic.code,
                        diagnostic.operation,
                        "%s (fallback failed: %s)"
                        % (diagnostic.message, redact_text(str(fallback_exc))),
                    ),
                    recovery_action=action,
                    outcome="failed",
                    attempt=max_attempts,
                    max_attempts=max_attempts,
                    fallback=fallback_description,
                    omissions=omissions,
                    affected_outputs=affected_outputs,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=True,
                    output_effects="the requested operation and its fallback both failed; the "
                    "output is incomplete",
                )
                return RecoveryResult(False, None, True, fallback_exc, max_attempts, record)
        return RecoveryResult(False, None, True, last_error, max_attempts, self.records[-1])

    def guard_work(
        self,
        operation: str,
        function: Callable[[], Any],
        *,
        code: str = "UNEXPECTED_EXCEPTION",
        tier: int = TIER_ERROR,
        recovery_action: str = "preserve available work, mark the affected item incomplete, "
        "and continue independent work",
        fallback: Optional[Callable[[BaseException], Any]] = None,
        fallback_description: str = "",
        affected_outputs: Sequence[str] = (),
        omissions: Sequence[str] = (),
        validation: str = "",
        remaining_uncertainty: str = "",
    ) -> RecoveryResult:
        """Run one work item and convert an unexpected exception into a record.

        This is the work-boundary recovery required by the shared policy: one
        failed item does not terminate unrelated processing.  The optional
        fallback runs first; exactly one record then states the actual outcome
        (``degraded`` only when the fallback produced a result, ``failed`` when
        neither the work nor the fallback completed).

        Example: ``controller.guard_work("stage verify", run_stage)`` keeps the
        remaining stages alive when one stage raises unexpectedly.

        Counterexample: swallowing the exception without recording it would
        make the item look successful and hide an omission.
        """
        try:
            value = function()
            return RecoveryResult(True, value, False, None, 0, None)
        except ThresholdExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 - the boundary owns the category
            if fallback is None:
                self.emit(
                    Diagnostic(
                        tier,
                        code,
                        operation,
                        "Unexpected %s: %s"
                        % (type(exc).__name__, redact_text(str(exc))),
                    ),
                    recovery_action=recovery_action,
                    outcome="failed",
                    fallback="",
                    affected_outputs=affected_outputs,
                    omissions=omissions,
                    validation=validation,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=True,
                    output_effects=(
                        "the affected work item is incomplete; independent work "
                        "continues"
                    ),
                    failure="%s: %s" % (type(exc).__name__, redact_text(str(exc))),
                )
                return RecoveryResult(False, None, True, exc, 1, self.records[-1])
            try:
                value = fallback(exc)
            except ThresholdExceeded:
                raise
            except Exception as fallback_exc:  # noqa: BLE001
                self.emit(
                    Diagnostic(
                        tier,
                        "RECOVERY_FALLBACK_FAILED",
                        operation,
                        "The fallback for %s failed: %s"
                        % (code, redact_text(str(fallback_exc))),
                    ),
                    recovery_action=recovery_action,
                    outcome="failed",
                    fallback=fallback_description,
                    affected_outputs=affected_outputs,
                    omissions=omissions,
                    validation=validation,
                    remaining_uncertainty=remaining_uncertainty,
                    human_review=True,
                    output_effects="neither the requested work nor its fallback "
                    "completed; the output is incomplete",
                    failure=redact_text(str(fallback_exc)),
                )
                return RecoveryResult(
                    False, None, True, fallback_exc, 1, self.records[-1]
                )
            # The fallback succeeded before this record was written, so
            # ``degraded`` states what actually reached the output.
            record = self.emit(
                Diagnostic(
                    tier,
                    code,
                    operation,
                    "Unexpected %s: %s"
                    % (type(exc).__name__, redact_text(str(exc))),
                ),
                recovery_action=recovery_action,
                outcome="degraded",
                fallback=fallback_description,
                affected_outputs=affected_outputs,
                omissions=omissions,
                validation=validation,
                remaining_uncertainty=remaining_uncertainty,
                human_review=True,
                output_effects="a fallback result replaced the failed item; the "
                "affected work and any listed omissions must be reviewed",
                failure="%s: %s" % (type(exc).__name__, redact_text(str(exc))),
            )
            return RecoveryResult(True, value, True, exc, 1, record)

    def flush_audit(self) -> List[Dict[str, Any]]:
        """Flush pending journal records and report audit failures once.

        The audit destination has its own recovery: records are retained and
        written to the documented fallback.  This method never raises, so an
        audit failure cannot defeat ``always-continue``.

        Example: a CLI calls it in ``finally`` and prints the fallback path.

        Counterexample: do not call :meth:`emit` from here, which would recurse
        into the same failing journal.
        """
        alerts: List[Dict[str, Any]] = []
        if self.journal is not None:
            self.journal.flush_pending()
            alerts = self.journal.drain_alerts()
            for alert in alerts:
                self._log(
                    "TIER-WARNING-AUDIT_WRITE_FAILED: %s (%s); records retained for fallback %s"
                    % (alert.get("code"), alert.get("message"), alert.get("fallback"))
                )
        return alerts

    def summary(self) -> Dict[str, Any]:
        """Counts and the affected-output/review picture for a final report.

        Example: ``summary()["outcomes"]`` lets a report state how many events
        were recovered, degraded or exhausted.

        Counterexample: counts alone are not a summary of consequences; the
        report also lists affected outputs and review needs.
        """
        counts: Dict[str, int] = {}
        outputs: List[str] = []
        review: List[str] = []
        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1
            for path in record.affected_outputs:
                if path not in outputs:
                    outputs.append(path)
            if record.needs_review:
                review.append(record.code)
        return {
            "events": len(self.records),
            "outcomes": counts,
            "affected_outputs": outputs,
            "review_codes": sorted(set(review)),
            "fail_tier": TIER_NAMES[self.policy.fail_tier],
            "fail_tier_rank": self.policy.fail_tier,
        }

    def _log_text(self, line: str) -> None:
        try:
            self.logger(line)
        except Exception:  # a logger must never break the diagnostic path
            pass

    def _log(self, record: RecoveryRecord) -> None:
        try:
            line = record.to_dict()["formatted"]
        except Exception:  # a malformed record must not break logging
            return
        self._log_text(line)


# ---------------------------------------------------------------------------
# Rendering helpers shared by both repositories
# ---------------------------------------------------------------------------


def recovery_summary_line(records: Sequence[RecoveryRecord]) -> str:
    """A concise one-line summary of recovery outcomes and review needs.

    Example: ``"3 recovery event(s): 1 recovered, 1 degraded, 1 exhausted; 2
    event(s) require human review."``

    Counterexample: "recovery succeeded" is not acceptable when any event is
    degraded or exhausted.
    """
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.outcome] = counts.get(record.outcome, 0) + 1
    parts = ", ".join("%d %s" % (count, outcome) for outcome, count in sorted(counts.items()))
    review = sum(1 for record in records if record.needs_review)
    return "%d recovery event(s): %s; %d event(s) require human review." % (
        len(records),
        parts or "none",
        review,
    )


def render_commit_section(
    records: Iterable[Any],
    *,
    heading: str = "Recovery (diagnostics, substitutions and omissions)",
) -> str:
    """Render the commit-message recovery section for generated Git commits.

    Accepts :class:`RecoveryRecord` objects or their JSON dictionaries, so a
    later process can render records from the run's journal.  Returns an empty
    string when there are no records, so a clean run's message is unchanged.

    Example: a commit after a failed commit attempt and a skipped test check
    gains a section listing both and what the committed change set omits.

    Counterexample: never render "all checks passed" here; a skipped check is
    stated as skipped even when the commit itself succeeded.
    """
    items = [_coerce_record(item) for item in (records or ())]
    items = [item for item in items if item is not None]
    if not items:
        return ""
    lines = ["", "%s" % heading, ""]
    lines.append(
        "The pipeline recorded %d diagnostic recovery event(s) relevant to this commit. "
        "Read this section before relying on the committed change set."
        % len(items)
    )
    lines.append("")
    lines.append(recovery_summary_line(items))
    lines.append("")
    for index, record in enumerate(items, 1):
        lines.append(record.to_markdown(index=index))
        lines.append("")
    affected = sorted({path for record in items for path in record.affected_outputs})
    if affected:
        lines.append("Affected outputs: %s" % ", ".join(affected))
        lines.append("")
    review = sorted({record.code for record in items if record.needs_review})
    lines.append(
        "Human review required for: %s"
        % (", ".join(review) if review else "no recovery event above")
    )
    return "\n".join(lines).rstrip() + "\n"


def render_run_markdown(
    run_id: str,
    *,
    status: str,
    records: Iterable[Any],
    started_at: str = "",
    finished_at: str = "",
    command: str = "",
    configuration: Optional[Mapping[str, Any]] = None,
    summary: Optional[Mapping[str, Any]] = None,
    empty_text: str = "No recovery events were recorded for this run.",
) -> str:
    """Render one run's self-contained markdown section for a recovery report.

    Example: ``paper-refine`` renders this under each run heading in
    ``<root>/RECOVERY.md`` with status, configuration, events and a summary.

    Counterexample: omitting the status would make an interrupted or
    threshold-terminated run look like a completed one.
    """
    items = [_coerce_record(item) for item in (records or ())]
    items = [item for item in items if item is not None]
    # The run id is caller/CLI supplied; collapse line breaks and render it as
    # a code span whose fence is longer than any backtick run in the value, so
    # it cannot forge a heading or an inline link.
    safe_run_id = _one_line(run_id)
    lines = [
        "## Run %s" % _inline_code(safe_run_id),
        "",
        "Status: **%s**" % _one_line(status),
        "",
        "- Started: %s" % (started_at or "not recorded"),
        "- Finished: %s" % (finished_at or "not recorded"),
        "- Command: %s" % _inline_code(_one_line(command or "not recorded")),
    ]
    config = redact_value(dict(configuration or {}))
    if config:
        lines.append(
            "- Configuration: %s"
            % _inline_code(json.dumps(config, sort_keys=True))
        )
    lines.append("")
    if not items:
        lines.extend([empty_text, ""])
    else:
        lines.append("### Summary")
        lines.append("")
        lines.append(recovery_summary_line(items))
        lines.append("")
        outcomes: Dict[str, int] = {}
        for item in items:
            outcomes[item.outcome] = outcomes.get(item.outcome, 0) + 1
        lines.append(
            "Outcomes: %s"
            % ", ".join("%s=%d" % (name, count) for name, count in sorted(outcomes.items()))
        )
        affected = sorted({path for item in items for path in item.affected_outputs})
        if affected:
            lines.append("Affected outputs: %s" % ", ".join(affected))
        review = sorted({item.code for item in items if item.needs_review})
        lines.append(
            "Human review needed for: %s" % (", ".join(review) if review else "no event")
        )
        lines.extend(["", "### Recovery events", ""])
        for index, item in enumerate(items, 1):
            lines.append(item.to_markdown(index=index))
            lines.append("")
    if summary:
        lines.extend(["### Run summary", "", "```json", json.dumps(redact_value(dict(summary)), indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _coerce_record(item: Any) -> Optional[RecoveryRecord]:
    if isinstance(item, RecoveryRecord):
        return item
    if isinstance(item, Mapping):
        payload = dict(item)
        payload.pop("formatted", None)
        fields = {field.name for field in dataclasses.fields(RecoveryRecord)}
        try:
            return RecoveryRecord(
                **{key: value for key, value in payload.items() if key in fields}
            )
        except (TypeError, ValueError):
            # A hand-edited/older journal entry may be missing required fields;
            # one malformed record must not abort every other record's section.
            return None
    return None


#: Public names re-exported for consumers that only need the taxonomy.
__all__ = [
    "TIER_DEBUG",
    "TIER_INFO",
    "TIER_WARNING",
    "TIER_ERROR",
    "TIER_CRITICAL",
    "TIER_NAMES",
    "TIER_RANKS",
    "DEFAULT_FAIL_TIER",
    "THRESHOLD_EXIT_CODE",
    "OUTCOMES",
    "RecoveryPolicyError",
    "FailTierError",
    "ThresholdExceeded",
    "ReportWriteError",
    "tier_rank",
    "tier_name",
    "parse_fail_tier",
    "fail_tier_help",
    "diagnostic_code",
    "format_diagnostic",
    "redact_value",
    "redact_text",
    "Diagnostic",
    "RecoveryRecord",
    "new_run_id",
    "RecoveryJournal",
    "MarkdownReport",
    "FailurePolicy",
    "threshold_failure_message",
    "RecoveryResult",
    "RecoveryController",
    "record_sequence",
    "recovery_summary_line",
    "render_commit_section",
    "render_run_markdown",
]
