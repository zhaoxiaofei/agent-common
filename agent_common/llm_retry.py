"""Shared per-LLM-call retry policy and a retrying HTTP proxy.

The policy is data (``llm_policy.json`` beside this module): exponential
backoff starts at 10 seconds, doubles, and is capped at 3600 seconds.  The
same policy drives three consumers:

* the generated DSH ``llm-pi-ai`` provider ``retryPolicy``;
* direct ``urllib`` clients in paper-refine and paper-refine-eval;
* :class:`RetryProxy`, a loopback HTTP proxy used for the compaction
  summarizer, whose DSH call site has no retry wrapper of its own.

Everything here is standard library only so a copied module works in a frozen
paper-refine root.
"""

from __future__ import annotations

import copy
import email.utils
import http.client
import http.server
import json
import math
import random
import ssl
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

#: Connection-scoped headers that must never be copied from the client to the
#: upstream (or in a response).  ``Proxy-Authorization`` belongs to the loopback
#: hop only; forwarding it would hand the proxy credential to the upstream.
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "proxy-connection",
        "proxy-authorization",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: Response headers the loopback server generates itself.  Copying the
#: upstream's value as well is what produced duplicate Server/Date values.
PROXY_RESPONSE_HEADERS = frozenset({"server", "date", "content-length"})


def _connection_tokens(values: Iterable[Any]) -> set:
    """Header names the Connection header declares hop-by-hop (RFC 7230 6.1)."""
    tokens = set()
    for value in values:
        for token in str(value or "").split(","):
            name = token.strip().lower()
            if name:
                tokens.add(name)
    return tokens


def _request_is_chunked(headers: Mapping[str, str]) -> bool:
    """Whether the client used unsupported chunked framing."""
    transfer = str(headers.get("Transfer-Encoding") or "").lower()
    return "chunked" in transfer


def _declared_content_length(headers: Mapping[str, str]) -> Optional[int]:
    """The request's Content-Length, ``None`` when absent.

    Raises ``ValueError`` for an unparsable or negative value; treating it as
    zero silently forwarded a corrupted request body.
    """
    raw = headers.get("Content-Length")
    if raw in (None, ""):
        return None
    try:
        length = int(raw)
    except (TypeError, ValueError):
        raise ValueError("invalid Content-Length header") from None
    if length < 0:
        raise ValueError("invalid Content-Length header")
    return length


def _read_request_body(
    rfile: Any, headers: Mapping[str, str]
) -> Optional[bytes]:
    """Read the declared request body, or ``None`` for an empty one."""
    length = _declared_content_length(headers)
    return rfile.read(length) if length else None


def _forwardable_request_headers(
    headers: Mapping[str, str], *, extra_skip: Iterable[str] = ()
) -> Dict[str, str]:
    """Copy request headers, dropping the hop-by-hop and framing ones."""
    skip = set(HOP_BY_HOP_HEADERS) | {"host", "content-length"}
    skip.update(str(name).lower() for name in extra_skip)
    # RFC 7230 also makes every header named by Connection hop-by-hop.  The
    # fixed set alone forwarded ``Connection: X-Secret`` together with
    # ``X-Secret: SECRET`` to the upstream.
    skip.update(
        _connection_tokens(
            value
            for name, value in headers.items()
            if name.lower() == "connection"
        )
    )
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in skip
    }


def _forwardable_response_headers(
    headers: Iterable[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Upstream response headers safe to re-emit from the loopback server."""
    items = list(headers)
    skip = set(HOP_BY_HOP_HEADERS) | set(PROXY_RESPONSE_HEADERS)
    skip.update(
        _connection_tokens(
            value for name, value in items if name.lower() == "connection"
        )
    )
    return [
        (name, value)
        for name, value in items
        if name.lower() not in skip
    ]


def _relay_response_body(response: Any, write: Callable[[bytes], None]) -> None:
    """Stream an upstream body to ``write`` until EOF, swallowing a broken pipe."""
    read_chunk = getattr(response, "read1", response.read)
    try:
        while True:
            chunk = read_chunk(65536)
            if not chunk:
                break
            write(chunk)
    except (OSError, ValueError, http.client.HTTPException):
        # The response headers are already on the wire; close the truncated
        # stream instead of letting the handler thread die.
        return


#: Fallback used when ``llm_policy.json`` is absent or malformed.  Keep this in
#: sync with the canonical JSON; ``agent-common/check_llm_policy.py`` verifies it.
DEFAULT_POLICY: Dict[str, Any] = {
    "schema_version": 1,
    "retry": {
        "initial_delay_seconds": 10,
        "multiplier": 2,
        "max_delay_seconds": 3600,
        "max_retries": 10,
        "jitter_ratio": 0.1,
        "retry_after_cap_seconds": 3600,
        "retryable_http_statuses": [408, 409, 425, 429, 500, 502, 503, 504, 529],
        "retryable_error_codes": [
            "RATE_LIMIT",
            "QUOTA",
            "SERVER",
            "TIMEOUT",
            "TRANSPORT",
            "EMPTY_RESPONSE",
        ],
    },
    "dots": {
        "regular_api_key_env": "DOTS_API_KEY",
        "compaction_api_key_env": "DOTS_API_KEY_1474",
        "compaction_api_key_aliases": [
            "DOTS_AI_API_KEY_1474",
            "PAPER_DOTS_COMPACTION_API_KEY",
            "DOTS_COMPACTION_API_KEY",
        ],
        "compaction_provider": "dots-compaction",
        "compaction_provider_label": "Dots API compaction (dots3-note-prev, dedicated key)",
    },
    "compaction": {
        "compaction_retries": 2,
        "max_overflow_retries": 2,
        "compaction_stream_idle_timeout_ms": 4200000,
        "summarization_max_tokens": 16384,
    },
}


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _merge_policy(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a policy document and fill absent fields from the defaults."""
    policy = _copy(DEFAULT_POLICY)
    for section in ("retry", "dots", "compaction"):
        incoming = document.get(section)
        if not isinstance(incoming, Mapping):
            continue
        policy[section].update(
            {key: value for key, value in incoming.items() if value is not None}
        )
    retry = policy["retry"]
    retry["initial_delay_seconds"] = float(retry["initial_delay_seconds"])
    retry["multiplier"] = float(retry["multiplier"])
    retry["max_delay_seconds"] = float(retry["max_delay_seconds"])
    retry["max_retries"] = int(retry["max_retries"])
    retry["jitter_ratio"] = float(retry["jitter_ratio"])
    retry["retry_after_cap_seconds"] = float(retry["retry_after_cap_seconds"])
    # A string is iterable, so the old code accepted "429" and expanded it to
    # the character codes [2, 4, 9] instead of the one HTTP status 429.
    statuses = retry.get("retryable_http_statuses") or []
    if not isinstance(statuses, (list, tuple)):
        raise ValueError("retry.retryable_http_statuses must be a list of integers")
    retry["retryable_http_statuses"] = sorted({int(status) for status in statuses})
    codes = retry.get("retryable_error_codes") or []
    if not isinstance(codes, (list, tuple)):
        raise ValueError("retry.retryable_error_codes must be a list of strings")
    retry["retryable_error_codes"] = sorted({str(code) for code in codes if code})
    for key in (
        "initial_delay_seconds",
        "multiplier",
        "max_delay_seconds",
        "jitter_ratio",
        "retry_after_cap_seconds",
    ):
        if not math.isfinite(retry[key]):
            # NaN comparisons are always false, so a NaN bound used to pass every
            # ``<``/``<=`` check and produce nonsense delays (or a hot retry loop).
            raise ValueError("retry.%s must be finite" % key)
    if retry["initial_delay_seconds"] <= 0:
        raise ValueError("retry.initial_delay_seconds must be positive")
    if retry["multiplier"] <= 0:
        raise ValueError("retry.multiplier must be positive")
    if retry["max_delay_seconds"] < retry["initial_delay_seconds"]:
        raise ValueError("retry.max_delay_seconds must not be below the initial delay")
    if retry["retry_after_cap_seconds"] < 0:
        raise ValueError("retry.retry_after_cap_seconds must be non-negative")
    if retry["max_retries"] < 0:
        raise ValueError("retry.max_retries must be non-negative")
    if not 0.0 <= retry["jitter_ratio"] <= 1.0:
        raise ValueError("retry.jitter_ratio must be between 0 and 1")
    # The dots section is merged straight into the policy and later consumed by
    # os.environ.get(...)/YAML generation.  A list/null key-env or provider used
    # to load "successfully" and crash on the first request.
    dots = policy["dots"]
    for key in (
        "regular_api_key_env",
        "compaction_api_key_env",
        "compaction_provider",
        "compaction_provider_label",
    ):
        if key in dots and dots[key] is not None:
            if not isinstance(dots[key], str) or not dots[key].strip():
                raise ValueError("dots.%s must be a non-empty string" % key)
    aliases = dots.get("compaction_api_key_aliases")
    if aliases is not None:
        if not isinstance(aliases, (list, tuple)) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError(
                "dots.compaction_api_key_aliases must be a list of non-empty strings"
            )
        dots["compaction_api_key_aliases"] = [str(alias) for alias in aliases]
    # The compaction tuning fields are consumed by int(...) in the generated
    # DSH patch; a typo used to load "successfully" and then crash patch
    # generation, defeating load_policy's documented fallback.
    compaction = policy["compaction"]
    for key in ("compaction_retries", "max_overflow_retries"):
        compaction[key] = int(compaction.get(key) or 0)
        if compaction[key] < 0:
            raise ValueError("compaction.%s must be non-negative" % key)
    if compaction.get("summarization_max_tokens") is not None:
        compaction["summarization_max_tokens"] = int(
            compaction["summarization_max_tokens"]
        )
        if compaction["summarization_max_tokens"] <= 0:
            raise ValueError("compaction.summarization_max_tokens must be positive")
    compaction["compaction_stream_idle_timeout_ms"] = int(
        compaction.get("compaction_stream_idle_timeout_ms") or 0
    )
    if compaction["compaction_stream_idle_timeout_ms"] < 0:
        raise ValueError(
            "compaction.compaction_stream_idle_timeout_ms must be non-negative"
        )
    return policy


def load_policy(path: Optional[Any] = None) -> Dict[str, Any]:
    """Read one policy document, falling back to the embedded defaults.

    A missing or malformed file is not fatal: an adapter copied into a frozen
    root may not carry the JSON beside it, and refusing to start a run over a
    tuning file would be worse than using the documented defaults.

    Example:
        ``load_policy("/nonexistent/llm_policy.json")`` returns the documented
        defaults (10 retries, 3600-second cap).

    Counterexample:
        A malformed file is not fatal either; both fall back to the defaults.  Do
        not use this function when the caller must reject a bad policy.
    """
    candidate = Path(path) if path is not None else Path(__file__).with_name(
        "llm_policy.json"
    )
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, RecursionError):
        # A valid but deeply nested document raises RecursionError, not ValueError; the
        # documented malformed-policy fallback must cover it instead of crashing import.
        return _copy(DEFAULT_POLICY)
    if not isinstance(document, Mapping):
        return _copy(DEFAULT_POLICY)
    try:
        return _merge_policy(document)
    except (TypeError, ValueError, OverflowError):
        return _copy(DEFAULT_POLICY)


def policy_paths(*candidates: Optional[Any]) -> Optional[Path]:
    """First existing policy path among ``candidates``, else the module copy.

    Example:
        ``policy_paths("/missing", "/etc/llm_policy.json")`` returns the first
        existing path.

    Counterexample:
        When no candidate exists it returns the module's own ``llm_policy.json``
        path, which may not exist; callers still need ``load_policy``'s fallback.
    """
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return Path(__file__).with_name("llm_policy.json")


class RetrySchedule:
    """The delay and retryable-failure policy for one process.

    Example:
        ``RetrySchedule(load_policy()).delays()`` returns the documented
        ``[10, 20, ..., 3600]`` schedule.

    Counterexample:
        ``RetrySchedule`` does not sleep; ``delay`` returns seconds.  Sleeping
        is the proxy's/caller's responsibility.
    """

    def __init__(self, policy: Optional[Mapping[str, Any]] = None):
        document = dict(policy or load_policy())
        retry = document.get("retry") or DEFAULT_POLICY["retry"]
        if not isinstance(retry, Mapping):
            raise ValueError("the retry policy must be a mapping")
        raw = dict(retry)
        # Validate through the same merge path as load_policy, but first fill
        # the optional fields with this constructor's historical defaults
        # (jitter off, empty retryable sets, Retry-After capped by max_delay)
        # so validating a direct schedule does not silently change it.
        base = {
            "initial_delay_seconds": raw.get(
                "initial_delay_seconds",
                DEFAULT_POLICY["retry"]["initial_delay_seconds"],
            ),
            "multiplier": raw.get("multiplier", 2.0),
            "max_delay_seconds": raw.get(
                "max_delay_seconds", DEFAULT_POLICY["retry"]["max_delay_seconds"]
            ),
            "max_retries": raw.get(
                "max_retries", DEFAULT_POLICY["retry"]["max_retries"]
            ),
            "jitter_ratio": raw.get("jitter_ratio", 0.0),
            "retry_after_cap_seconds": raw.get(
                "retry_after_cap_seconds",
                raw.get(
                    "max_delay_seconds",
                    DEFAULT_POLICY["retry"]["max_delay_seconds"],
                ),
            ),
            "retryable_http_statuses": raw.get("retryable_http_statuses") or [],
            "retryable_error_codes": raw.get("retryable_error_codes") or [],
        }
        retry = _merge_policy({"retry": base})["retry"]
        self.initial_delay_seconds = float(retry["initial_delay_seconds"])
        self.multiplier = float(retry["multiplier"])
        self.max_delay_seconds = float(retry["max_delay_seconds"])
        self.max_retries = int(retry["max_retries"])
        self.jitter_ratio = float(retry["jitter_ratio"])
        self.retry_after_cap_seconds = float(retry["retry_after_cap_seconds"])
        self.retryable_statuses = frozenset(
            int(status) for status in retry["retryable_http_statuses"]
        )
        self.retryable_codes = frozenset(
            str(code) for code in retry["retryable_error_codes"]
        )

    def delay(
        self,
        retry_number: int,
        *,
        retry_after: Optional[float] = None,
        jitter: bool = True,
    ) -> float:
        """Seconds before retry ``retry_number`` (1 = first retry).

        ``retry_after`` is the provider's own hint (for example from a 429
        ``Retry-After`` header); it wins up to the same 3600-second cap.

        Example:
            With the default policy, ``delay(1, jitter=False) == 10`` and
            ``delay(3, jitter=False) == 40``.  Jitter is on by default, so a plain
            ``delay(1)`` returns a random value within the policy's
            ``jitter_ratio`` of that schedule.

        Counterexample:
            ``delay(0)`` and ``delay(-1)`` both behave like the first retry; retry
            numbers are clamped at 1, not rejected.  A positive ``retry_after`` wins
            over exponential backoff up to the cap.
        """
        if retry_after is not None:
            try:
                requested = float(retry_after)
            except (TypeError, ValueError):
                requested = 0.0
            if requested > 0:
                # The cap is validated non-negative by _merge_policy; a
                # directly constructed schedule must not produce a negative
                # sleep either (time.sleep raises ValueError).
                return max(0.0, min(requested, self.retry_after_cap_seconds))
        exponent = max(0, int(retry_number) - 1)
        try:
            delay = self.initial_delay_seconds * (self.multiplier ** exponent)
        except OverflowError:
            # A long-running caller can reach an exponent whose float
            # exponentiation overflows; the documented cap is the answer, not
            # an exception.
            delay = math.inf
        delay = min(delay, self.max_delay_seconds)
        if jitter and self.jitter_ratio:
            factor = 1.0 - self.jitter_ratio + 2.0 * self.jitter_ratio * random.random()
            delay = min(delay * factor, self.max_delay_seconds)
        return max(0.0, delay)

    def delays(self, jitter: bool = False) -> List[float]:
        """The full configured schedule, useful for records and tests.

        Example:
            ``RetrySchedule(load_policy()).delays()`` returns
            ``[10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3600]``.

        Counterexample:
            ``delays(jitter=True)`` adds jitter and is not the documented schedule; use
            the default for records and tests.
        """
        return [
            self.delay(index + 1, jitter=jitter)
            for index in range(self.max_retries)
        ]

    def status_is_retryable(self, status: int) -> bool:
        """Whether an HTTP status is in the policy's retryable set.

        Example:
            The default policy returns ``True`` for 429 and 503.

        Counterexample:
            A client error such as 400 returns ``False``; not every failure is
            transient.
        """
        return int(status) in self.retryable_statuses

    def code_is_retryable(self, code: Optional[str]) -> bool:
        """Whether a provider error code is in the policy's retryable set.

        Example:
            ``code_is_retryable("RATE_LIMIT")`` is ``True`` by default.

        Counterexample:
            ``code_is_retryable(None)`` is ``False``; a missing code is not a
            retryable provider error.
        """
        return bool(code) and str(code) in self.retryable_codes


def retry_after_seconds(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    """Parse a ``Retry-After`` header as seconds from now.

    Example:
        ``retry_after_seconds({"Retry-After": "7"}) == 7.0``.

    Counterexample:
        An invalid or absent header returns ``None``; that means "no provider
        hint", not zero delay.
    """
    if not headers:
        return None
    raw = None
    for name, value in headers.items():
        if str(name).lower() == "retry-after":
            raw = str(value).strip()
            break
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        pass
    else:
        # The docstring contract: an invalid header means "no provider hint".
        # max(0.0, ...) turned a negative or NaN value into a false zero-delay
        # hint, which callers may then hot-loop on.
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return seconds
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    return max(0.0, parsed.timestamp() - time.time())


class _RetryProxyHandler(http.server.BaseHTTPRequestHandler):
    """One forwarded request, retried before any response byte reaches dsh."""

    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib name
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        self._handle("POST")

    def _chunked(self) -> bool:
        return _request_is_chunked(self.headers)

    def _body(self) -> Optional[bytes]:
        return _read_request_body(self.rfile, self.headers)

    def _handle(self, method: str) -> None:
        proxy: "RetryProxy" = self.server.proxy  # type: ignore[attr-defined]
        if self._chunked():
            # Reading only Content-Length silently forwarded a chunked POST without its
            # body. Reject the unsupported framing instead of corrupting a valid request.
            self._send_error(
                400, "chunked request bodies are not supported by the retry proxy"
            )
            return
        try:
            body = self._body()
        except ValueError as exc:
            # An unparsable Content-Length used to become a zero-length body,
            # silently forwarding a corrupted request downstream.
            self._send_error(400, str(exc))
            return
        schedule = proxy.schedule
        attempt = 0
        last_status = 0
        last_headers: List[Tuple[str, str]] = []
        last_body = b""
        while True:
            attempt += 1
            try:
                status, headers, response, connection = proxy.open(
                    method, self.path, self.headers, body
                )
            except ValueError as exc:
                # ``http.client`` encodes the request line as ASCII, so a
                # non-ASCII path raises UnicodeEncodeError (a ValueError); the
                # target_path validation above raises ValueError too.  Without
                # this guard the handler thread died and the client got an
                # empty response instead of a 4xx.
                self._send_error(400, str(exc))
                return
            except (
                OSError,
                TimeoutError,
                http.client.HTTPException,
                ssl.SSLError,
            ) as exc:
                if attempt > schedule.max_retries:
                    proxy.log(
                        "%s: upstream %s failed after %d attempt(s): %s"
                        % (proxy.name, method, attempt, exc)
                    )
                    self._send_error(502, "%s: %s" % (type(exc).__name__, exc))
                    return
                delay = schedule.delay(attempt)
                proxy.log(
                    "%s: upstream %s failed (%s); retry %d/%d in %.0fs"
                    % (
                        proxy.name,
                        method,
                        exc,
                        attempt,
                        schedule.max_retries,
                        delay,
                    )
                )
                time.sleep(delay)
                continue
            try:
                if 300 <= status < 400:
                    # Relay nothing: a client that follows the redirect
                    # re-sends the Authorization header to the Location host.
                    # The prep proxy refuses redirects for the same reason.
                    proxy.log(
                        "%s: upstream %s refused redirect HTTP %d"
                        % (proxy.name, method, status)
                    )
                    self._send_error(
                        502,
                        "retry proxy refused a redirect (HTTP %d); "
                        "credentials are not forwarded to another host" % status,
                    )
                    return
                if status < 400 or not schedule.status_is_retryable(status):
                    try:
                        self._relay(status, headers, response)
                    finally:
                        # The upstream connection is per-attempt; close it on
                        # the success path just like the retry path does.
                        response.close()
                        connection.close()
                    return
                last_status = status
                last_headers = headers
                try:
                    last_body = response.read()
                except (OSError, http.client.HTTPException) as exc:
                    # A truncated or stalled 5xx error body is exactly the
                    # failure class the proxy exists for: retry it instead of
                    # letting the exception kill the handler thread and close
                    # the client connection.  ``socket.timeout``/``TimeoutError``
                    # are ``OSError`` subclasses, not ``HTTPException``.
                    proxy.log(
                        "%s: upstream %s body read failed after HTTP %d: %s"
                        % (proxy.name, method, status, exc)
                    )
                    last_body = b""
            finally:
                response.close()
                connection.close()
            if attempt > schedule.max_retries:
                proxy.log(
                    "%s: upstream %s returned HTTP %d after %d attempt(s); "
                    "returning the last response"
                    % (proxy.name, method, status, attempt)
                )
                self._send(last_status, last_headers, last_body)
                return
            delay = schedule.delay(
                attempt, retry_after=retry_after_seconds(dict(last_headers))
            )
            proxy.log(
                "%s: upstream %s returned HTTP %d; retry %d/%d in %.0fs"
                % (
                    proxy.name,
                    method,
                    status,
                    attempt,
                    schedule.max_retries,
                    delay,
                )
            )
            time.sleep(delay)

    def _send_error(self, status: int, message: str) -> None:
        raw = json.dumps({"error": {"message": message}}).encode("utf-8")
        self._send(status, [("Content-Type", "application/json")], raw)

    def _send(
        self, status: int, headers: Iterable[Tuple[str, str]], body: bytes
    ) -> None:
        self.send_response(status)
        self._copy_headers(headers)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (OSError, ValueError):
            pass

    def _relay(
        self,
        status: int,
        headers: Iterable[Tuple[str, str]],
        response: http.client.HTTPResponse,
    ) -> None:
        # Buffer before writing the response line.  Streaming a body whose
        # upstream Content-Length was stripped let a truncated completion reach
        # the caller as a normal 200; an IncompleteRead must become a typed
        # error instead, and only a known-complete body may be sent.
        try:
            body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            self._send_error(
                502, "upstream response body was truncated: %s" % (exc,)
            )
            return
        self._send(status, headers, body)

    def _copy_headers(self, headers: Iterable[Tuple[str, str]]) -> None:
        # send_response() already wrote Server/Date/Content-Length; copying
        # them again produced duplicates.
        for name, value in _forwardable_response_headers(headers):
            self.send_header(name, value)


class _RetryProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


#: Credential-looking query names, normalised like
#: ``model_runtime._credential_key``.  This module deliberately stays
#: standalone (no sibling import), so the predicate is mirrored here rather
#: than importing model_runtime; the retry proxy must not accept a query
#: credential the prep proxy rejects.
_CREDENTIAL_QUERY_NAMES = frozenset(
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
        "passwd",
        "credential",
        "credentials",
        "authorization",
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
_CREDENTIAL_QUERY_SUFFIXES = (
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


def _credential_query_key(name: Any) -> bool:
    text = str(name or "").strip().lower()
    compact = "".join(character for character in text if character.isalnum())
    if compact in _CREDENTIAL_QUERY_NAMES:
        return True
    normalized = text.replace("-", "_").replace(" ", "_")
    return normalized.endswith(_CREDENTIAL_QUERY_SUFFIXES)


def _query_carries_credential(query: str) -> bool:
    """Whether an upstream base query contains a credential-named parameter."""
    if not query:
        return False
    try:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    except ValueError:
        # A query too malformed to inspect is not safe to forward verbatim.
        return True
    return any(_credential_query_key(key) for key, _value in pairs)


class RetryProxy:
    """A loopback HTTP proxy that retries each upstream LLM call.

    The proxy exists because the DSH compaction summarizer calls the LLM
    adapter directly, outside ``dsh-llm-retry``.  Pointing only the compaction
    provider at this proxy restores per-call backoff there without touching the
    installed harness.  The proxy never stores or rewrites credentials: the
    ``Authorization``/``api-key`` headers DSH sends are forwarded unchanged.

    Example:
        ``proxy = RetryProxy("https://api.example/v1"); base = proxy.start()``
        gives a loopback base URL to put in the provider patch.

    Counterexample:
        Do not use it as a general-purpose HTTP server: it only forwards to one
        configured upstream and only retries the policy's statuses.
    """

    def __init__(
        self,
        upstream_base: str,
        schedule: Optional[RetrySchedule] = None,
        *,
        name: str = "llm-retry-proxy",
        log: Optional[Callable[[str], None]] = None,
        timeout: float = 600.0,
    ):
        raw_base = str(upstream_base)
        parsed = urllib.parse.urlsplit(raw_base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            # Do not echo the raw URL: the bad-scheme path runs before the
            # userinfo check, so a URL such as ftp://user:secret@host would put
            # the password into the exception text and any traceback that
            # records it.
            raise ValueError("upstream_base must be an http(s) URL with a host")
        if parsed.username is not None or parsed.password is not None:
            # The proxy never authenticates the upstream hop with URL userinfo;
            # accepting it and dropping it silently lost a caller's credential.
            raise ValueError(
                "upstream_base must not carry userinfo; pass the credential in "
                "the forwarded Authorization header instead"
            )
        if parsed.fragment:
            raise ValueError("upstream_base must not carry a URL fragment")
        if _query_carries_credential(parsed.query):
            # The sibling prep proxy rejects the same target; accepting it here
            # stored the secret on the public ``upstream_base`` and appended it
            # to every forwarded request, contradicting the class docstring.
            raise ValueError(
                "upstream_base must not carry a credential query parameter; "
                "pass the credential in the forwarded Authorization header "
                "instead"
            )
        try:
            # ``parsed.port`` raises ValueError for an out-of-range port; keep
            # the dedicated message so a bad target fails at construction time.
            port = parsed.port
        except ValueError as exc:
            raise ValueError("upstream_base has an invalid port: %s" % exc) from None
        if port is not None and not 0 < port <= 65535:
            # Port 0 is not a usable client port.  endpoint_base already
            # rejects it; accepting it here delayed the failure to the first
            # forwarded request, which cannot connect.
            raise ValueError("upstream_base has an invalid port: %d" % port)
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = port
        self._prefix = parsed.path.rstrip("/")
        # A base URL may carry upstream parameters (for example
        # ``?api-version=2024``); every forwarded request keeps them.
        self._query = parsed.query
        # Strip trailing slashes from the path only: a whole-string
        # ``rstrip("/")`` would truncate a legitimate query value ending in a
        # slash (``?path=/``).
        self.upstream_base = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, self._prefix, self._query, "")
        )
        self.schedule = schedule or RetrySchedule()
        self.name = name
        self._log = log
        self.timeout = timeout
        self._server: Optional[_RetryProxyServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url = ""

    def log(self, message: str) -> None:
        """Send one retry line to the optional callback.

        Example:
            With ``RetryProxy(..., log=print)`` retry messages are printed.

        Counterexample:
            With no callback this is a silent no-op; it never raises because
            logging failed.
        """
        if self._log is not None:
            try:
                self._log(message)
            except Exception:
                # The documented contract is that logging never fails the
                # request.  A callback writing to a closed stream (or any other
                # bug) must not abort the retry loop and close the client.
                pass

    def target_path(self, path: str) -> str:
        """Forward ``path`` under the upstream's own base path.

        Example:
            A proxy for ``https://host/v1`` forwards ``/chat/completions`` to
            ``/v1/chat/completions``.

        Counterexample:
            A path already under the prefix is not prefixed twice; ``/v1/chat`` stays
            ``/v1/chat`` and ``/v1?x=1`` stays ``/v1?x=1``.
        """
        requested = path or "/"
        # A loopback proxy receives origin-form targets (``/v1/chat``).  An
        # absolute-form or network-path target names another origin, so it must
        # not be concatenated into a nonsense upstream path; reject it so the
        # handler can answer 400.  A slash-less origin-form path is not another
        # origin: normalise it to ``/...`` instead of refusing a legal client.
        parsed_request = urllib.parse.urlsplit(requested)
        if parsed_request.scheme or parsed_request.netloc or requested.startswith("//"):
            raise ValueError("absolute-form request targets are not supported")
        if not requested.startswith("/"):
            requested = "/" + requested
        # Compare only the path component: ``/v1?x=1`` is already under a
        # ``/v1`` prefix even though it does not start with ``/v1/``.
        path_only = requested.split("?", 1)[0]
        if self._prefix and (
            path_only == self._prefix
            or path_only.startswith(self._prefix + "/")
        ):
            target = requested
        else:
            target = self._prefix + requested
        if self._query:
            target += ("&" if "?" in target else "?") + self._query
        # http.client writes the request line as ASCII; a raw non-ASCII path
        # raised UnicodeEncodeError and killed the handler thread.  Percent
        # encode it, keeping existing escapes and reserved characters.
        return urllib.parse.quote(target, safe="/?&=%:@!$'()*+,;~-._[]")

    def open(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
    ) -> Tuple[int, List[Tuple[str, str]], http.client.HTTPResponse, http.client.HTTPConnection]:
        """Open one upstream request and return its unread response.

        Example:
            ``open("POST", "/chat/completions", headers, body)`` returns the status,
            headers, unread response and connection.

        Counterexample:
            The caller must read and close the response/connection (normally in a
            ``finally``); abandoning them leaks sockets.
        """
        target = self.target_path(path)
        forward = _forwardable_request_headers(headers)
        if body is not None:
            forward["Content-Length"] = str(len(body))
        if self._scheme == "https":
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                self._host, self._port, timeout=self.timeout
            )
        else:
            connection = http.client.HTTPConnection(
                self._host, self._port, timeout=self.timeout
            )
        try:
            connection.request(method, target, body=body, headers=forward)
            response = connection.getresponse()
        except BaseException:
            # The caller never receives the connection when this raises, so
            # open() is the only place that can close it.
            connection.close()
            raise
        return response.status, response.getheaders(), response, connection

    def start(self) -> str:
        """Start the proxy and return the base URL to put in a provider patch.

        Example:
            ``base = proxy.start()`` returns a loopback URL such as
            ``http://127.0.0.1:41234/v1``.

        Counterexample:
            Starting the same proxy twice without ``stop()`` leaves two servers; create
            one proxy per route.
        """
        server = _RetryProxyServer(("127.0.0.1", 0), _RetryProxyHandler)
        server.proxy = self  # type: ignore[attr-defined]
        thread = threading.Thread(
            target=server.serve_forever,
            name="%s-server" % self.name,
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        port = server.server_address[1]
        self.base_url = "http://127.0.0.1:%d%s" % (port, self._prefix)
        return self.base_url

    def stop(self) -> None:
        """Stop the loopback server and clear ``base_url``.

        Example:
            ``proxy.stop()`` releases the server thread after the last request.

        Counterexample:
            Calling it twice is safe, but requests made after it fail; stop only
            after the final upstream call.
        """
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        self.base_url = ""
        # A direct ``proxy.stop()`` is documented; leaving the dead object and
        # its base alias in the process-wide registry made len/contains report
        # a stopped proxy until some later registry-based stop.
        _PROXY_REGISTRY.remove(self)
        if server is not None:
            try:
                server.shutdown()
            except OSError:
                pass
            try:
                server.server_close()
            except OSError:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)


def default_log(message: str) -> None:
    """Write one proxy retry line to stderr without failing the request.

    Example:
        ``default_log("retry 1/10 in 10s")`` writes one line to stderr.

    Counterexample:
        A broken stderr does not propagate ``OSError``; logging must never fail the
        request.
    """
    try:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        # A closed/replaced stderr raises ValueError as well as OSError.
        pass


def retry_policy_yaml(policy: Optional[Mapping[str, Any]] = None, indent: str = "        ") -> str:
    """The DSH ``retryPolicy`` block for one provider entry.

    The block implements per-LLM-call retries on the agent loop's
    ``agent/request-error`` hook; it does not restart the harness or the stage.
    A partial policy is filled from ``DEFAULT_POLICY`` before rendering.

    Example:
        ``retry_policy_yaml(policy, "  ")`` emits a ``retryPolicy:`` block with
        attempts and delays for a DSH provider.

    Counterexample:
        It returns YAML text, not a dict; do not pass it to a YAML merge function
        as structured data.
    """
    document = _merge_policy(policy or load_policy())
    retry = document["retry"]
    lines = [
        "%sretryPolicy:" % indent,
        "%s  mode: normal" % indent,
        "%s  maxRetries: %d" % (indent, int(retry["max_retries"])),
        "%s  retryableCodes:" % indent,
    ]
    for code in retry["retryable_error_codes"]:
        # The scalar emitter quotes and escapes a code containing a colon,
        # newline or YAML boolean spelling so it cannot change the document shape.
        lines.append("%s    - %s" % (indent, _yaml_scalar(code)))
    # A positive sub-millisecond delay truncated to 0 ms disables the backoff
    # the policy explicitly asked for.  Round up, keeping at least 1 ms, and
    # never emit a maximum below the initial delay.
    initial_ms = max(
        1, int(math.ceil(float(retry["initial_delay_seconds"]) * 1000))
    )
    max_ms = max(
        initial_ms, int(math.ceil(float(retry["max_delay_seconds"]) * 1000))
    )
    lines += [
        "%s  backoff:" % indent,
        "%s    initialDelayMs: %d" % (indent, initial_ms),
        "%s    maxDelayMs: %d" % (indent, max_ms),
        "%s    jitterRatio: %s" % (indent, retry["jitter_ratio"]),
    ]
    return "\n".join(lines) + "\n"


def dots_compaction(policy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The second Dots route (dedicated compaction key) from the policy.

    Example:
        ``dots_compaction()["compaction_api_key_env"]`` is
        ``"DOTS_API_KEY_1474"`` in the default policy.

    Counterexample:
        The returned dict is a deep copy with absent fields filled from
        ``DEFAULT_POLICY``; do not assume it contains only the keys the caller
        passed.
    """
    return copy.deepcopy(_merge_policy(policy or load_policy())["dots"])


def compaction_basic_yaml(
    policy: Optional[Mapping[str, Any]] = None,
    indent: str = "",
    model: str = "",
) -> str:
    """The DSH ``compaction-basic`` entry that routes summarization.

    A partial policy is filled from ``DEFAULT_POLICY`` before rendering.

    Example:
        ``compaction_basic_yaml(policy, "", "dots3-note-prev")`` emits the
        ``compaction-basic`` DSH entry.

    Counterexample:
        An empty ``model`` emits an empty ``summarizationModel``; pass the real
        provider model id.
    """
    document = _merge_policy(policy or load_policy())
    dots = document["dots"]
    compaction = document["compaction"]
    return (
        "%s- id: compaction-basic\n"
        "%s  config:\n"
        "%s    summarizationProvider: %s\n"
        "%s    summarizationModel: %s\n"
        "%s    maxTokens: %d\n"
        "%s    compactionRetries: %d\n"
        "%s    maxOverflowRetries: %d\n"
        % (
            indent,
            indent,
            indent,
            _yaml_scalar(dots["compaction_provider"]),
            indent,
            json.dumps(model),
            indent,
            int(compaction.get("summarization_max_tokens") or 16384),
            indent,
            int(compaction["compaction_retries"]),
            indent,
            int(compaction["max_overflow_retries"]),
        )
    )


#: The Dots API exposes a single thinking level, spelled ``max`` on the wire.
#: Every thinking level a caller may request therefore maps onto it, while a
#: request that turns thinking off sets ``reasoningEfforts: false`` (the
#: absence of the parameter is the API's lower state).  ``high`` is the level
#: the shared catalog records as this deployment's default.
_DOTS_THINKING_EFFORTS: Tuple[str, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_DOTS_THINKING_OFF = frozenset({"", "0", "false", "no", "none", "off"})
DOTS_WIRE_EFFORT = "max"


#: Scalar spellings YAML 1.1/1.2 would read as a boolean or null instead of a
#: string.  Such a value must be quoted even though it looks like a plain id.
_YAML_AMBIGUOUS_SCALARS = frozenset(
    {
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
        "y",
        "n",
        "null",
        "none",
        "~",
        "inf",
        "nan",
    }
)


def _yaml_scalar(value: Any) -> str:
    """Render a generated scalar as plain YAML when that is unambiguous.

    Provider ids and key-env names are normally plain identifiers; keeping
    those unquoted preserves the generated patch exactly.  A value containing a
    colon, newline, comment marker, leading punctuation or a YAML
    boolean/number spelling is JSON-quoted so it cannot change the document
    structure.
    """
    text = str(value)
    plain = (
        bool(text)
        and (text[0].isalpha() or text[0] == "_")
        and all(
            character.isalnum() or character in "_.-/" for character in text
        )
        and text.lower() not in _YAML_AMBIGUOUS_SCALARS
    )
    if plain:
        try:
            float(text)
        except ValueError:
            return text
    return json.dumps(text)


def _overlay_base_url(value: Any) -> str:
    """A persisted base URL with userinfo, query and fragment removed.

    ``dots_dsh_overlay`` is a public generator, so a direct caller can pass a
    credential-bearing ``PAPER_*_BASE_URL``-shaped value.  The generated
    ``dsh_model_patch.yml`` is agent-readable; keep the live credential out of
    it, matching ``model_runtime.endpoint_base``.  This module stays standalone
    (no sibling import), so the normalisation is local.
    """
    text = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.netloc:
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")].rstrip("/")
        return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", ""))
    # A host-less or unparsable value still may carry userinfo/query text;
    # strip those parts textually rather than copying the secret.  The
    # userinfo always lives in the authority before the first "/", whether or
    # not a scheme is present, so a bare "user:pw@host/v1" is scrubbed too.
    without_fragment = text.split("#", 1)[0]
    before_query = without_fragment.split("?", 1)[0]
    prefix = ""
    remainder = before_query
    if "://" in remainder:
        scheme, _, remainder = remainder.partition("://")
        prefix = scheme + "://"
    elif remainder.startswith("//"):
        prefix = "//"
        remainder = remainder[2:]
    authority, separator, tail = remainder.partition("/")
    authority = authority.rsplit("@", 1)[-1]
    return (prefix + authority + separator + tail).rstrip("/")


def dots_dsh_overlay(
    policy: Optional[Mapping[str, Any]] = None,
    *,
    model: str,
    base_url: str,
    provider: str = "dots",
    context_window: int,
    max_tokens: int,
    regular_api_key_env: Optional[str] = None,
    compaction_base_url: Optional[str] = None,
    stream_idle_timeout_ms: Optional[int] = None,
    reasoning_effort: str = "high",
) -> str:
    """The complete DSH overlay for the Dots route.

    The regular ``dots`` provider carries the per-call ``retryPolicy``; the
    ``dots-compaction`` provider spends the dedicated key and points at the
    retrying proxy.  When the proxy is in the path it owns the retries, so the
    compaction provider carries no second ``retryPolicy``; without the proxy it
    keeps the policy.  ``compaction-basic`` routes summarization to it.

    The Dots API exposes a single thinking level, so the model declares every
    thinking level against the wire's ``max`` value and ``reasoning_effort``
    names the level the session requests (``high`` when the caller supplies
    none, the shared catalog's default for this model).  ``none``/``off``
    disables effort selection entirely, which is the API's lower state, and an
    unrecognised non-``off`` spelling is treated as the default (every
    thinking level reaches the same wire value).  ``provider`` is the
    catalog-owned DSH provider name for the regular route (default ``dots``);
    ``regular_api_key_env`` is the catalog-owned key variable for the regular
    Dots provider; the policy's ``regular_api_key_env`` remains the fallback.

    Example:
        ``dots_dsh_overlay(policy, model="dots3-note-prev", base_url=..., ...)``
        returns the complete DSH overlay text with ``reasoningEffort: "high"``.

    Counterexample:
        When a compaction proxy is configured, the compaction provider
        intentionally carries no second ``retryPolicy``; adding one doubles the
        retry budget.  The compaction model also keeps
        ``reasoningEfforts: false``: summarization spends its budget on the
        summary, not on a reasoning trace.
    """
    document = _merge_policy(policy or load_policy())
    dots = document["dots"]
    compaction = document["compaction"]
    # The overlay is a persisted artifact: a direct caller's credential-bearing
    # base URL must not reach baseURL:.  The internal dsh_patch_text path
    # already normalises it, but the public generator must protect itself.
    base_url = _overlay_base_url(base_url)
    compaction_base = _overlay_base_url(compaction_base_url or base_url)
    # Normalize first: a whitespace-only value is absent and must use the
    # default, not strip to "" and match the thinking-off sentinel.
    effort = str(reasoning_effort or "").strip().lower() or "high"
    thinking = effort not in _DOTS_THINKING_OFF
    if thinking and effort not in _DOTS_THINKING_EFFORTS:
        # The API has one thinking level, so an unrecognised non-off spelling
        # cannot change the request.  Use the deployment default rather than
        # declaring a level DSH's schema would reject.
        effort = "high"
    levels = list(_DOTS_THINKING_EFFORTS)
    idle = int(stream_idle_timeout_ms or 0)
    # Actual retry sleeps jitter the configured delay UP by ``jitter_ratio``,
    # and each attempt can also spend its own upstream timeout before the next
    # sleep starts.  The idle watchdog must cover both, or it aborts a retry
    # sequence the proxy would still have completed (F068).
    schedule = RetrySchedule(document)
    retry_sleep_ms = int(
        sum(schedule.delays()) * (1.0 + schedule.jitter_ratio) * 1000
    )
    idle = max(
        idle,
        int(compaction.get("compaction_stream_idle_timeout_ms") or 0),
        retry_sleep_ms + (schedule.max_retries + 1) * 600_000 + 600_000,
    )
    lines = [
        "# Generated by generic_debug_pipeline for this stage: "
        "--agent-model dots3-note-preview.",
        "- id: llm-pi-ai",
        "  config:",
        "    providers:",
        "      %s:" % _yaml_scalar(provider),
        "        displayName: %s" % json.dumps("Dots API (%s)" % model),
        "        apiKeyEnv: %s"
        % _yaml_scalar(
            regular_api_key_env
            or dots.get("regular_api_key_env", "DOTS_API_KEY")
        ),
        "        api: openai-completions",
        "        baseURL: %s" % json.dumps(base_url),
        "        models:",
        "          - id: %s" % json.dumps(model),
        "            name: %s" % json.dumps(model),
        "            contextWindow: %d" % int(context_window),
        "            maxTokens: %d" % int(max_tokens),
    ]
    if thinking:
        lines.append("            reasoningEfforts:")
        lines.extend(
            "              %s: %s" % (level, DOTS_WIRE_EFFORT) for level in levels
        )
    else:
        lines.append("            reasoningEfforts: false")
    lines += [
        "        compat:",
        "          maxTokensField: max_tokens",
        "          supportsStore: false",
    ]
    lines.extend(retry_policy_yaml(document, "        ").rstrip("\n").splitlines())
    lines += [
        "      %s:" % _yaml_scalar(dots["compaction_provider"]),
        "        displayName: %s"
        % json.dumps(dots.get("compaction_provider_label", "Dots compaction")),
        "        apiKeyEnv: %s" % _yaml_scalar(dots["compaction_api_key_env"]),
        "        api: openai-completions",
        "        baseURL: %s" % json.dumps(compaction_base),
        "        streamIdleTimeoutMs: %d" % idle,
        "        models:",
        "          - id: %s" % json.dumps(model),
        "            name: %s" % json.dumps(model),
        "            contextWindow: %d" % int(context_window),
        "            maxTokens: %d" % int(max_tokens),
        "            # Summaries spend the completion budget, not a reasoning trace.",
        "            reasoningEfforts: false",
        "        compat:",
        "          maxTokensField: max_tokens",
        "          supportsStore: false",
    ]
    if not compaction_base_url:
        # The proxied compaction route already retries every upstream call in RetryProxy.
        # A retryPolicy here would multiply the two budgets (up to 11x11 attempts).
        lines.extend(retry_policy_yaml(document, "        ").rstrip("\n").splitlines())
    lines += [
        "- id: agent-default-model",
        "  config:",
        "    provider: %s" % _yaml_scalar(provider),
        "    model: %s" % json.dumps(model),
    ]
    if thinking:
        lines.append("    reasoningEffort: %s" % json.dumps(effort))
    lines.extend(
        compaction_basic_yaml(document, "", model).rstrip("\n").splitlines()
    )
    return "\n".join(lines) + "\n"


class _ProxyRegistry:
    """Process-wide proxy ownership: one stop rule for every proxy kind.

    Every started proxy is tagged with its owner kind (``retry``,
    ``compaction``, ``prep``) and, when it has one, its loopback base URL.  A
    scoped stop call selects by kind or base; a global stop call selects
    everything of the retry family (``exclude={"prep"}``), so a compaction
    wrapper can never stop an unrelated proxy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: Dict[Any, Tuple[str, str]] = {}
        self.by_base: Dict[str, Any] = {}

    def add(self, item: Any, *, kind: str, base: str = "") -> None:
        with self._lock:
            self._items[item] = (str(kind), str(base or ""))
            if base:
                self.by_base[str(base)] = item

    def remove(self, item: Any) -> None:
        with self._lock:
            self._items.pop(item, None)
            for base, known in list(self.by_base.items()):
                if known is item:
                    self.by_base.pop(base, None)

    def __contains__(self, item: Any) -> bool:
        with self._lock:
            return item in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def targets(
        self,
        *,
        item: Any = None,
        kind: Optional[str] = None,
        base: Optional[str] = None,
        exclude: Iterable[str] = (),
    ) -> List[Any]:
        excluded = {str(value) for value in exclude}
        allowed = {str(kind)} if kind is not None else None
        with self._lock:
            if item is not None:
                selected = [item]
            elif base is not None:
                known = self.by_base.get(str(base))
                selected = [known] if known is not None else []
            else:
                selected = list(self._items)
            targets = []
            for candidate in selected:
                info = self._items.get(candidate)
                if info is None:
                    # An explicit item may predate the registry (or be stopped
                    # twice); stopping it is still harmless.
                    if item is not None and not excluded:
                        targets.append(candidate)
                    continue
                if allowed is not None and info[0] not in allowed:
                    continue
                if info[0] in excluded:
                    continue
                targets.append(candidate)
            for candidate in targets:
                self._items.pop(candidate, None)
                for known_base, known in list(self.by_base.items()):
                    if known is candidate:
                        self.by_base.pop(known_base, None)
            return targets


#: Every proxy this process started, tagged with its owner kind.
_PROXY_REGISTRY = _ProxyRegistry()


def start_retry_proxy(
    target_base: str,
    schedule: Optional[RetrySchedule] = None,
    *,
    name: str = "llm-retry-proxy",
    log: Optional[Callable[[str], None]] = None,
    kind: str = "retry",
) -> Tuple[Optional[RetryProxy], str]:
    """Start one proxy and return ``(proxy, base_url)``; ``(None, "")`` on failure.

    ``kind`` is the ownership tag a scoped stop call filters on (``retry`` by
    default, ``compaction`` for the summarizer route).

    Example:
        ``proxy, base = start_retry_proxy("https://api.example/v1")`` returns a
        loopback base URL and registers the proxy for cleanup.

    Counterexample:
        A bad or empty target returns ``(None, "")`` instead of raising; callers
        must check the pair.
    """
    if not target_base:
        return None, ""
    try:
        proxy = RetryProxy(target_base, schedule, name=name, log=log)
        base = proxy.start()
    except (OSError, ValueError):
        return None, ""
    _PROXY_REGISTRY.add(proxy, kind=kind, base=base)
    return proxy, base


def stop_retry_proxy(
    proxy: Optional[RetryProxy] = None, *, kind: Optional[str] = None
) -> None:
    """Stop one proxy, every proxy of ``kind``, or every retry-family proxy.

    Example:
        ``stop_retry_proxy(proxy)`` stops one proxy; ``stop_retry_proxy()`` stops
        every retry and compaction proxy this process started (a prep proxy has
        its own ``stop_prep_proxy``); ``stop_retry_proxy(kind="compaction")``
        stops only the compaction-owned proxies.

    Counterexample:
        Passing an already-stopped proxy is tolerated and does nothing.  An
        explicit proxy wins over ``kind``; it is never broadened to the whole
        registry.  A global ``stop_retry_proxy()`` deliberately leaves
        ``kind="prep"`` proxies running.
    """
    if proxy is not None:
        targets = _PROXY_REGISTRY.targets(item=proxy)
    elif kind is not None:
        targets = _PROXY_REGISTRY.targets(kind=kind)
    else:
        # The prep proxy has its own public stop function; a global retry stop
        # must not tear it down.
        targets = _PROXY_REGISTRY.targets(exclude=("prep",))
    for item in targets:
        if item is None:
            continue
        try:
            item.stop()
        except Exception:
            pass
