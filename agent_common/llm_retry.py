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
        "base_url": "https://note3-prev-api.askdiandian.com/v1",
        "model": "dots3-note-prev",
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
    except (TypeError, ValueError):
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
    """The delay and retryable-failure policy for one process."""

    def __init__(self, policy: Optional[Mapping[str, Any]] = None):
        document = dict(policy or load_policy())
        retry = document.get("retry") or DEFAULT_POLICY["retry"]
        self.initial_delay_seconds = float(retry["initial_delay_seconds"])
        self.multiplier = float(retry.get("multiplier", 2.0))
        self.max_delay_seconds = float(retry["max_delay_seconds"])
        self.max_retries = int(retry["max_retries"])
        self.jitter_ratio = float(retry.get("jitter_ratio", 0.0))
        self.retry_after_cap_seconds = float(
            retry.get("retry_after_cap_seconds", self.max_delay_seconds)
        )
        self.retryable_statuses = frozenset(
            int(status) for status in retry.get("retryable_http_statuses") or ()
        )
        self.retryable_codes = frozenset(
            str(code) for code in retry.get("retryable_error_codes") or ()
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
            With the default policy, ``delay(1) == 10`` and ``delay(3) == 40``.

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
        delay = self.initial_delay_seconds * (self.multiplier ** exponent)
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
        return max(0.0, float(raw))
    except ValueError:
        pass
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
        transfer = str(self.headers.get("Transfer-Encoding") or "").lower()
        return "chunked" in transfer

    def _body(self) -> Optional[bytes]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        return self.rfile.read(length) if length > 0 else None

    def _handle(self, method: str) -> None:
        proxy: "RetryProxy" = self.server.proxy  # type: ignore[attr-defined]
        if self._chunked():
            # Reading only Content-Length silently forwarded a chunked POST without its
            # body. Reject the unsupported framing instead of corrupting a valid request.
            self._send_error(
                400, "chunked request bodies are not supported by the retry proxy"
            )
            return
        body = self._body()
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
                if status < 400 or not schedule.status_is_retryable(status):
                    self._relay(status, headers, response)
                    return
                last_status = status
                last_headers = headers
                try:
                    last_body = response.read()
                except http.client.HTTPException as exc:
                    # A truncated 5xx error body is exactly the failure class the
                    # proxy exists for: retry it instead of letting the exception
                    # kill the handler thread and close the client connection.
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
        self.send_response(status)
        self._copy_headers(headers)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, ValueError, http.client.HTTPException):
            pass

    def _copy_headers(self, headers: Iterable[Tuple[str, str]]) -> None:
        for name, value in headers:
            if name.lower() in (
                "transfer-encoding",
                "connection",
                "content-length",
                "keep-alive",
            ):
                continue
            self.send_header(name, value)


class _RetryProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RetryProxy:
    """A loopback HTTP proxy that retries each upstream LLM call.

    The proxy exists because the DSH compaction summarizer calls the LLM
    adapter directly, outside ``dsh-llm-retry``.  Pointing only the compaction
    provider at this proxy restores per-call backoff there without touching the
    installed harness.  The proxy never stores or rewrites credentials: the
    ``Authorization``/``api-key`` headers DSH sends are forwarded unchanged.
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
        self.upstream_base = str(upstream_base).rstrip("/")
        parsed = urllib.parse.urlsplit(self.upstream_base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("upstream_base must be an http(s) URL: %r" % upstream_base)
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._prefix = parsed.path.rstrip("/")
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
            self._log(message)

    def target_path(self, path: str) -> str:
        """Forward ``path`` under the upstream's own base path.

        Example:
            A proxy for ``https://host/v1`` forwards ``/chat/completions`` to
            ``/v1/chat/completions``.

        Counterexample:
            A path already under the prefix is not prefixed twice; ``/v1/chat`` stays
            ``/v1/chat``.
        """
        requested = path or "/"
        if self._prefix and (
            requested == self._prefix or requested.startswith(self._prefix + "/")
        ):
            return requested
        return self._prefix + requested

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
        forward: Dict[str, str] = {}
        for name, value in headers.items():
            if name.lower() in (
                "host",
                "content-length",
                "connection",
                "transfer-encoding",
                "keep-alive",
            ):
                continue
            forward[name] = value
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
        connection.request(method, target, body=body, headers=forward)
        response = connection.getresponse()
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
    except OSError:
        pass


def retry_policy_yaml(policy: Optional[Mapping[str, Any]] = None, indent: str = "        ") -> str:
    """The DSH ``retryPolicy`` block for one provider entry.

    The block implements per-LLM-call retries on the agent loop's
    ``agent/request-error`` hook; it does not restart the harness or the stage.

    Example:
        ``retry_policy_yaml(policy, "  ")`` emits a ``retryPolicy:`` block with
        attempts and delays for a DSH provider.

    Counterexample:
        It returns YAML text, not a dict; do not pass it to a YAML merge function
        as structured data.
    """
    document = dict(policy or load_policy())
    retry = document["retry"]
    lines = [
        "%sretryPolicy:" % indent,
        "%s  mode: normal" % indent,
        "%s  maxRetries: %d" % (indent, int(retry["max_retries"])),
        "%s  retryableCodes:" % indent,
    ]
    for code in retry["retryable_error_codes"]:
        lines.append("%s    - %s" % (indent, code))
    lines += [
        "%s  backoff:" % indent,
        "%s    initialDelayMs: %d"
        % (indent, int(float(retry["initial_delay_seconds"]) * 1000)),
        "%s    maxDelayMs: %d"
        % (indent, int(float(retry["max_delay_seconds"]) * 1000)),
        "%s    jitterRatio: %s" % (indent, retry["jitter_ratio"]),
    ]
    return "\n".join(lines) + "\n"


def dots_compaction(policy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The second Dots route (dedicated compaction key) from the policy.

    Example:
        ``dots_compaction()["compaction_api_key_env"]`` is
        ``"DOTS_API_KEY_1474"`` in the default policy.

    Counterexample:
        It returns a shallow copy of the ``dots`` section; mutating nested values
        can still affect the cached policy.
    """
    return dict((policy or load_policy())["dots"])


def compaction_basic_yaml(
    policy: Optional[Mapping[str, Any]] = None,
    indent: str = "",
    model: str = "",
) -> str:
    """The DSH ``compaction-basic`` entry that routes summarization.

    Example:
        ``compaction_basic_yaml(policy, "", "dots3-note-prev")`` emits the
        ``compaction-basic`` DSH entry.

    Counterexample:
        An empty ``model`` emits an empty ``summarizationModel``; pass the real
        provider model id.
    """
    document = dict(policy or load_policy())
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
            dots["compaction_provider"],
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


def dots_dsh_overlay(
    policy: Optional[Mapping[str, Any]] = None,
    *,
    model: str,
    base_url: str,
    context_window: int,
    max_tokens: int,
    compaction_base_url: Optional[str] = None,
    stream_idle_timeout_ms: Optional[int] = None,
) -> str:
    """The complete DSH overlay for the Dots route.

    The regular ``dots`` provider carries the per-call ``retryPolicy``; the
    ``dots-compaction`` provider spends the dedicated key and points at the
    retrying proxy.  When the proxy is in the path it owns the retries, so the
    compaction provider carries no second ``retryPolicy``; without the proxy it
    keeps the policy.  ``compaction-basic`` routes summarization to it.

    Example:
        ``dots_dsh_overlay(policy, model="dots3-note-prev", base_url=..., ...)``
        returns the complete DSH overlay text.

    Counterexample:
        When a compaction proxy is configured, the compaction provider
        intentionally carries no second ``retryPolicy``; adding one doubles the
        retry budget.
    """
    document = dict(policy or load_policy())
    dots = document["dots"]
    compaction = document["compaction"]
    compaction_base = str(compaction_base_url or base_url).rstrip("/")
    idle = int(stream_idle_timeout_ms or 0)
    idle = max(
        idle,
        int(compaction.get("compaction_stream_idle_timeout_ms") or 0),
        int(sum(RetrySchedule(document).delays()) * 1000) + 600_000,
    )
    lines = [
        "# Generated by generic_debug_pipeline for this stage: "
        "--agent-model dots3-note-preview.",
        "- id: llm-pi-ai",
        "  config:",
        "    providers:",
        "      dots:",
        "        displayName: %s" % json.dumps("Dots API (%s)" % model),
        "        apiKeyEnv: %s" % dots.get("regular_api_key_env", "DOTS_API_KEY"),
        "        api: openai-completions",
        "        baseURL: %s" % json.dumps(base_url),
        "        models:",
        "          - id: %s" % json.dumps(model),
        "            name: %s" % json.dumps(model),
        "            contextWindow: %d" % int(context_window),
        "            maxTokens: %d" % int(max_tokens),
        "            reasoningEfforts: false",
        "        compat:",
        "          maxTokensField: max_tokens",
        "          supportsStore: false",
    ]
    lines.extend(retry_policy_yaml(document, "        ").rstrip("\n").splitlines())
    lines += [
        "      %s:" % dots["compaction_provider"],
        "        displayName: %s"
        % json.dumps(dots.get("compaction_provider_label", "Dots compaction")),
        "        apiKeyEnv: %s" % dots["compaction_api_key_env"],
        "        api: openai-completions",
        "        baseURL: %s" % json.dumps(compaction_base),
        "        streamIdleTimeoutMs: %d" % idle,
        "        models:",
        "          - id: %s" % json.dumps(model),
        "            name: %s" % json.dumps(model),
        "            contextWindow: %d" % int(context_window),
        "            maxTokens: %d" % int(max_tokens),
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
        "    provider: dots",
        "    model: %s" % json.dumps(model),
    ]
    lines.extend(
        compaction_basic_yaml(document, "", model).rstrip("\n").splitlines()
    )
    return "\n".join(lines) + "\n"


_PROXY_REGISTRY: set = set()
_PROXY_LOCK = threading.Lock()


def start_retry_proxy(
    target_base: str,
    schedule: Optional[RetrySchedule] = None,
    *,
    name: str = "llm-retry-proxy",
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[RetryProxy], str]:
    """Start one proxy and return ``(proxy, base_url)``; ``(None, "")`` on failure.

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
    with _PROXY_LOCK:
        _PROXY_REGISTRY.add(proxy)
    return proxy, base


def stop_retry_proxy(proxy: Optional[RetryProxy] = None) -> None:
    """Stop one proxy, or every proxy this process started.

    Example:
        ``stop_retry_proxy(proxy)`` stops one proxy; ``stop_retry_proxy()`` stops
        every proxy this process started.

    Counterexample:
        Passing an already-stopped proxy is tolerated and does nothing.
    """
    with _PROXY_LOCK:
        targets = [proxy] if proxy is not None else list(_PROXY_REGISTRY)
        for item in targets:
            _PROXY_REGISTRY.discard(item)
    for item in targets:
        if item is None:
            continue
        try:
            item.stop()
        except Exception:
            pass
