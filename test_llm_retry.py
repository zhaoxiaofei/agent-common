"""Tests for the shared retry policy and the loopback retrying proxy."""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from agent_common.llm_retry import (
    DEFAULT_POLICY,
    RetryProxy,
    RetrySchedule,
    default_log,
    load_policy,
)


class ScheduleTests(unittest.TestCase):
    def test_the_documented_schedule_starts_at_ten_and_caps_at_one_hour(self):
        schedule = RetrySchedule(load_policy())
        self.assertEqual(
            [int(delay) for delay in schedule.delays()],
            [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3600],
        )

    def test_only_transient_codes_and_statuses_are_retryable(self):
        schedule = RetrySchedule(load_policy())
        self.assertTrue(schedule.status_is_retryable(429))
        self.assertTrue(schedule.status_is_retryable(503))
        self.assertFalse(schedule.status_is_retryable(400))
        self.assertTrue(schedule.code_is_retryable("RATE_LIMIT"))
        self.assertFalse(schedule.code_is_retryable("INVALID_REQUEST"))

    def test_retry_after_is_honoured_up_to_the_cap(self):
        schedule = RetrySchedule(load_policy())
        self.assertEqual(schedule.delay(3, retry_after=7), 7)
        self.assertEqual(schedule.delay(3, retry_after=999999), 3600)

    def test_a_missing_policy_file_falls_back_to_the_defaults(self):
        policy = load_policy("/nonexistent/llm_policy.json")
        self.assertEqual(policy["retry"]["max_delay_seconds"], 3600)
        self.assertEqual(policy["retry"]["max_retries"], 10)


class ScheduleValidationTests(unittest.TestCase):
    def test_the_documented_examples_are_the_jitter_free_schedule(self):
        schedule = RetrySchedule(load_policy())
        self.assertEqual(schedule.delay(1, jitter=False), 10)
        self.assertEqual(schedule.delay(3, jitter=False), 40)
        self.assertIn("jitter=False", RetrySchedule.delay.__doc__)

    def test_a_directly_constructed_schedule_rejects_bad_delays(self):
        for retry in (
            {"initial_delay_seconds": -1, "max_delay_seconds": 2, "max_retries": 2},
            {
                "initial_delay_seconds": float("nan"),
                "max_delay_seconds": 2,
                "max_retries": 2,
            },
            {
                "initial_delay_seconds": 1,
                "max_delay_seconds": 2,
                "max_retries": 2,
                "jitter_ratio": -1,
            },
        ):
            with self.subTest(retry=retry):
                with self.assertRaises(ValueError):
                    RetrySchedule({"retry": retry})

    def test_validation_preserves_the_constructors_optional_defaults(self):
        schedule = RetrySchedule(
            {
                "retry": {
                    "initial_delay_seconds": 1,
                    "max_delay_seconds": 2,
                    "max_retries": 2,
                }
            }
        )
        self.assertEqual(schedule.jitter_ratio, 0.0)
        self.assertEqual(schedule.retry_after_cap_seconds, 2)
        self.assertEqual(schedule.retryable_statuses, frozenset())
        self.assertEqual(schedule.retryable_codes, frozenset())

    def test_a_large_retry_number_caps_instead_of_overflowing(self):
        schedule = RetrySchedule(load_policy())
        delay = schedule.delay(100000, jitter=False)
        self.assertEqual(delay, schedule.max_delay_seconds)


class ProxyTargetTests(unittest.TestCase):
    def test_a_base_query_is_preserved_on_every_forwarded_path(self):
        proxy = RetryProxy("https://host/v1?api-version=2024")
        self.assertEqual(proxy.target_path("/chat"), "/v1/chat?api-version=2024")
        self.assertEqual(
            proxy.target_path("/v1/chat?x=1"), "/v1/chat?x=1&api-version=2024"
        )

    def test_an_already_prefixed_path_with_a_query_is_not_prefixed_twice(self):
        proxy = RetryProxy("http://host/v1")
        self.assertEqual(proxy.target_path("/v1?x=1"), "/v1?x=1")
        self.assertEqual(proxy.target_path("/v1/chat?x=1"), "/v1/chat?x=1")

    def test_a_query_value_ending_in_a_slash_is_preserved(self):
        for base, expected in (
            ("https://host/v1?path=/", "/v1/chat?path=/"),
            ("https://host/v1?api-version=2024/", "/v1/chat?api-version=2024/"),
        ):
            with self.subTest(base=base):
                proxy = RetryProxy(base)
                self.assertEqual(proxy.upstream_base, base)
                self.assertEqual(proxy.target_path("/chat"), expected)

    def test_userinfo_and_fragment_are_rejected_instead_of_dropped(self):
        for url in (
            "https://user:secret@host/v1",
            "https://host/v1#fragment",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    RetryProxy(url)


class OpenFailureTests(unittest.TestCase):
    def test_a_failed_request_closes_the_connection(self):
        closed = []

        class FakeConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                raise OSError("connect refused")

            def close(self):
                closed.append(True)

        proxy = RetryProxy("http://127.0.0.1:1/v1")
        with mock.patch.object(http.client, "HTTPConnection", FakeConnection):
            with self.assertRaises(OSError):
                proxy.open("POST", "/v1/chat", {}, b"{}")
        self.assertTrue(closed)


class SuccessCloseTests(unittest.TestCase):
    def test_the_success_path_closes_the_response_and_connection(self):
        class FakeResponse:
            def __init__(self):
                self.payload = [b"ok"]
                self.closed = False

            def read(self, size=-1):
                return self.payload.pop(0) if self.payload else b""

            def close(self):
                self.closed = True

        class FakeConnection:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        response = FakeResponse()
        connection = FakeConnection()
        proxy = RetryProxy("http://127.0.0.1:1/v1")
        proxy.open = mock.Mock(return_value=(200, [], response, connection))
        base = proxy.start()
        try:
            with urllib.request.urlopen(base + "/v1/chat", timeout=5) as reply:
                self.assertEqual(reply.status, 200)
                self.assertEqual(reply.read(), b"ok")
        finally:
            proxy.stop()
        self.assertTrue(response.closed)
        self.assertTrue(connection.closed)


class _CountingUpstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.server.hits = getattr(self.server, "hits", 0) + 1
        payload = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class InvalidContentLengthTests(unittest.TestCase):
    def test_an_invalid_content_length_is_rejected_before_forwarding(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _CountingUpstream)
        upstream.hits = 0
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        proxy = RetryProxy("http://127.0.0.1:%d/v1" % upstream.server_address[1])
        base = proxy.start()
        try:
            parsed = urllib.parse.urlsplit(base)
            connection = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=5
            )
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Content-Length", "abc")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            connection.close()
            self.assertEqual(upstream.hits, 0)
        finally:
            proxy.stop()
            upstream.shutdown()
            upstream.server_close()


class _DuplicateHeaderUpstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.send_header("Server", "upstream-server")
        self.send_header("Date", "Thu, 01 Jan 1970 00:00:00 GMT")
        self.end_headers()
        self.wfile.write(b"{}")


class DuplicateResponseHeaderTests(unittest.TestCase):
    def test_the_retry_proxy_emits_one_server_and_date_pair(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), _DuplicateHeaderUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        proxy = RetryProxy("http://127.0.0.1:%d/v1" % upstream.server_address[1])
        base = proxy.start()
        try:
            parsed = urllib.parse.urlsplit(base)
            connection = http.client.HTTPConnection(
                parsed.hostname, parsed.port, timeout=5
            )
            connection.request("POST", "/v1/chat", body=b"{}")
            response = connection.getresponse()
            response.read()
            names = [name.lower() for name, _ in response.getheaders()]
            self.assertEqual(names.count("server"), 1)
            self.assertEqual(names.count("date"), 1)
            connection.close()
        finally:
            proxy.stop()
            upstream.shutdown()
            upstream.server_close()


class _SlowErrorBody(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        # Promise a body, send the headers, then stall past the proxy timeout.
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.flush()
        time.sleep(0.6)


class ErrorBodyTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SlowErrorBody)
        threading.Thread(
            target=self.upstream.serve_forever, daemon=True
        ).start()
        policy = json.loads(json.dumps(DEFAULT_POLICY))
        policy["retry"].update(
            {
                "initial_delay_seconds": 0.01,
                "max_delay_seconds": 0.02,
                "max_retries": 1,
                "jitter_ratio": 0.0,
            }
        )
        self.logs = []
        self.proxy = RetryProxy(
            "http://127.0.0.1:%d/v1" % self.upstream.server_address[1],
            RetrySchedule(policy),
            timeout=0.2,
            log=self.logs.append,
        )
        self.base = self.proxy.start()

    def tearDown(self):
        self.proxy.stop()
        self.upstream.shutdown()
        self.upstream.server_close()

    def test_a_timeout_while_reading_a_retryable_body_is_retried(self):
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=b'{"model":"m"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        # The structured last response reaches the client; the handler did not
        # die with a RemoteDisconnected.
        self.assertEqual(caught.exception.code, 503)
        self.assertTrue(
            any("body read failed" in line for line in self.logs), self.logs
        )


class MalformedPolicyFallbackTests(unittest.TestCase):
    def write_policy(self, document):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "llm_policy.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_an_infinite_numeric_value_falls_back_instead_of_raising(self):
        # JSON 1e400 parses as inf; int(inf) raises OverflowError, which used to
        # escape load_policy's documented fallback.
        policy = load_policy(self.write_policy({"retry": {"max_retries": 1e400}}))
        self.assertEqual(policy["retry"]["max_retries"], 10)
        self.assertEqual(policy["retry"]["max_delay_seconds"], 3600)

    def test_a_malformed_dots_section_falls_back_to_the_defaults(self):
        for dots in (
            {"regular_api_key_env": ["DOTS_API_KEY"]},
            {"compaction_api_key_env": 1474},
            {"compaction_provider": ""},
            {"compaction_api_key_aliases": "DOTS_API_KEY_1474"},
        ):
            with self.subTest(dots=dots):
                policy = load_policy(self.write_policy({"dots": dots}))
                self.assertEqual(
                    policy["dots"]["regular_api_key_env"], "DOTS_API_KEY"
                )
                self.assertEqual(
                    policy["dots"]["compaction_api_key_env"],
                    "DOTS_API_KEY_1474",
                )


class LoggingTests(unittest.TestCase):
    def test_a_raising_callback_does_not_escape_the_retry_proxy(self):
        proxy = RetryProxy(
            "https://api.example/v1",
            log=lambda _message: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        proxy.log("retry 1/10 in 10s")  # must not raise

    def test_a_closed_stderr_does_not_escape_default_log(self):
        class ClosedStderr:
            def write(self, _message):
                raise ValueError("I/O operation on closed file")

            def flush(self):
                raise ValueError("I/O operation on closed file")

        with mock.patch("sys.stderr", ClosedStderr()):
            default_log("retry 1/10 in 10s")  # must not raise


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        self.server.hits += 1  # type: ignore[attr-defined]
        if self.server.hits < 3:  # type: ignore[attr-defined]
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"tpm"}')
            return
        payload = json.dumps({"ok": True, "body": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        self.upstream.hits = 0
        threading.Thread(
            target=self.upstream.serve_forever, daemon=True
        ).start()
        policy = json.loads(json.dumps(DEFAULT_POLICY))
        policy["retry"].update(
            {
                "initial_delay_seconds": 0.01,
                "max_delay_seconds": 0.05,
                "max_retries": 4,
                "jitter_ratio": 0.0,
            }
        )
        self.proxy = RetryProxy(
            "http://127.0.0.1:%d/v1" % self.upstream.server_address[1],
            RetrySchedule(policy),
        )
        self.base = self.proxy.start()

    def tearDown(self):
        self.proxy.stop()
        self.upstream.shutdown()
        self.upstream.server_close()

    def test_a_429_is_retried_and_the_same_body_reaches_upstream(self):
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=b'{"model":"m"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read())
        self.assertTrue(body["ok"])
        self.assertEqual(self.upstream.hits, 3)


if __name__ == "__main__":
    unittest.main()
