"""Tests for the shared retry policy and the loopback retrying proxy."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
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
