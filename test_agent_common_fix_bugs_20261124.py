"""Regression tests for the 2026-11-24 verified findings in agent-common.

Each test pins one candidate from the verified-findings set.  The shared
package is intentionally dependency-free, so these are plain ``unittest``
tests with local loopback servers and temporary checkouts.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import check_cross_repo_consistency
import check_llm_policy
import check_model_catalog
import check_shared_package

from agent_common import dsh_events, llm_retry, model_runtime


# ---------------------------------------------------------------------------
# F20 / F37 / F38 / F21 / F22 -- model_runtime value safety and URL handling
# ---------------------------------------------------------------------------
class SelectionReprTests(unittest.TestCase):
    def test_prep_json_credentials_are_redacted_in_repr(self):
        selection = model_runtime.AgentSelection(
            harness="dsh", model="qwen38-27b", prep_json={"api_key": "SECRET"}
        )
        self.assertNotIn("SECRET", repr(selection))
        self.assertIn("[REDACTED]", repr(selection))


class EndpointRedactionTests(unittest.TestCase):
    def test_non_http_scheme_relative_and_hostless_urls_are_redacted(self):
        for url in (
            "//user:pw@host/v1?api_key=SECRET",
            "ftp://user:pw@host/v1?api_key=SECRET",
            "https:///v1?api_key=SECRET",
        ):
            with self.subTest(url=url):
                safe = model_runtime.redact_endpoint_url(url)
                self.assertNotIn("SECRET", safe)
                self.assertNotIn("pw@", safe)

    def test_a_url_shaped_model_name_does_not_reach_the_record(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh",
            model="https://host/v1",
            model_name="https://user:pw@host/model?api_key=SECRET",
        )
        record = selection.to_record()
        self.assertNotIn("SECRET", json.dumps(record))
        self.assertNotIn("pw@", json.dumps(record))


class EndpointBaseTests(unittest.TestCase):
    def test_a_double_slash_before_chat_completions_leaves_no_trailing_slash(self):
        self.assertEqual(
            model_runtime.endpoint_base("https://host/v1//chat/completions"),
            "https://host/v1",
        )


class ClaudeEnvironmentTests(unittest.TestCase):
    def _env(self, harness, model, level):
        selection = model_runtime.resolve_agent_selection(
            harness=harness, model=model
        )
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": level}, clear=True
        ):
            return model_runtime.claude_environment(selection)

    def test_every_claude_producing_branch_uses_claudes_vocabulary(self):
        for harness, model in (
            ("dsh", "default"),
            ("dsh", "gpt-sol"),
            ("claude-code", "claude-opus"),
        ):
            for level, expected in (("minimal", "low"), ("xhigh", "max")):
                with self.subTest(harness=harness, model=model, level=level):
                    env = self._env(harness, model, level)
                    self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], expected)

    def test_an_off_level_is_not_emitted(self):
        env = self._env("dsh", "default", "off")
        self.assertNotIn("CLAUDE_CODE_EFFORT_LEVEL", env)


# ---------------------------------------------------------------------------
# F24 / F25 / F26 / F42 / F52 -- prep proxy
# ---------------------------------------------------------------------------
def _start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stop_server(server):
    server.shutdown()
    server.server_close()


def _host_port(base):
    parsed = urllib.parse.urlsplit(base)
    return parsed.hostname, parsed.port


class _RateLimitTarget(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003 - stdlib name
        return

    def do_GET(self):  # noqa: N802 - stdlib name
        body = b'{"error":"rate limited"}'
        self.send_response(429)
        self.send_header("Retry-After", "7")
        self.send_header("X-RateLimit-Remaining", "0")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _TruncatedErrorTarget(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003 - stdlib name
        return

    def do_GET(self):  # noqa: N802 - stdlib name
        self.send_response(500)
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b"x")
        self.wfile.flush()
        self.close_connection = True


class PrepProxyTests(unittest.TestCase):
    def test_v1_with_a_query_is_not_double_prefixed(self):
        handler = object.__new__(model_runtime._PrepProxyHandler)
        handler.server = SimpleNamespace(target_base="http://host/v1")
        handler.path = "/v1?x=1"
        self.assertEqual(handler._target(), "http://host/v1?x=1")

    def test_error_response_headers_are_relayed(self):
        upstream = _start_server(_RateLimitTarget)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            request = urllib.request.Request(base + "/v1/models", method="GET")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 429)
            self.assertEqual(caught.exception.headers.get("Retry-After"), "7")
            self.assertEqual(
                caught.exception.headers.get("X-RateLimit-Remaining"), "0"
            )
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_a_truncated_upstream_error_is_a_typed_502(self):
        upstream = _start_server(_TruncatedErrorTarget)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            request = urllib.request.Request(base + "/v1/models", method="GET")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 502)
            caught.exception.read()
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_a_non_ascii_path_is_percent_encoded(self):
        handler = object.__new__(model_runtime._PrepProxyHandler)
        handler.server = SimpleNamespace(target_base="http://host/v1")
        handler.path = "/v1/chat/\u00e9"
        target = handler._target()
        self.assertTrue(target.isascii())
        self.assertIn("%C3%A9", target)

    def test_a_non_serialisable_prep_object_fails_at_startup(self):
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:1/v1", {"x": object()}
        )
        self.assertEqual(base, "")
        self.assertIn("JSON", error)


# ---------------------------------------------------------------------------
# F27 / F28 / F39 / F42 / F51 -- retry proxy
# ---------------------------------------------------------------------------
class _RedirectTarget(BaseHTTPRequestHandler):
    target_port = 0
    hits = 0

    def log_message(self, *args):  # noqa: A003 - stdlib name
        return

    def do_GET(self):  # noqa: N802 - stdlib name
        body = b"{}"
        self.send_response(302)
        self.send_header(
            "Location", "http://127.0.0.1:%d/steal" % self.target_port
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _RecordingTarget(BaseHTTPRequestHandler):
    seen_auth = None
    hits = 0

    def log_message(self, *args):  # noqa: A003 - stdlib name
        return

    def do_GET(self):  # noqa: N802 - stdlib name
        type(self).hits += 1
        type(self).seen_auth = self.headers.get("Authorization")
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _TruncatedSuccessTarget(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: A003 - stdlib name
        return

    def do_GET(self):  # noqa: N802 - stdlib name
        self.send_response(200)
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b"x")
        self.wfile.flush()
        self.close_connection = True


class RetryProxyTests(unittest.TestCase):
    def _proxy(self, upstream, retries=0):
        policy = json.loads(json.dumps(llm_retry.DEFAULT_POLICY))
        policy["retry"].update(
            {
                "initial_delay_seconds": 0.01,
                "max_delay_seconds": 0.05,
                "max_retries": retries,
                "jitter_ratio": 0.0,
            }
        )
        proxy = llm_retry.RetryProxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1],
            llm_retry.RetrySchedule(policy),
        )
        return proxy, proxy.start()

    def test_a_redirect_is_not_relayed_with_its_location(self):
        target = _start_server(_RecordingTarget)
        _RecordingTarget.hits = 0
        _RecordingTarget.seen_auth = None
        redirector = _start_server(_RedirectTarget)
        _RedirectTarget.target_port = target.server_address[1]
        proxy, base = self._proxy(redirector)
        try:
            request = urllib.request.Request(
                base + "/v1/models",
                headers={"Authorization": "Bearer SECRET"},
                method="GET",
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 502)
            caught.exception.read()
            self.assertIsNone(_RecordingTarget.seen_auth)
            self.assertEqual(_RecordingTarget.hits, 0)
        finally:
            proxy.stop()
            _stop_server(redirector)
            _stop_server(target)

    def test_a_truncated_success_body_is_not_accepted_as_a_short_200(self):
        upstream = _start_server(_TruncatedSuccessTarget)
        proxy, base = self._proxy(upstream)
        try:
            request = urllib.request.Request(base + "/v1/models", method="GET")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 502)
            caught.exception.read()
        finally:
            proxy.stop()
            _stop_server(upstream)

    def test_an_invalid_scheme_error_does_not_echo_the_credential(self):
        with self.assertRaises(ValueError) as caught:
            llm_retry.RetryProxy("ftp://user:secret@host/v1")
        self.assertNotIn("secret", str(caught.exception))

    def test_absolute_form_is_rejected_and_a_slash_less_path_is_normalised(self):
        proxy = llm_retry.RetryProxy("http://host/v1")
        with self.assertRaises(ValueError):
            proxy.target_path("http://other/v1/chat")
        self.assertEqual(proxy.target_path("chat"), "/v1/chat")

    def test_a_non_ascii_path_is_percent_encoded(self):
        proxy = llm_retry.RetryProxy("http://host/v1")
        target = proxy.target_path("/v1/chat/\u00e9")
        self.assertTrue(target.isascii())
        self.assertIn("%C3%A9", target)


# ---------------------------------------------------------------------------
# F40 -- turn-end failure classification
# ---------------------------------------------------------------------------
class TurnEndTests(unittest.TestCase):
    def test_an_error_reason_wins_over_a_successful_event_kind(self):
        event = {
            "type": "turn_end",
            "kind": "ok",
            "reason": {"kind": "error", "error": "boom"},
        }
        self.assertIs(dsh_events.turn_end_failed(event), True)
        self.assertEqual(dsh_events.turn_end_error(event["reason"]), "boom")

    def test_an_error_kind_still_wins_over_a_successful_reason(self):
        event = {
            "type": "turn_end",
            "kind": "error",
            "reason": {"kind": "ok"},
        }
        self.assertIs(dsh_events.turn_end_failed(event), True)


# ---------------------------------------------------------------------------
# F31 / F32 / F43 / F44 / F45 / F48 -- model catalog checker
# ---------------------------------------------------------------------------
class _CheckerHarness(unittest.TestCase):
    def _catalog_run(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(text, encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_model_catalog, "CANONICAL", path
        ), mock.patch.object(check_model_catalog, "COPIES", []):
            with contextlib.redirect_stdout(output):
                code = check_model_catalog.main()
        return code, output.getvalue()

    def _document(self):
        return json.loads(
            check_model_catalog.CANONICAL.read_text(encoding="utf-8")
        )


class ModelCatalogMalformedTests(_CheckerHarness):
    def test_invalid_json_reports_fail_instead_of_raising(self):
        code, output = self._catalog_run("{not json")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_a_non_object_document_reports_fail(self):
        code, output = self._catalog_run("[]")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_boolean_capacities_are_rejected(self):
        document = self._document()
        document["models"][0]["context_window"] = True
        document["models"][0]["max_tokens"] = True
        document["models"][0]["chat_max_tokens"] = True
        code, output = self._catalog_run(json.dumps(document))
        self.assertEqual(code, 1)
        self.assertIn("bad context_window", output)

    def test_an_unhashable_reasoning_level_reports_fail(self):
        document = self._document()
        document["models"][0]["default_reasoning_level"] = []
        code, output = self._catalog_run(json.dumps(document))
        self.assertEqual(code, 1)
        self.assertIn("bad default_reasoning_level", output)

    def test_a_non_string_model_id_reports_fail_instead_of_raising(self):
        document = self._document()
        document["models"][0]["id"] = ["x"]
        code, output = self._catalog_run(json.dumps(document))
        self.assertEqual(code, 1)
        self.assertIn("model id", output)

    def test_a_boolean_schema_version_is_rejected(self):
        document = self._document()
        document["schema_version"] = True
        code, output = self._catalog_run(json.dumps(document))
        self.assertEqual(code, 1)
        self.assertIn("unsupported schema_version", output)

    def test_a_deeply_nested_document_reports_fail(self):
        code, output = self._catalog_run("[" * 10000 + "]" * 10000)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_an_unreadable_catalog_reports_fail(self):
        document = self._document()
        code, output = self._catalog_run(json.dumps(document))
        self.assertEqual(code, 0, output)
        with mock.patch.object(
            check_model_catalog, "digest", side_effect=PermissionError("no")
        ):
            output = io.StringIO()
            with mock.patch.object(check_model_catalog, "COPIES", []):
                with contextlib.redirect_stdout(output):
                    code = check_model_catalog.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output.getvalue())


# ---------------------------------------------------------------------------
# F33 / F34 / F44 / F45 / F46 / F50 -- llm policy checker
# ---------------------------------------------------------------------------
class LlmPolicyMalformedTests(unittest.TestCase):
    def _checkout(self, policy_text=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        canonical = root / "agent_common"
        canonical.mkdir()
        shutil.copy(check_llm_policy.CANONICAL_JSON, canonical / "llm_policy.json")
        if policy_text is not None:
            (canonical / "llm_policy.json").write_text(
                policy_text, encoding="utf-8"
            )
        (canonical / "llm_retry.py").write_text(
            check_llm_policy.CANONICAL_MODULE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return root, canonical

    def _run(self, root, canonical):
        output = io.StringIO()
        with mock.patch.object(check_llm_policy, "ROOT", root), mock.patch.object(
            check_llm_policy, "CANONICAL_DIR", canonical
        ), mock.patch.object(
            check_llm_policy, "CANONICAL_JSON", canonical / "llm_policy.json"
        ), mock.patch.object(
            check_llm_policy, "CANONICAL_MODULE", canonical / "llm_retry.py"
        ), mock.patch.object(
            check_llm_policy, "COPIES", ()
        ):
            with contextlib.redirect_stdout(output):
                code = check_llm_policy.main()
        return code, output.getvalue()

    def test_invalid_json_reports_fail(self):
        root, canonical = self._checkout("{not json")
        code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_a_non_object_document_reports_fail(self):
        root, canonical = self._checkout("[]")
        code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_a_string_retry_value_reports_fail(self):
        document = json.loads(
            check_llm_policy.CANONICAL_JSON.read_text(encoding="utf-8")
        )
        document["retry"]["initial_delay_seconds"] = "abc"
        root, canonical = self._checkout(json.dumps(document))
        code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_extreme_finite_values_report_fail(self):
        document = json.loads(
            check_llm_policy.CANONICAL_JSON.read_text(encoding="utf-8")
        )
        document["retry"]["initial_delay_seconds"] = 1e308
        document["retry"]["multiplier"] = 1e308
        document["retry"]["max_delay_seconds"] = 1e308
        root, canonical = self._checkout(json.dumps(document))
        code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_a_deeply_nested_document_reports_fail(self):
        root, canonical = self._checkout("[" * 10000 + "]" * 10000)
        code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_an_unreadable_digest_reports_fail(self):
        root, canonical = self._checkout()
        with mock.patch.object(
            check_llm_policy, "digest", side_effect=PermissionError("no")
        ):
            code, output = self._run(root, canonical)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)

    def test_the_docstring_does_not_claim_duplicated_adapter_constants(self):
        text = Path(check_llm_policy.__file__).read_text(encoding="utf-8")
        self.assertNotIn(
            "adapters that cannot import a sibling module carry the same", text
        )


# ---------------------------------------------------------------------------
# F35 / F44 / F47 -- cross-repo checker
# ---------------------------------------------------------------------------
class CrossRepoMalformedTests(unittest.TestCase):
    def test_a_malformed_catalog_does_not_crash_import(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        bad = Path(directory.name) / "model_catalog.json"
        bad.write_text("[]", encoding="utf-8")
        # ``reload`` re-executes the module-level ``CATALOG = ...`` assignment,
        # so the malformation must be injected at the parser instead.
        with mock.patch.object(
            check_cross_repo_consistency.json,
            "loads",
            side_effect=ValueError("malformed"),
        ):
            try:
                reloaded = importlib.reload(check_cross_repo_consistency)
            except Exception as exc:  # noqa: BLE001 - the defect is the crash
                importlib.reload(check_cross_repo_consistency)
                self.fail("import crashed on a malformed catalog: %r" % (exc,))
            self.assertTrue(reloaded._CATALOGUE_ERROR)
        importlib.reload(check_cross_repo_consistency)

    def test_child_stderr_is_reported_on_failure(self):
        completed = mock.Mock(
            returncode=1, stdout="FAIL model catalog", stderr="traceback detail"
        )
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency.subprocess, "run", return_value=completed
        ):
            with contextlib.redirect_stdout(output):
                code = check_cross_repo_consistency.main()
        self.assertEqual(code, 1)
        self.assertIn("traceback detail", output.getvalue())


# ---------------------------------------------------------------------------
# F36 / F45 -- policy description and shared-package checker
# ---------------------------------------------------------------------------
class DocumentationConsistencyTests(unittest.TestCase):
    def test_the_policy_description_matches_the_single_source_model(self):
        document = json.loads(
            check_llm_policy.CANONICAL_JSON.read_text(encoding="utf-8")
        )
        description = str(document.get("description") or "")
        self.assertNotIn("copies beside each checkout are generated", description)


class SharedPackageMalformedTests(unittest.TestCase):
    def test_an_unreadable_copy_reports_fail(self):
        with mock.patch.object(
            check_shared_package, "digest", side_effect=PermissionError("no")
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = check_shared_package.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output.getvalue())


if __name__ == "__main__":
    unittest.main()
