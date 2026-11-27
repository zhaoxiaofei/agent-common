"""Regression tests for the fixes verified in run 20261127-215450.

Each class corresponds to one candidate from the verified-findings set
(F01-F12), except ``RepairR01RedactionTests``, which covers the later
verify-fixes review finding R01.  The tests are hermetic: loopback servers use
port 0 and temporary directories are removed after each test.
"""

import http.client
import http.server
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import check_cross_repo_consistency
from agent_common import llm_retry, model_runtime, recovery


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _write_json_response(handler, body, status=200):
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _start_server(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _nested(depth):
    root = {}
    cursor = root
    for _ in range(depth):
        child = {}
        cursor["x"] = child
        cursor = child
    return root


class DirectAliasSelectionTests(unittest.TestCase):
    """F01: a directly constructed AgentSelection resolves its alias."""

    def test_direct_alias_matches_resolved_selection(self):
        with mock.patch.dict(
            os.environ, {"PAPER_QWEN_API_KEY": "secret"}, clear=True
        ):
            direct = model_runtime.AgentSelection(
                harness="dsh", model="qwen3.8-27b"
            )
            resolved = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen3.8-27b"
            )
            self.assertEqual(
                model_runtime._normalized_family(direct),
                model_runtime._normalized_family(resolved),
            )
            self.assertIn("llm-pi-ai", model_runtime.dsh_patch_text(direct))
            self.assertEqual(
                model_runtime.selection_env(direct).get("PAPER_QWEN_API_KEY"),
                "secret",
            )
            self.assertTrue(direct.to_record()["api_key_set"])

    def test_direct_gpt_alias_still_builds_codex_arguments(self):
        codex = model_runtime.AgentSelection(
            harness="gpt-codex", model="gpt-6.1-sol"
        )
        self.assertTrue(model_runtime.codex_provider_args(codex))


class FragmentCredentialRedactionTests(unittest.TestCase):
    """F02: a credential in the fragment must not reach persisted artifacts."""

    def test_fragment_credential_is_stripped_everywhere(self):
        selection = model_runtime.AgentSelection(
            harness="dsh",
            model="https://host/v1",
            model_name="foo#api_key=SECRET",
        )
        record = json.dumps(selection.to_record())
        patch = model_runtime.dsh_patch_text(selection)
        for text in (record, patch, repr(selection)):
            self.assertNotIn("SECRET", text)
            self.assertNotIn("api_key=", text)

    def test_scheme_less_and_one_slash_fragment_forms_are_redacted(self):
        for value in (
            "foo#api_key=SECRET",
            "http:/host/v1#api_key=SECRET",
            "//host/v1#api_key=SECRET",
        ):
            with self.subTest(value=value):
                redacted = model_runtime._redacted_endpoint_text(value)
                self.assertNotIn("SECRET", redacted)

    def test_non_credential_fragment_is_unchanged(self):
        self.assertEqual(
            model_runtime._redacted_endpoint_text("foo#section"), "foo#section"
        )


class RecoveryRedactionTests(unittest.TestCase):
    """F03: scheme-relative/one-slash/bare URLs and fragments are scrubbed."""

    def test_structured_values_are_credential_free(self):
        for raw in (
            "//user:pw@host/v1?api_key=SECRET",
            "http:/user:pw@host/v1",
            "user:pw@host/v1",
            "https://host/v1#api_key=SECRET",
        ):
            with self.subTest(raw=raw):
                rendered = json.dumps(recovery.redact_value({"endpoint": raw}))
                self.assertNotIn("pw", rendered)
                self.assertNotIn("SECRET", rendered)

    def test_free_text_scheme_relative_and_one_slash_forms_are_scrubbed(self):
        for raw in ("//user:pw@host/v1?api_key=SECRET", "http:/user:pw@host/v1"):
            with self.subTest(raw=raw):
                redacted = recovery.redact_text("endpoint " + raw)
                self.assertNotIn("pw", redacted)

    def test_ordinary_addresses_are_not_over_redacted(self):
        for value in ("user@example.com", "mailto:user@example.com"):
            with self.subTest(value=value):
                self.assertEqual(recovery.redact_value(value), value)

    def test_journal_write_cannot_persist_a_scheme_relative_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = recovery.RecoveryJournal(Path(directory) / "j", "r")
            journal.start()
            controller = recovery.RecoveryController(
                run_id="r", journal=journal, logger=lambda line: None
            )
            controller.emit(
                recovery.Diagnostic(
                    recovery.TIER_ERROR,
                    "UPSTREAM_FAILED",
                    "call",
                    "failed",
                    context={
                        "endpoint": "//user:pw@host/v1?api_key=SECRET"
                    },
                ),
                recovery_action="retry",
            )
            text = (Path(directory) / "j" / "recovery.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("pw", text)
            self.assertNotIn("SECRET", text)


class BackslashTraversalTests(unittest.TestCase):
    """F04: both proxy path filters reject raw and encoded backslashes."""

    def test_both_filters_reject_backslash_forms(self):
        for function in (
            model_runtime._reject_dot_segments,
            llm_retry._reject_dot_segments,
        ):
            for path in (
                "/v1/..%5cadmin",
                "/v1/%2e%2e%5cadmin",
                "/v1/..%5Csecret",
                "/v1/..\\admin",
            ):
                with self.subTest(function=function.__module__, path=path):
                    with self.assertRaises(ValueError):
                        function(path)

    def test_a_normal_path_is_still_accepted(self):
        model_runtime._reject_dot_segments("/v1/chat/completions")
        llm_retry._reject_dot_segments("/v1/chat/completions")

    def test_retry_proxy_target_path_rejects_the_escape(self):
        proxy = llm_retry.RetryProxy("https://upstream.example/v1")
        with self.assertRaises(ValueError):
            proxy.target_path("/v1/%2e%2e%5cadmin")


class DeepEndpointListingTests(unittest.TestCase):
    """F05: a deeply nested /models body is a probe failure, not a crash."""

    def test_deep_listing_becomes_a_diagnostic(self):
        body = b"[" * 100000 + b"]" * 100000

        class Handler(_QuietHandler):
            def do_GET(self):
                _write_json_response(self, body)

        server, thread = _start_server(Handler)
        try:
            url = "http://127.0.0.1:%d/v1" % server.server_address[1]
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model=url
            )
            problem = model_runtime.selection_problem(selection)
        finally:
            _stop_server(server, thread)
        self.assertIsInstance(problem, str)
        self.assertIn("/models listing failed", problem)


class RouteBaseValidationTests(unittest.TestCase):
    """F06: Codex/Claude must not export a base URL DSH rejects."""

    def test_invalid_host_and_port_raise(self):
        for value in ("https://:443/v1", "https://host:99999/v1"):
            with self.subTest(value=value):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime._normalized_route_base_url(value)

    def test_unparsable_url_error_stays_credential_free(self):
        with self.assertRaises(model_runtime.ConfigError) as caught:
            model_runtime._normalized_route_base_url(
                "https://user:pw@[bad/v1"
            )
        self.assertNotIn("pw", str(caught.exception))

    def test_claude_route_rejects_the_invalid_environment_url(self):
        with mock.patch.dict(
            os.environ, {"PAPER_QWEN_BASE_URL": "https://:443/v1"}, clear=True
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="claude-code", model="qwen38-27b"
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.claude_environment(selection)

    def test_a_valid_base_still_keeps_one_non_secret_query(self):
        self.assertEqual(
            model_runtime._normalized_route_base_url(
                "http://host/v1?api-version=2024"
            ),
            "http://host/v1?api-version=2024",
        )


class PrepProxyDeepObjectTests(unittest.TestCase):
    """F07: a deep prep/request gets a structured 400, not a dead handler."""

    def test_deep_merge_returns_a_structured_400(self):
        class Handler(_QuietHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                _write_json_response(self, b'{"ok":true}')

        upstream, up_thread = _start_server(Handler)
        base = ""
        try:
            base, error = model_runtime.start_prep_proxy(
                "http://127.0.0.1:%d/v1" % upstream.server_address[1],
                _nested(70),
            )
            self.assertEqual(error, "")
            self.assertTrue(base)
            port = int(base.split(":")[2].split("/")[0])
            connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=5
            )
            try:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(_nested(70)).encode(),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                payload = json.loads(response.read().decode())
            finally:
                connection.close()
            self.assertEqual(response.status, 400)
            self.assertIn("nested too deeply", payload["error"]["message"])
        finally:
            if base:
                model_runtime.stop_prep_proxy(base)
            _stop_server(upstream, up_thread)

    def test_a_deep_request_body_returns_a_structured_400(self):
        class Handler(_QuietHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                _write_json_response(self, b'{"ok":true}')

        upstream, up_thread = _start_server(Handler)
        base = ""
        try:
            base, _error = model_runtime.start_prep_proxy(
                "http://127.0.0.1:%d/v1" % upstream.server_address[1],
                {"temperature": 0},
            )
            port = int(base.split(":")[2].split("/")[0])
            body = b"[" * 100000 + b"]" * 100000
            connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=5
            )
            try:
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                response.read()
            finally:
                connection.close()
            self.assertEqual(response.status, 400)
        finally:
            if base:
                model_runtime.stop_prep_proxy(base)
            _stop_server(upstream, up_thread)


class RunMarkdownCodeSpanTests(unittest.TestCase):
    """F08: backticks in command/configuration cannot close the code span."""

    def test_backticks_in_command_and_configuration(self):
        rendered = recovery.render_run_markdown(
            "run-1",
            status="completed",
            records=[],
            command="echo `whoami`",
            configuration={"k": "v`w"},
        )
        self.assertIn("- Command: `` echo `whoami` ``", rendered)
        self.assertIn('- Configuration: ``{"k": "v`w"}``', rendered)
        self.assertNotIn("- Command: `echo `whoami``", rendered)


class DeepJournalLineTests(unittest.TestCase):
    """F09: a deeply nested JSONL line is skipped like any malformed line."""

    def test_records_and_next_event_id_skip_a_deep_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "recovery.jsonl").write_text(
                '{"event_id":"r-1","sequence":1,"context":'
                + "[" * 100000
                + "]" * 100000
                + "}\n",
                encoding="utf-8",
            )
            journal = recovery.RecoveryJournal(path, "r")
            self.assertEqual(journal.records(), [])
            self.assertEqual(journal.next_event_id(), "r-0001")


class CrossRepoRedactorTests(unittest.TestCase):
    """F10: probe-output redaction covers ftp and scheme-relative URLs."""

    def test_ftp_and_scheme_relative_forms_are_redacted(self):
        for text in (
            "probe failed: ftp://user:pw@host/v1",
            "probe failed: //user:pw@host/v1",
        ):
            with self.subTest(text=text):
                redacted = check_cross_repo_consistency._redacted_text(text)
                self.assertNotIn("pw", redacted)

    def test_https_form_still_redacts(self):
        self.assertEqual(
            check_cross_repo_consistency._redacted_text(
                "probe failed: https://user:pw@host/v1"
            ),
            "probe failed: https://host/v1",
        )


class OperationRedactionTests(unittest.TestCase):
    """F11: the operation is redacted in the record, report and threshold."""

    def test_record_operation_is_credential_free(self):
        record = recovery.RecoveryRecord(
            run_id="r",
            event_id="e",
            timestamp="t",
            sequence=1,
            code="UPSTREAM_FAILED",
            tier=3,
            operation="POST https://user:pw@host?api_key=SECRET",
            failure="",
            recovery_action="retry",
            output_effects="none",
        )
        rendered = json.dumps(record.to_dict()) + record.to_markdown()
        self.assertNotIn("pw", rendered)
        self.assertNotIn("SECRET", rendered)

    def test_threshold_message_operation_is_credential_free(self):
        diagnostic = recovery.Diagnostic(
            recovery.TIER_ERROR,
            "UPSTREAM_FAILED",
            "POST https://user:pw@host?api_key=SECRET",
            "boom",
        )
        message = recovery.threshold_failure_message(diagnostic, 2)
        self.assertNotIn("pw", message)
        self.assertNotIn("SECRET", message)

    def test_a_plain_operation_is_unchanged(self):
        diagnostic = recovery.Diagnostic(
            recovery.TIER_ERROR, "UPSTREAM_FAILED", "GET /v1/models", "boom"
        )
        self.assertEqual(diagnostic.operation, "GET /v1/models")


class RepairR01RedactionTests(unittest.TestCase):
    """R01: scheme-less query/fragment and bare-userinfo credentials leaked."""

    def test_scheme_less_fragment_and_query_credentials_are_masked(self):
        for raw in ("foo#api_key=SECRET", "foo?api_key=SECRET"):
            with self.subTest(raw=raw):
                redacted = recovery.redact_value(raw)
                self.assertNotIn("SECRET", redacted)

    def test_non_credential_scheme_less_values_are_unchanged(self):
        for raw in ("foo#section", "foo?max_tokens=512"):
            with self.subTest(raw=raw):
                self.assertEqual(recovery.redact_value(raw), raw)

    def test_bare_userinfo_in_free_text_is_masked(self):
        redacted = recovery.redact_text("POST user:s3cr3t@host/v1")
        self.assertNotIn("s3cr3t", redacted)
        self.assertNotIn("user:s3cr3t", redacted)

    def test_bare_userinfo_operation_does_not_reach_the_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = recovery.RecoveryJournal(Path(directory) / "j", "r")
            journal.start()
            controller = recovery.RecoveryController(
                run_id="r", journal=journal, logger=lambda line: None
            )
            controller.emit(
                recovery.Diagnostic(
                    recovery.TIER_ERROR,
                    "UPSTREAM_FAILED",
                    "POST user:s3cr3t@host/v1",
                    "failed",
                ),
                recovery_action="retry",
            )
            text = (Path(directory) / "j" / "recovery.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("s3cr3t", text)

    def test_ordinary_email_without_a_path_is_unchanged(self):
        for text in ("contact user@example.com", "mailto:user@example.com"):
            with self.subTest(text=text):
                self.assertEqual(recovery.redact_text(text), text)


class DotsEffortValidationTests(unittest.TestCase):
    """F12: the Dots branch validates an explicit reasoning level."""

    def test_unknown_dots_effort_is_rejected_like_every_family(self):
        with mock.patch.dict(os.environ, {"DOTS_API_KEY": "x"}, clear=True):
            dots = model_runtime.resolve_agent_selection(
                harness="dsh", model="dots3-note-preview"
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.dsh_patch_text(dots, effort_override="turbo")
            patch = model_runtime.dsh_patch_text(
                dots, effort_override="high"
            )
            self.assertIn('reasoningEffort: "high"', patch)

    def test_qwen_still_rejects_the_unknown_effort(self):
        with mock.patch.dict(
            os.environ, {"PAPER_QWEN_API_KEY": "secret"}, clear=True
        ):
            qwen = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.dsh_patch_text(qwen, effort_override="turbo")


if __name__ == "__main__":
    unittest.main()
