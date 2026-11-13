"""Tests for the shared model-runtime catalog policy."""

from __future__ import annotations

import gzip
import http.client
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from agent_common import llm_retry, model_runtime


class CatalogPolicyTests(unittest.TestCase):
    def setUp(self):
        self._path = model_runtime._MODEL_CATALOG_PATH
        self._fatal = model_runtime.MISSING_CATALOG_IS_FATAL

    def tearDown(self):
        model_runtime._MODEL_CATALOG_PATH = self._path
        model_runtime.MISSING_CATALOG_IS_FATAL = self._fatal

    def test_reload_raises_for_a_missing_catalog_when_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            model_runtime._MODEL_CATALOG_PATH = Path(directory) / "missing.json"
            model_runtime.MISSING_CATALOG_IS_FATAL = True
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.reload_model_catalog()

    def test_reload_tolerates_a_missing_catalog_when_not_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            model_runtime._MODEL_CATALOG_PATH = Path(directory) / "missing.json"
            model_runtime.MISSING_CATALOG_IS_FATAL = False
            model_runtime.reload_model_catalog()


class DefaultReasoningLevelTests(unittest.TestCase):
    """The catalog's defaults are the one policy every harness reads."""

    def selection(self, model, harness="dsh"):
        return model_runtime.resolve_agent_selection(harness=harness, model=model)

    def test_the_catalog_records_the_documented_defaults(self):
        self.assertEqual(model_runtime.DEEPSEEK_DEFAULT_EFFORT, "high")
        self.assertEqual(model_runtime.DOTS_DEFAULT_EFFORT, "high")
        self.assertEqual(model_runtime.QWEN_DEFAULT_EFFORT, "medium")
        self.assertEqual(
            model_runtime.default_reasoning_level("deepseek-flash"), "high"
        )
        self.assertEqual(
            model_runtime.default_reasoning_level("dots3-note-preview"), "high"
        )
        self.assertEqual(
            model_runtime.default_reasoning_level("qwen38-27b"), "medium"
        )

    def test_the_catalog_facts_are_read_through_one_helper(self):
        # Context length, harness cap, chat cap and reasoning level come from
        # the catalog, so a consumer never hard-codes one of them.
        self.assertEqual(
            model_runtime.model_fact("qwen38-27b", "context_window"), 150000
        )
        self.assertEqual(model_runtime.model_fact("qwen38-27b", "max_tokens"), 32768)
        self.assertEqual(
            model_runtime.model_fact("qwen38-27b", "chat_max_tokens"),
            model_runtime.QWEN_CHAT_MAX_TOKENS,
        )
        self.assertEqual(
            model_runtime.model_fact("dots3-note-preview", "context_window"),
            model_runtime.DOTS_CONTEXT_WINDOW,
        )
        self.assertEqual(
            model_runtime.model_fact("dots3-note-preview", "chat_max_tokens"),
            model_runtime.DOTS_CHAT_MAX_TOKENS,
        )
        self.assertEqual(
            model_runtime.model_fact("deepseek-flash", "default_reasoning_level"),
            "high",
        )
        self.assertEqual(
            model_runtime.model_fact("unknown-model", "context_window", 7), 7
        )

    def test_a_model_that_keeps_its_own_default_has_no_catalog_level(self):
        # GPT and Claude entries deliberately omit the field; the generated
        # configuration must not substitute a literal for them.
        for model in ("gpt-sol", "gpt-terra", "claude-opus", "claude-sonnet"):
            self.assertIsNone(model_runtime.default_reasoning_level(model))
        self.assertIsNone(model_runtime.default_reasoning_level("default"))
        self.assertIsNone(model_runtime.default_reasoning_level("https://host/v1"))

    def test_the_dsh_patch_writes_the_catalog_level(self):
        deepseek = model_runtime.dsh_patch_text(self.selection("deepseek-flash"))
        self.assertIn('reasoningEffort: "high"', deepseek)
        dots = model_runtime.dsh_patch_text(self.selection("dots3-note-preview"))
        self.assertIn("              high: max", dots)
        self.assertIn('    reasoningEffort: "high"', dots)
        qwen = model_runtime.dsh_patch_text(self.selection("qwen38-27b"))
        self.assertIn('    reasoningEffort: "medium"', qwen)

    def test_the_gpt_and_claude_patches_leave_the_effort_unset(self):
        for model in ("gpt-sol", "claude-sonnet"):
            patch = model_runtime.dsh_patch_text(self.selection(model))
            self.assertNotIn("reasoningEffort", patch)

    def test_the_codex_route_passes_the_same_defaults(self):
        deepseek = model_runtime.codex_provider_args(self.selection("deepseek-flash"))
        self.assertIn('model_reasoning_effort="high"', deepseek)
        dots = model_runtime.codex_provider_args(
            self.selection("dots3-note-preview")
        )
        self.assertIn('model_reasoning_effort="high"', dots)
        qwen = model_runtime.codex_provider_args(self.selection("qwen38-27b"))
        self.assertIn('model_reasoning_effort="medium"', qwen)
        self.assertEqual(
            model_runtime.codex_provider_args(self.selection("gpt-sol")),
            ["-c", 'model_provider="openai"'],
        )

    def test_the_operator_override_still_wins(self):
        patch = model_runtime.dsh_patch_text(
            self.selection("deepseek-flash"), effort_override="low"
        )
        self.assertIn('reasoningEffort: "low"', patch)

    def test_claude_code_gets_the_deepseek_default_only(self):
        env = model_runtime.claude_environment(self.selection("deepseek-flash"))
        self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], "high")
        # GPT and Claude keep their own effort: no override is exported.
        self.assertEqual(
            model_runtime.claude_environment(self.selection("claude-opus")), {}
        )
        self.assertEqual(
            model_runtime.claude_environment(self.selection("gpt-sol")), {}
        )

    def test_an_unknown_dots_effort_falls_back_to_the_default_level(self):
        overlay = llm_retry.dots_dsh_overlay(
            model="dots3-note-prev",
            base_url="https://note3-prev-api.askdiandian.com/v1",
            context_window=393000,
            max_tokens=65536,
            reasoning_effort="turbo",
        )
        # The overlay must never declare a level DSH's schema rejects; the API
        # has one thinking level, so the unknown spelling is the default.
        self.assertIn('    reasoningEffort: "high"', overlay)
        self.assertNotIn("turbo", overlay)

    def test_thinking_off_disables_dots_effort_selection(self):
        overlay = llm_retry.dots_dsh_overlay(
            model="dots3-note-prev",
            base_url="https://note3-prev-api.askdiandian.com/v1",
            context_window=393000,
            max_tokens=65536,
            reasoning_effort="none",
        )
        self.assertIn("reasoningEfforts: false", overlay)
        self.assertNotIn("    reasoningEffort: ", overlay)


class ReasoningEffortOverrideTests(unittest.TestCase):
    """The shared --reasoning-effort override reaches every family."""

    def selection(self, model, harness="dsh"):
        return model_runtime.resolve_agent_selection(harness=harness, model=model)

    def with_override(self, level):
        return mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": level}
        )

    def test_the_override_wins_and_maps_to_each_family(self):
        with self.with_override("medium"):
            self.assertEqual(model_runtime.reasoning_effort_override(), "medium")
            deepseek = model_runtime.dsh_patch_text(
                self.selection("deepseek-flash")
            )
            self.assertIn('reasoningEffort: "high"', deepseek)
            qwen = model_runtime.dsh_patch_text(self.selection("qwen38-27b"))
            self.assertIn('reasoningEffort: "medium"', qwen)
            dots = model_runtime.dsh_patch_text(
                self.selection("dots3-note-preview")
            )
            self.assertIn('reasoningEffort: "medium"', dots)
        with self.with_override("max"):
            qwen = model_runtime.dsh_patch_text(self.selection("qwen38-27b"))
            self.assertIn('reasoningEffort: "xhigh"', qwen)

    def test_off_is_expressed_by_every_route(self):
        with self.with_override("off"):
            deepseek = model_runtime.dsh_patch_text(
                self.selection("deepseek-flash")
            )
            self.assertIn('reasoningEffort: "off"', deepseek)
            qwen = model_runtime.dsh_patch_text(self.selection("qwen38-27b"))
            self.assertIn('reasoningEffort: "off"', qwen)
            self.assertIn('              "off":', qwen)
            dots = model_runtime.dsh_patch_text(
                self.selection("dots3-note-preview")
            )
            self.assertIn("reasoningEfforts: false", dots)
            self.assertNotIn('    reasoningEffort: "', dots)

    def test_gpt_and_claude_receive_the_override_on_every_route(self):
        with self.with_override("high"):
            gpt = model_runtime.dsh_patch_text(self.selection("gpt-sol"))
            self.assertIn('reasoningEffort: "high"', gpt)
            claude = model_runtime.dsh_patch_text(
                self.selection("claude-opus")
            )
            self.assertIn('reasoningEffort: "high"', claude)
            codex = model_runtime.codex_provider_args(
                self.selection("gpt-sol", harness="gpt-codex")
            )
            self.assertIn('model_reasoning_effort="high"', codex)
            env = model_runtime.claude_environment(
                self.selection("claude-opus", harness="claude-code")
            )
            self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], "high")
            # A config-owned model still receives the codex level.
            owned = model_runtime.codex_provider_args(
                self.selection("default", harness="gpt-codex")
            )
            self.assertIn('model_reasoning_effort="high"', owned)

    def test_the_override_is_carried_in_the_child_environment(self):
        with self.with_override("low"):
            env = model_runtime.selection_env(
                self.selection("deepseek-flash")
            )
        self.assertEqual(env["PAPER_AGENT_REASONING_EFFORT"], "low")

    def test_none_is_off_and_an_unknown_level_is_an_error(self):
        with self.with_override("none"):
            self.assertEqual(model_runtime.reasoning_effort_override(), "off")
        with self.with_override("turbo"):
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.reasoning_effort_override()
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.normalize_reasoning_effort("deepseek-flash", "turbo")

    def test_normalization_is_the_one_translation_table(self):
        self.assertEqual(
            model_runtime.normalize_reasoning_effort("deepseek-flash", "medium"),
            "high",
        )
        self.assertEqual(
            model_runtime.normalize_reasoning_effort("qwen38-27b", "max"), "xhigh"
        )
        self.assertEqual(
            model_runtime.normalize_reasoning_effort("dots3-note-preview", "medium"),
            "medium",
        )
        self.assertEqual(
            model_runtime.normalize_reasoning_effort("gpt-sol", "xhigh"), "xhigh"
        )


class _RecordingTarget(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.server.seen_auth = self.headers.get("Authorization")
        self.server.hits = getattr(self.server, "hits", 0) + 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _RedirectingTarget(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.server.hits = getattr(self.server, "hits", 0) + 1
        self.send_response(302)
        self.send_header(
            "Location", "http://127.0.0.1:%d/landed" % self.server.target_port
        )
        self.send_header("Content-Length", "0")
        self.end_headers()


class _ChunkRecorder(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.server.called = True
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class _GzipTarget(BaseHTTPRequestHandler):
    payload = b'{"hello":"world"}'

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib name
        return

    def do_POST(self):  # noqa: N802 - stdlib name
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = gzip.compress(self.payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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


class PrepProxyTests(unittest.TestCase):
    def test_stopping_an_unknown_base_url_is_a_noop(self):
        base, error = model_runtime.start_prep_proxy(
            "https://upstream.example/v1", {"temperature": 0}
        )
        self.assertEqual(error, "")
        try:
            model_runtime.stop_prep_proxy("http://127.0.0.1:1/v1")
            self.assertIn(base, model_runtime._PROXY_BY_BASE)
        finally:
            model_runtime.stop_prep_proxy(base)

    def test_a_redirect_is_refused_instead_of_forwarding_the_authorization(self):
        target = _start_server(_RecordingTarget)
        redirector = _start_server(_RedirectingTarget)
        redirector.target_port = target.server_address[1]
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % redirector.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            request = urllib.request.Request(
                base + "/v1/chat/completions",
                data=b"{}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer SECRET",
                },
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(caught.exception.code, 502)
            self.assertIsNone(getattr(target, "seen_auth", None))
            self.assertEqual(getattr(target, "hits", 0), 0)
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(target)
            _stop_server(redirector)

    def test_a_chunked_post_is_rejected_instead_of_forwarded_without_its_body(self):
        upstream = _start_server(_ChunkRecorder)
        upstream.called = False
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = _host_port(base)
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            connection.send(b"5\r\nhello\r\n0\r\n\r\n")
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            connection.close()
            self.assertFalse(upstream.called)
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_content_encoding_is_forwarded_with_the_encoded_body(self):
        upstream = _start_server(_GzipTarget)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = _host_port(base)
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.getheader("Content-Encoding"), "gzip")
            self.assertEqual(gzip.decompress(body), _GzipTarget.payload)
            connection.close()
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)


class RedactionTests(unittest.TestCase):
    def test_an_endpoint_record_is_redacted_and_reports_api_key_set(self):
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_MODEL_API_KEY": "sk-live"}, clear=True
        ):
            record = model_runtime.effective_agent_model(
                [
                    "--endpoint",
                    "https://user:pw@host/v1?api_key=SECRET",
                    "--model-name",
                    "m",
                ]
            )
        self.assertEqual(record["endpoint"], "https://host/v1")
        self.assertTrue(record["api_key_set"])
        text = json.dumps(record)
        self.assertNotIn("pw", text)
        self.assertNotIn("SECRET", text)

    def test_a_raw_endpoint_model_is_redacted_in_the_harness_record(self):
        record = model_runtime.effective_agent_model(
            [
                "dsh",
                "--harness",
                "dsh",
                "--model",
                "https://user:pw@host/v1?api_key=SECRET",
            ]
        )
        self.assertNotIn("pw", json.dumps(record))
        self.assertNotIn("SECRET", json.dumps(record))

    def test_the_query_splitter_and_validator_share_one_credential_table(self):
        for key in (
            "apikey",
            "api-key",
            "api_key",
            "x-api-key",
            "accessKeyId",
            "auth",
            "authorization",
            "passwd",
            "client_secret",
        ):
            url = "https://host/v1?%s=SECRET" % key
            with self.subTest(key=key):
                safe, secret = model_runtime.split_endpoint_credentials(url)
                self.assertEqual(secret, "SECRET")
                self.assertNotIn("SECRET", safe)
                self.assertTrue(model_runtime.endpoint_credentials(url))
        safe, secret = model_runtime.split_endpoint_credentials(
            "https://host/v1?model=qwen"
        )
        self.assertEqual(safe, "https://host/v1?model=qwen")
        self.assertEqual(secret, "")
        self.assertEqual(model_runtime.endpoint_credentials(safe), "")

    def test_qwen_and_dots_base_url_credentials_do_not_reach_artifacts(self):
        with mock.patch.dict(
            os.environ,
            {"PAPER_QWEN_BASE_URL": "https://user:secret@qwen.example/v1"},
            clear=True,
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            patch = model_runtime.dsh_patch_text(selection)
            argv = " ".join(model_runtime.codex_provider_args(selection))
            claude = model_runtime.claude_environment(selection)
            env = model_runtime.selection_env(selection)
        self.assertNotIn("secret", patch)
        self.assertNotIn("secret", argv)
        self.assertNotIn("secret", claude["ANTHROPIC_BASE_URL"])
        # The secret is moved to the key channel rather than dropped.
        self.assertEqual(env["PAPER_QWEN_API_KEY"], "secret")

        with mock.patch.dict(
            os.environ,
            {"PAPER_DOTS_BASE_URL": "https://user:secret@dots.example/v1"},
            clear=True,
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="dots3-note-preview"
            )
            patch = model_runtime.dsh_patch_text(selection)
            claude = model_runtime.claude_environment(selection)
            env = model_runtime.selection_env(selection)
        self.assertNotIn("secret", patch)
        self.assertNotIn("secret", claude["ANTHROPIC_BASE_URL"])
        self.assertEqual(env["DOTS_API_KEY"], "secret")

    def test_endpoint_errors_do_not_echo_the_credential(self):
        for url in (
            "https://user:pw@host:bad/v1",
            "https://user:pw@/v1",
            "https://user:pw@host:99999/v1",
            "http://[::1",
            "https://user:pw@[bad/v1?api_key=SECRET",
        ):
            with self.subTest(url=url):
                with self.assertRaises(model_runtime.ConfigError) as caught:
                    model_runtime.endpoint_base(url)
                self.assertNotIn("pw", str(caught.exception))
                self.assertNotIn("SECRET", str(caught.exception))

    def test_a_scheme_relative_malformed_base_url_does_not_reach_artifacts(self):
        with mock.patch.dict(
            os.environ,
            {"PAPER_QWEN_BASE_URL": "//user:pw@[bad/v1?api_key=SECRET"},
            clear=True,
        ):
            claude_selection = model_runtime.resolve_agent_selection(
                harness="claude-code", model="qwen38-27b"
            )
            claude = model_runtime.claude_environment(claude_selection)
            codex_selection = model_runtime.resolve_agent_selection(
                harness="gpt-codex", model="qwen38-27b"
            )
            argv = " ".join(model_runtime.codex_provider_args(codex_selection))
            dsh_selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            with self.assertRaises(model_runtime.ConfigError) as caught:
                model_runtime.dsh_patch_text(dsh_selection)
        for text in (claude["ANTHROPIC_BASE_URL"], argv, str(caught.exception)):
            self.assertNotIn("pw", text)
            self.assertNotIn("SECRET", text)

    def test_prep_redaction_covers_common_credential_names(self):
        redacted = model_runtime.redact_prep_object(
            {
                "auth": "A",
                "bearer": "B",
                "accessKeyId": "C",
                "access_key_id": "D",
                "api_key": {"value": "E"},
                "max_tokens": 5,
                "temperature": 0,
            }
        )
        for key in ("auth", "bearer", "accessKeyId", "access_key_id", "api_key"):
            self.assertEqual(redacted[key], "[REDACTED]")
        self.assertEqual(redacted["max_tokens"], 5)
        self.assertEqual(redacted["temperature"], 0)

    def test_the_selection_repr_never_prints_the_live_api_key(self):
        selection = model_runtime.AgentSelection(
            harness="dsh", model="qwen38-27b", api_key="sk-live"
        )
        self.assertNotIn("sk-live", repr(selection))
        self.assertTrue(selection.to_record()["api_key_set"])


class AliasHelperTests(unittest.TestCase):
    def test_helpers_resolve_aliases_before_looking_up_the_pair(self):
        self.assertTrue(
            model_runtime.agent_pair_supported("codex", "gpt-6.1-sol")
        )
        self.assertEqual(
            model_runtime.agent_harness_for_model("gpt-6.1-sol"), "gpt-codex"
        )
        self.assertEqual(model_runtime.agent_model_for_harness("codex"), "gpt-sol")
        self.assertEqual(
            model_runtime.pair_from_options(None, "gpt-6.1-sol"),
            ("gpt-codex", "gpt-sol"),
        )
        self.assertEqual(
            model_runtime.pair_from_options("codex", None),
            ("gpt-codex", "gpt-sol"),
        )
        self.assertEqual(
            model_runtime.pair_from_options(None, "opus"),
            ("claude-code", "claude-opus"),
        )


class SelectionProblemDryRunTests(unittest.TestCase):
    def test_dry_run_does_not_evaluate_an_invalid_operator_override(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "turbo"}, clear=True
        ):
            self.assertEqual(
                model_runtime.selection_problem(selection, dry_run=True), ""
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.selection_problem(selection)


class CatalogReloadTests(unittest.TestCase):
    def setUp(self):
        self._path = model_runtime._MODEL_CATALOG_PATH

    def tearDown(self):
        model_runtime._MODEL_CATALOG_PATH = self._path
        model_runtime.reload_model_catalog()

    @staticmethod
    def _entry(catalog, model_id):
        return next(entry for entry in catalog["models"] if entry["id"] == model_id)

    def _write_catalog(self, mutate, reload=True):
        catalog = json.loads(self._path.read_text(encoding="utf-8"))
        mutate(catalog)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        model_runtime._MODEL_CATALOG_PATH = path
        if reload:
            model_runtime.reload_model_catalog()

    def test_reload_refreshes_the_exported_constants_and_request_routes(self):
        def mutate(catalog):
            qwen = self._entry(catalog, "qwen3.8-27b")
            qwen["context_window"] = 123456
            qwen["max_tokens"] = 11111
            qwen["default_reasoning_level"] = "low"
            qwen["base_url"] = "https://user:pw@qwen.example/v1"

        self._write_catalog(mutate)
        self.assertEqual(model_runtime.QWEN_CONTEXT_WINDOW, 123456)
        self.assertEqual(model_runtime.QWEN_MAX_TOKENS, 11111)
        self.assertEqual(model_runtime.QWEN_DEFAULT_EFFORT, "low")
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        patch = model_runtime.dsh_patch_text(selection)
        self.assertIn("123456", patch)
        self.assertIn("11111", patch)
        self.assertIn('reasoningEffort: "low"', patch)
        self.assertNotIn("pw", patch)

    def test_a_failed_reload_leaves_the_previous_values_in_place(self):
        def mutate(catalog):
            # The invalid entry must be iterated *after* the changed one, or the
            # old per-model write loop would raise before writing anything and
            # this test could pass without the atomic update.
            self._entry(catalog, "deepseek-flash")["context_window"] = 777777
            self._entry(catalog, "dots3-note-prev")["max_tokens"] = 999999999

        self._write_catalog(mutate, reload=False)
        before = (
            model_runtime.AGENT_MODELS["deepseek-flash"]["context_window"],
            model_runtime.DEEPSEEK_CONTEXT_WINDOW,
        )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.reload_model_catalog()
        self.assertEqual(
            before,
            (
                model_runtime.AGENT_MODELS["deepseek-flash"]["context_window"],
                model_runtime.DEEPSEEK_CONTEXT_WINDOW,
            ),
        )

    def test_the_catalog_owns_base_url_provider_and_key_env(self):
        self.assertEqual(
            model_runtime.AGENT_MODELS["qwen38-27b"]["dsh_provider"], "qwen"
        )
        self.assertEqual(
            model_runtime.AGENT_MODELS["qwen38-27b"]["base_url"],
            model_runtime.QWEN_DEFAULT_BASE_URL,
        )
        self.assertEqual(
            model_runtime.AGENT_MODELS["dots3-note-preview"]["api_key_env"],
            "DOTS_API_KEY",
        )
        self.assertNotIn("base_url", llm_retry.DEFAULT_POLICY["dots"])
        self.assertNotIn("model", llm_retry.DEFAULT_POLICY["dots"])

    def test_a_catalog_key_env_reaches_patch_argv_and_environment(self):
        def mutate(catalog):
            qwen = self._entry(catalog, "qwen3.8-27b")
            qwen["base_url"] = "https://qwen.example/v1"
            qwen["api_key_env"] = "CUSTOM_QWEN_KEY"

        self._write_catalog(mutate)
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        patch = model_runtime.dsh_patch_text(selection)
        self.assertIn('baseURL: "https://qwen.example/v1"', patch)
        self.assertIn("apiKeyEnv: CUSTOM_QWEN_KEY", patch)
        argv = model_runtime.codex_provider_args(selection)
        self.assertIn('model_providers.qwen.env_key="CUSTOM_QWEN_KEY"', argv)
        self.assertIn(
            "CUSTOM_QWEN_KEY", model_runtime.selection_env(selection)
        )

    def test_a_catalog_dots_key_env_reaches_the_overlay_and_environment(self):
        def mutate(catalog):
            dots = self._entry(catalog, "dots3-note-prev")
            dots["api_key_env"] = "CUSTOM_DOTS_KEY"

        self._write_catalog(mutate)
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="dots3-note-preview"
        )
        with mock.patch.dict(
            os.environ, {"CUSTOM_DOTS_KEY": "dots-secret"}, clear=True
        ):
            patch = model_runtime.dsh_patch_text(selection)
            env = model_runtime.selection_env(selection)
        self.assertIn("apiKeyEnv: CUSTOM_DOTS_KEY", patch)
        self.assertEqual(env["CUSTOM_DOTS_KEY"], "dots-secret")

    def test_a_field_removed_from_the_catalog_does_not_survive_a_reload(self):
        def mutate(catalog):
            qwen = self._entry(catalog, "qwen3.8-27b")
            for key in ("base_url", "default_reasoning_level", "chat_max_tokens"):
                qwen.pop(key, None)

        self._write_catalog(mutate)
        qwen = model_runtime.AGENT_MODELS["qwen38-27b"]
        self.assertIsNone(qwen.get("base_url"))
        self.assertIsNone(qwen.get("default_reasoning_level"))
        self.assertIsNone(qwen.get("chat_max_tokens"))
        # The exported constant falls back to the harness cap instead of
        # raising KeyError for the removed optional field.
        self.assertEqual(
            model_runtime.QWEN_CHAT_MAX_TOKENS, model_runtime.QWEN_MAX_TOKENS
        )

    def test_a_duplicate_catalog_id_is_rejected(self):
        def mutate(catalog):
            duplicate = dict(self._entry(catalog, "deepseek-flash"))
            duplicate["context_window"] = 1000
            duplicate["max_tokens"] = 100
            duplicate["chat_max_tokens"] = 200
            catalog["models"].append(duplicate)

        self._write_catalog(mutate, reload=False)
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.reload_model_catalog()

    def test_a_catalog_provider_reaches_each_route_builder(self):
        def mutate(catalog):
            self._entry(catalog, "qwen3.8-27b")["provider"] = "qwen-renamed"
            self._entry(catalog, "dots3-note-prev")["provider"] = "dots-renamed"
            self._entry(catalog, "deepseek-flash")["provider"] = "deepseek-renamed"

        self._write_catalog(mutate)
        qwen = model_runtime.dsh_patch_text(
            model_runtime.resolve_agent_selection(harness="dsh", model="qwen38-27b")
        )
        self.assertIn("      qwen-renamed:", qwen)
        self.assertIn("    provider: qwen-renamed", qwen)
        dots = model_runtime.dsh_patch_text(
            model_runtime.resolve_agent_selection(
                harness="dsh", model="dots3-note-preview"
            )
        )
        self.assertIn("      dots-renamed:", dots)
        self.assertIn("    provider: dots-renamed", dots)
        deepseek = model_runtime.dsh_patch_text(
            model_runtime.resolve_agent_selection(
                harness="dsh", model="deepseek-flash"
            )
        )
        self.assertIn('    provider: "deepseek-renamed"', deepseek)


class EndpointCredentialDocTests(unittest.TestCase):
    def test_the_docstring_describes_redacted_records_not_a_leak(self):
        doc = model_runtime.endpoint_credentials.__doc__ or ""
        self.assertIn("redacted", doc)
        self.assertNotIn("would be written to the run", doc)


class SelectionReprRedactionTests(unittest.TestCase):
    def test_an_endpoint_repr_redacts_userinfo_and_query_credentials(self):
        previous = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://user:pw@host/v1?api_key=SECRET"
            )
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = previous
        text = repr(selection)
        self.assertNotIn("pw", text)
        self.assertNotIn("SECRET", text)
        self.assertIn("https://host/v1", text)


class EndpointResolutionTests(unittest.TestCase):
    def test_a_named_model_returns_the_documented_empty_resolution(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        self.assertEqual(
            model_runtime.resolve_endpoint_model(selection), ("", None, "")
        )

    def test_an_infinite_context_length_is_ignored(self):
        self.assertIsNone(
            model_runtime._endpoint_context(
                [{"id": "m", "context_length": 1e400}], "m"
            )
        )

    def test_a_malformed_url_never_returns_its_credential(self):
        safe, secret = model_runtime.split_endpoint_credentials(
            "https://user:pw@[bad/v1?api_key=S"
        )
        self.assertNotIn("pw", safe)
        self.assertNotIn("api_key", safe)
        self.assertNotIn("S", safe)
        self.assertEqual(secret, "")

    def test_a_scheme_relative_or_scheme_less_url_never_returns_its_credential(self):
        for url in (
            "//user:pw@[bad/v1?api_key=SECRET",
            "user:pw@[bad/v1?api_key=SECRET",
        ):
            with self.subTest(url=url):
                safe, secret = model_runtime.split_endpoint_credentials(url)
                self.assertNotIn("pw", safe)
                self.assertNotIn("SECRET", safe)
                self.assertNotIn("api_key", safe)
                self.assertEqual(secret, "")

    def test_a_non_http_url_never_returns_its_credential(self):
        for url in (
            "ftp://user:pw@host/v1?api_key=SECRET",
            "//user:pw@host/v1?api_key=SECRET",
        ):
            with self.subTest(url=url):
                safe, _secret = model_runtime.split_endpoint_credentials(url)
                self.assertNotIn("pw", safe)
                self.assertNotIn("SECRET", safe)

    def test_a_failed_models_probe_is_an_error_not_a_default_model(self):
        class Unauthorized(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - stdlib name
                return

            def do_GET(self):  # noqa: N802 - stdlib name
                body = b'{"error":"unauthorized"}'
                self.send_response(401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = _start_server(Unauthorized)
        try:
            base = "http://127.0.0.1:%d/v1" % server.server_address[1]
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model=base
            )
            model, context, error = model_runtime.resolve_endpoint_model(selection)
            self.assertEqual(model, "")
            self.assertIsNone(context)
            self.assertIn("401", error)
            self.assertNotEqual(model_runtime.selection_problem(selection), "")
        finally:
            _stop_server(server)

    def test_a_missing_models_route_is_not_a_probe_failure(self):
        for status in (404, 405):
            with self.subTest(status=status):

                class NoModelsRoute(BaseHTTPRequestHandler):
                    def log_message(self, *args):  # noqa: A003 - stdlib name
                        return

                    def do_GET(self):  # noqa: N802 - stdlib name
                        body = b'{"error":"no models route"}'
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                    def do_POST(self):  # noqa: N802 - stdlib name
                        self.rfile.read(
                            int(self.headers.get("Content-Length") or 0)
                        )
                        body = b'{"choices":[{"message":{"content":"ok"}}]}'
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                server = _start_server(NoModelsRoute)
                try:
                    base = "http://127.0.0.1:%d/v1" % server.server_address[1]
                    selection = model_runtime.resolve_agent_selection(
                        harness="dsh", model=base
                    )
                    self.assertEqual(
                        model_runtime.resolve_endpoint_model(selection),
                        ("default", None, ""),
                    )
                    self.assertEqual(
                        model_runtime.selection_problem(selection), ""
                    )
                    # The endpoint itself works; only the optional listing route
                    # is missing.
                    request = urllib.request.Request(
                        base + "/chat/completions",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                finally:
                    _stop_server(server)

    def test_selection_problem_reports_an_ambiguous_endpoint_listing(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://h/v1"
        )
        with mock.patch.object(
            model_runtime, "_endpoint_catalog", return_value=[{"id": "a"}, {"id": "b"}]
        ):
            self.assertIn(
                "pass --agent-model-name",
                model_runtime.selection_problem(selection),
            )


class PatchTextOverrideTests(unittest.TestCase):
    def selection(self, model="qwen38-27b", harness="dsh"):
        return model_runtime.resolve_agent_selection(harness=harness, model=model)

    def test_a_credential_bearing_base_override_is_not_written(self):
        patch = model_runtime.dsh_patch_text(
            self.selection(),
            base_url_override="https://user:secret@qwen.example/v1",
        )
        self.assertNotIn("secret", patch)
        self.assertIn('baseURL: "https://qwen.example/v1"', patch)

    def test_a_chat_completions_override_is_normalized(self):
        for model in ("qwen38-27b", "dots3-note-preview"):
            with self.subTest(model=model):
                patch = model_runtime.dsh_patch_text(
                    self.selection(model),
                    base_url_override="https://qwen.example/v1/chat/completions",
                )
                self.assertIn('baseURL: "https://qwen.example/v1"', patch)
                self.assertNotIn("/chat/completions", patch)

    def test_provider_override_reaches_every_family(self):
        cases = {
            "qwen38-27b": "      gateway:\n",
            "dots3-note-preview": "      gateway:\n",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                patch = model_runtime.dsh_patch_text(
                    self.selection(model), provider_override="gateway"
                )
                self.assertIn(expected, patch)
                self.assertIn("provider: gateway", patch)
        endpoint = model_runtime.dsh_patch_text(
            self.selection("https://h/v1"), provider_override="gateway"
        )
        self.assertIn("      gateway:\n", endpoint)
        self.assertIn("provider: gateway", endpoint)

    def test_the_default_sentinel_plus_a_raw_model_infers_the_provider(self):
        patch = model_runtime.dsh_patch_text(
            self.selection("default", "gpt-codex"), raw_model="gpt-6.1-sol"
        )
        self.assertIn('provider: "openai"', patch)
        self.assertIn('model: "gpt-6.1-sol"', patch)
        claude = model_runtime.dsh_patch_text(
            self.selection("default", "claude-code"), raw_model="claude-opus-5-5"
        )
        self.assertIn('provider: "anthropic"', claude)

    def test_invalid_capacity_overrides_are_rejected(self):
        for kwargs in (
            {"context_window_override": -5},
            {"context_window_override": 0},
            {"max_tokens_override": 0},
            {"context_window_override": 100, "max_tokens_override": 100},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.dsh_patch_text(self.selection(), **kwargs)


class ClaudeEffortTests(unittest.TestCase):
    def selection(self, model):
        return model_runtime.resolve_agent_selection(
            harness="claude-code", model=model
        )

    def test_an_explicit_qwen_or_dots_override_uses_claudes_vocabulary(self):
        for model in ("qwen38-27b", "dots3-note-preview"):
            for requested, expected in (
                ("max", "max"),
                ("high", "high"),
                ("xhigh", "max"),
                ("minimal", "low"),
            ):
                with self.subTest(model=model, requested=requested):
                    with mock.patch.dict(
                        os.environ,
                        {"PAPER_AGENT_REASONING_EFFORT": requested},
                        clear=True,
                    ):
                        env = model_runtime.claude_environment(self.selection(model))
                    self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], expected)

    def test_an_off_override_does_not_emit_an_ignored_level(self):
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "off"}, clear=True
        ):
            env = model_runtime.claude_environment(self.selection("qwen38-27b"))
        self.assertNotIn("CLAUDE_CODE_EFFORT_LEVEL", env)


class PrepProxyHeaderTests(unittest.TestCase):
    def test_an_invalid_target_is_rejected_before_binding(self):
        for target in ("", "not-a-url", "ftp://host/v1"):
            with self.subTest(target=target):
                base, error = model_runtime.start_prep_proxy(target, {})
                self.assertEqual(base, "")
                self.assertTrue(error)

    def test_hop_by_hop_and_proxy_headers_are_not_forwarded(self):
        class Recorder(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - stdlib name
                return

            def do_POST(self):  # noqa: N802 - stdlib name
                self.server.headers = dict(self.headers)
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = _start_server(Recorder)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = _host_port(base)
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Content-Length", "2")
            connection.putheader("Connection", "keep-alive")
            connection.putheader("Proxy-Connection", "keep-alive")
            connection.putheader("Keep-Alive", "timeout=5")
            connection.putheader("TE", "trailers")
            connection.putheader("Proxy-Authorization", "Basic SECRET")
            connection.endheaders()
            connection.send(b"{}")
            connection.getresponse().read()
            connection.close()
            forwarded = {key.lower() for key in upstream.headers}
            for name in (
                "proxy-connection",
                "keep-alive",
                "te",
                "proxy-authorization",
            ):
                self.assertNotIn(name, forwarded)
            # urllib adds its own Connection: close; the client's keep-alive
            # value must not survive.
            self.assertNotEqual(
                upstream.headers.get("Connection", "").lower(), "keep-alive"
            )
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_the_prep_proxy_emits_one_server_and_date_pair(self):
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - stdlib name
                return

            def do_POST(self):  # noqa: N802 - stdlib name
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Server", "upstream-server")
                self.end_headers()
                self.wfile.write(body)

        upstream = _start_server(Upstream)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = _host_port(base)
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.request("POST", "/v1/chat/completions", body=b"{}")
            response = connection.getresponse()
            response.read()
            names = [name.lower() for name, _ in response.getheaders()]
            self.assertEqual(names.count("server"), 1)
            self.assertEqual(names.count("date"), 1)
            connection.close()
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_an_invalid_content_length_is_rejected(self):
        class Recorder(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - stdlib name
                return

            def do_POST(self):  # noqa: N802 - stdlib name
                self.server.called = True
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = _start_server(Recorder)
        upstream.called = False
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = _host_port(base)
            connection = http.client.HTTPConnection(host, port, timeout=5)
            connection.putrequest("POST", "/v1/chat/completions")
            connection.putheader("Content-Length", "abc")
            connection.endheaders()
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
            connection.close()
            self.assertFalse(upstream.called)
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)


class ProxyOwnershipTests(unittest.TestCase):
    def test_stop_compaction_proxy_only_stops_compaction_proxies(self):
        compaction, _ = model_runtime.start_compaction_proxy(
            "http://127.0.0.1:1/v1"
        )
        unrelated, _ = llm_retry.start_retry_proxy(
            "http://127.0.0.1:1/v2", name="unrelated"
        )
        self.assertIsNotNone(compaction)
        self.assertIsNotNone(unrelated)
        try:
            model_runtime.stop_compaction_proxy()
            self.assertNotIn(compaction, llm_retry._PROXY_REGISTRY)
            self.assertIn(unrelated, llm_retry._PROXY_REGISTRY)
        finally:
            llm_retry.stop_retry_proxy()

    def test_a_global_retry_stop_leaves_prep_proxies_running(self):
        base, error = model_runtime.start_prep_proxy(
            "https://upstream.example/v1", {"temperature": 0}
        )
        self.assertEqual(error, "")
        retry, _ = llm_retry.start_retry_proxy("http://127.0.0.1:1/v1")
        self.assertIsNotNone(retry)
        try:
            llm_retry.stop_retry_proxy()
            self.assertNotIn(retry, llm_retry._PROXY_REGISTRY)
            self.assertIn(base, model_runtime._PROXY_BY_BASE)
        finally:
            model_runtime.stop_prep_proxy(base)


if __name__ == "__main__":
    unittest.main()
