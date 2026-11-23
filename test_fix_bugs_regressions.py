"""Regression tests for the confirmed fix-bugs candidates (wave 1)."""

from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import check_cross_repo_consistency
import check_llm_policy
import check_model_catalog
import check_shared_package
from agent_common import chat, dsh_events, llm_retry, model_runtime


def _real_digest(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _start_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stop_server(server):
    server.shutdown()
    server.server_close()


def _raw_exchange(port, request_line, *, body=b""):
    payload = request_line.encode("latin-1") + b"\r\nHost: x\r\n"
    if body:
        payload += b"Content-Length: %d\r\n" % len(body)
    payload += b"\r\n" + body
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(payload)
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks)


class RedactionRegressionTests(unittest.TestCase):
    def test_redact_endpoint_url_never_returns_a_credential(self):
        cases = {
            "//user:pw@host/v1?api_key=SECRET": "//host/v1",
            "ftp://user:pw@host/v1?api_key=SECRET": "ftp://host/v1",
            "https:///v1?api_key=SECRET": "https:///v1",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                safe = model_runtime.redact_endpoint_url(raw)
                self.assertEqual(safe, expected)
                self.assertNotIn("pw", safe)
                self.assertNotIn("SECRET", safe)

    def test_an_unparsable_url_is_redacted_textually(self):
        safe = model_runtime.redact_endpoint_url(
            "https://user:pw@[bad/v1?api_key=SECRET"
        )
        self.assertNotIn("pw", safe)
        self.assertNotIn("SECRET", safe)

    def test_a_url_shaped_model_name_is_redacted_everywhere(self):
        selection = model_runtime.AgentSelection(
            harness="dsh",
            model="https://host/v1",
            model_name="https://user:pw@host/model?api_key=S",
        )
        record = selection.to_record()
        self.assertNotIn("pw", json.dumps(record))
        self.assertNotIn("api_key=S", json.dumps(record))
        self.assertEqual(record["model_name"], "https://host/model")
        self.assertEqual(record["id"], "https://host/model")
        self.assertNotIn("pw", repr(selection))
        self.assertNotIn("api_key=S", repr(selection))
        patch = model_runtime.dsh_patch_text(selection)
        self.assertNotIn("pw", patch)
        self.assertNotIn("api_key=S", patch)

    def test_non_http_url_shaped_models_are_redacted_everywhere(self):
        cases = {
            "//user:pw@host/v1?api_key=S": "//host/v1",
            "ftp://user:pw@host/v1?api_key=S": "ftp://host/v1",
        }
        for raw, safe in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(model_runtime.agent_model_label(raw), safe)
                selection = model_runtime.AgentSelection(harness="dsh", model=raw)
                record = selection.to_record()
                self.assertEqual(record["model"], safe)
                self.assertEqual(record["id"], safe)
                self.assertNotIn("pw", json.dumps(record))
                self.assertNotIn("pw", repr(selection))
                info = model_runtime.effective_agent_model(
                    ["dsh", "--harness", "dsh", "--model", raw]
                )
                self.assertEqual(info["name"], safe)
                self.assertEqual(info["id"], safe)
                self.assertNotIn("pw", json.dumps(info))
                patch = model_runtime.dsh_patch_text(selection)
                self.assertNotIn("pw", patch)
                self.assertNotIn("api_key=S", patch)

    def test_unparsable_non_http_urls_are_redacted_textually(self):
        for raw in (
            "//user:pw@[bad/v1?api_key=S",
            "ftp://user:pw@[bad/v1?api_key=S",
        ):
            with self.subTest(raw=raw):
                selection = model_runtime.AgentSelection(harness="dsh", model=raw)
                self.assertNotIn("pw", repr(selection))
                self.assertNotIn("pw", json.dumps(selection.to_record()))
                self.assertNotIn("pw", model_runtime.dsh_patch_text(selection))

    def test_a_non_http_url_shaped_model_name_is_redacted_in_the_patch(self):
        selection = model_runtime.AgentSelection(
            harness="dsh",
            model="https://host/v1",
            model_name="ftp://user:pw@host/model?api_key=S",
        )
        self.assertNotIn("pw", repr(selection))
        record = selection.to_record()
        self.assertEqual(record["model_name"], "ftp://host/model")
        self.assertNotIn("pw", json.dumps(record))
        patch = model_runtime.dsh_patch_text(selection)
        self.assertNotIn("pw", patch)
        self.assertNotIn("api_key=S", patch)
        self.assertIn('id: "ftp://host/model"', patch)


class CatalogCredentialRegressionTests(unittest.TestCase):
    def setUp(self):
        self._path = model_runtime._MODEL_CATALOG_PATH

    def tearDown(self):
        model_runtime._MODEL_CATALOG_PATH = self._path
        model_runtime.reload_model_catalog()

    def test_a_catalog_base_url_credential_is_not_exported(self):
        catalog = json.loads(self._path.read_text(encoding="utf-8"))
        for entry in catalog["models"]:
            if entry["id"] == "qwen3.8-27b":
                entry["base_url"] = "https://user:pw@qwen.example/v1"
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        model_runtime._MODEL_CATALOG_PATH = path
        model_runtime.reload_model_catalog()

        self.assertEqual(model_runtime.QWEN_DEFAULT_BASE_URL, "https://qwen.example/v1")
        self.assertEqual(
            model_runtime.AGENT_MODELS["qwen38-27b"]["base_url"],
            "https://qwen.example/v1",
        )
        self.assertEqual(
            model_runtime.API_PROVIDER_DEFAULTS["qwen"]["base_url"],
            "https://qwen.example/v1",
        )
        # The live route still receives the credential through the key channel.
        with mock.patch.dict(os.environ, {}, clear=True):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            env = model_runtime.selection_env(selection)
        self.assertEqual(env["PAPER_QWEN_API_KEY"], "pw")

    def test_a_non_http_catalog_base_url_imports_credential_free(self):
        package_dir = Path(model_runtime.__file__).resolve().parent
        cases = {
            "ftp://user:pw@qwen.example/v1": "ftp://qwen.example/v1",
            "//user:pw@qwen.example/v1": "//qwen.example/v1",
        }
        for raw, expected in cases.items():
            with self.subTest(base_url=raw):
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / "agent_common"
                    shutil.copytree(
                        package_dir,
                        target,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    )
                    catalog_path = target / "model_catalog.json"
                    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                    for entry in catalog["models"]:
                        if entry["id"] == "qwen3.8-27b":
                            entry["base_url"] = raw
                    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
                    env = dict(os.environ)
                    env["PYTHONPATH"] = directory
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            "import agent_common.model_runtime as m; "
                            "print(m.QWEN_DEFAULT_BASE_URL)",
                        ],
                        cwd=directory,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                self.assertEqual(
                    completed.returncode, 0, completed.stderr
                )
                self.assertEqual(completed.stdout.strip(), expected)


class PrepProxyRegressionTests(unittest.TestCase):
    def test_a_truncated_success_body_keeps_the_declared_length(self):
        class TruncatedSuccess(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Length", "100")
                self.end_headers()
                self.wfile.write(b"x")
                self.wfile.flush()
                self.close_connection = True

        upstream = _start_server(TruncatedSuccess)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            host, port = base.split("//", 1)[1].split("/", 1)[0].rsplit(":", 1)
            connection = http.client.HTTPConnection(host, int(port), timeout=5)
            connection.request("POST", "/v1/chat/completions", body=b"{}")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Length"), "100")
            with self.assertRaises(http.client.IncompleteRead):
                response.read()
            connection.close()
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_a_non_json_prep_object_is_rejected_before_binding(self):
        base, error = model_runtime.start_prep_proxy(
            "https://host/v1", {"bad": {1, 2}}
        )
        self.assertEqual(base, "")
        self.assertIn("not JSON-serialisable", error)

    def test_a_credential_bearing_target_is_rejected_before_binding(self):
        base, error = model_runtime.start_prep_proxy(
            "https://user:pw@host/v1", {}
        )
        self.assertEqual(base, "")
        self.assertIn("must not carry a credential", error)

    def test_a_non_ascii_path_gets_a_400(self):
        upstream = _start_server(_QuietHandler)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            port = int(base.rsplit(":", 1)[1].split("/", 1)[0])
            response = _raw_exchange(port, "GET /caf\xe9 HTTP/1.0")
            self.assertIn(b"400 Bad Request", response)
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)

    def test_an_absolute_form_path_gets_a_400(self):
        upstream = _start_server(_QuietHandler)
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:%d/v1" % upstream.server_address[1], {}
        )
        self.assertEqual(error, "")
        try:
            port = int(base.rsplit(":", 1)[1].split("/", 1)[0])
            response = _raw_exchange(port, "GET http://other/v1/chat HTTP/1.0")
            self.assertIn(b"400 Bad Request", response)
        finally:
            model_runtime.stop_prep_proxy(base)
            _stop_server(upstream)


class _QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class RetryProxyRegressionTests(unittest.TestCase):
    def test_the_bad_scheme_error_does_not_echo_the_credential(self):
        with self.assertRaises(ValueError) as caught:
            llm_retry.RetryProxy("ftp://user:secret@host/v1")
        self.assertNotIn("secret", str(caught.exception))

    def test_malformed_targets_are_rejected(self):
        proxy = llm_retry.RetryProxy("https://host/v1")
        for target in ("http://other/v1/chat", "chat", "//other/v1"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    proxy.target_path(target)

    def test_absolute_form_and_non_ascii_targets_get_a_400(self):
        proxy = llm_retry.RetryProxy("https://upstream.example/v1")
        base = proxy.start()
        try:
            port = int(base.rsplit(":", 1)[1].split("/", 1)[0])
            for target in ("GET http://other/v1/chat HTTP/1.0", "GET /caf\xe9 HTTP/1.0"):
                with self.subTest(target=target):
                    response = _raw_exchange(port, target)
                    self.assertIn(b"400 Bad Request", response)
        finally:
            llm_retry.stop_retry_proxy(proxy)


class DshEventRegressionTests(unittest.TestCase):
    def test_a_failing_reason_wins_over_a_stale_success_marker(self):
        cases = (
            {"kind": "completed", "reason": {"kind": "error"}},
            {"kind": "completed", "reason": {"error": {"message": "boom"}}},
            {"kind": "completed", "reason": {"kind": "completed", "error": "boom"}},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                event = {
                    "type": "turn_end",
                    "status": "ok",
                    "reason": {"kind": "completed"},
                }
                event.update(extra)
                self.assertIs(dsh_events.turn_end_failed(event), True)

    def test_a_clean_completed_turn_is_still_success(self):
        event = {
            "type": "turn_end",
            "status": "ok",
            "kind": "completed",
            "reason": {"kind": "completed"},
        }
        self.assertIs(dsh_events.turn_end_failed(event), False)


class ChatRegressionTests(unittest.TestCase):
    def test_non_string_text_parts_are_not_stringified(self):
        for value in ({"a": 1}, [1], 3, True):
            with self.subTest(value=value):
                self.assertEqual(
                    chat.message_text([{"type": "text", "text": value}]), ""
                )
        self.assertEqual(
            chat.message_text(
                [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
            ),
            "ab",
        )


class ModelPatchRegressionTests(unittest.TestCase):
    def test_a_config_owned_selection_emits_no_patch(self):
        self.assertEqual(
            model_runtime.dsh_patch_text(
                model_runtime.AgentSelection(
                    harness="gpt-codex", model=None, legacy="codex"
                )
            ),
            "",
        )
        self.assertEqual(
            model_runtime.dsh_patch_text(
                model_runtime.AgentSelection(harness="gpt-codex", model=None)
            ),
            "",
        )

    def test_default_plus_raw_qwen_gets_the_family_builder(self):
        patch = model_runtime.dsh_patch_text(
            model_runtime.AgentSelection(harness="dsh", model="default"),
            raw_model="qwen3.8-27b",
        )
        self.assertIn("- id: llm-pi-ai", patch)
        self.assertIn("baseURL:", patch)
        self.assertIn("provider: qwen", patch)
        self.assertIn("reasoningEffort: \"medium\"", patch)

    def test_default_plus_raw_dots_and_deepseek_use_their_builders(self):
        dots = model_runtime.dsh_patch_text(
            model_runtime.AgentSelection(harness="dsh", model="default"),
            raw_model="dots3-note-preview",
        )
        self.assertIn("- id: llm-pi-ai", dots)
        self.assertIn("provider: dots", dots)
        deepseek = model_runtime.dsh_patch_text(
            model_runtime.AgentSelection(harness="dsh", model="default"),
            raw_model="deepseek-flash",
        )
        self.assertIn('provider: "deepseek-official"', deepseek)
        self.assertIn('reasoningEffort: "high"', deepseek)

    def test_model_none_with_a_raw_gpt_model_uses_the_gpt_provider(self):
        patch = model_runtime.dsh_patch_text(
            model_runtime.AgentSelection(harness="gpt-codex", model=None),
            raw_model="gpt-6.1-sol",
        )
        self.assertIn('provider: "openai"', patch)
        self.assertIn('model: "gpt-6.1-sol"', patch)

    def test_effort_override_reaches_gpt_and_claude(self):
        for model, harness in (
            ("gpt-sol", "gpt-codex"),
            ("claude-opus", "claude-code"),
        ):
            with self.subTest(model=model):
                patch = model_runtime.dsh_patch_text(
                    model_runtime.resolve_agent_selection(
                        harness=harness, model=model
                    ),
                    effort_override="high",
                )
                self.assertIn('reasoningEffort: "high"', patch)

    def test_codex_honours_a_raw_model_for_a_config_owned_selection(self):
        for selection in (
            model_runtime.AgentSelection(harness="gpt-codex", model="default"),
            model_runtime.AgentSelection(harness="gpt-codex", model=None),
        ):
            with self.subTest(model=selection.model):
                args = model_runtime.codex_provider_args(
                    selection, raw_model="gpt-6.1-terra"
                )
                self.assertIn('model="gpt-6.1-terra"', args)

    def test_codex_emits_the_gpt_model_override(self):
        args = model_runtime.codex_provider_args(
            model_runtime.resolve_agent_selection(
                harness="gpt-codex", model="gpt-sol"
            ),
            raw_model="gpt-6.1-terra",
        )
        self.assertIn('model="gpt-6.1-terra"', args)

    def test_a_chat_completions_base_url_is_normalized_on_every_route(self):
        with mock.patch.dict(
            os.environ,
            {"PAPER_QWEN_BASE_URL": "https://q.example/v1/chat/completions"},
            clear=True,
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            patch = model_runtime.dsh_patch_text(selection)
            codex = model_runtime.codex_provider_args(selection)
            claude = model_runtime.claude_environment(selection)
        self.assertIn('baseURL: "https://q.example/v1"', patch)
        self.assertIn(
            'model_providers.qwen.base_url="https://q.example/v1"', codex
        )
        self.assertEqual(claude["ANTHROPIC_BASE_URL"], "https://q.example")


class _CheckerCase(unittest.TestCase):
    def catalog_document(self):
        return json.loads(
            check_model_catalog.CANONICAL.read_text(encoding="utf-8")
        )

    def run_catalog(self, document, *, copies=()):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        if isinstance(document, str):
            path.write_text(document, encoding="utf-8")
        else:
            path.write_text(json.dumps(document), encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_model_catalog, "CANONICAL", path
        ), mock.patch.object(check_model_catalog, "COPIES", copies):
            with contextlib.redirect_stdout(output):
                code = check_model_catalog.main()
        return code, output.getvalue()

    def run_policy(self, document, *, module_text=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        canonical = root / "agent_common"
        canonical.mkdir()
        (canonical / "llm_policy.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        (canonical / "llm_retry.py").write_text(
            module_text
            if module_text is not None
            else check_llm_policy.CANONICAL_MODULE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
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
                try:
                    code = check_llm_policy.main()
                except Exception as exc:  # pragma: no cover - the bug
                    self.fail(
                        "checker crashed: %s: %s" % (type(exc).__name__, exc)
                    )
        return code, output.getvalue()

    def policy_document(self):
        return json.loads(
            check_llm_policy.CANONICAL_JSON.read_text(encoding="utf-8")
        )


class CheckerRegressionTests(_CheckerCase):
    def test_an_unhashable_reasoning_level_is_reported(self):
        document = self.catalog_document()
        for entry in document["models"]:
            if entry["id"] == "deepseek-flash":
                entry["default_reasoning_level"] = ["x"]
        code, output = self.run_catalog(document)
        self.assertEqual(code, 1)
        self.assertIn("bad default_reasoning_level", output)
        self.assertNotIn("Traceback", output)

    def test_a_deeply_nested_catalog_is_reported(self):
        code, output = self.run_catalog("[" * 100000 + "]" * 100000)
        self.assertEqual(code, 1)
        self.assertIn("FAIL cannot read canonical catalog", output)

    def test_a_deeply_nested_policy_is_reported(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "llm_policy.json"
        path.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_llm_policy, "CANONICAL_JSON", path
        ), mock.patch.object(check_llm_policy, "COPIES", ()):
            with contextlib.redirect_stdout(output):
                code = check_llm_policy.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL cannot read canonical policy", output.getvalue())

    def test_a_deeply_nested_catalog_does_not_break_the_cross_check_import(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
        catalogue, error = check_cross_repo_consistency._read_catalogue(path)
        self.assertEqual(catalogue, {})
        self.assertIn("cannot read catalog", error)

    def test_an_unreadable_catalog_digest_is_a_fail_not_a_crash(self):
        document = self.catalog_document()
        code, _output = self.run_catalog(document)
        self.assertEqual(code, 0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_model_catalog, "CANONICAL", path
        ), mock.patch.object(check_model_catalog, "COPIES", ()), mock.patch.object(
            check_model_catalog, "digest", side_effect=PermissionError("denied")
        ):
            with contextlib.redirect_stdout(output):
                code = check_model_catalog.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL cannot read canonical catalog", output.getvalue())

    def test_an_unreadable_shared_package_file_is_a_fail_not_a_crash(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        canonical = root / "canonical"
        canonical.mkdir()
        (canonical / "chat.py").write_text("chat", encoding="utf-8")
        copy = root / "copy" / "agent_common"
        copy.mkdir(parents=True)
        (copy / "chat.py").write_text("chat", encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_shared_package, "CANONICAL_DIR", canonical
        ), mock.patch.object(
            check_shared_package, "COPIES", (copy,)
        ), mock.patch.object(
            check_shared_package, "digest", side_effect=PermissionError("denied")
        ):
            with contextlib.redirect_stdout(output):
                code = check_shared_package.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL cannot read", output.getvalue())

    def test_an_unreadable_policy_copy_is_a_fail_not_a_crash(self):
        document = self.policy_document()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        canonical = root / "canonical"
        canonical.mkdir()
        (canonical / "llm_policy.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        (canonical / "llm_retry.py").write_text(
            check_llm_policy.CANONICAL_MODULE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        copy = root / "copy"
        copy.mkdir()
        (copy / "llm_policy.json").write_text("{}", encoding="utf-8")
        (copy / "llm_retry.py").write_text("", encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_llm_policy, "ROOT", root
        ), mock.patch.object(
            check_llm_policy, "CANONICAL_DIR", canonical
        ), mock.patch.object(
            check_llm_policy, "CANONICAL_JSON", canonical / "llm_policy.json"
        ), mock.patch.object(
            check_llm_policy, "CANONICAL_MODULE", canonical / "llm_retry.py"
        ), mock.patch.object(
            check_llm_policy, "COPIES", (copy,)
        ), mock.patch.object(
            check_llm_policy,
            "digest",
            side_effect=lambda path: (
                (_ for _ in ()).throw(PermissionError("denied"))
                if str(path).startswith(str(copy))
                else _real_digest(path)
            ),
        ):
            with contextlib.redirect_stdout(output):
                code = check_llm_policy.main()
        self.assertEqual(code, 1)
        self.assertIn("FAIL cannot read", output.getvalue())

    def test_an_overflowing_schedule_is_reported(self):
        document = self.policy_document()
        document["retry"]["multiplier"] = 1e308
        code, output = self.run_policy(document)
        self.assertEqual(code, 1)
        self.assertIn("cannot derive the retry schedule", output)

    def test_a_failing_child_check_prints_its_stderr(self):
        completed = mock.Mock(
            returncode=1, stdout="", stderr="Traceback: boom"
        )
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency.subprocess, "run", return_value=completed
        ):
            with contextlib.redirect_stdout(output):
                code = check_cross_repo_consistency.main()
        self.assertEqual(code, 1)
        self.assertIn("Traceback: boom", output.getvalue())

    def test_a_boolean_schema_version_is_rejected(self):
        document = self.catalog_document()
        document["schema_version"] = True
        code, output = self.run_catalog(document)
        self.assertEqual(code, 1)
        self.assertIn("unsupported schema_version True", output)

    def test_a_default_policy_divergence_is_reported(self):
        document = self.policy_document()
        saved = llm_retry.DEFAULT_POLICY["retry"]["initial_delay_seconds"]
        try:
            llm_retry.DEFAULT_POLICY["retry"]["initial_delay_seconds"] = 99
            code, output = self.run_policy(document)
        finally:
            llm_retry.DEFAULT_POLICY["retry"]["initial_delay_seconds"] = saved
        self.assertEqual(code, 1)
        self.assertIn("differs from llm_retry.DEFAULT_POLICY", output)

    def test_the_policy_docstring_describes_the_shared_runtime(self):
        doc = check_llm_policy.__doc__ or ""
        self.assertNotIn("carry the same constants", doc)
        self.assertIn("shared model runtime", doc)
        self.assertIn("DEFAULT_POLICY", doc)


if __name__ == "__main__":
    unittest.main()
