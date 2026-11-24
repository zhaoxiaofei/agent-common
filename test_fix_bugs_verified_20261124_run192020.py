"""Regression tests for the wave-1 verified findings of run 20261124-192020.

One test class per candidate (F009 .. F098).  Every test was written against the
unmodified tree and failed there; each pins the smallest observable contract
the fix must restore.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import check_llm_policy
from agent_common import llm_retry, model_runtime

REPO_ROOT = Path(__file__).resolve().parent


class TempCatalogMixin:
    """Point the runtime at a throwaway catalog and restore the real one."""

    def setUp(self):
        super().setUp()
        self._original_path = model_runtime._MODEL_CATALOG_PATH
        self._original_models = {
            name: dict(entry) for name, entry in model_runtime.AGENT_MODELS.items()
        }
        self.addCleanup(self._restore)

    def _restore(self):
        model_runtime._MODEL_CATALOG_PATH = self._original_path
        model_runtime.reload_model_catalog()

    def _mutated_catalog(self, model_id, **changes):
        document = json.loads(self._original_path.read_text(encoding="utf-8"))
        for entry in document["models"]:
            if entry["id"] == model_id:
                entry.update(changes)
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        with handle:
            json.dump(document, handle)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        model_runtime._MODEL_CATALOG_PATH = Path(handle.name)
        return Path(handle.name)


class SelectionEnvCallerNamedKeyTests(unittest.TestCase):
    """F009: a caller-named endpoint key variable must be read, not overwritten."""

    def test_the_caller_named_variable_is_preserved(self):
        with mock.patch.dict(os.environ, {"OTHER_KEY": "K"}, clear=True):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1"
            )
            env = model_runtime.selection_env(
                selection, endpoint_api_key_env="OTHER_KEY"
            )
        self.assertEqual(env["OTHER_KEY"], "K")


class ToRecordApiKeySetTests(unittest.TestCase):
    """F010: a live URL/custom-env credential must be reported as api_key_set."""

    def test_url_credential_sets_the_flag(self):
        original = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                selection = model_runtime.resolve_agent_selection(
                    harness="dsh", model="https://user:pw@host/v1"
                )
                self.assertTrue(selection.to_record()["api_key_set"])
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = original

    def test_a_custom_key_environment_sets_the_flag(self):
        with mock.patch.dict(os.environ, {"OTHER_KEY": "K"}, clear=True):
            selection = model_runtime.AgentSelection(
                harness="dsh", model="https://host/v1"
            )
            self.assertTrue(
                selection.to_record(endpoint_api_key_env="OTHER_KEY")["api_key_set"]
            )

    def test_a_sentinel_selection_with_a_raw_endpoint_sets_the_flag(self):
        # R02: the live DSH route builds the endpoint block from exactly this
        # sentinel + raw URL shape, so the record must follow the effective
        # endpoint instead of looking only at ``self.model``.
        original = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            with mock.patch.dict(os.environ, {}, clear=True):
                sentinel = model_runtime.AgentSelection(harness="dsh", model="default")
                record = sentinel.to_record(raw_model="https://user:pw@host/v1")
                self.assertTrue(record["api_key_set"])
                self.assertEqual(record["id"], "https://host/v1")
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = original

    def test_a_sentinel_selection_with_a_caller_named_key_sets_the_flag(self):
        with mock.patch.dict(os.environ, {"OTHER_KEY": "K"}, clear=True):
            sentinel = model_runtime.AgentSelection(harness="dsh", model="default")
            self.assertTrue(
                sentinel.to_record(
                    raw_model="https://host/v1", endpoint_api_key_env="OTHER_KEY"
                )["api_key_set"]
            )


class EndpointProbeCredentialTests(unittest.TestCase):
    """F011: the /models preflight must use the URL credential the request uses."""

    def test_a_url_credential_passes_the_preflight(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer pw":
                    self.send_response(401)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({"data": [{"id": "m1"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        original = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            url = "http://user:pw@127.0.0.1:%d/v1" % server.server_address[1]
            with mock.patch.dict(os.environ, {}, clear=True):
                selection = model_runtime.resolve_agent_selection(
                    harness="dsh", model=url
                )
                self.assertEqual(model_runtime.selection_problem(selection), "")
                self.assertEqual(
                    model_runtime.resolve_endpoint_model(selection)[0], "m1"
                )
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = original
            server.shutdown()
            server.server_close()


class ReprPrepRedactionTests(unittest.TestCase):
    """F012: prep values under non-credential keys must still be redacted."""

    def test_url_values_and_header_pairs_are_redacted(self):
        selection = model_runtime.AgentSelection(
            harness="dsh",
            model="https://host/v1",
            prep_json={
                "endpoint": "https://user:pw@host/v1?api_key=SECRET",
                "headers": [
                    {"name": "Authorization", "value": "Bearer SECRET"}
                ],
            },
        )
        text = repr(selection)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("pw", text)
        self.assertIn("[REDACTED]", text)


class ProviderEnvModelRedactionTests(unittest.TestCase):
    """F013: PAPER_QWEN_MODEL/PAPER_DOTS_MODEL must not leak into artifacts."""

    def test_qwen_and_dots_model_values_are_redacted(self):
        secret = "https://user:pw@host/v1?api_key=SECRET"
        with mock.patch.dict(
            os.environ,
            {"PAPER_QWEN_MODEL": secret, "PAPER_DOTS_MODEL": secret},
            clear=True,
        ):
            qwen = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            dots = model_runtime.resolve_agent_selection(
                harness="dsh", model="dots3-note-preview"
            )
            patch = model_runtime.dsh_patch_text(qwen) + model_runtime.dsh_patch_text(
                dots
            )
            claude = model_runtime.claude_environment(
                model_runtime.resolve_agent_selection(
                    harness="claude-code", model="qwen38-27b"
                )
            )
        for text in (patch, json.dumps(claude)):
            self.assertNotIn("SECRET", text)
            self.assertNotIn("pw", text)


class RawEndpointPatchTests(unittest.TestCase):
    """F016: a sentinel plus a raw endpoint URL must build the endpoint route."""

    def test_the_endpoint_builder_is_used(self):
        selection = model_runtime.AgentSelection(harness="dsh", model="default")
        patch = model_runtime.dsh_patch_text(selection, raw_model="https://host/v1")
        self.assertIn("provider: endpoint", patch)
        self.assertIn('baseURL: "https://host/v1"', patch)
        self.assertNotIn('model: "https://host/v1"', patch)


class ClaudeRawModelTests(unittest.TestCase):
    """F020: the Claude route must consult raw_model for a config-owned selection."""

    def test_raw_model_reaches_the_claude_route(self):
        selection = model_runtime.AgentSelection(
            harness="claude-code", model="default"
        )
        self.assertEqual(
            model_runtime.claude_model_arg(selection, raw_model="claude-opus"),
            "claude-opus",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            env = model_runtime.claude_environment(
                selection, raw_model="qwen38-27b"
            )
        self.assertEqual(env["ANTHROPIC_MODEL"], "qwen38-27b")


class CodexUnsupportedRawModelTests(unittest.TestCase):
    """F021: unsupported raw models on the Codex route must raise, not vanish."""

    def test_endpoint_and_claude_raw_models_are_rejected(self):
        selection = model_runtime.AgentSelection(harness="dsh", model="default")
        for raw in ("https://host/v1", "claude-opus"):
            with self.subTest(raw=raw):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.codex_provider_args(selection, raw_model=raw)


class PrepJsonDictValidationTests(unittest.TestCase):
    """F039: the dict fast path must reject non-finite numbers like the JSON path."""

    def test_non_finite_values_are_rejected(self):
        for value in (
            {"temperature": float("nan")},
            {"temperature": float("inf")},
            {"nested": [{"x": float("-inf")}]},
        ):
            with self.subTest(value=value):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.parse_prep_json(value)

    def test_a_valid_dict_is_returned(self):
        self.assertEqual(
            model_runtime.parse_prep_json({"temperature": 0}), {"temperature": 0}
        )


class RetryProxyPortTests(unittest.TestCase):
    """F048: RetryProxy must reject port 0 like endpoint_base does."""

    def test_port_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            llm_retry.RetryProxy("https://host:0/v1")

    def test_a_real_port_is_accepted(self):
        self.assertEqual(llm_retry.RetryProxy("https://host:8443/v1")._port, 8443)


class CatalogLevelValidationTests(TempCatalogMixin, unittest.TestCase):
    """F049: a falsy malformed default_reasoning_level must fail the reload."""

    def test_a_falsy_malformed_level_is_rejected(self):
        self._mutated_catalog("deepseek-flash", default_reasoning_level=[])
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.reload_model_catalog()

    def test_a_missing_level_remains_absent_or_unchanged(self):
        document = json.loads(self._original_path.read_text(encoding="utf-8"))
        for entry in document["models"]:
            if entry["id"] == "deepseek-flash":
                entry.pop("default_reasoning_level", None)
        path = Path(tempfile.mkdtemp()) / "c.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        model_runtime._MODEL_CATALOG_PATH = path
        model_runtime.reload_model_catalog()
        self.assertIsNone(
            model_runtime.AGENT_MODELS["deepseek-flash"].get(
                "default_reasoning_level"
            )
        )


class CatalogProviderValidationTests(TempCatalogMixin, unittest.TestCase):
    """F051: malformed provider/api_key_env values must fail the reload."""

    def test_a_non_string_provider_is_rejected(self):
        self._mutated_catalog("qwen3.8-27b", provider=["x"])
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.reload_model_catalog()

    def test_a_non_string_key_env_is_rejected(self):
        self._mutated_catalog("qwen3.8-27b", api_key_env=123)
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.reload_model_catalog()

    def test_a_falsy_malformed_provider_is_rejected(self):
        # R01: a present-but-falsy value (``[]``, ``0``) is malformed, not
        # absent; the reload must reject it and keep the previous entry.
        for value in ([], 0):
            with self.subTest(value=value):
                before = dict(model_runtime.AGENT_MODELS["qwen38-27b"])
                self._mutated_catalog("qwen3.8-27b", provider=value)
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.reload_model_catalog()
                self.assertEqual(
                    model_runtime.AGENT_MODELS["qwen38-27b"], before
                )

    def test_a_falsy_malformed_key_env_is_rejected(self):
        for value in (0, ""):
            with self.subTest(value=value):
                before = dict(model_runtime.AGENT_MODELS["qwen38-27b"])
                self._mutated_catalog("qwen3.8-27b", api_key_env=value)
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.reload_model_catalog()
                self.assertEqual(
                    model_runtime.AGENT_MODELS["qwen38-27b"], before
                )


class RetryAfterValidationTests(unittest.TestCase):
    """F055: invalid Retry-After numeric hints must return None, not 0.0."""

    def test_negative_and_non_finite_values_return_none(self):
        for raw in ("-5", "nan", "inf", "-inf"):
            with self.subTest(raw=raw):
                self.assertIsNone(
                    llm_retry.retry_after_seconds({"Retry-After": raw})
                )

    def test_a_positive_hint_is_returned(self):
        self.assertEqual(
            llm_retry.retry_after_seconds({"Retry-After": "7"}), 7.0
        )
        self.assertIsNone(
            llm_retry.retry_after_seconds({"Retry-After": "not-a-date"})
        )


class RetryPolicyMillisecondTests(unittest.TestCase):
    """F057: a positive sub-millisecond delay must not render as zero."""

    def test_a_sub_millisecond_delay_rounds_up(self):
        yaml = llm_retry.retry_policy_yaml(
            {"retry": {"initial_delay_seconds": 0.0001, "max_delay_seconds": 3600}}
        )
        self.assertIn("initialDelayMs: 1", yaml)

    def test_the_maximum_is_not_below_the_initial_delay(self):
        yaml = llm_retry.retry_policy_yaml(
            {"retry": {"initial_delay_seconds": 0.0001, "max_delay_seconds": 0.0001}}
        )
        values = {
            line.strip().split(": ")[0]: int(line.strip().split(": ")[1])
            for line in yaml.splitlines()
            if "DelayMs:" in line
        }
        self.assertGreaterEqual(values["maxDelayMs"], values["initialDelayMs"])
        self.assertGreaterEqual(values["initialDelayMs"], 1)


class PolicyCheckerDotsShapeTests(unittest.TestCase):
    """F079: a non-mapping dots section must be reported as a shape error."""

    def test_the_shape_error_replaces_the_duplicate_key_report(self):
        original = check_llm_policy.CANONICAL_JSON
        document = json.loads(original.read_text(encoding="utf-8"))
        document["dots"] = []
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        with handle:
            json.dump(document, handle)
        check_llm_policy.CANONICAL_JSON = Path(handle.name)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                code = check_llm_policy.main()
        finally:
            check_llm_policy.CANONICAL_JSON = original
            Path(handle.name).unlink(missing_ok=True)
        self.assertEqual(code, 1)
        output = out.getvalue()
        self.assertNotIn(
            "regular and compaction Dots keys are the same variable", output
        )
        self.assertIn("dots must be a mapping", output)


class ScriptNoCheckoutTests(unittest.TestCase):
    """F080: both sync wrappers must fail when no checkout was processed."""

    def test_both_scripts_exit_nonzero_without_checkouts(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "wrapper_demo" / "agent-common"
            wrapper.mkdir(parents=True)
            for name in ("sync_llm_policy.sh", "update_model_catalog.sh"):
                target = wrapper / name
                target.write_bytes((REPO_ROOT / name).read_bytes())
                result = subprocess.run(
                    ["sh", str(target)],
                    capture_output=True,
                    text=True,
                    cwd=str(wrapper),
                )
                with self.subTest(script=name):
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertNotIn("OK: agent-common submodule checked out", result.stdout)


class DotsCompactionCopyTests(unittest.TestCase):
    """F092: the documented deep copy must not share nested values."""

    def test_mutating_the_result_does_not_change_the_input(self):
        policy = {"dots": {"extra": {"x": 1}}}
        result = llm_retry.dots_compaction(policy)
        result["extra"]["x"] = 2
        self.assertEqual(policy["dots"]["extra"]["x"], 1)


class DotsWhitespaceEffortTests(unittest.TestCase):
    """F093: a whitespace-only effort is absent, not thinking off."""

    def test_whitespace_uses_the_default(self):
        overlay = llm_retry.dots_dsh_overlay(
            None,
            model="m",
            base_url="https://h/v1",
            context_window=1000,
            max_tokens=100,
            reasoning_effort=" ",
        )
        # The compaction provider always keeps effort off; only the regular
        # provider's declaration may disappear when the override is off.
        self.assertEqual(overlay.count("reasoningEfforts: false"), 1)
        self.assertIn('reasoningEffort: "high"', overlay)


class PolicyCheckerAdapterReadTests(unittest.TestCase):
    """F096: an unreadable adapter is a FAIL report, never a traceback."""

    def test_a_non_utf8_adapter_is_reported(self):
        original = check_llm_policy.CANONICAL_JSON
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refine = root / "paper-refine"
            refine.mkdir()
            (refine / "paper_dots_agent.py").write_bytes(
                b"\xff\xfeimport model_runtime as rt\n"
            )
            (refine / "paper_qwen_agent.py").write_text(
                "import model_runtime as rt\n", encoding="utf-8"
            )
            (refine / "paper_model_agent.py").write_text(
                "import model_runtime as rt\n", encoding="utf-8"
            )
            check_llm_policy.CANONICAL_JSON = root / "p.json"
            check_llm_policy.CANONICAL_JSON.write_text(
                original.read_text(encoding="utf-8"), encoding="utf-8"
            )
            out = io.StringIO()
            try:
                with mock.patch.object(check_llm_policy, "COPIES", ()), mock.patch.object(
                    check_llm_policy, "ROOT", root
                ):
                    with contextlib.redirect_stdout(out):
                        code = check_llm_policy.main()
            finally:
                check_llm_policy.CANONICAL_JSON = original
        self.assertEqual(code, 1)
        self.assertIn("cannot read adapter", out.getvalue())


class DuplicateEndpointModelTests(unittest.TestCase):
    """F098: a /models listing that repeats one id is not ambiguous."""

    def test_duplicate_ids_resolve_to_one_model(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1"
        )
        with mock.patch.object(
            model_runtime, "_endpoint_catalog", return_value=[{"id": "m"}, {"id": "m"}]
        ):
            self.assertEqual(
                model_runtime.resolve_endpoint_model(selection, api_key="k")[0], "m"
            )


if __name__ == "__main__":
    unittest.main()
