"""Regression tests for the wave-1 verified findings (2026-11-24 run).

Each test pins one candidate from the verified-findings set: F01, F02, F06,
F09, F22, F28, F30, F31, F33, F40, F42 and F47.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_llm_policy
from agent_common import dsh_events, llm_retry, model_runtime

URL = "https://user:SECRET@host/v1?api_key=QKEY"
SAFE = "https://host/v1"


class EffectiveAgentModelRedactionTests(unittest.TestCase):
    """F01: the documented secret-free record agent must redact codex/claude argv."""

    def test_codex_and_claude_model_values_are_redacted(self):
        for argv, harness in (
            (["codex", "exec", "-m", URL, "hi"], "gpt-codex"),
            (["/usr/bin/claude", "--model", URL, "-p", "x"], "claude-code"),
        ):
            with self.subTest(argv=argv[0]):
                record = model_runtime.effective_agent_model(argv)
                self.assertEqual(record["harness"], harness)
                self.assertEqual(record["name"], SAFE)
                self.assertNotIn("SECRET", json.dumps(record))


class RouteRawModelRedactionTests(unittest.TestCase):
    """F02: Codex argv and Claude argv/environment must not emit raw_model."""

    def test_codex_args_claude_arg_and_claude_env_redact_raw_model(self):
        codex = model_runtime.resolve_agent_selection(
            harness="gpt-codex", model="deepseek-flash"
        )
        args = model_runtime.codex_provider_args(codex, raw_model=URL)
        self.assertIn('-c', args)
        self.assertIn('model="%s"' % SAFE, args)
        self.assertNotIn("SECRET", " ".join(args))

        claude = model_runtime.resolve_agent_selection(
            harness="claude-code", model="qwen38-27b"
        )
        self.assertEqual(model_runtime.claude_model_arg(claude, raw_model=URL), SAFE)
        env = model_runtime.claude_environment(claude, raw_model=URL)
        self.assertEqual(env["ANTHROPIC_MODEL"], SAFE)
        self.assertNotIn("SECRET", json.dumps(env))


class SchemeLessUserinfoRedactionTests(unittest.TestCase):
    """R02: a bare `userinfo@host` value must be redacted at every F01/F02 sink."""

    BARE = "user:pw@host/v1"
    BARE_SAFE = "host/v1"

    def test_effective_agent_model_redacts_codex_and_claude_argv(self):
        for argv in (
            ["codex", "exec", "-m", self.BARE, "hi"],
            ["/usr/bin/claude", "--model", self.BARE, "-p", "x"],
        ):
            with self.subTest(exe=argv[0]):
                record = model_runtime.effective_agent_model(argv)
                self.assertEqual(record["name"], self.BARE_SAFE)
                self.assertNotIn("pw", json.dumps(record))

    def test_codex_args_and_claude_arg_env_redact_bare_userinfo(self):
        codex = model_runtime.resolve_agent_selection(
            harness="gpt-codex", model="deepseek-flash"
        )
        args = model_runtime.codex_provider_args(codex, raw_model=self.BARE)
        self.assertIn('model="%s"' % self.BARE_SAFE, args)
        self.assertNotIn("pw", " ".join(args))

        claude = model_runtime.resolve_agent_selection(
            harness="claude-code", model="qwen38-27b"
        )
        self.assertEqual(
            model_runtime.claude_model_arg(claude, raw_model=self.BARE),
            self.BARE_SAFE,
        )
        env = model_runtime.claude_environment(claude, raw_model=self.BARE)
        self.assertEqual(env["ANTHROPIC_MODEL"], self.BARE_SAFE)
        self.assertNotIn("pw", json.dumps(env))

    def test_plain_model_ids_are_unchanged(self):
        for value in ("deepseek-flash", "qwen38-27b", "deepseek-flash[1m]"):
            with self.subTest(value=value):
                self.assertEqual(model_runtime._redacted_endpoint_text(value), value)


class DotsOverlayCredentialTests(unittest.TestCase):
    """F06: the public dots_dsh_overlay must normalise its own base URLs."""

    def test_direct_call_does_not_embed_a_credential(self):
        overlay = llm_retry.dots_dsh_overlay(
            None,
            model="dots3-note-prev",
            base_url=URL,
            context_window=393000,
            max_tokens=65536,
            compaction_base_url=URL.replace("host", "compaction.example"),
        )
        self.assertNotIn("SECRET", overlay)
        base_lines = [line.strip() for line in overlay.splitlines() if "baseURL" in line]
        self.assertEqual(
            base_lines,
            [
                'baseURL: "%s"' % SAFE,
                'baseURL: "https://compaction.example/v1"',
            ],
        )

    def test_a_hostless_userinfo_value_is_scrubbed_too(self):
        # R02: without a scheme the value is still a credential-bearing
        # ``userinfo@host/path``; only the authority userinfo must go.
        for value in ("user:pw@host/v1", "user:pw@host/v1?api_key=QKEY"):
            with self.subTest(value=value):
                overlay = llm_retry.dots_dsh_overlay(
                    None,
                    model="dots3-note-prev",
                    base_url=value,
                    context_window=393000,
                    max_tokens=65536,
                )
                self.assertNotIn("pw", overlay)
                self.assertNotIn("QKEY", overlay)
                base_lines = [
                    line.strip() for line in overlay.splitlines() if "baseURL" in line
                ]
                self.assertEqual(
                    base_lines, ['baseURL: "host/v1"', 'baseURL: "host/v1"']
                )


class ClaudeMissingKeyTests(unittest.TestCase):
    """F09: a required Dots/DeepSeek key must not be replaced by 'dummy'."""

    def _clean_env(self):
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("DOTS", "PAPER_", "DEEPSEEK", "QWEN"))
        }

    def test_missing_dots_and_deepseek_keys_emit_no_token(self):
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            for model in ("dots3-note-preview", "deepseek-flash"):
                with self.subTest(model=model):
                    selection = model_runtime.resolve_agent_selection(
                        harness="claude-code", model=model
                    )
                    self.assertTrue(model_runtime.selection_problem(selection))
                    env = model_runtime.claude_environment(selection)
                    self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
                    self.assertNotIn("ANTHROPIC_API_KEY", env)

    def test_the_local_qwen_route_keeps_its_dummy_key(self):
        with mock.patch.dict(os.environ, self._clean_env(), clear=True):
            selection = model_runtime.resolve_agent_selection(
                harness="claude-code", model="qwen38-27b"
            )
            env = model_runtime.claude_environment(selection)
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "dummy")


class EndpointBasePortTests(unittest.TestCase):
    """F22: the invalid-port branch must be reachable and specific."""

    def test_bad_ports_get_the_invalid_port_message(self):
        for url in ("https://host:abc/v1", "https://host:99999/v1"):
            with self.subTest(url=url):
                with self.assertRaises(model_runtime.ConfigError) as caught:
                    model_runtime.endpoint_base(url)
                self.assertIn("invalid port", str(caught.exception))
                self.assertNotIn("not an http(s) URL with a host", str(caught.exception))


class EndpointEffortOverrideTests(unittest.TestCase):
    """F28: an explicit override must reach the custom-endpoint patch."""

    def _patch(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1"
        )
        return model_runtime.dsh_patch_text(selection)

    def test_an_explicit_override_is_applied_to_the_endpoint_patch(self):
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "high"}, clear=True
        ):
            patch = self._patch()
        self.assertIn("reasoningEfforts:", patch)
        self.assertIn("              high: high", patch)
        self.assertIn('    reasoningEffort: "high"', patch)
        self.assertNotIn("reasoningEfforts: false", patch)

    def test_without_an_override_the_endpoint_still_disables_effort(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            patch = self._patch()
        self.assertIn("reasoningEfforts: false", patch)
        self.assertNotIn("    reasoningEffort: ", patch)

    def test_a_catalog_shaped_model_name_does_not_borrow_another_family_mapping(self):
        # The custom endpoint passes the level through; a model name that also
        # exists in the catalog must not turn "high" into Qwen's "xhigh".
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "high"}, clear=True
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1", model_name="qwen38-27b"
            )
            patch = model_runtime.dsh_patch_text(selection)
        self.assertIn('    reasoningEffort: "high"', patch)
        self.assertNotIn('    reasoningEffort: "xhigh"', patch)

    def test_an_explicit_override_argument_is_applied_to_the_endpoint_patch(self):
        # R01: the argument is the public channel; the environment variable is
        # the operator channel.  Both must reach the custom endpoint.
        with mock.patch.dict(os.environ, {}, clear=True):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1"
            )
            patch = model_runtime.dsh_patch_text(selection, effort_override="low")
        self.assertIn("reasoningEfforts:", patch)
        self.assertIn('    reasoningEffort: "low"', patch)
        self.assertNotIn("reasoningEfforts: false", patch)

    def test_the_explicit_argument_wins_over_the_environment(self):
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "high"}, clear=True
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1"
            )
            patch = model_runtime.dsh_patch_text(selection, effort_override="low")
        self.assertIn('    reasoningEffort: "low"', patch)
        self.assertNotIn('    reasoningEffort: "high"', patch)


class CatalogValidationTests(unittest.TestCase):
    """F30/F31/F33: malformed catalog values fail atomically as ConfigError."""

    def setUp(self):
        self._path = model_runtime._MODEL_CATALOG_PATH
        self._catalog = json.loads(Path(self._path).read_text(encoding="utf-8"))

    def tearDown(self):
        model_runtime._MODEL_CATALOG_PATH = self._path
        model_runtime.reload_model_catalog()

    def _reload_with(self, **changes):
        catalog = json.loads(json.dumps(self._catalog))
        for entry in catalog["models"]:
            if entry["id"] == "qwen3.8-27b":
                entry.update(changes)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        model_runtime._MODEL_CATALOG_PATH = path
        return model_runtime.reload_model_catalog()

    def test_an_unknown_default_reasoning_level_fails_at_load(self):
        with self.assertRaises(model_runtime.ConfigError) as caught:
            self._reload_with(default_reasoning_level="banana")
        self.assertIn("default_reasoning_level", str(caught.exception))
        # Atomic: the previous catalog value is still in place.
        self.assertEqual(model_runtime.QWEN_DEFAULT_EFFORT, "medium")

    def test_an_infinite_capacity_is_a_config_error_not_an_overflow(self):
        with self.assertRaises(model_runtime.ConfigError) as caught:
            self._reload_with(context_window=1e400)
        self.assertNotIsInstance(caught.exception, OverflowError)

    def test_a_hostless_base_url_fails_at_load(self):
        with self.assertRaises(model_runtime.ConfigError) as caught:
            self._reload_with(base_url="not a url")
        self.assertIn("base_url", str(caught.exception))
        self.assertEqual(
            model_runtime.QWEN_DEFAULT_BASE_URL, "http://127.0.0.1:8000/v1"
        )

    def test_a_non_http_base_url_does_not_reach_claude_code(self):
        # R03: a legacy non-http/scheme-relative catalog value stays importable
        # (credential-free) but the Claude route must reject it instead of
        # exporting an unusable ANTHROPIC_BASE_URL.
        for raw in ("ftp://user:pw@qwen.example/v1", "//user:pw@qwen.example/v1"):
            with self.subTest(base_url=raw):
                self._reload_with(base_url=raw)
                self.assertNotIn("pw", model_runtime.QWEN_DEFAULT_BASE_URL)
                selection = model_runtime.resolve_agent_selection(
                    harness="claude-code", model="qwen38-27b"
                )
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.claude_environment(selection)


class DshEventRecursionTests(unittest.TestCase):
    """F40: a deeply nested line is a non-event, not a RecursionError."""

    def test_a_deeply_nested_line_returns_none(self):
        line = '{"a":' * 100000 + "]" * 100000
        self.assertIsNone(dsh_events.parse_event(line))


class PrepNonFiniteTests(unittest.TestCase):
    """F42: NaN/Infinity must not be accepted as a JSON prep object."""

    def test_non_finite_json_is_rejected(self):
        for value in ('{"temperature": NaN}', '{"temperature": 1e400}'):
            with self.subTest(value=value):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.parse_prep_json(value)

    def test_a_directly_supplied_nan_does_not_start_a_proxy(self):
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:9/v1", {"temperature": float("nan")}
        )
        self.assertEqual(base, "")
        self.assertIn("not JSON-serialisable", error)


class PolicyCheckerNullTests(unittest.TestCase):
    """F47: a runtime-accepted null numeric field must not crash the checker."""

    def _run(self, field):
        canonical = json.loads(
            (Path(check_llm_policy.CANONICAL_JSON)).read_text(encoding="utf-8")
        )
        canonical["retry"][field] = None
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "llm_policy.json"
        path.write_text(json.dumps(canonical), encoding="utf-8")
        original_json = check_llm_policy.CANONICAL_JSON
        original_copies = check_llm_policy.COPIES
        try:
            check_llm_policy.CANONICAL_JSON = path
            check_llm_policy.COPIES = ()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = check_llm_policy.main()
        finally:
            check_llm_policy.CANONICAL_JSON = original_json
            check_llm_policy.COPIES = original_copies
        return code, output.getvalue()

    def test_null_numeric_fields_report_fail_instead_of_crashing(self):
        for field in ("max_retries", "initial_delay_seconds"):
            with self.subTest(field=field):
                code, output = self._run(field)
                self.assertEqual(code, 1)
                self.assertIn("FAIL", output)


if __name__ == "__main__":
    unittest.main()
