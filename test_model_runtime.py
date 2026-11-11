"""Tests for the shared model-runtime catalog policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
