"""Regression tests for the 2026-11-27 verified fix-bugs wave.

Each test names the verified finding it pins.  They are hermetic: no network,
no real provider, and temporary directories only.
"""
from __future__ import annotations

import ast
import json
import os
import re
import tempfile
import time
import unittest
import datetime
from pathlib import Path
from unittest import mock

import check_cross_repo_consistency
import check_llm_policy
import check_model_catalog
from agent_common import llm_retry, model_runtime, recovery

REPO_ROOT = Path(__file__).resolve().parent


def _record(**overrides):
    values = {
        "run_id": "r",
        "event_id": "r-0001",
        "timestamp": "2026-11-27T00:00:00Z",
        "sequence": 1,
        "code": "SAMPLE",
        "tier": recovery.TIER_ERROR,
        "operation": "op",
        "failure": "failed",
        "recovery_action": "acted",
    }
    values.update(overrides)
    return recovery.RecoveryRecord(**values)


class CredentialPredicateTests(unittest.TestCase):
    """F01/F02/F57: one credential table for every consumer."""

    def test_plural_credential_spellings_are_recognised(self):
        for key in ("passwords", "passwds", "authorizations"):
            url = "https://h/v1?%s=SECRET" % key
            with self.subTest(key=key):
                self.assertTrue(model_runtime.endpoint_credentials(url))
                safe, secret = model_runtime.split_endpoint_credentials(url)
                self.assertEqual(secret, "SECRET")
                self.assertNotIn("SECRET", safe)
                self.assertEqual(
                    model_runtime.redact_prep_object({key: "SECRET"})[key],
                    "[REDACTED]",
                )
                self.assertEqual(
                    llm_retry._credential_query_key(key), True
                )
                # The proxy refuses a credential query rather than forwarding
                # the secret it now recognises.
                with self.assertRaises(ValueError):
                    llm_retry.RetryProxy(url)
                self.assertNotIn(
                    "SECRET", check_cross_repo_consistency._redacted(url)
                )

    def test_plural_credential_is_not_emitted_in_the_patch(self):
        selection = model_runtime.AgentSelection(
            harness="dsh", model="https://h/v1?passwords=SECRET"
        )
        patch = model_runtime.dsh_patch_text(selection)
        self.assertNotIn("SECRET", patch)

    def test_legacy_semicolon_query_is_split(self):
        url = "https://host/v1?model=x;api_key=SECRET"
        self.assertEqual(model_runtime.endpoint_query_model(url), "x")
        selection = model_runtime.AgentSelection(harness="dsh", model=url)
        patch = model_runtime.dsh_patch_text(selection)
        record = selection.to_record()
        self.assertNotIn("SECRET", patch)
        self.assertNotIn("SECRET", json.dumps(record))
        self.assertEqual(record["id"], "x")

    def test_catalog_checker_keeps_non_secret_query(self):
        self.assertEqual(
            check_model_catalog._redacted_url("https://h/v1?api-version=2023"),
            "https://h/v1?api-version=2023",
        )
        self.assertEqual(
            check_model_catalog._redacted_url(
                "https://h/v1?api-version=2024&api_key=SECRET"
            ),
            "https://h/v1?api-version=2024",
        )


class RecoveryRedactionTests(unittest.TestCase):
    """F03/F04/F05/F21: credential-shaped text and values are masked."""

    def test_free_text_forms_are_masked(self):
        cases = {
            "Authorization: Bearer abc123": "abc123",
            "https://user:pw@host/v1?api_key=SECRET": "pw",
            '{"api_key": "SECRET"}': "SECRET",
            "passwords=SECRET": "SECRET",
            "x access_key=SECRET y": "SECRET",
            '{"api key": "SECRET"}': "SECRET",
        }
        for text, secret in cases.items():
            with self.subTest(text=text):
                redacted = recovery.redact_text(text)
                self.assertNotIn(secret, redacted)
                self.assertIn("<redacted>", redacted)

    def test_generation_counters_remain_readable(self):
        for text in ("max_tokens=512", "prompt_tokens=99", "the token expired"):
            with self.subTest(text=text):
                self.assertEqual(recovery.redact_text(text), text)
        value = recovery.redact_value(
            {
                "max_tokens": 5,
                "prompt_tokens": 5,
                "tokens": 5,
                "reasoning_tokens": 5,
                "secretary": "x",
            }
        )
        self.assertEqual(value["max_tokens"], 5)
        self.assertEqual(value["tokens"], 5)
        self.assertEqual(value["secretary"], "x")

    def test_prefixed_and_nested_assignments_are_masked(self):
        cases = {
            "HTTP error: api_key = SECRET": "HTTP error: api_key = <redacted>",
            "request failed: api_key=SECRET": "request failed: api_key=<redacted>",
            "argv=--token=SECRET": "argv=--token=<redacted>",
            "body=api_key=SECRET": "body=api_key=<redacted>",
            "a=b&token=SECRET": "a=b&token=<redacted>",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                redacted = recovery.redact_text(text)
                self.assertEqual(redacted, expected)
                self.assertNotIn("SECRET", redacted)

    def test_diagnostic_secret_does_not_reach_the_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = recovery.RecoveryJournal(Path(directory) / "j", "r")
            journal.start({"command": "x"})
            controller = recovery.RecoveryController(
                run_id="r", journal=journal, logger=lambda line: None
            )
            record = controller.emit(
                recovery.Diagnostic(
                    recovery.TIER_ERROR,
                    "UPSTREAM_FAILED",
                    "call",
                    "HTTP error: api_key = SECRET",
                ),
                recovery_action="retry",
            )
            self.assertNotIn("SECRET", record.failure)
            journal_text = (Path(directory) / "j" / "recovery.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("SECRET", journal_text)

    def test_bearer_and_basic_prose_is_not_over_redacted(self):
        for text in (
            "the basic configuration is fine",
            "Basic authentication failed",
            "basic validation passed",
            "Basic Authentication failed",
            "the basic Configuration is fine",
            "the bearer token expired",
            "the bearer responsibility expired",
        ):
            with self.subTest(text=text):
                self.assertEqual(recovery.redact_text(text), text)
        for text in ("Bearer abc123", "Bearer SECRETVALUE", "Bearer sk-abc123"):
            with self.subTest(text=text):
                self.assertNotIn(text.split()[-1], recovery.redact_text(text))

    def test_mapping_keys_and_url_values_are_masked(self):
        value = recovery.redact_value(
            {
                "key": "SECRET",
                "access_key": "SECRET",
                "signature": "SECRET",
                "endpoint": "https://u:p@h/v1?token=T",
            }
        )
        for key in ("key", "access_key", "signature"):
            self.assertEqual(value[key], "<redacted>")
        self.assertNotIn("u:p@", value["endpoint"])
        self.assertNotIn("token=T", value["endpoint"])

    def test_record_free_text_and_consequences_are_redacted(self):
        record = _record(
            recovery_action="used token=SECRET",
            fallback="Authorization: Bearer abc123",
            validation="api_key=SECRET",
            remaining_uncertainty="passwords=SECRET",
            output_effects="https://u:p@h/v1?token=T",
            omissions=["access_key=SECRET"],
            substitutions=["signature=SECRET"],
        )
        rendered = json.dumps(record.to_dict()) + record.to_markdown()
        for secret in ("SECRET", "abc123", "u:p@"):
            self.assertNotIn(secret, rendered)


class RecoveryJournalTests(unittest.TestCase):
    """F06/F10/F11/F12/F13: path, serialization and resume safety."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_path_components_cannot_escape(self):
        for run_id in ("../escape", "/abs", "a/b", "..", ""):
            with self.subTest(run_id=run_id):
                with self.assertRaises(recovery.RecoveryPolicyError):
                    recovery.RecoveryJournal(self.root / "j", run_id)
        with self.assertRaises(recovery.RecoveryPolicyError):
            recovery.RecoveryJournal(
                self.root / "j", "r", filename="/tmp/recovery.jsonl"
            )

    def test_non_json_metadata_does_not_raise(self):
        journal = recovery.RecoveryJournal(self.root / "j", "r")
        self.assertTrue(journal.start({"path": Path("/x"), "items": {1, 2}}))
        self.assertTrue(
            journal.append(
                {"event_id": "r-0001", "sequence": 1, "path": Path("/x")}
            )
        )
        persisted = journal.records()
        self.assertEqual(persisted[0]["path"], "/x")

    def test_invalid_utf8_is_skipped_and_readable(self):
        directory = self.root / "j"
        directory.mkdir()
        (directory / "recovery.jsonl").write_bytes(
            b'{"event_id":"a"}\n\xff\xfe\n'
        )
        self.assertEqual(
            [item["event_id"] for item in recovery.RecoveryJournal(
                directory, "r"
            ).records()],
            ["a"],
        )
        report = directory / "RECOVERY.md"
        report.write_bytes(b"# x\n\xff\xfe")
        self.assertIn("# x", recovery.MarkdownReport(report).read())

    def test_reopening_a_run_keeps_event_ids_unique(self):
        directory = self.root / "j"
        first = recovery.RecoveryJournal(directory, "r")
        first.append(
            {"event_id": first.next_event_id(), "sequence": 1, "code": "A"}
        )
        second = recovery.RecoveryJournal(directory, "r")
        self.assertEqual(second.next_event_id(), "r-0002")
        second.append(
            {"event_id": "r-0001", "sequence": 1, "code": "B"}
        )
        entries = [(item["event_id"], item["code"]) for item in second.records()]
        self.assertEqual([code for _event_id, code in entries], ["A", "B"])
        self.assertEqual(len({event_id for event_id, _code in entries}), 2)

    def test_finalize_keeps_started_at_when_start_fell_back(self):
        blocker = self.root / "not-a-dir"
        blocker.write_text("x")
        fallback = self.root / "fallback"
        journal = recovery.RecoveryJournal(
            blocker, "r", fallback_directory=fallback
        )
        self.assertFalse(journal.start({"command": "c"}))
        manifest = journal.finalize("completed")
        self.assertIsNotNone(manifest["started_at"])
        saved = json.loads((fallback / "r-run.json").read_text())
        self.assertEqual(saved["started_at"], manifest["started_at"])


class RecoveryReportTests(unittest.TestCase):
    """F07/F25: report markers and malformed journal entries."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_embedded_end_marker_cannot_split_a_section(self):
        report = recovery.MarkdownReport(self.root / "RECOVERY.md")
        body = (
            "first\n<!-- /recovery-run:r1 -->\n"
            "Status: **IN PROGRESS**\nEVIL"
        )
        report.upsert_section("r1", body)
        report.upsert_section("r1", "second")
        text = report.read()
        self.assertNotIn("EVIL\n", text)
        self.assertEqual(text.count("<!-- /recovery-run:r1 -->"), 1)

    def test_unsafe_run_id_is_rejected(self):
        report = recovery.MarkdownReport(self.root / "RECOVERY.md")
        for run_id in ("a b", "a\nb", "a-->"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(recovery.RecoveryPolicyError):
                    report.upsert_section(run_id, "body")

    def test_partial_json_record_is_skipped(self):
        self.assertEqual(
            recovery.render_commit_section([{"run_id": "r"}]), ""
        )
        self.assertIn(
            "Recovery",
            recovery.render_commit_section(
                [_record(run_id="r", event_id="e")]
            ),
        )

    def test_run_id_with_backtick_cannot_close_the_heading_code_span(self):
        run_id = "x` [pwn](https://evil) `"
        heading = recovery.render_run_markdown(
            run_id, status="completed", records=[]
        ).splitlines()[0]
        opening = re.match(r"## Run (`+)", heading)
        self.assertIsNotNone(opening)
        fence = opening.group(1)
        body = heading[len("## Run ") + len(fence) : -len(fence)]
        runs = re.findall(r"`+", body)
        # A CommonMark code span closes at the first run as long as its fence;
        # the fence must therefore be longer than every run inside the value.
        self.assertGreater(len(fence), max((len(run) for run in runs), default=0))
        self.assertIn("[pwn](https://evil)", body)


class RecoveryControllerTests(unittest.TestCase):
    """F16/F17/F20: honest audit records and threshold ordering."""

    def test_unpersisted_record_emits_a_warning_before_the_stop(self):
        class BadJournal:
            def next_event_id(self):
                return "r-0001"

            def append(self, payload):
                return False

        lines = []
        controller = recovery.RecoveryController(
            run_id="r",
            policy=recovery.FailurePolicy(recovery.TIER_WARNING),
            journal=BadJournal(),
            logger=lines.append,
        )
        with self.assertRaises(recovery.ThresholdExceeded):
            controller.emit(
                recovery.Diagnostic(recovery.TIER_ERROR, "E", "op", "m"),
                recovery_action="act",
            )
        self.assertTrue(any("AUDIT_WRITE_FAILED" in line for line in lines))

    def test_first_attempt_success_records_nothing(self):
        controller = recovery.RecoveryController(run_id="r", logger=lambda line: None)
        result = controller.run_with_recovery(
            recovery.Diagnostic(recovery.TIER_ERROR, "E", "op", "m"),
            "act",
            lambda attempt: "ok",
            max_attempts=2,
        )
        self.assertTrue(result.ok)
        self.assertEqual(controller.records, [])

    def test_successful_retry_records_recovered(self):
        def flaky(attempt):
            if attempt == 1:
                raise ValueError("transient")
            return "ok"

        controller = recovery.RecoveryController(
            run_id="r", logger=lambda line: None
        )
        result = controller.run_with_recovery(
            recovery.Diagnostic(recovery.TIER_WARNING, "W", "op", "m"),
            "act",
            flaky,
            max_attempts=2,
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            [item.outcome for item in controller.records],
            ["retried", "recovered"],
        )

    def test_guard_work_records_the_actual_fallback_outcome(self):
        def boom():
            raise ValueError("x")

        def bad_fallback(exc):
            raise ValueError("fallback")

        controller = recovery.RecoveryController(
            run_id="r", logger=lambda line: None
        )
        result = controller.guard_work("op", boom, fallback=bad_fallback)
        self.assertFalse(result.ok)
        self.assertEqual([item.outcome for item in controller.records], ["failed"])
        self.assertNotIn("replaced", controller.records[0].output_effects)

        controller = recovery.RecoveryController(
            run_id="r2", logger=lambda line: None
        )
        result = controller.guard_work(
            "op", boom, fallback=lambda exc: "fallback value"
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.degraded)
        self.assertEqual(
            [item.outcome for item in controller.records], ["degraded"]
        )


class ModelRuntimeRegressionTests(unittest.TestCase):
    """F27/F28/F30/F31/F32/F34/F41/F42: runtime routes and records."""

    def test_non_object_catalog_raises_config_error(self):
        original = model_runtime._MODEL_CATALOG_PATH
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text("[]")
            model_runtime._MODEL_CATALOG_PATH = path
            try:
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime._apply_model_catalog(model_runtime.AGENT_MODELS)
            finally:
                model_runtime._MODEL_CATALOG_PATH = original

    def test_tolerated_route_base_keeps_one_non_secret_query(self):
        self.assertEqual(
            model_runtime._normalized_route_base_url(
                "http://host:99999/v1?api-version=2024"
            ),
            "http://host:99999/v1?api-version=2024",
        )
        self.assertEqual(
            model_runtime._normalized_route_base_url("http://:8080/v1?x=1"),
            "http://:8080/v1?x=1",
        )

    def test_non_secret_query_survives_every_generated_route(self):
        qwen = model_runtime.resolve_agent_selection(model="qwen38-27b")
        qwen_patch = model_runtime.dsh_patch_text(
            qwen, base_url_override="https://gw/v1?api-version=2024"
        )
        self.assertIn('baseURL: "https://gw/v1?api-version=2024"', qwen_patch)

        dots = model_runtime.resolve_agent_selection(model="dots3-note-preview")
        dots_patch = model_runtime.dsh_patch_text(
            dots, compaction_base_url="https://gw/v1?api-version=2024"
        )
        self.assertIn('baseURL: "https://gw/v1?api-version=2024"', dots_patch)

        original = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            endpoint = model_runtime.resolve_agent_selection(
                model="https://h/v1?api-version=2024"
            )
            endpoint_patch = model_runtime.dsh_patch_text(
                endpoint, base_url_override="https://gw/v1?api-version=2025"
            )
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = original
        self.assertIn(
            'baseURL: "https://gw/v1?api-version=2025"', endpoint_patch
        )

        with mock.patch.dict(
            os.environ, {"PAPER_QWEN_BASE_URL": "https://gw/v1?api-version=2024"}
        ):
            claude = model_runtime.claude_environment(qwen)
            codex = model_runtime.codex_provider_args(qwen)
        self.assertEqual(
            claude["ANTHROPIC_BASE_URL"], "https://gw?api-version=2024"
        )
        self.assertIn(
            'model_providers.qwen.base_url="https://gw/v1?api-version=2024"',
            codex,
        )

    def test_direct_claude_selection_is_rejected_on_codex(self):
        selection = model_runtime.AgentSelection(
            harness="gpt-codex", model="claude-opus"
        )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.codex_provider_args(selection)

    def test_raw_endpoint_on_named_selection_has_a_consistent_record(self):
        selection = model_runtime.resolve_agent_selection(
            harness="gpt-codex", model="gpt-sol"
        )
        record = selection.to_record(raw_model="https://user:pw@host/v1")
        self.assertEqual(record["model"], "https://host/v1")
        self.assertEqual(record["id"], "default")
        self.assertEqual(record["endpoint"], "https://host/v1")
        self.assertTrue(record["api_key_set"])

        sentinel = model_runtime.AgentSelection(harness="dsh", model="default")
        self.assertEqual(
            sentinel.to_record(raw_model="https://host/v1")["id"],
            "https://host/v1",
        )

    def test_preflight_uses_the_caller_named_key_variable(self):
        seen = {}

        def fake_catalog(base_url, api_key="", timeout=5.0):
            seen["key"] = api_key
            return [{"id": "one"}]

        original = model_runtime._endpoint_catalog
        model_runtime._endpoint_catalog = fake_catalog
        try:
            with mock.patch.dict(os.environ, {"OTHER_KEY": "K"}, clear=False):
                os.environ.pop("PAPER_AGENT_MODEL_API_KEY", None)
                selection = model_runtime.resolve_agent_selection(
                    model="https://h/v1"
                )
                resolved = model_runtime.resolve_endpoint_model(
                    selection, endpoint_api_key_env="OTHER_KEY"
                )
                self.assertEqual(resolved, ("one", None, ""))
                self.assertEqual(seen["key"], "K")
                self.assertEqual(
                    model_runtime.selection_problem(
                        selection, endpoint_api_key_env="OTHER_KEY"
                    ),
                    "",
                )
        finally:
            model_runtime._endpoint_catalog = original

    def test_direct_compaction_url_keeps_the_retry_policy(self):
        selection = model_runtime.resolve_agent_selection(
            model="dots3-note-preview"
        )

        def compaction_block(text):
            lines = text.splitlines()
            start = lines.index("      dots-compaction:")
            block = []
            for line in lines[start + 1:]:
                if line.startswith("      ") and not line.startswith("        "):
                    break
                block.append(line)
            return block

        direct = model_runtime.dsh_patch_text(
            selection, compaction_base_url="https://direct.example/v1"
        )
        proxy = model_runtime.dsh_patch_text(
            selection, compaction_base_url="http://127.0.0.1:9999/v1"
        )
        self.assertTrue(
            any("retryPolicy" in line for line in compaction_block(direct))
        )
        self.assertFalse(
            any("retryPolicy" in line for line in compaction_block(proxy))
        )

    def test_endpoint_ids_are_one_line_and_credential_free(self):
        original = model_runtime._endpoint_catalog
        model_runtime._endpoint_catalog = lambda base_url, api_key="", timeout=5.0: [
            {"id": "evil\nTIER-ERROR-FAKE: x"},
            {"id": "api_key=SECRET"},
        ]
        try:
            _name, _context, error = model_runtime.resolve_endpoint_model(
                model_runtime.resolve_agent_selection(model="https://h/v1")
            )
        finally:
            model_runtime._endpoint_catalog = original
        self.assertNotIn("\n", error)
        self.assertNotIn("SECRET", error)
        self.assertIn("<redacted>", error)


class ProxyPathTests(unittest.TestCase):
    """F08: both loopback proxies reject double-encoded dot segments."""

    def test_double_encoded_dot_segments_are_rejected(self):
        for module in (model_runtime, llm_retry):
            with self.subTest(module=module.__name__):
                with self.assertRaises(ValueError):
                    module._reject_dot_segments("/v1/%252e%252e/etc")
                module._reject_dot_segments("/v1/models")

    def test_retry_proxy_target_path_rejects_double_encoding(self):
        proxy = llm_retry.RetryProxy("https://h/v1")
        with self.assertRaises(ValueError):
            proxy.target_path("/v1/%252e%252e/etc")


class RetryPolicyTests(unittest.TestCase):
    """F44/F45/F46: strict policy types, empty list rendering, HTTP dates."""

    def test_invalid_member_types_are_rejected(self):
        for document in (
            {"retry": {"max_retries": 2.9}},
            {"retry": {"retryable_http_statuses": [True]}},
            {"retry": {"retryable_http_statuses": [1.9]}},
            {"retry": {"retryable_http_statuses": ["429"]}},
            {"retry": {"retryable_error_codes": [1]}},
        ):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    llm_retry._merge_policy(document)

    def test_empty_retryable_codes_render_as_an_empty_list(self):
        rendered = llm_retry.retry_policy_yaml(
            {"retry": {"retryable_error_codes": []}}
        )
        self.assertIn("retryableCodes: []", rendered)
        self.assertNotIn("retryableCodes:\n", rendered)

    def test_timezone_less_http_date_is_utc(self):
        had_tzset = hasattr(time, "tzset")
        if not had_tzset:
            self.skipTest("time.tzset is not available")
        previous = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            time.tzset()
            raw = "Wed, 21 Oct 2030 07:28:00"
            fixed = time.time()
            with mock.patch.object(llm_retry.time, "time", return_value=fixed):
                observed = llm_retry.retry_after_seconds({"Retry-After": raw})
            expected = (
                datetime.datetime(
                    2030, 10, 21, 7, 28, tzinfo=datetime.timezone.utc
                ).timestamp()
                - fixed
            )
            self.assertAlmostEqual(observed, expected, delta=2)
        finally:
            if previous is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous
            time.tzset()


class CheckerTests(unittest.TestCase):
    """F53/F54: AST wiring and bytecode-free probes."""

    def test_from_import_and_chained_module_use_are_valid_wiring(self):
        cases = {
            "from-import": (
                "from agent_common.model_runtime import RETRY_SCHEDULE, dsh_patch_text\n"
                "x = RETRY_SCHEDULE\n"
                "y = dsh_patch_text\n"
            ),
            "chained": (
                "import agent_common.model_runtime\n"
                "x = agent_common.model_runtime.RETRY_SCHEDULE\n"
                "y = agent_common.model_runtime.dsh_patch_text\n"
            ),
            "alias": (
                "from agent_common import model_runtime as rt\n"
                "x = rt.RETRY_SCHEDULE\n"
                "y = rt.dsh_patch_text\n"
            ),
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                import ast

                tree = ast.parse(source)
                aliases = check_llm_policy._model_runtime_aliases(tree)
                self.assertTrue(
                    check_llm_policy._uses_attribute(
                        tree, aliases, "RETRY_SCHEDULE"
                    )
                )
                self.assertTrue(
                    check_llm_policy._uses_attribute(
                        tree, aliases, "dsh_patch_text"
                    )
                )

    def test_aliased_use_passes_and_unused_import_does_not(self):
        aliased = ast.parse(
            "from agent_common.model_runtime import RETRY_SCHEDULE as RS, "
            "dsh_patch_text as dp\n"
            "x = RS\n"
            "y = dp\n"
        )
        aliases = check_llm_policy._model_runtime_aliases(aliased)
        self.assertTrue(
            check_llm_policy._uses_attribute(aliased, aliases, "RETRY_SCHEDULE")
        )
        self.assertTrue(
            check_llm_policy._uses_attribute(aliased, aliases, "dsh_patch_text")
        )

        unused = ast.parse(
            "from agent_common.model_runtime import RETRY_SCHEDULE, dsh_patch_text\n"
            "x = 1\n"
        )
        aliases = check_llm_policy._model_runtime_aliases(unused)
        self.assertFalse(
            check_llm_policy._uses_attribute(unused, aliases, "RETRY_SCHEDULE")
        )
        self.assertFalse(
            check_llm_policy._uses_attribute(unused, aliases, "dsh_patch_text")
        )

    def test_probe_does_not_write_bytecode_into_the_checkout(self):
        import subprocess  # noqa: F401 - documentation of the probe mechanism

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "mod.py").write_text("X = 1\n")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
                observed = check_cross_repo_consistency.probe(
                    "demo",
                    repo,
                    'import json, mod; print(json.dumps({"x": mod.X}))',
                )
            self.assertEqual(observed, {"x": 1})
            self.assertFalse((repo / "__pycache__").exists())


class DocumentationTests(unittest.TestCase):
    """F61/F62/F64: the README inventory and API recipe are accurate."""

    def test_readme_inventory_and_shared_logic_list_recovery(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("recovery.py", text)
        self.assertIn("`recovery.py`", text)

    def test_readme_scopes_the_single_source_claim(self):
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("not catalog fields", text)

    def test_api_recovery_example_does_not_finalize_a_degraded_result_as_completed(self):
        text = (REPO_ROOT / "API.md").read_text(encoding="utf-8")
        self.assertNotIn(
            'journal.finalize("completed" if result.ok else "blocked")', text
        )
        self.assertIn("not result.degraded", text)


if __name__ == "__main__":
    unittest.main()
