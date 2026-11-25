"""Regression tests for the 2026-11-25 verified findings (wave 1).

Each test pins one candidate from the verified-findings set: F01-F61 and
F68-F70.  Items the fix stage declined (F62-F67) are recorded in the ledger,
not tested here.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from unittest import mock

import check_cross_repo_consistency
import check_llm_policy
import check_model_catalog
from agent_common import agent_cli, chat, dsh_events, llm_retry, model_runtime


class RedactionTests(unittest.TestCase):
    """F01/F02/F60: malformed URLs and bare userinfo stay credential-free."""

    def test_malformed_authority_redacts_everywhere(self):
        for raw in (
            "http:/user:pw@host/v1",
            "http:///user:pw@host/v1",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError) as caught:
                    model_runtime.agent_model_choice(raw)
                self.assertNotIn("pw", str(caught.exception))
                safe = model_runtime.redact_endpoint_url(raw)
                self.assertNotIn("pw", safe)
                self.assertNotIn("@", safe)

    def test_a_bare_username_is_the_live_secret(self):
        safe, secret = model_runtime.split_endpoint_credentials(
            "https://TOKEN@host/v1"
        )
        self.assertEqual(safe, "https://host/v1")
        self.assertEqual(secret, "TOKEN")
        with mock.patch.dict(
            os.environ,
            {"PAPER_AGENT_MODEL_API_KEY": "TOKEN"},
            clear=True,
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1"
            )
            self.assertEqual(
                model_runtime._provider_family_settings(selection)["api_key"],
                "TOKEN",
            )

    def test_start_prep_proxy_rejects_a_bare_userinfo_target(self):
        base, error = model_runtime.start_prep_proxy(
            "https://TOKEN@host/v1", {}
        )
        self.assertEqual(base, "")
        self.assertIn("credential", error)
        self.assertNotIn("TOKEN", error)

    def test_semicolon_query_credentials_are_rejected(self):
        url = "https://host/v1?x=1;api_key=SECRET"
        self.assertTrue(model_runtime.endpoint_credentials(url))
        safe, secret = model_runtime.split_endpoint_credentials(url)
        self.assertEqual(secret, "SECRET")
        self.assertNotIn("SECRET", safe)
        with self.assertRaises(ValueError):
            llm_retry.RetryProxy(url)

    def test_an_at_sign_in_a_path_segment_is_not_userinfo(self):
        for url in ("https://host/user@example/v1", "https://host/v1/@latest"):
            with self.subTest(url=url):
                self.assertEqual(model_runtime.endpoint_credentials(url), "")
                self.assertEqual(model_runtime.agent_model_choice(url), url)

    def test_the_resolve_docstring_names_this_repository_error(self):
        doc = inspect.getdoc(model_runtime.resolve_agent_selection) or ""
        self.assertNotIn("gdp.errors", doc)
        self.assertIn("ConfigError", doc)


class DshEventTests(unittest.TestCase):
    """F12/F13/F14: turn-end diagnostics and parser edge cases."""

    def test_cancelled_reason_keeps_its_message(self):
        self.assertEqual(
            dsh_events.turn_end_error(
                {"kind": "cancelled", "message": "operator cancelled"}
            ),
            "operator cancelled",
        )
        self.assertEqual(dsh_events.turn_end_error({"kind": "completed"}), "")

    def test_invalid_utf8_is_not_an_event(self):
        self.assertIsNone(dsh_events.parse_event(b'\xff{"type":"turn_end"}'))

    def test_non_mapping_is_not_a_turn_end(self):
        for value in (None, [], "turn_end", 0):
            with self.subTest(value=value):
                self.assertFalse(dsh_events.is_turn_end(value))


class ChatBudgetTests(unittest.TestCase):
    """F15: the non-positive output-cap branch."""

    def test_non_positive_caps_return_zero(self):
        self.assertEqual(chat.completion_budget(0, 0, 100), 0)
        self.assertEqual(chat.completion_budget(0, -1, 100), 0)
        self.assertEqual(chat.completion_budget(-1, 10, 100), 0)


class ArgvTests(unittest.TestCase):
    """F16: a plain string extra_args value is one argument."""

    def test_a_string_is_not_split_into_characters(self):
        self.assertEqual(
            agent_cli.build_dsh_argv(extra_args="--foo"),
            ["dsh", "--profile", "headless", "--json", "--foo"],
        )


class EndpointPatchTests(unittest.TestCase):
    """F06/F10/F17: generated patches keep the right base/model/effort."""

    def test_a_non_secret_query_is_kept_in_the_patch(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1?api-version=2024"
        )
        patch = model_runtime.dsh_patch_text(selection)
        self.assertIn('baseURL: "https://host/v1?api-version=2024"', patch)
        self.assertNotIn("api_key", patch)

    def test_a_raw_endpoint_never_becomes_a_model_id(self):
        for name in (
            "deepseek-flash",
            "qwen38-27b",
            "dots3-note-preview",
            "gpt-sol",
        ):
            with self.subTest(name=name):
                selection = model_runtime.resolve_agent_selection(
                    harness="dsh", model=name
                )
                patch = model_runtime.dsh_patch_text(
                    selection, raw_model="https://host/v1"
                )
                self.assertNotIn('model: "https://host/v1"', patch)
                self.assertIn("provider: endpoint", patch)

    def test_a_raw_endpoint_keeps_the_explicit_model_name(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", model_name="my-model"
        )
        patch = model_runtime.dsh_patch_text(
            selection, raw_model="https://other/v2"
        )
        self.assertIn('baseURL: "https://other/v2"', patch)
        self.assertIn('model: "my-model"', patch)
        effective = model_runtime._effective_selection(
            selection, "https://other/v2"
        )
        self.assertEqual(effective.model_name, "my-model")

    def test_an_unknown_endpoint_effort_is_rejected(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", api_key="k"
        )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.dsh_patch_text(selection, effort_override="turbo")

    def test_claude_rejects_a_raw_endpoint(self):
        selection = model_runtime.resolve_agent_selection(
            harness="claude-code", model="default"
        )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.claude_model_arg(
                selection, raw_model="https://host/v1"
            )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.claude_environment(
                selection, raw_model="https://host/v1"
            )


class ProxyTests(unittest.TestCase):
    """F05/F18-F23: loopback proxy framing, ownership and path handling."""

    def test_prep_proxy_query_merge_prefers_the_client(self):
        base, error = model_runtime.start_prep_proxy(
            "https://host/v1?api-version=2024", {}
        )
        self.assertEqual(error, "")
        try:
            server = model_runtime._PROXY_BY_BASE[base]
            handler = object.__new__(model_runtime._PrepProxyHandler)
            handler.server = server
            handler.path = "/v1/chat?api-version=2023"
            self.assertEqual(
                handler._target(),
                "https://host/v1/chat?api-version=2023",
            )
        finally:
            model_runtime.stop_prep_proxy(base)

    def test_retry_proxy_query_merge_prefers_the_client(self):
        proxy = llm_retry.RetryProxy("https://host/v1?api-version=2024")
        self.assertEqual(
            proxy.target_path("/v1/chat?api-version=2023"),
            "/v1/chat?api-version=2023",
        )

    def test_dot_segments_are_rejected(self):
        proxy = llm_retry.RetryProxy("https://host/v1")
        for path in ("/v1/../admin", "/../admin", "/v1/%2e%2e/admin"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    proxy.target_path(path)

    def test_a_short_body_is_an_error(self):
        with self.assertRaises(ValueError):
            llm_retry._read_request_body(
                io.BytesIO(b"abc"), {"Content-Length": "10"}
            )

    def test_a_short_body_gets_http_400_from_the_proxy(self):
        proxy = llm_retry.RetryProxy("http://127.0.0.1:9/v1")
        base = proxy.start()
        try:
            port = int(base.rsplit(":", 1)[1].split("/")[0])
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(
                    b"POST /v1/x HTTP/1.1\r\nHost: x\r\n"
                    b"Content-Length: 10\r\n\r\nabc"
                )
                sock.shutdown(socket.SHUT_WR)
                data = b""
                while True:
                    part = sock.recv(65536)
                    if not part:
                        break
                    data += part
            self.assertIn(b"400", data)
        finally:
            proxy.stop()

    def test_duplicate_request_headers_are_merged(self):
        headers = Message()
        headers["Cookie"] = "a=1"
        headers["Cookie"] = "b=2"
        headers["X-Test"] = "one"
        headers["X-Test"] = "two"
        forwarded = llm_retry._forwardable_request_headers(headers)
        self.assertEqual(forwarded["Cookie"], "a=1; b=2")
        self.assertEqual(forwarded["X-Test"], "one, two")

    def test_stop_retry_proxy_stops_a_prep_server(self):
        base, error = model_runtime.start_prep_proxy("http://127.0.0.1:9/v1", {})
        self.assertEqual(error, "")
        server = model_runtime._PROXY_BY_BASE[base]
        thread = server._gdp_thread
        llm_retry.stop_retry_proxy(kind="prep")
        self.assertNotIn(base, llm_retry._PROXY_REGISTRY.by_base)
        self.assertFalse(thread.is_alive())
        port = int(base.rsplit(":", 1)[1].split("/")[0])
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=1).close()

    def test_stop_prep_proxy_joins_the_serving_thread(self):
        base, error = model_runtime.start_prep_proxy("http://127.0.0.1:9/v1", {})
        self.assertEqual(error, "")
        server = model_runtime._PROXY_BY_BASE[base]
        serving_thread = server._gdp_thread

        class RecordingThread:
            def __init__(self):
                self.join_calls = []

            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.join_calls.append(timeout)

        recorder = RecordingThread()
        # shutdown() already stops serve_forever, so is_alive() on the real
        # thread cannot show whether stop_prep_proxy waits for it.  Replace the
        # stored thread with one that only records the join call.
        server._gdp_thread = recorder
        try:
            model_runtime.stop_prep_proxy(base)
        finally:
            serving_thread.join(timeout=5)
        self.assertEqual(len(recorder.join_calls), 1)

    def test_an_http_error_response_is_closed(self):
        body = io.BytesIO(b"error")
        errors = []

        class Opener:
            def open(self, request, timeout=None):  # noqa: A003 - stdlib name
                error = urllib.error.HTTPError(
                    request.full_url, 500, "boom", {}, body
                )
                # Retain the exception: without a live reference CPython
                # collects it before the assertion and closes the body during
                # deallocation, which made this test pass on the pre-fix tree.
                errors.append(error)
                raise error

        with mock.patch.object(
            model_runtime.urllib.request, "build_opener", return_value=Opener()
        ):
            with self.assertRaises(model_runtime._EndpointProbeError):
                model_runtime._endpoint_catalog("https://host/v1")
        self.assertEqual(len(errors), 1)
        self.assertTrue(body.closed)

    def test_a_nested_frozen_prep_can_start_a_proxy(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh",
            model="https://host/v1",
            prep_json={"temperature": 0, "nested": {"a": [1, 2]}},
        )
        base, error = model_runtime.start_prep_proxy(
            "http://127.0.0.1:9/v1", selection.prep_json
        )
        self.assertEqual(error, "")
        try:
            server = model_runtime._PROXY_BY_BASE[base]
            self.assertEqual(server.prep["nested"]["a"], [1, 2])
        finally:
            model_runtime.stop_prep_proxy(base)

    def test_a_prep_proxy_upstream_error_response_is_closed(self):
        errors = []

        class Opener:
            def open(self, request, timeout=None):  # noqa: A003 - stdlib name
                body = io.BytesIO(b"upstream failure")
                error = urllib.error.HTTPError(
                    request.full_url, 500, "boom", {}, body
                )
                # Keep the error alive so only the handler's explicit close can
                # close the upstream body; GC would otherwise hide the bug.
                errors.append(error)
                raise error

        handler = object.__new__(model_runtime._PrepProxyHandler)
        handler.server = mock.Mock(
            target_base="https://host/v1", target_query="", prep={}
        )
        handler.path = "/v1/chat"
        handler.headers = Message()
        handler.wfile = io.BytesIO()
        handler.request_version = "HTTP/1.0"
        handler.requestline = "POST /v1/chat HTTP/1.0"
        handler._headers_buffer = []
        handler.command = "POST"
        with mock.patch.object(model_runtime, "_PREP_OPENER", Opener()):
            handler._forward("POST", b"{}")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].fp.closed)

    def test_an_oversized_models_body_is_rejected(self):
        class Response:
            def __init__(self):
                self.closed = False

            def read(self, limit=None):
                return b"x" * (model_runtime._MAX_MODELS_BODY_BYTES + 1)

            def close(self):
                self.closed = True

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        class Opener:
            def open(self, request, timeout=None):  # noqa: A003 - stdlib name
                return Response()

        with mock.patch.object(
            model_runtime.urllib.request, "build_opener", return_value=Opener()
        ):
            with self.assertRaises(model_runtime._EndpointProbeError):
                model_runtime._endpoint_catalog("https://host/v1")


class RetryPolicyTests(unittest.TestCase):
    """F03/F04/F35/F37: schedule/registry/policy edge cases."""

    def test_present_but_malformed_retry_lists_are_rejected(self):
        for bad in ({}, "", 0, False):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    llm_retry.RetrySchedule(
                        {"retry": {"retryable_http_statuses": bad}}
                    )

    def test_idle_bound_covers_a_retry_after_cap_above_max_delay(self):
        policy = {
            "retry": {
                "initial_delay_seconds": 10,
                "multiplier": 2,
                "max_delay_seconds": 20,
                "max_retries": 2,
                "jitter_ratio": 0.1,
                "retry_after_cap_seconds": 3600,
                "retryable_http_statuses": [429],
                "retryable_error_codes": ["RATE_LIMIT"],
            },
            "compaction": {"compaction_stream_idle_timeout_ms": 0},
        }
        schedule = llm_retry.RetrySchedule(policy)
        self.assertEqual(schedule.delay(1, retry_after=3600), 3600.0)
        overlay = llm_retry.dots_dsh_overlay(
            policy,
            model="m",
            base_url="https://h/v1",
            context_window=100000,
            max_tokens=1000,
        )
        import re

        idle = int(re.search(r"streamIdleTimeoutMs:\s*(\d+)", overlay).group(1))
        worst = int(
            (
                3600.0 * schedule.max_retries * (1.0 + schedule.jitter_ratio)
                + (schedule.max_retries + 1) * 600.0
            )
            * 1000
        )
        self.assertGreaterEqual(idle, worst)

    def test_reregistering_an_item_drops_its_old_base(self):
        class Item:
            pass

        item = Item()
        registry = llm_retry._ProxyRegistry()
        registry.add(item, kind="x", base="old")
        registry.add(item, kind="x", base="new")
        self.assertEqual(sorted(registry.by_base), ["new"])

    def test_a_bytes_policy_path_uses_the_documented_fallback(self):
        self.assertIn("retry", llm_retry.load_policy(b"/tmp/x.json"))
        self.assertIsNotNone(llm_retry.policy_paths(b"/tmp/x.json"))

    def test_a_direct_compaction_url_keeps_the_retry_policy(self):
        overlay = llm_retry.dots_dsh_overlay(
            None,
            model="m",
            base_url="https://h/v1",
            context_window=1000,
            max_tokens=100,
            compaction_base_url="https://h/v1",
        )
        self.assertEqual(overlay.count("retryPolicy:"), 2)

    def test_overlay_rejects_unroutable_bases_and_zero_capacities(self):
        for kwargs in (
            {"base_url": "not a url", "context_window": 1000, "max_tokens": 100},
            {"base_url": "https://h/v1", "context_window": 0, "max_tokens": 100},
            {"base_url": "https://h/v1", "context_window": 1000, "max_tokens": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    llm_retry.dots_dsh_overlay(None, model="m", **kwargs)


class SelectionTests(unittest.TestCase):
    """F09/F24-F34: selection validation and raw-model consistency."""

    def test_selection_env_infers_a_raw_catalog_model(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="default"
        )
        env = model_runtime.selection_env(selection, raw_model="qwen38-27b")
        self.assertIn("PAPER_QWEN_API_KEY", env)
        with mock.patch.dict(os.environ, {"DOTS_API_KEY": "K"}, clear=True):
            env = model_runtime.selection_env(
                selection, raw_model="dots3-note-preview"
            )
        self.assertEqual(env["DOTS_API_KEY"], "K")
        env = model_runtime.selection_env(
            selection,
            raw_model="https://host/v1",
            endpoint_api_key_env="OTHER",
        )
        self.assertIn("OTHER", env)

    def test_selection_problem_reports_instead_of_raising(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        with mock.patch.dict(
            os.environ, {"PAPER_AGENT_REASONING_EFFORT": "turbo"}, clear=True
        ):
            self.assertIn(
                "unknown reasoning effort",
                model_runtime.selection_problem(selection),
            )

    def test_selection_problem_checks_capacities(self):
        with mock.patch.dict(
            os.environ, {"PAPER_QWEN_CONTEXT_WINDOW": "-1"}, clear=True
        ):
            selection = model_runtime.resolve_agent_selection(
                harness="dsh", model="qwen38-27b"
            )
            self.assertIn(
                "context window must be positive",
                model_runtime.selection_problem(selection),
            )

    def test_non_string_options_are_rejected(self):
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1", api_key=123
            )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1", model_name=456
            )

    def test_a_non_string_legacy_agent_is_a_config_error(self):
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.resolve_agent_selection(legacy_agent=["x"])

    def test_the_resolved_prep_mapping_is_read_only(self):
        source = {"a": {"b": 1}, "items": [1, 2]}
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", prep_json=source
        )
        self.assertEqual(selection.prep_json, source)
        with self.assertRaises(TypeError):
            selection.prep_json["api_key"] = "SECRET"
        with self.assertRaises(TypeError):
            selection.prep_json["a"]["b"] = 99
        with self.assertRaises(TypeError):
            selection.prep_json["items"].append(3)
        # The validated copy is detached from the caller's object as well.
        source["a"]["b"] = 99
        source["items"].append(3)
        self.assertEqual(selection.prep_json["a"]["b"], 1)
        self.assertEqual(selection.prep_json["items"], [1, 2])

    def test_a_mapping_prep_json_round_trips(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", prep_json={"a": {"b": 1}}
        )
        again = model_runtime.resolve_agent_selection(
            harness="dsh",
            model="https://host/v1",
            prep_json=selection.prep_json,
        )
        self.assertEqual(again.prep_json, {"a": {"b": 1}})

        class CustomMapping(Mapping):
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        custom = CustomMapping({"temperature": 0})
        resolved = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", prep_json=custom
        )
        self.assertEqual(resolved.prep_json, {"temperature": 0})

    def test_a_directly_constructed_alias_resolves(self):
        selection = model_runtime.AgentSelection(
            harness="dsh", model="gpt-6.1-sol"
        )
        self.assertEqual(selection.model_info()["id"], "gpt-6.1-sol")
        self.assertEqual(selection.effective_model_id(), "gpt-6.1-sol")

    def test_deep_merge_rejects_non_mappings_and_cycles(self):
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.deep_merge({}, [("a", 1)])
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.deep_merge({}, cyclic)

    def test_nan_prep_and_pair_lists_are_rejected(self):
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.resolve_agent_selection(
                harness="dsh",
                model="https://host/v1",
                prep_json={"temperature": float("nan")},
            )
        with self.assertRaises(model_runtime.ConfigError):
            model_runtime.resolve_agent_selection(
                harness="dsh", model="https://host/v1", prep_json=[("a", 1)]
            )

    def test_non_integer_capacity_overrides_are_rejected(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        for kwargs in (
            {"context_window_override": 150000.9},
            {"max_tokens_override": 100.9},
            {"context_window_override": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.dsh_patch_text(selection, **kwargs)

    def test_effective_agent_model_reads_a_dsh_model_flag(self):
        self.assertEqual(
            model_runtime.effective_agent_model(
                ["dsh", "--profile", "headless", "--model", "gpt-sol"]
            )["name"],
            "gpt-sol",
        )

    def test_scheme_less_route_base_is_an_error(self):
        with mock.patch.dict(
            os.environ,
            {"PAPER_QWEN_BASE_URL": "user:pw@host/v1"},
            clear=True,
        ):
            codex = model_runtime.resolve_agent_selection(
                harness="gpt-codex", model="qwen38-27b"
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.codex_provider_args(codex)
            claude = model_runtime.resolve_agent_selection(
                harness="claude-code", model="qwen38-27b"
            )
            with self.assertRaises(model_runtime.ConfigError):
                model_runtime.claude_environment(claude)


class CatalogTests(unittest.TestCase):
    """F36/F41/F44/F45/F56: catalog loader/checker contract."""

    def test_the_runtime_rejects_an_unknown_schema_version(self):
        document = json.loads(model_runtime._MODEL_CATALOG_PATH.read_text())
        document["schema_version"] = 99
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(document))
            old = model_runtime._MODEL_CATALOG_PATH
            try:
                model_runtime._MODEL_CATALOG_PATH = path
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.reload_model_catalog()
            finally:
                model_runtime._MODEL_CATALOG_PATH = old
                model_runtime.reload_model_catalog()

    def _run_catalog(self, document, *, copies=()):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(document))
            output = io.StringIO()
            with mock.patch.object(check_model_catalog, "CANONICAL", path), \
                    mock.patch.object(check_model_catalog, "COPIES", copies):
                with contextlib.redirect_stdout(output):
                    code = check_model_catalog.main()
            return code, output.getvalue()

    def test_the_checker_accepts_a_missing_chat_cap(self):
        document = json.loads(check_model_catalog.CANONICAL.read_text())
        for entry in document["models"]:
            if entry["id"] == "deepseek-flash":
                entry.pop("chat_max_tokens", None)
        code, output = self._run_catalog(document)
        self.assertEqual(code, 0, output)

    def test_the_checker_runs_the_runtime_loader(self):
        document = json.loads(check_model_catalog.CANONICAL.read_text())
        with mock.patch.object(
            model_runtime,
            "_apply_model_catalog",
            side_effect=RuntimeError("loader boom"),
        ):
            code, output = self._run_catalog(document)
        self.assertEqual(code, 1)
        self.assertIn("shared runtime does not accept", output)

    def test_the_checker_validates_names_and_dates(self):
        document = json.loads(check_model_catalog.CANONICAL.read_text())
        document["updated_at"] = "not-a-date"
        for entry in document["models"]:
            if entry["id"] == "qwen3.8-27b":
                entry["name"] = ""
                entry["verified_at"] = "not-a-date"
        code, output = self._run_catalog(document)
        self.assertEqual(code, 1)
        self.assertIn("bad updated_at", output)
        self.assertIn("bad name", output)
        self.assertIn("bad verified_at", output)

    def test_checker_diagnostics_redact_a_credential_url(self):
        document = json.loads(check_model_catalog.CANONICAL.read_text())
        for entry in document["models"]:
            if entry["id"] == "qwen3.8-27b":
                entry["base_url"] = "https://user:pw@host:bad/v1"
        code, output = self._run_catalog(document)
        self.assertEqual(code, 1)
        self.assertNotIn("pw", output)
        self.assertIn("https://host:bad/v1", output)

    def test_the_checker_rejects_bad_provider_and_key_env(self):
        for field, value in (("provider", []), ("api_key_env", 123)):
            with self.subTest(field=field):
                document = json.loads(check_model_catalog.CANONICAL.read_text())
                for entry in document["models"]:
                    if entry["id"] == "qwen3.8-27b":
                        entry[field] = value
                code, output = self._run_catalog(document)
                self.assertEqual(code, 1)
                self.assertIn("bad %s" % field, output)

    def test_the_checker_accepts_good_base_urls_and_rejects_bad_ones(self):
        good = json.loads(check_model_catalog.CANONICAL.read_text())
        code, output = self._run_catalog(good)
        self.assertEqual(code, 0, output)
        bad = json.loads(check_model_catalog.CANONICAL.read_text())
        for entry in bad["models"]:
            if entry["id"] == "qwen3.8-27b":
                entry["base_url"] = "https://host:bad/v1"
        code, output = self._run_catalog(bad)
        self.assertEqual(code, 1)


class CheckerTests(unittest.TestCase):
    """F07/F46-F54/F57/F70: checker robustness and coverage."""

    def test_drift_redaction_keeps_a_non_secret_query(self):
        first = check_cross_repo_consistency._redacted(
            "https://host/v1?api-version=2023"
        )
        second = check_cross_repo_consistency._redacted(
            "https://host/v1?api-version=2024"
        )
        self.assertNotEqual(first, second)
        hidden = check_cross_repo_consistency._redacted(
            "https://user:pw@host/v1?api_key=SECRET"
        )
        self.assertNotIn("pw", hidden)
        self.assertNotIn("SECRET", hidden)

    def test_expected_fields_include_provider_and_key_env(self):
        catalogue = {
            "m": {
                "context_window": 1,
                "max_tokens": 1,
                "provider": "p",
                "api_key_env": "K",
            }
        }
        fields = check_cross_repo_consistency.expected_fields("m", catalogue)
        self.assertIn("provider", fields)
        self.assertIn("api_key_env", fields)
        self.assertNotIn("chat_max_tokens", fields)

    def test_unexpected_observation_fields_fail(self):
        catalogue = {
            "m": {
                "id": "m",
                "context_window": 1,
                "max_tokens": 1,
                "chat_max_tokens": 1,
            }
        }
        observed = {
            "m": {
                "context_window": 1,
                "max_tokens": 1,
                "chat_max_tokens": 1,
                "default_reasoning_level": "high",
            }
        }
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency, "_read_catalogue",
            return_value=(catalogue, ""),
        ), mock.patch.object(
            check_cross_repo_consistency, "probe", return_value=observed
        ), mock.patch.object(
            check_cross_repo_consistency.subprocess,
            "run",
            return_value=mock.Mock(returncode=0, stdout="OK", stderr=""),
        ):
            with contextlib.redirect_stdout(output):
                code = check_cross_repo_consistency.main()
        self.assertEqual(code, 1)
        self.assertIn("unexpected field", output.getvalue())

    def test_a_non_mapping_observation_fails_without_crashing(self):
        catalogue = {
            "m": {
                "id": "m",
                "context_window": 1,
                "max_tokens": 1,
                "chat_max_tokens": 1,
            }
        }
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency, "_read_catalogue",
            return_value=(catalogue, ""),
        ), mock.patch.object(
            check_cross_repo_consistency, "probe", return_value={"m": 5}
        ), mock.patch.object(
            check_cross_repo_consistency.subprocess,
            "run",
            return_value=mock.Mock(returncode=0, stdout="OK", stderr=""),
        ):
            with contextlib.redirect_stdout(output):
                code = check_cross_repo_consistency.main()
        self.assertEqual(code, 1)
        self.assertIn("not an object", output.getvalue())

    def test_invalid_utf8_probe_output_does_not_crash(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            observed = check_cross_repo_consistency.probe(
                "demo",
                Path("."),
                'import sys; sys.stdout.buffer.write(b"\\xff"); '
                "sys.stdout.buffer.flush()",
            )
        self.assertIsNone(observed)
        self.assertIn("invalid JSON", output.getvalue())

    def test_a_probe_timeout_is_a_failure(self):
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency.subprocess,
            "run",
            side_effect=__import__("subprocess").TimeoutExpired(["x"], 1),
        ):
            with contextlib.redirect_stdout(output):
                observed = check_cross_repo_consistency.probe(
                    "demo", Path("."), "print(1)"
                )
        self.assertIsNone(observed)
        self.assertIn("timed out", output.getvalue())

    def test_probe_stderr_is_redacted(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check_cross_repo_consistency.probe(
                "demo",
                Path("."),
                'import sys; sys.stderr.write('
                '"https://user:pw@host/v1?api_key=SECRET\\n"); sys.exit(1)',
            )
        self.assertNotIn("pw", output.getvalue())
        self.assertNotIn("SECRET", output.getvalue())

    def test_adapter_wiring_is_parsed_not_substring_matched(self):
        tree = ast.parse(
            "from agent_common import model_runtime as rt\n"
            "x = rt.RETRY_SCHEDULE\nrt.dsh_patch_text()\n"
        )
        aliases = check_llm_policy._model_runtime_aliases(tree)
        self.assertIn("rt", aliases)
        self.assertTrue(check_llm_policy._uses_attribute(tree, aliases, "dsh_patch_text"))
        comment_only = ast.parse("# import model_runtime as rt\n")
        self.assertFalse(check_llm_policy._model_runtime_aliases(comment_only))

    def test_the_policy_checker_pins_schema_and_description(self):
        document = json.loads(check_llm_policy.CANONICAL_JSON.read_text())
        document["schema_version"] = 99
        document.pop("description", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.json"
            path.write_text(json.dumps(document))
            output = io.StringIO()
            with mock.patch.object(check_llm_policy, "CANONICAL_JSON", path), \
                    mock.patch.object(check_llm_policy, "COPIES", ()):
                with contextlib.redirect_stdout(output):
                    code = check_llm_policy.main()
        self.assertEqual(code, 1)
        self.assertIn("unsupported schema_version", output.getvalue())
        self.assertIn("description", output.getvalue())

    def test_deepseek_selection_exports_its_key(self):
        cases = (
            ({"DEEPSEEK_API_KEY": "A"}, "A"),
            ({"PAPER_DEEPSEEK_API_KEY": "B"}, "B"),
            ({"DEEPSEEK_API_KEY": "A", "PAPER_DEEPSEEK_API_KEY": "B"}, "A"),
            ({}, None),
        )
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="deepseek-flash"
        )
        for environment, expected in cases:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    env = model_runtime.selection_env(selection)
                if expected is None:
                    self.assertEqual(env, {})
                else:
                    self.assertEqual(env["DEEPSEEK_API_KEY"], expected)


class DocumentationTests(unittest.TestCase):
    """F58/F59: the public documentation states the real contracts."""

    ROOT = Path(__file__).resolve().parent

    def test_the_docs_qualify_the_example_contract(self):
        readme = (self.ROOT / "README.md").read_text(encoding="utf-8")
        api = (self.ROOT / "API.md").read_text(encoding="utf-8")
        self.assertIn("callable method", readme)
        self.assertIn("public properties", readme)
        self.assertIn("callable method", api)

    def test_the_readme_qualifies_the_claude_effort_default(self):
        readme = (self.ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("PAPER_AGENT_REASONING_EFFORT", readme)
        self.assertIn("including qwen and dots", readme)


if __name__ == "__main__":
    unittest.main()
