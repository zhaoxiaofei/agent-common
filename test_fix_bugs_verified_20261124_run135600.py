"""Regression tests for the verified findings of run 20261124-135600.

Each test pins one candidate kept by the verify-findings stage: F01, F02,
F04, F05, F06, F10, F14, F17, F29, F48, F49 and F54.  The tests are
hermetic: loopback HTTP servers only, temporary catalog files only, and all
process-wide state (registry, proxy flag, catalog path) is restored.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_common import chat, llm_retry, model_runtime

REPO = Path(__file__).resolve().parent


@contextlib.contextmanager
def _server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


class KeyBearingProbeProxyTests(unittest.TestCase):
    """F01: environment proxies must not receive key-bearing requests."""

    def test_probe_and_prep_opener_bypass_environment_proxies(self):
        script = r"""
import json, os, sys, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

seen = {"origin": [], "proxy": []}

class Origin(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        seen["origin"].append(self.headers.get("Authorization"))
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "m1"}]}).encode()
        else:
            body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        seen["proxy"].append(self.path)
        body = json.dumps({"data": [{"id": "from-proxy"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

origin = serve(Origin)
proxy = serve(Proxy)
origin_url = "http://127.0.0.1:%d/v1" % origin.server_address[1]
proxy_url = "http://127.0.0.1:%d" % proxy.server_address[1]
os.environ["HTTP_PROXY"] = proxy_url
os.environ["http_proxy"] = proxy_url
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
sys.path.insert(0, sys.argv[1])
from agent_common import model_runtime as mr
urllib.request.proxy_bypass = lambda host: False
try:
    models = mr._endpoint_catalog(origin_url, "SECRET-KEY")
except Exception as exc:
    models = "ERR:%s" % type(exc).__name__
request = urllib.request.Request(
    origin_url + "/prep", headers={"Authorization": "Bearer PREP-KEY"}, method="GET"
)
try:
    with mr._PREP_OPENER.open(request, timeout=5) as response:
        prep = response.read().decode()
except Exception as exc:
    prep = "ERR:%s" % type(exc).__name__
print(json.dumps({"models": models, "prep": prep, "seen": seen}))
"""
        env = {
            key: value
            for key, value in os.environ.items()
            if key.lower()
            not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "pythonpath"}
        }
        env["PYTHONPATH"] = str(REPO)
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script, str(REPO)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["models"], [{"id": "m1"}])
        self.assertEqual(payload["prep"], "ok")
        self.assertEqual(payload["seen"]["proxy"], [])
        self.assertIn("Bearer SECRET-KEY", payload["seen"]["origin"])
        self.assertIn("Bearer PREP-KEY", payload["seen"]["origin"])


class EndpointCredentialFallbackTests(unittest.TestCase):
    """F02: with rejection disabled the URL credential must reach the request."""

    def test_disabled_flag_uses_userinfo_and_query_credentials(self):
        original = model_runtime.REJECT_ENDPOINT_CREDENTIALS
        model_runtime.REJECT_ENDPOINT_CREDENTIALS = False
        try:
            for url, expected in (
                ("https://user:SECRET@host/v1?api_key=QKEY", "SECRET"),
                ("https://host/v1?api_key=QKEY", "QKEY"),
            ):
                with self.subTest(url=url):
                    selection = model_runtime.resolve_agent_selection(
                        harness="dsh", model=url
                    )
                    env = model_runtime.selection_env(selection)
                    patch = model_runtime.dsh_patch_text(selection)
                    self.assertEqual(
                        env[model_runtime.AGENT_MODEL_API_KEY_ENV], expected
                    )
                    self.assertNotIn(expected, patch)
        finally:
            model_runtime.REJECT_ENDPOINT_CREDENTIALS = original


class GeneratedYamlQuotingTests(unittest.TestCase):
    """F04: operator/policy controlled scalars must be quoted."""

    def test_retry_codes_and_provider_names_are_quoted(self):
        policy = {"retry": {"retryable_error_codes": ["OK\n    - injected: true"]}}
        text = llm_retry.retry_policy_yaml(policy, "  ")
        self.assertIn('      - "OK\\n    - injected: true"', text)
        self.assertNotIn("\n    - injected: true\n", text)

        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="qwen38-27b"
        )
        patch = model_runtime.dsh_patch_text(
            selection, provider_override="x: y"
        )
        self.assertIn('      "x: y":', patch)
        self.assertIn('    provider: "x: y"', patch)

        endpoint = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", api_key="K"
        )
        endpoint_patch = model_runtime.dsh_patch_text(
            endpoint, endpoint_api_key_env="x: y"
        )
        self.assertIn('        apiKeyEnv: "x: y"', endpoint_patch)

        overlay = llm_retry.dots_dsh_overlay(
            model="dots3-note-prev",
            base_url="https://dots.example/v1",
            provider="x: y",
            context_window=1000,
            max_tokens=100,
        )
        self.assertIn('      "x: y":', overlay)
        self.assertIn("apiKeyEnv: DOTS_API_KEY", overlay)

    def test_dots_overlay_quotes_policy_owned_provider_and_key_env(self):
        policy = json.loads(json.dumps(llm_retry.DEFAULT_POLICY))
        policy["dots"]["compaction_provider"] = "x: y"
        policy["dots"]["compaction_api_key_env"] = "z: w"
        overlay = llm_retry.dots_dsh_overlay(
            policy,
            model="dots3-note-prev",
            base_url="https://dots.example/v1",
            context_window=1000,
            max_tokens=100,
        )
        self.assertIn('      "x: y":', overlay)
        self.assertIn('        apiKeyEnv: "z: w"', overlay)


class CredentialPredicateTests(unittest.TestCase):
    """F05: signed-URL spellings and tuples must be treated as credentials."""

    def test_signature_names_and_tuples_are_redacted(self):
        redacted = model_runtime.redact_prep_object(
            {
                "signature": "SECRET",
                "sig": "SECRET",
                "cookie": "SECRET",
                "refresh_tokens": "SECRET",
                "headers": ({"api_key": "SECRET"},),
            }
        )
        for key in ("signature", "sig", "cookie", "refresh_tokens"):
            self.assertEqual(redacted[key], "[REDACTED]")
        self.assertEqual(redacted["headers"], ({"api_key": "[REDACTED]"},))

    def test_signed_url_secret_is_split_out(self):
        safe, secret = model_runtime.split_endpoint_credentials(
            "https://host/v1?signature=SECRET&model=m"
        )
        self.assertEqual(secret, "SECRET")
        self.assertNotIn("SECRET", safe)
        self.assertIn("model=m", safe)


class RetryProxyQueryCredentialTests(unittest.TestCase):
    """F06: the retry proxy must reject query credentials like the prep proxy."""

    def test_credential_query_is_rejected(self):
        for base in (
            "https://host/v1?api_key=SECRET",
            "https://host/v1?x-api-key=SECRET",
            "https://host/v1?signature=SECRET",
            "https://host/v1?sig=SECRET",
            "https://host/v1?cookie=SECRET",
        ):
            with self.subTest(base=base):
                with self.assertRaises(ValueError):
                    llm_retry.RetryProxy(base)

    def test_non_secret_query_is_still_preserved(self):
        proxy = llm_retry.RetryProxy("https://host/v1?api-version=2024")
        self.assertEqual(proxy.upstream_base, "https://host/v1?api-version=2024")
        self.assertEqual(proxy.target_path("/chat"), "/v1/chat?api-version=2024")


class EndpointKeyEnvPairingTests(unittest.TestCase):
    """F10: selection_env must be able to name the patch's apiKeyEnv."""

    def test_selection_env_honours_the_patch_override(self):
        selection = model_runtime.resolve_agent_selection(
            harness="dsh", model="https://host/v1", api_key="K"
        )
        patch = model_runtime.dsh_patch_text(
            selection, endpoint_api_key_env="OTHER_KEY"
        )
        env = model_runtime.selection_env(
            selection, endpoint_api_key_env="OTHER_KEY"
        )
        self.assertIn("apiKeyEnv: OTHER_KEY", patch)
        self.assertEqual(env["OTHER_KEY"], "K")
        self.assertEqual(
            model_runtime.selection_env(selection)[
                model_runtime.AGENT_MODEL_API_KEY_ENV
            ],
            "K",
        )


class ReprRecursionTests(unittest.TestCase):
    """F14: cyclic and very deep prep objects must not crash the repr."""

    def test_cyclic_prep_object_is_marked(self):
        cycle = {}
        cycle["self"] = cycle
        selection = model_runtime.AgentSelection(
            harness="dsh", model="qwen38-27b", prep_json=cycle
        )
        text = repr(selection)
        self.assertIn("[CYCLE]", text)

    def test_deep_prep_object_is_truncated(self):
        deep = {}
        cursor = deep
        for _ in range(2000):
            cursor["x"] = {}
            cursor = cursor["x"]
        redacted = model_runtime.redact_prep_object(deep)
        self.assertIn("[TRUNCATED]", json.dumps(redacted))

    def test_plain_prep_object_is_unchanged(self):
        self.assertEqual(
            model_runtime.redact_prep_object({"temperature": 0}), {"temperature": 0}
        )


class RetryProxyRegistryTests(unittest.TestCase):
    """F17: RetryProxy.stop() must remove the proxy from the registry."""

    def test_stop_removes_the_registry_and_base_alias(self):
        proxy, base = llm_retry.start_retry_proxy("http://127.0.0.1:9/v1")
        try:
            self.assertIsNotNone(proxy)
            self.assertIn(proxy, llm_retry._PROXY_REGISTRY)
            self.assertIn(base, llm_retry._PROXY_REGISTRY.by_base)
            proxy.stop()
            self.assertNotIn(proxy, llm_retry._PROXY_REGISTRY)
            self.assertNotIn(base, llm_retry._PROXY_REGISTRY.by_base)
        finally:
            if proxy is not None:
                proxy.stop()
            llm_retry.stop_retry_proxy()


class CatalogCapacityTypeTests(unittest.TestCase):
    """F29: a float/string capacity must be rejected, not truncated."""

    def _reload_with(self, mutate):
        original_path = model_runtime._MODEL_CATALOG_PATH
        document = json.loads(original_path.read_text(encoding="utf-8"))
        mutate(document)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            model_runtime._MODEL_CATALOG_PATH = path
            try:
                with self.assertRaises(model_runtime.ConfigError):
                    model_runtime.reload_model_catalog()
            finally:
                model_runtime._MODEL_CATALOG_PATH = original_path
                model_runtime.reload_model_catalog()

    def test_float_and_string_capacities_are_rejected(self):
        def mutate(document):
            for entry in document["models"]:
                if entry["id"] == "qwen3.8-27b":
                    entry["max_tokens"] = 32768.9
                    entry["context_window"] = "150000"

        self._reload_with(mutate)


class EndpointCatalogPayloadTests(unittest.TestCase):
    """F48: a malformed 200 /models payload is a failed probe, not no models."""

    def _payload(self, payload):
        class Handler(BaseHTTPRequestHandler):
            body = payload

            def log_message(self, *args):  # noqa: D401
                pass

            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", str(len(self.body)))
                self.end_headers()
                self.wfile.write(self.body)

        with _server(Handler) as server:
            return model_runtime._endpoint_catalog(
                "http://127.0.0.1:%d/v1" % server.server_address[1], "K"
            )

    def test_scalar_bodies_raise_probe_errors(self):
        for payload in (b'{"data":"oops"}', b"[]", b'{"data":5}', b"{}"):
            with self.subTest(payload=payload):
                with self.assertRaises(model_runtime._EndpointProbeError):
                    self._payload(payload)

    def test_valid_and_empty_listings_still_work(self):
        self.assertEqual(self._payload(b'{"data":[]}'), [])
        self.assertEqual(
            self._payload(b'{"data":[{"id":"m1"},{"no_id":true}]}'), [{"id": "m1"}]
        )


class CompletionBudgetNegativeReserveTests(unittest.TestCase):
    """F49: a negative reserve must not exceed the declared context window."""

    def test_negative_reserve_is_clamped(self):
        self.assertEqual(
            chat.completion_budget(1000, 100000, 0, reserve_tokens=-1000), 1000
        )
        self.assertEqual(
            chat.completion_budget(1000, 100000, 0, reserve_tokens=-10**9), 1000
        )
        self.assertEqual(
            chat.completion_budget(0, 4096, 10000, reserve_tokens=-5), 4096
        )


class PartialPolicyMergeTests(unittest.TestCase):
    """F54: public generators must merge DEFAULT_POLICY for partial policies."""

    def test_partial_and_none_sections_use_defaults(self):
        full = llm_retry.load_policy()
        for policy in ({"retry": {}}, {"retry": None, "dots": None, "compaction": None}):
            with self.subTest(policy=policy):
                text = llm_retry.retry_policy_yaml(policy)
                self.assertIn("maxRetries: 10", text)
                compaction = llm_retry.dots_compaction(policy)
                self.assertEqual(
                    compaction["compaction_provider"],
                    full["dots"]["compaction_provider"],
                )
                summary = llm_retry.compaction_basic_yaml(policy, "", "m")
                self.assertIn(
                    "summarizationProvider: %s"
                    % full["dots"]["compaction_provider"],
                    summary,
                )
                overlay = llm_retry.dots_dsh_overlay(
                    policy,
                    model="dots3-note-prev",
                    base_url="https://dots.example/v1",
                    context_window=1000,
                    max_tokens=100,
                )
                self.assertIn("compaction-basic", overlay)


if __name__ == "__main__":
    unittest.main()
