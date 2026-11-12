"""Tests for the repository-level consistency check scripts."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_cross_repo_consistency
import check_llm_policy
import check_model_catalog
import check_shared_package


class CrossRepoProbeTests(unittest.TestCase):
    def _canonical_observation(self):
        observation = {}
        for model_id in check_cross_repo_consistency.CATALOGUE:
            observation[model_id] = {
                field: check_cross_repo_consistency.CATALOGUE[model_id].get(field)
                for field in check_cross_repo_consistency.expected_fields(model_id)
            }
        return observation

    def _run(self, observed):
        completed = mock.Mock(returncode=0, stdout="OK", stderr="")
        output = io.StringIO()
        with mock.patch.object(
            check_cross_repo_consistency.subprocess, "run", return_value=completed
        ), mock.patch.object(
            check_cross_repo_consistency, "probe", return_value=observed
        ):
            with contextlib.redirect_stdout(output):
                code = check_cross_repo_consistency.main()
        return code, output.getvalue()

    def test_a_probe_that_cannot_run_is_a_failed_check(self):
        code, output = self._run(None)
        self.assertEqual(code, 1)
        self.assertIn("gdp probe failed", output)
        self.assertNotIn("OK gdp", output)

    def test_a_missing_model_is_a_problem(self):
        observed = self._canonical_observation()
        del observed["deepseek-flash"]
        code, output = self._run(observed)
        self.assertEqual(code, 1)
        self.assertIn("missing model deepseek-flash", output)

    def test_a_missing_field_is_a_problem(self):
        observed = self._canonical_observation()
        del observed["qwen3.8-27b"]["max_tokens"]
        code, output = self._run(observed)
        self.assertEqual(code, 1)
        self.assertIn("qwen3.8-27b is missing max_tokens", output)

    def test_a_wrong_value_is_still_a_problem(self):
        observed = self._canonical_observation()
        observed["qwen3.8-27b"]["context_window"] = 1
        code, output = self._run(observed)
        self.assertEqual(code, 1)
        self.assertIn("catalog says", output)

    def test_a_complete_observation_passes(self):
        code, output = self._run(self._canonical_observation())
        self.assertEqual(code, 0)
        self.assertIn("OK gdp, paper-refine-eval and paper-refine", output)


class ModelCatalogCheckTests(unittest.TestCase):
    def _document(self):
        return json.loads(
            check_model_catalog.CANONICAL.read_text(encoding="utf-8")
        )

    def _run(self, document):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "model_catalog.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        output = io.StringIO()
        with mock.patch.object(
            check_model_catalog, "CANONICAL", path
        ), mock.patch.object(check_model_catalog, "COPIES", []):
            with contextlib.redirect_stdout(output):
                code = check_model_catalog.main()
        return code, output.getvalue()

    def test_a_catalog_missing_an_expected_model_fails(self):
        document = self._document()
        document["models"] = [
            entry
            for entry in document["models"]
            if entry["id"] != "qwen3.8-27b"
        ]
        code, output = self._run(document)
        self.assertEqual(code, 1)
        self.assertIn("missing expected model id qwen3.8-27b", output)

    def test_a_catalog_with_an_extra_model_fails(self):
        document = self._document()
        extra = json.loads(json.dumps(document["models"][0]))
        extra["id"] = "extra-model"
        extra["name"] = "extra-model"
        document["models"].append(extra)
        code, output = self._run(document)
        self.assertEqual(code, 1)
        self.assertIn("unexpected model id extra-model", output)

    def test_the_canonical_catalog_passes(self):
        code, output = self._run(self._document())
        self.assertEqual(code, 0)
        self.assertIn("OK 7 models", output)


class LlmPolicyCheckTests(unittest.TestCase):
    def test_a_missing_adapter_in_an_initialized_checkout_fails(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "paper-refine").mkdir()
        output = io.StringIO()
        with mock.patch.object(check_llm_policy, "ROOT", root), mock.patch.object(
            check_llm_policy, "COPIES", ()
        ):
            with contextlib.redirect_stdout(output):
                code = check_llm_policy.main()
        self.assertEqual(code, 1)
        self.assertIn("missing adapter", output.getvalue())


class SharedPackageCheckTests(unittest.TestCase):
    def _tree(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        canonical = root / "canonical"
        canonical.mkdir()
        (canonical / "model_runtime.py").write_text("runtime", encoding="utf-8")
        (canonical / "chat.py").write_text("chat", encoding="utf-8")
        copies = []
        for index in range(2):
            copy = root / ("copy%d" % index) / "agent_common"
            copy.mkdir(parents=True)
            for name in ("model_runtime.py", "chat.py"):
                shutil.copy2(canonical / name, copy / name)
            cache = copy / "__pycache__"
            cache.mkdir()
            (cache / "chat.cpython-312.pyc").write_bytes(b"compiled")
            copies.append(copy)
        return canonical, copies

    def _run(self, canonical, copies):
        output = io.StringIO()
        with mock.patch.object(
            check_shared_package, "CANONICAL_DIR", canonical
        ), mock.patch.object(check_shared_package, "COPIES", tuple(copies)):
            with contextlib.redirect_stdout(output):
                code = check_shared_package.main()
        return code, output.getvalue()

    def test_every_shared_file_is_compared(self):
        canonical, copies = self._tree()
        code, output = self._run(canonical, copies)
        self.assertEqual(code, 0)
        self.assertIn("OK 2 shared package files", output)

    def test_a_modified_shared_module_is_detected(self):
        canonical, copies = self._tree()
        (copies[1] / "chat.py").write_text("drifted", encoding="utf-8")
        code, output = self._run(canonical, copies)
        self.assertEqual(code, 1)
        self.assertIn("copy differs from canonical", output)
        self.assertIn("chat.py", output)

    def test_a_missing_shared_module_is_detected(self):
        canonical, copies = self._tree()
        (copies[0] / "model_runtime.py").unlink()
        code, output = self._run(canonical, copies)
        self.assertEqual(code, 1)
        self.assertIn("missing copy", output)
        self.assertIn("model_runtime.py", output)


if __name__ == "__main__":
    unittest.main()
