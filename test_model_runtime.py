"""Tests for the shared model-runtime catalog policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_common import model_runtime


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


if __name__ == "__main__":
    unittest.main()
