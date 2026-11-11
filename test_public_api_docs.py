"""Every public agent_common callable must document usage and misuse."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import unittest

import agent_common

# Discover every module in the package so a newly exported helper cannot evade
# the documentation contract.
MODULES = tuple(
    importlib.import_module(f"agent_common.{module.name}")
    for module in pkgutil.iter_modules(agent_common.__path__)
    if not module.name.startswith("_")
)


def public_functions(module):
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield name, obj
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for method_name, method in vars(obj).items():
                if method_name.startswith("_") or isinstance(method, property):
                    continue
                if inspect.isfunction(method):
                    yield f"{name}.{method_name}", method


class PublicApiDocumentationTests(unittest.TestCase):
    def test_every_public_function_has_examples_and_counterexamples(self):
        missing = []
        for module in MODULES:
            for name, function in public_functions(module):
                doc = inspect.getdoc(function) or ""
                if "Example" not in doc or "Counterexample" not in doc:
                    missing.append(f"{module.__name__}.{name}")
                if len(doc) < 120:
                    missing.append(f"{module.__name__}.{name} (too short)")
        self.assertEqual(missing, [], "undocumented public API: %r" % missing)

    def test_every_public_class_has_a_docstring(self):
        missing = []
        for module in MODULES:
            for name, obj in inspect.getmembers(module):
                if (
                    not name.startswith("_")
                    and inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                    and not (inspect.getdoc(obj) or "").strip()
                ):
                    missing.append(f"{module.__name__}.{name}")
        self.assertEqual(missing, [], "undocumented public classes: %r" % missing)


if __name__ == "__main__":
    unittest.main()
