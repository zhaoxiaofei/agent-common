"""Tests for the shared dsh argv construction."""

from __future__ import annotations

import unittest

from agent_common.agent_cli import build_dsh_argv


class DshArgvTests(unittest.TestCase):
    def test_launcher_flags_precede_app_flags(self):
        argv = build_dsh_argv(
            dsh_bin="/usr/bin/dsh",
            patch_path="/tmp/model.yml",
            extra_patch_path="/tmp/mcp.yml",
            extra_args=["--foo", "bar"],
            json_stream=True,
        )
        self.assertEqual(
            argv,
            [
                "/usr/bin/dsh",
                "--profile",
                "headless",
                "--patch",
                "/tmp/model.yml",
                "--patch",
                "/tmp/mcp.yml",
                "--json",
                "--foo",
                "bar",
            ],
        )

    def test_optional_layers_can_be_omitted(self):
        self.assertEqual(build_dsh_argv(json_stream=False), ["dsh", "--profile", "headless"])


if __name__ == "__main__":
    unittest.main()
