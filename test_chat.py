"""Tests for the shared chat budgeting/response helpers."""

from __future__ import annotations

import unittest

from agent_common.chat import (
    CONTEXT_RESERVE_TOKENS,
    completion_budget,
    estimated_prompt_tokens,
    finish_reason_error,
    message_text,
)


class BudgetTests(unittest.TestCase):
    def test_prompt_estimate_uses_the_conservative_ratio(self):
        self.assertEqual(estimated_prompt_tokens(340), 100)
        self.assertEqual(estimated_prompt_tokens(341), 101)
        self.assertEqual(estimated_prompt_tokens(-5), 0)

    def test_no_declared_window_leaves_the_output_cap(self):
        self.assertEqual(completion_budget(0, 4096, 10_000), 4096)

    def test_the_answer_is_capped_by_the_remaining_room(self):
        # 1000 chars -> 295 prompt tokens; 10000 - 295 - 2048 = 7657 room.
        self.assertEqual(completion_budget(10_000, 16_000, 1_000), 7_657)
        self.assertEqual(
            completion_budget(10_000, 16_000, 1_000, reserve_tokens=0),
            10_000 - estimated_prompt_tokens(1_000),
        )

    def test_a_prompt_that_cannot_fit_has_no_budget(self):
        self.assertEqual(completion_budget(1_000, 4_096, 10_000), 0)

    def test_the_reserve_default_is_exposed(self):
        self.assertEqual(CONTEXT_RESERVE_TOKENS, 2048)


class ResponseTests(unittest.TestCase):
    def test_message_text_accepts_a_string_or_typed_parts(self):
        self.assertEqual(message_text("hello"), "hello")
        self.assertEqual(
            message_text(
                [
                    {"type": "text", "text": "a"},
                    {"type": "image", "text": "ignored"},
                    {"type": "text", "text": "b"},
                ]
            ),
            "ab",
        )
        self.assertEqual(message_text(None), "")
        self.assertEqual(message_text({"type": "text"}), "")

    def test_length_and_unknown_finish_reasons_are_errors(self):
        self.assertIn("truncated", finish_reason_error("length", 99))
        self.assertIn("did not finish normally", finish_reason_error("content_filter", 99))

    def test_normal_finish_reasons_are_accepted(self):
        for reason in (None, "stop", "end_turn", "stop_sequence", "tool_calls"):
            self.assertEqual(finish_reason_error(reason, 99), "")

    def test_a_non_string_finish_reason_is_a_controlled_error(self):
        self.assertIn(
            "did not finish normally", finish_reason_error(["length"], 99)
        )


if __name__ == "__main__":
    unittest.main()
