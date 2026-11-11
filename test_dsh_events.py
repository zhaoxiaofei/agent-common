"""Tests for the shared dsh --json event parsing."""

from __future__ import annotations

import unittest

from agent_common.dsh_events import (
    line_turn_end_failed,
    parse_event,
    turn_end_error,
    turn_end_failed,
)


class ParseEventTests(unittest.TestCase):
    def test_a_json_object_line_parses_from_bytes_and_text(self):
        line = b'{"type":"status","phase":"turn_end","status":"ok"}'
        self.assertEqual(parse_event(line)["status"], "ok")
        self.assertEqual(parse_event(line.decode())["status"], "ok")

    def test_non_event_lines_do_not_parse(self):
        for line in (b"", b"not json", b"[1,2]", b'"text"'):
            self.assertIsNone(parse_event(line))


class TurnEndTests(unittest.TestCase):
    def test_the_old_turn_end_shape(self):
        self.assertIs(turn_end_failed({"type": "turn_end", "status": "ok"}), False)
        self.assertIs(turn_end_failed({"type": "turn_end", "status": "error"}), True)
        self.assertIs(
            turn_end_failed({"type": "turn_end", "status": "cancelled"}), True
        )

    def test_the_current_status_phase_shape(self):
        event = {
            "type": "status",
            "phase": "turn_end",
            "reason": {"kind": "completed"},
        }
        self.assertIs(turn_end_failed(event), False)
        event["reason"] = {"kind": "error"}
        self.assertIs(turn_end_failed(event), True)

    def test_non_turn_end_events_are_none(self):
        self.assertIsNone(turn_end_failed({"type": "text"}))
        self.assertIsNone(line_turn_end_failed(b"not json"))
        self.assertIsNone(line_turn_end_failed(b'{"type":"text"}'))

    def test_no_turn_end_event_is_not_success(self):
        # Callers treat a missing turn_end separately; None is not False.
        self.assertIsNone(line_turn_end_failed(b'{"type":"session"}'))


class TurnEndErrorTests(unittest.TestCase):
    def test_error_message_prefers_the_error_object(self):
        self.assertEqual(
            turn_end_error({"kind": "error", "error": {"message": "boom"}}), "boom"
        )
        self.assertEqual(
            turn_end_error({"kind": "error", "error": {"code": "E1"}}), "E1"
        )

    def test_string_error_and_reason_message_are_used(self):
        self.assertEqual(turn_end_error({"kind": "error", "error": "bad"}), "bad")
        self.assertEqual(turn_end_error({"kind": "error", "message": "why"}), "why")

    def test_non_error_reason_is_empty(self):
        self.assertEqual(turn_end_error({"kind": "completed"}), "")
        self.assertEqual(turn_end_error(None), "")


if __name__ == "__main__":
    unittest.main()
