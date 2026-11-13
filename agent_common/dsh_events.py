"""Parsing helpers for the ``dsh --json`` event stream.

Both GDP (``gdp.runner``) and the paper-refine adapters consume this stream.
The accepted event shapes and the success allowlist live here so one checkout
cannot quietly accept a different terminal state than the other.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

#: Turn-end statuses that mean the turn completed.  This is an allowlist: a
#: cancelled, interrupted, aborted or future non-success status is incomplete.
SUCCESS_STATUSES = frozenset(
    {"ok", "success", "succeeded", "completed", "complete", "done", "finished"}
)

#: Reason kinds that are unambiguously failures.
FAILURE_KINDS = frozenset({"error", "failed"})


def parse_event(line: Any) -> Optional[dict]:
    """The JSON object on one NDJSON line, or ``None`` for a non-event line.

    Example:
        ``parse_event(b'{"type":"status","phase":"turn_end"}')`` returns the dict.

    Counterexample:
        A JSON array or a plain text line returns ``None``; "not an event" must not
        be treated as a completed turn.
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = str(line or "")
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        event = json.loads(text)
    except (ValueError, TypeError):
        return None
    return event if isinstance(event, dict) else None


def is_turn_end(event: Any) -> bool:
    """Whether ``event`` is one of the two turn-end shapes dsh has emitted.

    Example:
        Both ``{"type":"turn_end"}`` and
        ``{"type":"status","phase":"turn_end"}`` return ``True``.

    Counterexample:
        A ``text`` or ``session`` event returns ``False``; this is a turn-end
        check, not a generic JSON-event check.
    """
    if not isinstance(event, Mapping):
        return False
    phase = str(event.get("phase") or "").lower()
    return event.get("type") == "turn_end" or (
        event.get("type") == "status" and phase == "turn_end"
    )


def turn_end_failed(event: Any) -> Optional[bool]:
    """``True``/``False`` for a turn-end event, ``None`` for every other event.

    Example:
        A turn-end with ``status="error"`` or ``reason.kind="error"`` returns
        ``True``; an ``ok`` turn returns ``False``.

    Counterexample:
        A non-turn-end event returns ``None``, not ``False``; callers must handle
        the three-valued result.
    """
    if not is_turn_end(event):
        return None
    assert isinstance(event, Mapping)
    status = str(event.get("status") or "").lower()
    reason = event.get("reason")
    reason_kind = ""
    if isinstance(reason, Mapping):
        reason_kind = str(reason.get("kind") or "").lower()
    kind = str(event.get("kind") or reason_kind).lower()
    # Any explicit non-success marker is a failure, even when the other field
    # says ok: a cancelled turn that also carries a stale ``status: ok`` is not
    # a completed turn.  Success therefore needs positive evidence in every
    # field that is present, and a turn-end with no status/kind at all is not
    # treated as success either (SUCCESS_STATUSES is an allowlist).
    if kind and kind not in SUCCESS_STATUSES:
        return True
    if status and status not in SUCCESS_STATUSES:
        return True
    if not status and not kind:
        return True
    return False


def line_turn_end_failed(line: Any) -> Optional[bool]:
    """``turn_end_failed`` for one raw NDJSON line.

    Example:
        ``line_turn_end_failed(b'{"type":"turn_end","status":"ok"}')`` returns
        ``False``.

    Counterexample:
        A malformed or truncated line returns ``None``; an unparsable completion
        must not become a success.
    """
    event = parse_event(line)
    if event is None:
        return None
    return turn_end_failed(event)


def turn_end_error(reason: Any) -> str:
    """Message of a dsh turn-end reason that describes an error, else ``""``.

    Example:
        ``turn_end_error({"kind":"error","error":{"message":"boom"}}) == "boom"``.

    Counterexample:
        A non-failure reason such as ``{"kind":"completed"}`` returns ``""``; the
        caller adds its own fallback text.
    """
    if not isinstance(reason, Mapping):
        return ""
    kind = str(reason.get("kind") or "").strip().lower()
    if kind not in FAILURE_KINDS and not reason.get("error"):
        return ""
    error = reason.get("error")
    if isinstance(error, Mapping):
        text = str(error.get("message") or error.get("code") or "")
    elif error:
        text = str(error)
    else:
        text = ""
    return text or str(reason.get("message") or "") or "turn ended with an error"
