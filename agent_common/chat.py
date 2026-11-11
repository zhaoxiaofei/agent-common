"""Provider-neutral chat-completion budgeting and response parsing.

paper-refine's direct adapters and paper-refine-eval's ``pr_eval.llm`` both
estimate prompt tokens, reserve room for the answer, and validate the
provider's terminal state.  Keeping the arithmetic and the allowlist here
prevents the two checkouts from drifting on a context-window edge case.
"""

from __future__ import annotations

import math

#: Code, JSON and tool output run closer to 3.4 characters per token than to
#: four; dividing by four underestimates a long agent prompt.
CHARS_PER_TOKEN = 3.4

#: Room reserved for protocol overhead and the provider's own formatting.
CONTEXT_RESERVE_TOKENS = 2048

#: finish_reason values that mean the answer is complete.
ACCEPTED_FINISH_REASONS = frozenset(
    {None, "stop", "end_turn", "stop_sequence", "tool_calls", "function_call"}
)


def estimated_prompt_tokens(chars: int) -> int:
    """A conservative input-token estimate from a character count.

    Example:
        ``estimated_prompt_tokens(340) == 100``.

    Counterexample:
        Dividing code/tool prompts by four underestimates them; this helper uses
        3.4 characters per token on purpose.  Negative counts return 0 instead of
        raising.
    """
    return int(math.ceil(max(0, chars) / CHARS_PER_TOKEN))


def completion_budget(
    context_window: int,
    output_cap: int,
    prompt_chars: int,
    *,
    reserve_tokens: int = CONTEXT_RESERVE_TOKENS,
    minimum: int = 1,
) -> int:
    """The usable output budget, or ``0`` when the prompt cannot fit.

    ``context_window == 0`` means the caller supplied no window (older
    regression fixtures do); in that case the output cap is the only limit.

    Example:
        With a 10,000-token window, a 16,000-token cap and a 1,000-character
        prompt, ``completion_budget(...)`` returns 7,657.

    Counterexample:
        ``context_window=0`` does not mean "no room"; it means the caller supplied
        no window, so the output cap is returned unchanged.
    """
    context = int(context_window or 0)
    cap = int(output_cap or 0)
    if not context:
        return max(minimum, cap)
    room = context - estimated_prompt_tokens(prompt_chars) - int(reserve_tokens)
    if room <= 0:
        return 0
    return max(minimum, min(cap, room))


def message_text(content: object) -> str:
    """The text of an OpenAI-style message ``content`` value.

    Providers may return a string or a list of typed content parts.  A
    non-text part (an image, a tool object) is not an answer and is ignored.

    Example:
        ``message_text([{"type":"text","text":"a"}, {"type":"text","text":"b"}])``
        returns ``"ab"``.

    Counterexample:
        A dict, ``None`` or an image-only part returns ``""``; those values are not
        answers and must not be ``str()``-ified into one.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def finish_reason_error(finish_reason: object, max_tokens: int) -> str:
    """Why ``finish_reason`` is not a deliverable answer, or ``""`` when it is.

    ``length`` is a max_tokens prefix and every unknown reason is a partial
    answer the provider refused to finish; neither may be returned as success.

    Example:
        ``finish_reason_error("stop", 4096) == ""``.

    Counterexample:
        ``"length"`` means the answer is a truncated prefix, and an unknown or
        non-string reason is also an error; neither may be returned as success.
    """
    if finish_reason is not None and not isinstance(finish_reason, str):
        return (
            "the chat completion did not finish normally "
            "(finish_reason=%r): the answer is incomplete" % (finish_reason,)
        )
    if finish_reason == "length":
        return (
            "the chat completion was truncated (finish_reason='length', "
            "max_tokens=%d): the answer is incomplete" % max_tokens
        )
    if finish_reason not in ACCEPTED_FINISH_REASONS:
        return (
            "the chat completion did not finish normally "
            "(finish_reason=%r): the answer is incomplete" % (finish_reason,)
        )
    return ""
