"""Argv construction shared by the checkouts that launch agent CLIs.

GDP's runner and paper-refine's provider adapters both launch the same
non-interactive ``dsh`` profile.  The launcher parses only its own flags and
hands everything after the first unrecognized token to the booted profile, so
the launcher flags (``--profile``, ``--patch``) must come first and the app's
flags follow.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

#: The non-interactive dsh profile used for one task / final answer.
DSH_DEFAULT_PROFILE = "headless"


def build_dsh_argv(
    *,
    dsh_bin: str = "dsh",
    profile: str = DSH_DEFAULT_PROFILE,
    patch_path: Optional[str] = None,
    extra_patch_path: Optional[str] = None,
    extra_args: Sequence[str] = (),
    json_stream: bool = True,
) -> List[str]:
    """argv for one non-interactive ``dsh`` run (one task, final answer, exit).

    Example:
        ``build_dsh_argv(patch_path="/tmp/model.yml")`` returns
        ``["dsh", "--profile", "headless", "--patch", "/tmp/model.yml", "--json"]``.

    Counterexample:
        Do not put application flags in ``patch_path``; the launcher would treat
        them as a file name.  Put application flags in ``extra_args``.
    """
    argv: List[str] = [dsh_bin]
    if profile:
        argv += ["--profile", str(profile)]
    if patch_path:
        # Later patch layers win, so the overlay outranks the profile's own model selection.
        argv += ["--patch", str(patch_path)]
    if extra_patch_path:
        # A second overlay (for example a per-session MCP disable list) is applied after the
        # model overlay; it targets different row ids, so the order does not change the model.
        argv += ["--patch", str(extra_patch_path)]
    if json_stream:
        # NDJSON on stdout (session/text/tool/final events) instead of the bare final answer.
        argv.append("--json")
    argv += [str(item) for item in extra_args]
    return argv
