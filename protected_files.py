"""Refuse a change that rewrites a line already landed in a protected file.

Three files here are records of what was known when: `DECISIONS.md`,
`research/gate_d4_registration.md` and `research/spinello_stilwell_rung.md`. Their
worth is that a reader can tell a claim made before a result existed from one made
after, and an in-place edit destroys that silently. Corrections, qualifications and
new results go in as a new dated entry appended in the file's own convention.

Landed means reached `main`. A line added and then reworked on a branch was never
visible to anyone, so the comparison point is the merge base with `main` rather than
`HEAD`. Drafting across several commits on a branch is left alone.

Two callers, one rule. The `commit-msg` hook in `.pre-commit-config.yaml` sees the
index and the message about to be recorded. The `append-only` CI job sees the whole
pull request. Both run `python protected_files.py [<message-file>]`, and without a
message file there is no override, which is the CI case.

A typo or broken markdown is the documented exception, and even then not to a
number, a date, a claim or a result. Put `[in-place]` in the commit subject to take
it. That leaves the exception in the log where a reviewer meets it, which
`--no-verify` does not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROTECTED_PATHS = (
    "DECISIONS.md",
    "research/gate_d4_registration.md",
    "research/spinello_stilwell_rung.md",
)
"""The files whose landed lines may not be rewritten.

The one place this list lives. Both the hook and the CI job read it from here, and
`CONTRIBUTING.md` names the file rather than restating the paths.
"""

IN_PLACE_MARKER = "[in-place]"
"""What a commit subject carries to take the typo exception."""

LANDED_REFS = ("origin/main", "main")
"""Where to look for the landed state, first match wins."""

REPORT_LIMIT = 10
"""How many rewritten lines to print before saying how many more there were."""

_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


class GitFailed(RuntimeError):
    """A git command this check depends on did not succeed."""


@dataclass(frozen=True)
class Removal:
    """One line the staged change would take out of a protected file."""

    path: str
    line_number: int
    text: str


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run git and hand back the finished process, successful or not."""
    return subprocess.run(
        ("git", *arguments), capture_output=True, text=True, check=False
    )


def _git(*arguments: str) -> str:
    """Run git and return its output, raising on a non-zero exit."""
    finished = _run_git(*arguments)
    if finished.returncode != 0:
        raise GitFailed(f"git {' '.join(arguments)}: {finished.stderr.strip()}")
    return finished.stdout


def _diff_path(raw: str) -> str:
    """The repository path a `---`/`+++` header names, empty for `/dev/null`."""
    stripped = raw.strip()
    if stripped == "/dev/null":
        return ""
    prefix, _, rest = stripped.partition("/")
    return rest if prefix in {"a", "b"} else stripped


def parse_removals(diff: str) -> list[Removal]:
    """Every line a zero-context unified diff removes, with where it stood.

    Header lines are only read as headers outside a hunk. Inside one, a leading
    `---` is a removed line whose own text began with `--`, and reading it as a
    header would silently retarget everything after it.

    Args:
        diff: the output of `git diff -U0`.

    Returns:
        One `Removal` per removed line, in the order the diff gives them.
    """
    removals: list[Removal] = []
    path = ""
    old_line = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = "", False
            continue
        hunk = _HUNK.match(line)
        if hunk:
            old_line, in_hunk = int(hunk.group(1)), True
            continue
        if not in_hunk:
            if line.startswith("--- ") or (line.startswith("+++ ") and not path):
                path = _diff_path(line[4:])
            continue
        if line.startswith("-"):
            removals.append(Removal(path, old_line, line[1:]))
            old_line += 1
    return removals


def landed_base(refs: tuple[str, ...] = LANDED_REFS) -> str | None:
    """The merge base with the first reachable landed ref.

    Args:
        refs: candidate names for the landed state, tried in order.

    Returns:
        The merge-base commit, or `None` when no candidate is reachable. `None`
        means the comparison falls back to `HEAD`, which is stricter.
    """
    for ref in refs:
        finished = _run_git("merge-base", "HEAD", ref)
        if finished.returncode == 0:
            return finished.stdout.strip()
    return None


def staged_diff(base: str | None, paths: tuple[str, ...] = PROTECTED_PATHS) -> str:
    """The zero-context diff of the index against `base`, over `paths` only.

    Args:
        base: the commit to compare against, or `None` for `HEAD`.
        paths: the files to restrict the diff to.

    Returns:
        Unified diff text, empty when nothing under `paths` changed.
    """
    arguments = [
        "diff",
        "--cached",
        "-U0",
        "-M",
        "--no-ext-diff",
        "--no-color",
        "--src-prefix=a/",
        "--dst-prefix=b/",
    ]
    if base is not None:
        arguments.append(base)
    arguments.extend(["--", *paths])
    return _git(*arguments)


def commit_subject(message: str) -> str:
    """The first line of a commit message that is neither blank nor a comment."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def override_requested(message: str) -> bool:
    """Whether the commit subject takes the typo exception."""
    return IN_PLACE_MARKER in commit_subject(message)


def _report(removals: list[Removal], base: str | None, overridable: bool) -> None:
    """Print the rewritten lines and what the author is expected to do instead."""
    against = base[:12] if base is not None else "HEAD"
    print(f"Protected files are append-only. Compared against {against}.")
    print(f"{len(removals)} landed line(s) would be rewritten:\n")
    for removal in removals[:REPORT_LIMIT]:
        print(f"  {removal.path}:{removal.line_number}  {removal.text}")
    if len(removals) > REPORT_LIMIT:
        print(f"  ... and {len(removals) - REPORT_LIMIT} more")
    print(
        "\nAppend a new dated entry in the file's own convention instead. The "
        "superseded\nline keeps its place, and the new entry says what replaced "
        "it and why."
    )
    if overridable:
        print(
            f"\nA typo or broken markdown is the exception, and even then not to "
            f"a number,\na date, a claim or a result. Put {IN_PLACE_MARKER} in "
            f"the commit subject to take it."
        )


def main(argv: list[str] | None = None) -> int:
    """Check the index, and report anything it would rewrite.

    Args:
        argv: command-line arguments. A single optional path to a commit message
            file, which is what the `commit-msg` hook passes.

    Returns:
        A process exit status: zero when nothing landed is rewritten.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if _run_git("rev-parse", "--verify", "HEAD").returncode != 0:
        return 0

    base = landed_base()
    removals = parse_removals(staged_diff(base))
    if not removals:
        return 0

    if arguments:
        message = Path(arguments[0]).read_text(encoding="utf-8")
        if override_requested(message):
            print(
                f"protected files: {len(removals)} landed line(s) rewritten, "
                f"allowed by {IN_PLACE_MARKER} in the subject."
            )
            return 0

    _report(removals, base, overridable=bool(arguments))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
