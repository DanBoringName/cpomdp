"""Refuse a change that rewrites a line already landed in a protected file.

Some files here are records of what was known when. What a reader gets from them is the
ability to tell a claim made before a result existed from one made after, and an
in-place edit destroys that silently. No diff restores it. Corrections, qualifications,
retractions and new results go in as a new dated entry appended in the file's own
convention.

Landed means reached the branch this change will land on. A line added and then reworked
on a branch was never visible to anyone, so the comparison point is the merge base
rather than `HEAD`. Drafting across several commits is left alone.

Two callers, one rule. The `commit-msg` hook in `.pre-commit-config.yaml` sees the index
and the message about to be recorded. The `append-only` CI job sees the whole pull
request. Both run `python protected_files.py [<message-file>]`.

A typo or broken markdown is the documented exception, and even then not to a number, a
date, a claim or a result. Put `[in-place]` in the commit subject to take it. The marker
is read from the message being written and from every commit already on the branch, so
one marked commit clears the branch rather than obliging every commit after it to carry
the marker too. CI reads the same subjects, which is what stops the exception passing
the hook and then failing the job.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROTECTED_PATHS = ("DECISIONS.md", "research/gate_d4_registration.md")
"""The files whose landed lines may not be rewritten.

The one place this list lives. Both the hook and the CI job read it from here, and
`CONTRIBUTING.md` names the file rather than restating the paths. Every entry has to be
tracked: a pathspec matching nothing makes `git diff` exit zero without complaining, so
a path that moved would be guarded by an entry that quietly does nothing. Adding a
record belongs in the change that lands the record.
"""

IN_PLACE_MARKER = "[in-place]"
"""What a commit subject carries to take the typo exception."""

LANDED_REFS = ("origin/main", "main")
"""Where to look for the landed state when nothing else names it, first match wins."""

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
    """Run git and hand back the finished process, successful or not.

    The encoding is pinned rather than read from the locale. `DECISIONS.md` carries real
    mathematical notation, so under `LC_ALL=C` a diff of it would decode as ASCII and
    the hook would die where it is meant to return a verdict.
    """
    return subprocess.run(
        ("git", *arguments),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
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

    Header lines are only read as headers outside a hunk. Inside one, a leading `---` is
    a removed line whose own text began with `--`, and reading it as a header would
    silently retarget everything after it.

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


def landed_refs() -> tuple[str, ...]:
    """Candidate names for the landed state, in the order to try them.

    A pull request's base branch is what the change will land on, and it is not always
    `main`. A stacked branch targets the one below it, whose lines are as landed as
    anything for the purposes of this check. GitHub names it in `GITHUB_BASE_REF`.

    Returns:
        The candidates, most specific first.
    """
    base_ref = os.environ.get("GITHUB_BASE_REF", "").strip()
    if not base_ref:
        return LANDED_REFS
    return (f"origin/{base_ref}", base_ref, *LANDED_REFS)


def landed_base(refs: tuple[str, ...] | None = None) -> str | None:
    """The merge base with the first reachable landed ref.

    Args:
        refs: candidate names, or `None` to take them from `landed_refs`.

    Returns:
        The merge-base commit, or `None` when no candidate is reachable.
    """
    for ref in refs if refs is not None else landed_refs():
        finished = _run_git("merge-base", "HEAD", ref)
        if finished.returncode == 0:
            return finished.stdout.strip()
    return None


def missing_paths(paths: tuple[str, ...] = PROTECTED_PATHS) -> list[str]:
    """Declared paths that git is not tracking.

    A pathspec matching nothing is not an error to `git diff`, so a protected file that
    was renamed or never added would be guarded by an entry doing nothing at all.

    Args:
        paths: the declared paths.

    Returns:
        The ones not in the index, in declared order.
    """
    tracked = set(_git("ls-files", "--", *paths).splitlines())
    return [path for path in paths if path not in tracked]


def staged_diff(base: str | None, paths: tuple[str, ...] = PROTECTED_PATHS) -> str:
    """The zero-context diff of the index against `base`, over `paths` only.

    Rename detection is not asked for. The pathspec names the source and not the
    destination, so a protected file moved out from under it reads as a whole-file
    deletion however the diff is configured. Failing on that is the intended answer.

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


def branch_subjects(base: str | None) -> list[str]:
    """The subject of every commit made since `base`.

    An exception taken on one commit holds for the branch. The diff against the merge
    base is cumulative, so without this a marked in-place edit would oblige every commit
    after it to carry the marker as well, and a plain append would have to be labelled
    an in-place edit to get itself recorded.

    Args:
        base: the merge base, or `None` when none was found.

    Returns:
        The subjects, newest first, or empty when there is no base.
    """
    if base is None:
        return []
    return _git("log", "--format=%s", f"{base}..HEAD").splitlines()


def override_requested(subjects: list[str]) -> bool:
    """Whether any commit subject takes the typo exception."""
    return any(IN_PLACE_MARKER in subject for subject in subjects)


def _report(removals: list[Removal], base: str | None) -> None:
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
        "superseded\nline keeps its place, and the new entry says what replaced it "
        "and why."
    )
    print(
        f"\nA typo or broken markdown is the exception, and even then not to a "
        f"number,\na date, a claim or a result. Put {IN_PLACE_MARKER} in the subject "
        f"of any commit\non this branch to take it."
    )


def main(argv: list[str] | None = None) -> int:
    """Check the index, and report anything it would rewrite.

    Args:
        argv: command-line arguments. A single optional path to a commit message file,
            which is what the `commit-msg` hook passes.

    Returns:
        A process exit status: zero when nothing landed is rewritten.
    """
    arguments = sys.argv[1:] if argv is None else argv
    if _run_git("rev-parse", "--verify", "HEAD").returncode != 0:
        return 0

    absent = missing_paths()
    if absent:
        print(
            f"protected files: {', '.join(absent)} is declared in "
            f"{Path(__file__).name} and not tracked. A pathspec matching nothing "
            "guards nothing and says nothing about it. Correct the list, or add the "
            "file in the change that needs it protected."
        )
        return 1

    base = landed_base()
    if base is None and not arguments:
        print(
            f"protected files: none of {', '.join(landed_refs())} is reachable, so "
            "nothing was compared and nothing was checked. Fetch the base branch "
            "before running this."
        )
        return 1

    removals = parse_removals(staged_diff(base))
    if not removals:
        return 0

    subjects = branch_subjects(base)
    if arguments:
        subjects.append(commit_subject(Path(arguments[0]).read_text(encoding="utf-8")))
    if override_requested(subjects):
        print(
            f"protected files: {len(removals)} landed line(s) rewritten, allowed by "
            f"{IN_PLACE_MARKER} in a commit subject on this branch."
        )
        return 0

    _report(removals, base)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
