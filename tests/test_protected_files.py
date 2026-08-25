"""The append-only check over `DECISIONS.md` and the two research records."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from protected_files import (
    IN_PLACE_MARKER,
    PROTECTED_PATHS,
    commit_subject,
    main,
    override_requested,
    parse_removals,
)

INSERTION = """\
diff --git a/DECISIONS.md b/DECISIONS.md
--- a/DECISIONS.md
+++ b/DECISIONS.md
@@ -120,0 +121,2 @@
+### ADR-099
+Something new.
"""

REWRITE = """\
diff --git a/DECISIONS.md b/DECISIONS.md
--- a/DECISIONS.md
+++ b/DECISIONS.md
@@ -42 +42 @@
-The gap is 3.2e-14.
+The gap is 3.2e-13.
"""

DELETION = """\
diff --git a/research/gate_d4_registration.md b/research/gate_d4_registration.md
--- a/research/gate_d4_registration.md
+++ /dev/null
@@ -1,2 +0,0 @@
-# Registration
-Bar: 0.05 nats.
"""

# A removed line whose own text starts with `--` reaches the parser as `--- ...`,
# which is also the shape of a file header. The second file is here so a header read
# by mistake has somewhere wrong to point.
DASHES_IN_CONTENT = """\
diff --git a/DECISIONS.md b/DECISIONS.md
--- a/DECISIONS.md
+++ b/DECISIONS.md
@@ -7,2 +7 @@
--- a/somewhere/else.md
-+++ b/somewhere/else.md
+A replacement.
diff --git a/research/spinello_stilwell_rung.md b/research/spinello_stilwell_rung.md
--- a/research/spinello_stilwell_rung.md
+++ b/research/spinello_stilwell_rung.md
@@ -3 +3 @@
-Q2 is open.
+Q2 is closed.
"""


def test_a_pure_insertion_removes_nothing():
    assert parse_removals(INSERTION) == []


def test_an_empty_diff_removes_nothing():
    assert parse_removals("") == []


def test_a_rewritten_line_is_reported_where_it_stood():
    (removal,) = parse_removals(REWRITE)
    assert removal.path == "DECISIONS.md"
    assert removal.line_number == 42
    assert removal.text == "The gap is 3.2e-14."


def test_deleting_the_file_reports_every_line_under_its_old_path():
    removals = parse_removals(DELETION)
    assert [r.line_number for r in removals] == [1, 2]
    assert {r.path for r in removals} == {"research/gate_d4_registration.md"}


def test_header_shaped_content_stays_inside_its_own_hunk():
    removals = parse_removals(DASHES_IN_CONTENT)
    assert [(r.path, r.line_number) for r in removals] == [
        ("DECISIONS.md", 7),
        ("DECISIONS.md", 8),
        ("research/spinello_stilwell_rung.md", 3),
    ]


@pytest.mark.parametrize(
    ("subjects", "taken"),
    [
        ([f"docs: fix a typo {IN_PLACE_MARKER}"], True),
        (["docs: fix a typo"], False),
        ([], False),
        (["docs: adr-002", f"docs: fix a typo {IN_PLACE_MARKER}"], True),
    ],
)
def test_one_marked_subject_anywhere_takes_the_exception(subjects, taken: bool):
    assert override_requested(subjects) is taken


@pytest.mark.parametrize(
    ("message", "subject"),
    [
        ("docs: fix a typo [in-place]", "docs: fix a typo [in-place]"),
        (f"# {IN_PLACE_MARKER}\n\ndocs: fix a typo\n", "docs: fix a typo"),
        ("", ""),
    ],
)
def test_the_subject_is_what_the_marker_is_read_from(message: str, subject: str):
    assert commit_subject(message) == subject


def test_the_subject_skips_blanks_and_comments():
    assert commit_subject("\n# a comment\n\nfeat: the subject\nmore\n") == (
        "feat: the subject"
    )


def git(repository: Path, *arguments: str) -> None:
    """Run git in `repository`, failing the test on a non-zero exit."""
    subprocess.run(("git", "-C", str(repository), *arguments), check=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A repository on `main` with one protected file already landed."""
    git(tmp_path, "init", "-b", "main", "--quiet")
    git(tmp_path, "config", "user.email", "noreply@github.com")
    git(tmp_path, "config", "user.name", "DanBoringName")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.hooksPath", str(tmp_path / "no-hooks"))
    for path in PROTECTED_PATHS:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(
            "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.05.\n"
        )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "--quiet", "-m", "docs: the first decision")
    return tmp_path


def stage(repository: Path, text: str) -> None:
    """Write the protected file and stage it."""
    (repository / PROTECTED_PATHS[0]).write_text(text)
    git(repository, "add", "-A")


def test_a_clean_index_passes(repository: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(repository)
    assert main([]) == 0


def test_appending_to_a_landed_file_passes(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    stage(
        repository,
        "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.05.\n"
        "\n### ADR-002 2026-02-01\nADR-001's bar was replaced.\n",
    )
    assert main([]) == 0


def test_rewriting_a_landed_line_fails(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    assert main([]) == 1


def test_the_marker_lets_a_rewrite_through(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    message = repository / "COMMIT_EDITMSG"
    message.write_text(f"docs: correct the heading {IN_PLACE_MARKER}\n")
    assert main([str(message)]) == 0


def test_a_message_without_the_marker_still_fails(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    message = repository / "COMMIT_EDITMSG"
    message.write_text("docs: correct the bar\n")
    assert main([str(message)]) == 1


def test_a_block_drafted_on_a_branch_can_still_be_reworked(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    stage(
        repository,
        "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.05.\n"
        "\n### ADR-002 2026-02-01\nDarft.\n",
    )
    git(repository, "commit", "--quiet", "-m", "docs: adr-002")
    stage(
        repository,
        "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.05.\n"
        "\n### ADR-002 2026-02-01\nDraft.\n",
    )
    # The typo was fixed in a line that never reached `main`, so nothing a reader
    # could have seen was rewritten.
    assert main([]) == 0


def test_rewriting_a_landed_line_from_a_branch_fails(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    assert main([]) == 1


def test_a_declared_path_that_is_not_tracked_fails_loudly(
    repository: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.chdir(repository)
    git(repository, "rm", "--quiet", PROTECTED_PATHS[1])
    # Nothing is rewritten in the file that remains, so the run would otherwise pass
    # while one entry in the list silently guarded nothing.
    assert main([]) == 1
    assert "not tracked" in capsys.readouterr().out


def test_moving_a_protected_file_out_from_under_the_rule_fails(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    git(repository, "mv", PROTECTED_PATHS[0], "ADRS.md")
    # The pathspec names the source, so a move reads as a whole-file deletion however
    # the diff is configured. Every line of it counts as rewritten, which is the
    # intended answer rather than an accident of the shape.
    assert main([]) == 1


def test_the_override_survives_the_next_commit_on_the_branch(
    repository: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    message = repository / "COMMIT_EDITMSG"
    message.write_text(f"docs: correct the bar {IN_PLACE_MARKER}\n")
    assert main([str(message)]) == 0
    git(
        repository,
        "commit",
        "--quiet",
        "-m",
        f"docs: correct the bar {IN_PLACE_MARKER}",
    )
    # The diff against the merge base still carries that removal. A plain append after
    # it must not have to call itself an in-place edit to be recorded.
    stage(
        repository,
        "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n"
        "\n### ADR-002 2026-02-01\nAppended.\n",
    )
    message.write_text("docs: adr-002\n")
    assert main([str(message)]) == 0
    assert "allowed by" in capsys.readouterr().out


def test_the_ci_shape_honours_a_marker_already_on_the_branch(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    git(
        repository,
        "commit",
        "--quiet",
        "-m",
        f"docs: correct the bar {IN_PLACE_MARKER}",
    )
    # No message file, which is how the job runs it. The exception has to survive into
    # CI or the hook lets a commit through that the job then refuses for good.
    assert main([]) == 0


def test_the_ci_shape_still_refuses_an_unmarked_branch(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    git(repository, "commit", "--quiet", "-m", "docs: correct the bar")
    assert main([]) == 1


def test_the_base_branch_is_read_from_the_environment(
    repository: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repository)
    # `feat/lower` rewrites a landed line and owns that decision. `feat/upper` stacks on
    # it and only appends.
    git(repository, "switch", "--quiet", "-c", "feat/lower")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    git(repository, "commit", "--quiet", "-m", "docs: correct the bar")
    git(repository, "switch", "--quiet", "-c", "feat/upper")
    stage(
        repository,
        "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n"
        "\n### ADR-002 2026-02-01\nAppended.\n",
    )
    git(repository, "commit", "--quiet", "-m", "docs: adr-002")
    # Against `main` the cumulative diff carries the rewrite `feat/upper` never made,
    # and there is no subject on this branch that could clear it.
    assert main([]) == 1
    monkeypatch.setenv("GITHUB_BASE_REF", "feat/lower")
    assert main([]) == 0


def test_the_ci_shape_refuses_to_pass_when_it_compared_nothing(
    repository: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.chdir(repository)
    git(repository, "switch", "--quiet", "-c", "feat/next")
    git(repository, "branch", "--quiet", "-D", "main")
    stage(repository, "# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.10.\n")
    # With no landed ref the diff against `HEAD` is what a fresh checkout would produce:
    # empty. Passing on that reports green having looked at nothing.
    assert main([]) == 1
    assert "nothing was compared" in capsys.readouterr().out
