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

PURE_RENAME = """\
diff --git a/DECISIONS.md b/ADRS.md
similarity index 100%
rename from DECISIONS.md
rename to ADRS.md
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


def test_a_pure_rename_removes_nothing():
    assert parse_removals(PURE_RENAME) == []


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
    ("message", "taken"),
    [
        ("docs: fix a typo [in-place]", True),
        ("docs: fix a typo", False),
        (f"# {IN_PLACE_MARKER}\n\ndocs: fix a typo\n", False),
        (f"\n\ndocs: fix a typo {IN_PLACE_MARKER}\n# comment\n", True),
        ("", False),
    ],
)
def test_the_exception_is_taken_from_the_subject_alone(message: str, taken: bool):
    assert override_requested(message) is taken


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
    protected = tmp_path / PROTECTED_PATHS[0]
    protected.write_text("# Decisions\n\n### ADR-001 2026-01-01\nThe bar is 0.05.\n")
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
