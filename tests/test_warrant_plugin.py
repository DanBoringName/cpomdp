"""What a run says, rather than what its checks returned.

Every other test here reads a `CheckReport` back from the function that built it. None
of them can see what pytest then does with it, and that is the whole job of the plugin:
five outcomes reaching a terminal that has three, an accounting line at the end, and the
same result whether or not the run was split across workers.

So these run pytest inside pytest. `pytester` writes a throwaway suite, runs it, and
hands back the outcomes and the output. The plugin loads in the inner run through its
entry point, the same way it will for anyone who installs it.
"""

import pytest

#: Imported by every generated suite. Kept here so a change to the vocabulary breaks
#: one string rather than a dozen.
_PREAMBLE = """
from warrantlib import CheckReport, Outcome, Tier, Warrant

def _report(check_id, outcome, warrant=Warrant.CORROBORATED, name="a check"):
    if outcome in (Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE):
        warrant = None
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=warrant,
        outcome=outcome,
        tier=Tier.COMPUTED,
        detail="why it reports what it reports",
    )
"""


def _suite(pytester: pytest.Pytester, body: str) -> None:
    """Write a one-file suite that records checks.

    Args:
        pytester: the inner run's workspace.
        body: the test functions, appended to the shared preamble.
    """
    pytester.makepyfile(_PREAMBLE + body)


class TestTheOutcomeAReportTakes:
    """A check's outcome decides the test's, in pytest's own three."""

    def test_a_surviving_check_passes(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
""",
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_a_fired_check_fails(self, pytester):
        # The refutation is the result, and a run that reports it green is the failure
        # this whole vocabulary exists to prevent.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.fired", Outcome.FIRED))
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    def test_an_unresolved_check_fails(self, pytester):
        # It ran and could not decide. That is not the claim surviving.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.unresolved", Outcome.NOT_RESOLVED))
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    @pytest.mark.parametrize("outcome", ["NOT_APPLICABLE", "NOT_RUN_HERE"])
    def test_a_check_that_never_ran_here_skips(self, pytester, outcome):
        _suite(
            pytester,
            f"""
def test_it(record_check):
    record_check(_report("s.unrun", Outcome.{outcome}))
""",
        )
        pytester.runpytest().assert_outcomes(skipped=1)

    def test_a_test_recording_several_takes_the_worst(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(
        _report("s.a", Outcome.NOT_TRIGGERED),
        _report("s.b", Outcome.FIRED),
        _report("s.c", Outcome.NOT_TRIGGERED),
    )
""",
        )
        pytester.runpytest().assert_outcomes(failed=1)

    def test_a_test_recording_nothing_is_left_alone(self, pytester):
        # A suite that uses none of this must read exactly as it did before.
        _suite(
            pytester,
            """
def test_it():
    assert True
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)
        assert "warrant summary" not in result.stdout.str()

    def test_a_raising_test_still_fails_on_its_own_error(self, pytester):
        # The check survived and the test around it did not. Taking the check's outcome
        # here reports a crash as a survivor, which is worse than any label being wrong.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
    raise RuntimeError("the test itself broke")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*RuntimeError: the test itself broke*"])

    def test_a_raising_test_is_not_relabelled_in_the_vocabulary(self, pytester):
        # The outcome and the word it prints are set by two different hooks, so the
        # verdict can be right while the row still reads NOT TRIGGERED beside it.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.survived", Outcome.NOT_TRIGGERED))
    raise RuntimeError("the test itself broke")
""",
        )
        # Scoped to the test's own row. The summary block below it says NOT TRIGGERED
        # and is right to: the check survived, and the test failed around it.
        rows = [
            line
            for line in pytester.runpytest("-v").stdout.lines
            if "::test_it" in line
        ]
        assert rows
        assert not any("NOT TRIGGERED" in row for row in rows)
        assert any("FAILED" in row for row in rows)

    def test_a_check_that_never_ran_still_reaches_the_summary(self, pytester):
        # The record is attached whatever the verdict, so a run's accounting counts a
        # check the test then failed around rather than losing it.
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check(_report("s.void", Outcome.NOT_APPLICABLE))
    raise RuntimeError("the test itself broke")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*1 registered, 0 tested here, none fired*"])


class TestTheVocabularyReachesTheTerminal:
    """`PASSED` and `SKIPPED` are not what a falsifier did."""

    @pytest.mark.parametrize(
        ("outcome", "word"),
        [
            ("NOT_TRIGGERED", "NOT TRIGGERED"),
            ("FIRED", "FIRED"),
            ("NOT_RESOLVED", "NOT RESOLVED"),
            ("NOT_APPLICABLE", "NOT APPLICABLE"),
            ("NOT_RUN_HERE", "NOT RUN HERE"),
        ],
    )
    def test_the_verbose_word_is_the_checks_own(self, pytester, outcome, word):
        _suite(
            pytester,
            f"""
def test_it(record_check):
    record_check(_report("s.check", Outcome.{outcome}))
""",
        )
        result = pytester.runpytest("-v")
        result.stdout.fnmatch_lines([f"*{word}*"])

    def test_the_two_that_never_ran_stay_distinct(self, pytester):
        # Both are skips underneath. Collapsing them loses the survivor accounting,
        # which is the reading ADR-029 split them to prevent.
        _suite(
            pytester,
            """
def test_void(record_check):
    record_check(_report("s.void", Outcome.NOT_APPLICABLE))

def test_elsewhere(record_check):
    record_check(_report("s.elsewhere", Outcome.NOT_RUN_HERE))
""",
        )
        output = pytester.runpytest("-v").stdout.str()
        assert "NOT APPLICABLE" in output
        assert "NOT RUN HERE" in output


class TestTheRunSummary:
    """The accounting a reader needs first: registered, tested here, fired."""

    def test_the_summary_counts_the_run(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))

def test_b(record_check):
    record_check(_report("s.b", Outcome.NOT_APPLICABLE))

def test_c(record_check):
    record_check(_report("s.c", Outcome.NOT_RUN_HERE))
""",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*3 registered, 1 tested here, none fired*"])

    def test_a_fired_check_reaches_the_summary(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.FIRED))
""",
        )
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(["*1 registered, 1 tested here, 1 fired*"])


class TestTheRunSurvivesWorkers:
    """The suite runs under `-n auto`, so a worker boundary is the ordinary path."""

    def test_the_summary_is_the_same_split_across_workers(self, pytester):
        _suite(
            pytester,
            """
def test_a(record_check):
    record_check(_report("s.a", Outcome.NOT_TRIGGERED))

def test_b(record_check):
    record_check(_report("s.b", Outcome.NOT_APPLICABLE))

def test_c(record_check):
    record_check(_report("s.c", Outcome.NOT_TRIGGERED))
""",
        )
        # A report crosses a process boundary as JSON. Records that were dataclasses
        # would arrive as nothing and the summary would pass by counting none.
        line = "*3 registered, 2 tested here, none fired*"
        pytester.runpytest().stdout.fnmatch_lines([line])
        split = pytester.runpytest("-n", "2", "--dist", "worksteal")
        split.stdout.fnmatch_lines([line])


class TestTheFixtureRefusesWhatItCannotRecord:
    def test_something_that_is_not_a_report_is_refused(self, pytester):
        _suite(
            pytester,
            """
def test_it(record_check):
    record_check("s.a: NOT TRIGGERED")
""",
        )
        result = pytester.runpytest()
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(["*record_check was given a str*"])
