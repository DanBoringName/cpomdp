"""The warrant vocabulary in a pytest run, instead of a column of dots.

pytest has three outcomes and a check has five. Collapsing the five loses the two the
vocabulary exists to keep apart: a falsifier void by construction and one measured
elsewhere both read as a skip, and a run that survived everything without deciding
anything reads the same as one that decided it all.

So a check's outcome is carried alongside pytest's rather than replacing it.
`pytest_report_teststatus` sets the progress letter and the verbose word, which is how
`xfail` and `xpass` are spelled too, and the run's own accounting is printed at the end
in the vocabulary the checks reported in.

The pytest outcome underneath stays one of its three. `junitxml` branches on that and on
nothing else, so a fourth value writes the row as nothing at all and every CI parser
downstream sees a test that never ran.

Loaded through the ``pytest11`` entry point, so it is active wherever both packages are
installed. ``warrantlib`` itself imports nothing from here, and importing ``warrantlib``
does not import pytest.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Literal

import pytest

from warrantlib._serialise import report_from_dict, report_to_dict
from warrantlib._vocabulary import CheckReport, Outcome, check_summary

if TYPE_CHECKING:
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

#: The three outcomes pytest itself has. Spelled out so the table below cannot drift
#: into a fourth, which `junitxml` would write as no row at all.
_PytestOutcome = Literal["passed", "failed", "skipped"]

#: What each outcome becomes in a run: the pytest outcome, the counting category, the
#: progress letter and the verbose word.
#:
#: `NOT_RESOLVED` fails. It ran and the ordering came out undetermined, which is a
#: question the run could not answer rather than one it answered in the claim's favour.
#: The two that never ran skip, because they did.
#:
#: The category is pytest's own rather than a new one, and this is the one place the
#: vocabulary gives ground. A category is what the closing tally counts under, so a
#: fresh one moves every check out of `N passed` into a name no existing tool reads:
#: `assert_outcomes` stops seeing them and `parseoutcomes` cannot split a two-word name.
#: The distinctions survive in the three places a reader meets them anyway. The letter
#: separates `v` from `e` and from `.` as the run goes, `-v` prints the check's own word
#: instead of `PASSED`, and the warrant summary carries the accounting that decides
#: anything. Buying a headline count with a broken tally is the wrong trade.
_STATUS: dict[Outcome, tuple[_PytestOutcome, str, str, str]] = {
    Outcome.NOT_TRIGGERED: ("passed", "passed", ".", "NOT TRIGGERED"),
    Outcome.FIRED: ("failed", "failed", "F", "FIRED"),
    Outcome.NOT_RESOLVED: ("failed", "failed", "?", "NOT RESOLVED"),
    Outcome.NOT_APPLICABLE: ("skipped", "skipped", "v", "NOT APPLICABLE"),
    Outcome.NOT_RUN_HERE: ("skipped", "skipped", "e", "NOT RUN HERE"),
}

#: The outcome a test takes when it recorded several, worst first. A test recording one
#: fired check and four survivors is a test that found a refutation.
_SEVERITY: tuple[Outcome, ...] = (
    Outcome.FIRED,
    Outcome.NOT_RESOLVED,
    Outcome.NOT_TRIGGERED,
    Outcome.NOT_APPLICABLE,
    Outcome.NOT_RUN_HERE,
)

#: What a test recorded, read by the report hook once its call phase is over.
_RECORDS: pytest.StashKey[list[CheckReport]] = pytest.StashKey()

#: The attribute a `TestReport` carries the records on, and the reason they are carried
#: as mappings rather than as `CheckReport` objects.
#:
#: Under `-n auto` a report crosses a process boundary. `_report_to_json` copies the
#: report's `__dict__` wholesale and `TestReport.__init__` puts the extras back, so an
#: attribute survives the trip on one condition: everything in it is JSON. Records that
#: were dataclasses would need a pair of `pytest_report_*_serializable` hooks to
#: survive. Records that are already the wire form need none, and the run behaves the
#: same way with workers as without.
_REPORT_ATTRIBUTE = "warrant_records"

#: The outcome a report's own status was taken from, set only where it was taken. A test
#: that recorded a check and then raised carries records and keeps pytest's verdict, so
#: the status hook cannot re-derive the word from the records: it would relabel a real
#: error as the outcome of a check that survived, and the error would read as a pass.
_STATUS_ATTRIBUTE = "warrant_status"


def pytest_configure(config: pytest.Config) -> None:
    """Register the run's collector, which needs config and the report stream both.

    Args:
        config: the run's configuration.
    """
    config.pluginmanager.register(_WarrantRun(), "warrant-run")


@pytest.fixture
def record_check(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Hand a `CheckReport` to the run, which reports it in the warrant vocabulary.

    A test that records one check takes that check's outcome. A test that records
    several takes the worst of them, and every one reaches the summary.

        def test_the_coefficient_is_closed_form(record_check):
            record_check(measure_c2())

    Args:
        request: the running test, whose stash the records live on.

    Returns:
        A callable taking one `CheckReport`, or several.
    """
    records: list[CheckReport] = []
    request.node.stash[_RECORDS] = records

    def record(*reports: CheckReport) -> None:
        for report in reports:
            if not isinstance(report, CheckReport):
                raise TypeError(
                    f"record_check was given a {type(report).__name__}. It records a "
                    "CheckReport, which is the only thing carrying the outcome and the "
                    "warrant this run reports in."
                )
        records.extend(reports)

    return record


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, TestReport, TestReport]:
    """Move a test's records onto its report, and take the outcome from them.

    The records are attached whatever the phase, so a run that writes them out later has
    the setup and teardown rows too. The outcome is only taken from the call phase: a
    test that recorded a fired check and then raised in teardown has two problems, and
    overwriting the second with the first hides it.

    Args:
        item: the test.
        call: its phase.

    Returns:
        The report, with the records on it.
    """
    report = yield
    records = item.stash.get(_RECORDS, [])
    if not records:
        return report
    setattr(report, _REPORT_ATTRIBUTE, [report_to_dict(one) for one in records])
    if report.when == "call" and not call.excinfo:
        _apply_outcome(report, item, records)
    return report


def _apply_outcome(
    report: TestReport, item: pytest.Item, records: list[CheckReport]
) -> None:
    """Give a passing test the outcome its checks reported, and something to render.

    An outcome on its own is not enough. The terminal reads a skip's reason out of a
    three-part `longrepr` and asserts on its shape, and a failure with nothing to show
    prints as a row with no cause. Both are filled from the checks that decided it, so
    `-rs` and the failure block say which check and why rather than naming the test.

    Args:
        report: the call phase's report, passing so far.
        item: the test, for the location a skip reason carries.
        records: what it recorded.
    """
    worst = _worst(records)
    outcome, _, _, word = _STATUS[worst]
    report.outcome = outcome
    setattr(report, _STATUS_ATTRIBUTE, worst.value)
    deciding = [record for record in records if record.outcome is worst]
    reason = "; ".join(f"{record.check_id}: {record.detail}" for record in deciding)
    if outcome == "skipped":
        # The shape `_get_raw_skip_reason` asserts on: path, line, and a reason it
        # strips its own prefix from.
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {reason}")
    elif outcome == "failed":
        report.longrepr = f"{word}\n{reason}"


def _worst(records: list[CheckReport]) -> Outcome:
    """The outcome a test takes, given everything it recorded.

    Args:
        records: what the test recorded, in any order.

    Returns:
        The most severe outcome present.
    """
    present = {record.outcome for record in records}
    return next(outcome for outcome in _SEVERITY if outcome in present)


def pytest_report_teststatus(
    report: pytest.CollectReport | pytest.TestReport, config: pytest.Config
) -> tuple[str, str, str] | None:
    """Say what a check-carrying report is called, in the vocabulary the check used.

    Only a report whose own outcome came from its checks is relabelled. A test that
    recorded a check and then failed on its own keeps pytest's word for it, because the
    check did not decide that row.

    Args:
        report: the report being rendered.
        config: the run's configuration.

    Returns:
        The category, the letter and the word, or ``None`` to leave the report alone.
    """
    status = getattr(report, _STATUS_ATTRIBUTE, None)
    if status is None:
        return None
    _, category, letter, word = _STATUS[Outcome(status)]
    return category, letter, word


def _records_of(report: object) -> list[CheckReport]:
    """The reports a test recorded, read back from whatever carried them.

    Args:
        report: a pytest report, which may carry none.

    Returns:
        The records, empty where the report carries none.
    """
    raw = getattr(report, _REPORT_ATTRIBUTE, None)
    if not raw:
        return []
    return [report_from_dict(one) for one in raw]


class _WarrantRun:
    """One run's checks, gathered as they are reported and summarised at the end.

    A plugin object rather than module state, because the hooks that gather and the hook
    that renders need the same store and pytest hands `config` to only some of them.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.records: list[CheckReport] = []

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        """Gather a report's checks, on the controller and on a worker alike.

        Args:
            report: the report, which may carry none.
        """
        if report.when != "call":
            return
        self.records.extend(_records_of(report))

    def pytest_terminal_summary(
        self,
        terminalreporter: TerminalReporter,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        """Print the run's accounting, in the vocabulary the checks reported in.

        Silent when the run carried no checks, so a suite that uses none of this reads
        exactly as it did before the plugin was installed.

        Args:
            terminalreporter: where the section goes.
            exitstatus: the run's status, unused.
            config: the run's configuration, unused.
        """
        if not self.records:
            return
        terminalreporter.write_sep("=", "warrant summary")
        for line in check_summary(self.records).splitlines():
            terminalreporter.write_line(line)
