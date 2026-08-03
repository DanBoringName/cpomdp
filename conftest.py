"""Pytest configuration.

Put ``examples/`` on the import path so the acceptance tests can import the models that
are built *with* cpomdp (e.g. the chemotaxis network) rather than shipped inside the
installable library. The examples stay plain, runnable scripts; this only affects how
the test process resolves them.

Also police the ``slow`` marker. The rule is a wall-clock one: past
``SLOW_TEST_SECONDS`` a test belongs off the pull-request path. Remembering to apply
it is the part that rots, so the run measures itself and reports what has drifted over.
"""

import sys
from pathlib import Path

_examples = Path(__file__).parent / "examples"
sys.path.insert(0, str(_examples))
sys.path.insert(0, str(_examples / "ffg"))

# The threshold the `slow` marker means, in seconds. Change it here. The marker's
# description in pyproject.toml points at this name rather than repeating the number.
SLOW_TEST_SECONDS = 20.0

_over_threshold: list[tuple[str, float]] = []


def pytest_runtest_logreport(report) -> None:
    """Record any unmarked test whose call phase ran past the threshold.

    Only the call phase counts. Setup and teardown are fixture cost, which marking the
    test would not move off the pull-request path anyway.
    """
    if report.when != "call" or report.duration <= SLOW_TEST_SECONDS:
        return
    if "slow" in report.keywords:
        return
    _over_threshold.append((report.nodeid, report.duration))


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Report the drift, without failing on it.

    A hard failure here would turn ordinary CI timing noise into a red build, and the
    threshold is a housekeeping rule rather than a correctness one. Reporting keeps it
    visible on the merge and release runs, which are the ones that execute the slow
    tests and so are the only ones that can see the real durations.
    """
    if not _over_threshold:
        return
    terminalreporter.write_sep(
        "=", "unmarked tests over the slow threshold", yellow=True
    )
    for nodeid, duration in sorted(_over_threshold, key=lambda row: -row[1]):
        terminalreporter.write_line(f"  {duration:6.1f}s  {nodeid}")
    terminalreporter.write_line(
        f"  threshold {SLOW_TEST_SECONDS:.0f}s (conftest.SLOW_TEST_SECONDS). Add "
        f"@pytest.mark.slow, or raise the threshold if the rule has moved."
    )
