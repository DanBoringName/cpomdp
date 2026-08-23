import research.explorations.c6_window as window


def test_the_window_exploration_runs_and_its_assertions_hold(capsys):
    # The module asserts its two validations rather than only printing them, so running
    # it is the check. Without this the write-up's numbers rot silently.
    window.main()
    printed = capsys.readouterr().out
    assert "registered: D* = 0.5200" in printed
    assert "D moves by a factor of 0.895" in printed


def test_the_exploration_reports_no_warrant():
    # It is not a check suite and must not look like one: nothing here may be collected
    # into the manifest or read as carrying a warrant.
    assert not hasattr(window, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(window))
