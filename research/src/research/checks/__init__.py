"""Registered falsifiers for the programme's declared numbers.

Every module here is runnable on its own::

    python -m research.checks.<name> --check

That runs the module's suite, prints the summary in the warrant vocabulary, and exits
non-zero if a check fired.

**Adding or renaming a check means rewriting the manifest.**
``research/registered_checks.toml`` declares every id each suite is registered to
report, and CI reconciles a run against it::

    python -m warrantlib.manifest research/registered_checks.toml

A new `CheckReport` whose id is not declared there fails the run, naming it. So does a
declared id that stops being reported, which is the point: a stage dropped from a runner
used to leave a shorter run that still read green, and a count of the survivors could
not say which one had gone (ADR-046).

Run the reconciliation the way CI does::

    pytest research/registered_checks.toml --warrant-detail
"""
