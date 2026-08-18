"""Registered falsifiers for the programme's declared numbers.

Every module here is runnable on its own::

    python -m research.checks.<name> --check

That runs the module's suite, prints the summary in the warrant vocabulary, and exits
non-zero if a check fired. CI pins each suite's registered count on top of the exit
status, so a stage dropped from a runner shows up as a failure instead of as a shorter
run that still reads green.
"""
