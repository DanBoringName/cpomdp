"""Registered falsifiers for the programme's declared numbers.

Every module here is runnable on its own::

    python -m research.checks.<name> --check

That runs the module's suite, prints the summary in the warrant vocabulary, and exits
non-zero if a check fired.

``research/registered_checks.toml`` declares every check id each suite is registered to
report. A declared id that stops being reported fails, naming it, and so does a reported
id nobody declared. A stage dropped from a runner used to leave a shorter run that still
read green, and a count of the survivors could not say which one had gone (ADR-046).

**Reconcile a run against the manifest.** This is what CI runs, and it changes nothing::

    pytest research/registered_checks.toml --warrant-detail

**Rewrite the manifest, after adding or renaming a check.** This one edits the file, so
it is the deliberate half. Run it only once the new ids are the ids you meant, and let
the diff be the review::

    python -m warrantlib.manifest research/registered_checks.toml

Running the rewrite in place of the reconciliation makes the manifest agree with
whatever the suites currently report, which is the failure the manifest exists to
catch. ``python -m warrantlib.manifest --check`` is the read-only form, and it is what
CI uses to notice a manifest nobody regenerated.
"""
