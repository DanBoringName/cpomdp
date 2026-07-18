"""Runs the example ``--check`` gates so their paper-facing assertions gate in CI.

Each demo's ``check()`` asserts the numbers the paper quotes. Calling them here turns
the manual ``--check`` into a regression gate: the assertions fire on every test run,
not only when the script is run by hand. Both ``check()`` are plotting-free (matplotlib
is imported lazily in the render path), so they run in the base test environment without
the ``examples`` extra. ``conftest.py`` puts ``examples/`` and ``examples/ffg/`` on the
path.
"""

import efe_collapse_figure
import epistemic_dissociation_figure


def test_single_chain_theorem_check():
    """Theorem 1 (i)/(iii) on the single-chain model class: the epistemic value varies,
    its frozen-R twin is flat, and the sensor noise traces a curve across the grid."""
    efe_collapse_figure.check()


def test_epistemic_dissociation_check():
    """The four T-maze results, including Result 4's horizon-1 pull < gradient order."""
    epistemic_dissociation_figure.check()
