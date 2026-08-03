"""Runs the example ``--check`` gates so their paper-facing assertions gate in CI.

Each demo's ``check()`` asserts the numbers the paper quotes. Calling them here turns
the manual ``--check`` into a regression gate: the assertions fire on every test run,
not only when the script is run by hand. Both ``check()`` are plotting-free (matplotlib
is imported lazily in the render path), so they run in the base test environment without
the ``examples`` extra. ``conftest.py`` puts ``examples/`` and ``examples/ffg/`` on the
path.
"""

import crossover
import crossover_horizon_figure
import crossover_sweep
import efe_collapse_figure
import epistemic_dissociation_figure
import pytest


def test_single_chain_theorem_check():
    """Theorem 1 (i)/(iii) on the single-chain model class: the epistemic value varies,
    its frozen-R twin is flat, and the sensor noise traces a curve across the grid."""
    efe_collapse_figure.check()


def test_epistemic_dissociation_check():
    """The four T-maze results, including Result 4's horizon-1 pull < gradient order."""
    epistemic_dissociation_figure.check()


def test_crossover_sweep_check():
    """The constant reach/walk pair never crosses over — the search-family artefact that
    makes the exhaustive varying-sequence search necessary."""
    crossover_sweep.check()


def test_crossover_horizon_check():
    """The open-plane margin between a direct plan and a detour. The two animated
    agents choose differently. The margin crosses zero exactly once as the horizon
    grows. The epistemic pull stays flat while the pragmatic gradient decays under it,
    and a frozen-R twin never crosses at any horizon in range."""
    crossover_horizon_figure.check()


@pytest.mark.slow
def test_crossover_check():
    """The exhaustive argmin flips reach -> two-phase walk at H*=7: the crossover, its
    flat-pull / decaying-gradient mechanism, and the headline number against a NumPy
    oracle. Enumerates ~150k policies (H=6, H=7), so it is marked slow and deselected on
    PRs; it gates on merge-to-main and release."""
    crossover.check()
