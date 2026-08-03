"""Runs the example ``--check`` gates so their paper-facing assertions gate in CI.

Each demo's ``check()`` asserts the numbers the paper quotes. Calling them here turns
the manual ``--check`` into a regression gate: the assertions fire on every test run,
not only when the script is run by hand. Every ``check()`` here is plotting-free
(matplotlib is imported lazily in the render path), so they run in the base test
environment without the ``examples`` extra. ``conftest.py`` puts ``examples/`` and
``examples/ffg/`` on the path.
"""

import bacillus_uncertain_food
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


def test_uncertain_food_backend_agreement():
    """`KalmanBackend` and the FFG `ChainBackend` agree on the flagship's sensor.

    The channel reads one state block while its noise is keyed on another, a topology
    neither backend's own suite covers. Both get one scripted `(observation, action)`
    sequence rather than two closed loops, so a near-tied argmin cannot send them down
    different trajectories and be misread as a disagreement.

    This is the flagship's only gate. It is also the README's hero demo, so importing
    the module here fails the suite if the demo stops loading at all.
    """
    max_mean_diff, max_cov_diff = bacillus_uncertain_food.check_backend_agreement()
    assert max_mean_diff < 1e-7, f"means diverge by {max_mean_diff:.2e}"
    assert max_cov_diff < 1e-7, f"covariances diverge by {max_cov_diff:.2e}"


def test_crossover_sweep_check():
    """The constant reach/walk pair never crosses over — the search-family artefact that
    makes the exhaustive varying-sequence search necessary."""
    crossover_sweep.check()


@pytest.mark.slow
def test_crossover_horizon_check():
    """The open-plane margin between a direct plan and a detour. The two animated
    agents choose differently. The margin crosses zero exactly once as the horizon
    grows. The epistemic pull stays flat while the pragmatic gradient decays under it,
    and a frozen-R twin never crosses at any horizon in range.

    Two 16-horizon sweeps, live and frozen, at ~20s."""
    crossover_horizon_figure.check()


@pytest.mark.slow
def test_crossover_check():
    """The exhaustive argmin flips reach -> two-phase walk at H*=7: the crossover, its
    flat-pull / decaying-gradient mechanism, and the headline number against a NumPy
    oracle. Enumerates ~150k policies (H=6, H=7), so it is marked slow and deselected on
    PRs; it gates on merge-to-main and release."""
    crossover.check()
