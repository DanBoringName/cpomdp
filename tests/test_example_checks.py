"""Runs the example ``--check`` gates so their paper-facing assertions gate in CI.

Each demo's ``check()`` asserts the numbers the paper quotes. Calling them here turns
the manual ``--check`` into a regression gate: the assertions fire on every test run,
not only when the script is run by hand. Every ``check()`` here is plotting-free
(matplotlib is imported lazily in the render path), so they run in the base test
environment without the ``examples`` extra. ``conftest.py`` puts ``examples/`` and
``examples/ffg/`` on the path.
"""

import bacillus_uncertain_food
import chemotaxis_figure
import coupling_graph_figure
import crossover
import crossover_horizon_figure
import crossover_sweep
import efe_collapse_figure
import epistemic_dissociation_figure
import pytest

from cpomdp.warrant import Outcome, Tier, Warrant, check_summary


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


def test_coupling_graph_two_routes_agree():
    """`CouplingGraph.infer` on a branching tree matches the hand-flattened Kalman.

    The demo prints the verdict, and until it was wired up here nothing consumed it: a
    disagreement printed `FAIL` and exited zero.
    """
    coupling_graph_figure.check()


def test_chemotaxis_two_routes_agree():
    """The same equivalence on the five-node chemotaxis network, hub inferred through
    its leaves. Its verdict was equally unconsumed."""
    chemotaxis_figure.check()


def test_crossover_sweep_check():
    """The constant reach/walk pair never crosses over — the search-family artefact that
    makes the exhaustive varying-sequence search necessary."""
    crossover_sweep.check()


def test_crossover_horizon_check():
    """The open-plane margin between a direct plan and a detour. The two animated
    agents choose differently. The margin crosses zero exactly once as the horizon
    grows. The epistemic pull stays flat while the pragmatic gradient decays under it,
    and a frozen-R twin never crosses at any horizon in range.

    Two 16-horizon sweeps, live and frozen. ~12s on the reference machine, under the
    marker's threshold, and this is the demo's only gate. It stays on the PR path."""
    crossover_horizon_figure.check()


def test_crossover_falsifiers_are_reports():
    """The four registered falsifiers, as `CheckReport`s the summary can count.

    The `PROVED` rows must carry the real completeness certificate off the H* search.
    A fabricated one would satisfy the constructor and claim an enumeration that never
    ran, which is the failure the precondition exists to catch. Costs one search
    construction (~1s), so it stays on the pull-request path where `check()` does not.
    """
    reports = crossover.falsifiers()
    assert len(reports) == 4
    proved = [r for r in reports if r.warrant is Warrant.PROVED]
    assert len(proved) == 2
    for report in proved:
        assert report.evidence is not None
        assert report.evidence.complete
        assert report.evidence.expected == 5**crossover.FLIP_H
        # The other axis: decided by enumeration, and measured against a stated bar.
        # A Tier B row has to name the bar, or the tier is a label with nothing under
        # it. `cond` is the ceiling the bound is propagated from.
        assert report.tier is Tier.B
        assert "cond" in report.detail
        # The margin clears the bound today, so the flip is not a tie. A regression to
        # one shows up here as NOT RESOLVED rather than as an exception inside check().
        assert report.outcome is Outcome.NOT_TRIGGERED
        # The qualifier travels with the number. H* = 7 is an upper bound because the
        # declared set clips the reach at -2, and a Tier B row reads stronger than the
        # claim is without it.
        assert "upper bound" in report.detail

    # The two that never ran stay distinct, and neither claims a prover.
    unrun = {r.outcome: r for r in reports if r.warrant is None}
    assert set(unrun) == {Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE}

    summary = check_summary(reports)
    assert "PROVED" in summary
    assert "FIRED" not in summary
    assert "4 registered, 2 tested here, none fired" in summary


@pytest.mark.slow
def test_crossover_check():
    """The exhaustive argmin flips reach -> two-phase walk at H*=7: the crossover, its
    flat-pull / decaying-gradient mechanism, and the headline number against a NumPy
    oracle. Enumerates ~150k policies (H=6, H=7), so it is marked slow and deselected on
    PRs; it gates on merge-to-main and release."""
    crossover.check()
