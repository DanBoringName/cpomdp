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

from cpomdp.enumeration import CompletenessCertificate
from cpomdp.warrant import Outcome, Tier, Warrant, check_summary


def test_single_chain_theorem_check():
    """Theorem 1 (i)/(iii) on the single-chain model class: the epistemic value varies,
    its frozen-R twin is flat, and the observation noise traces a curve across the
    grid."""
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


def _measurement(horizon, *, delta_g, cue_ward, bound=1e-5):
    """A synthetic `FlipMeasurement`, so the reporting can be driven off any result.

    The live run only ever produces one shape, so every branch that a refuting or tied
    result would take is unreachable from it. Injecting the measurement is what turns
    those branches into something a test can execute.
    """
    return crossover.FlipMeasurement(
        horizon=horizon,
        certificate=CompletenessCertificate(
            expected=5**horizon,
            visited=5**horizon,
            warrant=Warrant.PROVED,
            action_set_size=5,
            horizon=horizon,
            action_set_version="v1",
        ),
        delta_g=delta_g,
        bound=bound,
        cue_ward=cue_ward,
    )


def _rows(at_prior, at_flip):
    """The two measured falsifiers, keyed by name prefix."""
    reports = crossover.falsifiers(at_prior, at_flip)
    return reports[0], reports[1]


class TestCrossoverFalsifierReporting:
    """The reporting logic, driven off measurements rather than off a live run."""

    # The measured shape: reach wins at H*-1, walk wins at H*, both clear of the bound.
    SURVIVING = (
        {"delta_g": +0.5, "cue_ward": False},
        {"delta_g": -0.152, "cue_ward": True},
    )

    def _pair(self, prior=None, flip=None):
        prior = {**self.SURVIVING[0], **(prior or {})}
        flip = {**self.SURVIVING[1], **(flip or {})}
        return _measurement(6, **prior), _measurement(7, **flip)

    def test_the_measured_shape_survives_both(self):
        one, two = _rows(*self._pair())
        assert one.outcome is Outcome.NOT_TRIGGERED
        assert two.outcome is Outcome.NOT_TRIGGERED

    def test_both_rows_name_the_action_mode(self):
        # `RecedingHorizonSelector` and `OpenLoopSelector` genuinely differ, and the
        # same sentence reads true under either, so a row that does not name its seam
        # is quotable as the mode the reader had in mind. The ledger requires R10 under
        # its seam. Without this the `{seam}` interpolation can be deleted and the whole
        # suite stays green, which is how the qualifier would rot.
        for report in _rows(*self._pair()):
            assert "open-loop" in report.detail
            assert "no re-planning" in report.detail

    def test_a_reversed_flip_fires_row_one(self):
        # The defect this replaces: `abs(ΔG) > bound` cleared on a reversed result too,
        # so a refutation printed as a survivor beside a hardcoded "argmin is cue-ward".
        one, _ = _rows(*self._pair(flip={"delta_g": +0.152, "cue_ward": False}))
        assert one.outcome is Outcome.FIRED
        assert "prior-ward at H = 7" in one.detail

    def test_a_flip_that_was_already_cue_ward_fires_row_two(self):
        # Cue-ward at both horizons is not a clean flip, however wide the margins.
        _, two = _rows(*self._pair(prior={"delta_g": -0.4, "cue_ward": True}))
        assert two.outcome is Outcome.FIRED

    def test_a_tie_at_the_flip_resolves_neither_row(self):
        one, two = _rows(*self._pair(flip={"delta_g": 1e-9, "cue_ward": True}))
        assert one.outcome is Outcome.NOT_RESOLVED
        assert two.outcome is Outcome.NOT_RESOLVED

    def test_a_tie_at_the_prior_horizon_resolves_only_row_two(self):
        # Row 2 quantifies over both horizons, row 1 over H* alone. Reading row 2 off
        # H* would let it survive while ΔG(H*-1) sat inside its own bound.
        one, two = _rows(*self._pair(prior={"delta_g": 1e-9, "cue_ward": False}))
        assert one.outcome is Outcome.NOT_TRIGGERED
        assert two.outcome is Outcome.NOT_RESOLVED

    def test_row_two_carries_a_certificate_for_each_horizon(self):
        _, two = _rows(*self._pair())
        assert [c.expected for c in two.evidence] == [5**6, 5**7]

    def test_row_one_carries_the_flip_horizon_certificate(self):
        one, _ = _rows(*self._pair())
        assert [c.expected for c in one.evidence] == [5**7]

    def test_the_detail_is_generated_from_the_measurement(self):
        # Nothing about the result is hardcoded, so a changed measurement changes the
        # sentence rather than leaving a stale one beside a new outcome.
        one, _ = _rows(*self._pair(flip={"delta_g": -0.9, "cue_ward": True}))
        assert "0.9000" in one.detail

    def test_a_fired_row_reaches_the_summary(self):
        reports = crossover.falsifiers(
            *self._pair(flip={"delta_g": +0.152, "cue_ward": False})
        )
        # A reversed flip fires both: row 1 because the argmin is not cue-ward at H*,
        # row 2 because prior-ward-then-cue-ward is what "clean" means.
        summary = check_summary(reports)
        assert "FIRED" in summary
        assert "2 fired" in summary


@pytest.mark.slow
def test_crossover_falsifiers_are_reports():
    """The four registered falsifiers on a live measurement of the H* boundary.

    The `PROVED` rows must carry the real completeness certificates. A fabricated one
    would satisfy the constructor and claim an enumeration that never ran, which is the
    failure the precondition exists to catch. Enumerating both horizons is the cost
    `check()` already pays, so this is marked slow and gates on merge; the reporting
    logic itself is covered on the pull-request path by the class above.
    """
    reports = crossover.falsifiers()
    assert len(reports) == 5
    proved = [r for r in reports if r.warrant is Warrant.PROVED]
    assert len(proved) == 3
    for report in proved:
        assert report.evidence
        # Every PROVED row names where its bar was registered, so a reader can check
        # the ordering rather than take it. Rows 1 and 2 registered and measured in one
        # commit and say so; row 5 was registered a commit ahead of its measurement.
        assert report.provenance
        # Evidence widened to two kinds when `SymbolicReduction` landed. These rows are
        # decided by enumeration, so which kind they carry is part of the assertion.
        certificates = [
            item
            for item in report.evidence
            if isinstance(item, CompletenessCertificate)
        ]
        assert len(certificates) == len(report.evidence)
        assert all(certificate.complete for certificate in certificates)
        # The other axis: decided by enumeration, and measured against a stated bar.
        # A `BOUNDED` row has to name the bar, or the tier is a label with nothing under
        # it.
        assert report.tier is Tier.BOUNDED
        assert report.outcome is Outcome.NOT_TRIGGERED

    # The crossover rows carry a qualifier the extension row does not: H* = 7 is an
    # upper bound because the declared set clips the reach at -2, and a `BOUNDED` row
    # reads stronger than the claim is without it. `bound` is what the margin was read
    # against. Row 5 is measured against a registered bar instead, so it says `bar`.
    for report in proved:
        if report.name.startswith(("1.", "2.")):
            assert "bound" in report.detail
            assert "upper bound" in report.detail
        else:
            assert "registered bar" in report.detail

    # The two that never ran stay distinct, and neither claims a prover.
    unrun = {r.outcome for r in reports if r.warrant is None}
    assert unrun == {Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE}

    summary = check_summary(reports)
    assert "PROVED" in summary
    assert "FIRED" not in summary
    assert "5 registered, 3 tested here, none fired" in summary


@pytest.mark.slow
def test_crossover_check():
    """The exhaustive argmin flips reach -> two-phase walk at H*=7: the crossover, its
    flat-pull / decaying-gradient mechanism, and the headline number against a NumPy
    oracle. Enumerates ~150k policies (H=6, H=7), so it is marked slow and deselected on
    PRs; it gates on merge-to-main and release."""
    crossover.check()
