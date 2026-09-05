"""Route 3: what the printed scheme does either side of its pole.

`research/spinello_stilwell_rung.md` states Q3 as an argument. This is the
measurement. The module is not the rung and reports no warrant.

Running a module is checking it, since every claim it prints is asserted inside it.
These call the same functions so a failure names the claim rather than the whole run.
"""

import math

import pytest

from research.spinello_stilwell import invariance, pole_failure, scheme


def test_it_runs_and_its_assertions_hold(capsys):
    pole_failure.main()
    assert capsys.readouterr().out


def test_it_reports_no_warrant():
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(pole_failure, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(pole_failure))


class TestTheFailureFromAbove:
    """Route 3's silent side. The step collapses and the run calls that convergence."""

    def test_the_printed_curvature_has_no_value_on_the_pole(self):
        # `1/ln σ` at σ = 1. The rung's only external check is the fixed-noise
        # reduction, and it cannot be evaluated in the printed form there.
        with pytest.raises(ZeroDivisionError):
            scheme.gauss_newton_curvature(1.0, 1.0, 1.0, 1.2, log_block=True)
        assert scheme.gauss_newton_curvature(1.0, 1.0, 1.0, 1.2, log_block=False) > 0.0

    def test_the_curvature_runs_away_on_both_sides(self):
        rows = pole_failure.curvature_across_the_pole()
        assert all(above > 0.0 for _, above, _ in rows), rows
        assert all(below < 0.0 for _, _, below in rows), rows
        # Four decades in, and still growing by two decades per two decades of offset.
        assert rows[-1][1] > 1e6, rows[-1]
        assert rows[-1][2] < -1e6, rows[-1]

    def test_the_stall_reports_the_prediction_as_converged(self):
        # The severity-one case: it produces a number, the number is the prediction,
        # and stopping early is exactly what a converged run does.
        case = pole_failure.case_at(pole_failure.OFFSETS[-1])
        spread = math.sqrt(case["prior_variance"])
        collapsed = abs(
            pole_failure.trajectory(
                case, log_block=True, max_iterations=1, tolerance=0.0
            )[0].step
        )
        stalled = pole_failure.trajectory(
            case,
            log_block=True,
            max_iterations=invariance.PROBE_BUDGET,
            tolerance=10.0 * collapsed / spread,
        )
        assert len(stalled) == 1
        assert abs(stalled[-1].estimate - case["prior_mean"]) < 1e-7

    def test_the_answer_the_stall_hides_is_far_from_the_prediction(self):
        case = pole_failure.case_at(pole_failure.OFFSETS[-1])
        settled = pole_failure.trajectory(
            case,
            log_block=True,
            max_iterations=invariance.PROBE_BUDGET,
            tolerance=invariance.PROBE_TOLERANCE,
        )
        assert abs(settled[-1].estimate - case["prior_mean"]) > 1e-2

    def test_the_tolerance_that_would_have_caught_it_depends_on_the_case(self):
        # Which is why no declared tolerance is safe for every case. Closer to the
        # pole, the step to catch is smaller in the same proportion.
        near = pole_failure.trajectory(
            pole_failure.case_at(1e-8), log_block=True, max_iterations=1, tolerance=0.0
        )[0]
        nearer = pole_failure.trajectory(
            pole_failure.case_at(1e-10), log_block=True, max_iterations=1, tolerance=0.0
        )[0]
        assert abs(nearer.step) < 0.02 * abs(near.step)

    def test_the_modification_has_no_such_neighbourhood(self):
        moved = pole_failure.trajectory(
            pole_failure.case_at(pole_failure.OFFSETS[-1]),
            log_block=False,
            max_iterations=1,
            tolerance=0.0,
        )[0]
        assert abs(moved.step) > 1e-2


class TestTheFailureFromBelow:
    """Route 3's loud side. The step matrix passes through singular and turns."""

    def test_the_step_is_unbounded_at_the_crossing(self):
        crossing = pole_failure.singular_state(pole_failure.RIDGE)
        steps = [
            pole_failure.step_at(
                pole_failure.RIDGE, crossing * (1.0 - distance), log_block=True
            ).step
            for distance in (1e-8, 1e-10)
        ]
        assert abs(steps[1]) > 5.0 * abs(steps[0])
        assert abs(steps[1]) > 1e5, steps

    def test_the_printed_step_climbs_the_objective_past_the_crossing(self):
        # An indefinite step matrix is not a descent direction. That is the whole of
        # Q3's second half, and it is why `P⁻¹ + R_mod ≻ 0` was worth buying.
        ridge = pole_failure.RIDGE
        crossing = pole_failure.singular_state(ridge)
        edge = math.sqrt((1.0 - ridge["base_noise"]) / ridge["curvature"])
        uphill = 0.5 * (crossing + edge)
        before = pole_failure.objective_at(ridge, uphill)
        climbed = pole_failure.step_at(ridge, uphill, log_block=True)
        descended = pole_failure.step_at(ridge, uphill, log_block=False)
        assert climbed.objective > before
        assert descended.objective < before
