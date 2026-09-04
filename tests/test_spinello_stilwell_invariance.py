"""The checks behind `research/spinello_stilwell_rung.md`, routes 1 and 2.

Running the module is checking it: every claim it prints is asserted inside it.
"""

import numpy as np
import pytest

from research.checks.gap_kernel import FAMILIES, SIGMAS
from research.spinello_stilwell import invariance, reachable_noise, scheme
from research.spinello_stilwell.invariance import AT_THE_PROBE_BUDGET


def test_it_runs_and_its_assertions_hold(capsys):
    invariance.main()
    printed = capsys.readouterr().out
    assert "the estimate and the variance are unmoved" in printed
    assert "R0 = 1/2 puts the pole on the operating point" in printed


def test_exactly_one_term_moves_under_rescaling():
    survives = invariance.term_invariance()
    moving = [name for name, held in survives.items() if not held]
    assert moving == ["gn: (1/4sigma^2)(1/ln sigma) grad sigma^2"]
    assert sum(survives.values()) == 7


def test_the_converged_estimate_is_unit_free():
    # What Q2 rests on. If rescaling moved the answer, picking units to clear the pole
    # would be picking an answer.
    estimates = [
        invariance.iterate(**invariance.CASE, scale=scale, **AT_THE_PROBE_BUDGET)[0]
        for scale in invariance.SCALES
    ]
    assert max(estimates) - min(estimates) < 1e-12


def test_the_iteration_count_is_not_unit_free():
    # The other half: the path does move, which is why the pole is reachable in some
    # units and not others.
    counts = {
        invariance.iterate(**invariance.CASE, scale=scale, **AT_THE_PROBE_BUDGET)[2]
        for scale in invariance.SCALES
    }
    assert len(counts) > 1, counts
    assert all(count < invariance.PROBE_BUDGET for count in counts), counts


@pytest.mark.parametrize("module", [invariance, reachable_noise])
def test_it_reports_no_warrant(module):
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(module, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(module))


@pytest.mark.parametrize(
    "iterate", [invariance.iterate, scheme.iterate, scheme.iterate_with]
)
@pytest.mark.parametrize("owed", ["max_iterations", "tolerance"])
def test_the_rung_s_open_decisions_stay_arguments(iterate, owed):
    # The budget and the tolerance are undeclared on purpose: ADR-056 routes them to a
    # measurement that has not been made. A default here would settle by default what
    # the record says is open, so every caller has to say which budget it ran at.
    import inspect

    parameter = inspect.signature(iterate).parameters[owed]
    assert parameter.default is inspect.Parameter.empty
    assert not hasattr(invariance, owed.upper())
    assert not hasattr(scheme, owed.upper())


def test_a_run_that_exhausts_the_budget_says_so():
    # The count coming back equal to the budget is the only signal, so it has to be
    # there. Tolerance zero can never be met, so the run spends the whole budget.
    _, _, taken = scheme.iterate(
        **invariance.CASE, scale=1.0, tolerance=0.0, max_iterations=3, log_block=True
    )
    assert taken == 3


class TestReachableNoise:
    """Route 2: whether one unit choice clears the pole for the declared families."""

    def test_it_runs_and_its_assertions_hold(self, capsys):
        reachable_noise.main()
        printed = capsys.readouterr().out
        assert "One lambda for every family at every declared spread" in printed
        assert "no lambda" in printed

    def test_four_families_never_reach_below_the_pole(self):
        # A property of how they were declared, not of any sweep width.
        for key in ("quadratic", "tanh", "sin", "constant"):
            floor = reachable_noise._infimum_over_the_line(FAMILIES[key])
            assert floor >= 1.0 - 1e-9, (key, floor)

    @pytest.mark.parametrize("key", ["quadratic", "sin"])
    def test_two_families_attain_the_pole_exactly(self, key):
        # 1 + x^2 reaches 1.0 at the origin and 1.5 + 0.5 sin(x) where sin(x) = -1.
        # At lambda = 1 the pole is on both families rather than near them, so a
        # units-only repair would have had to move it, not merely widen a margin.
        floor = reachable_noise._infimum_over_the_line(FAMILIES[key])
        assert abs(floor - 1.0) < 1e-6, floor

    def test_the_quadratic_reaches_the_pole_in_more_of_the_grid_than_the_sine(self):
        # Both priors sit at one. The quadratic's minimum is one unit away and the
        # sine's nearest is at -pi/2, so the declared windows reach the first in five
        # cells of six and the second in two.
        cells = {
            key: sum(
                reachable_noise.reachable_noise(FAMILIES[key], spread)[0] <= 1.0 + 1e-6
                for spread in SIGMAS
            )
            for key in ("quadratic", "sin")
        }
        assert cells == {"quadratic": 5, "sin": 2}, cells

    def test_the_exponential_cannot_be_cleared_over_the_whole_line(self):
        # Its infimum is zero, so no lambda puts every state clear. That is why
        # ADR-057 removed the block instead: the shipped rung has no pole to guard.
        floor = reachable_noise._infimum_over_the_line(FAMILIES["exponential"])
        assert floor < 1e-9, floor

    def test_one_scale_clears_every_family_over_the_reachable_window(self):
        # What a declared unit choice would have rested on. ADR-057 declares none.
        surveyed = {**FAMILIES, reachable_noise.RIDGE.key: reachable_noise.RIDGE}
        floors = [
            reachable_noise.reachable_noise(family, spread)[0]
            for family in surveyed.values()
            for spread in SIGMAS
        ]
        needed = max(reachable_noise.clearing_scale(low) for low in floors)
        assert all(needed**2 * low >= reachable_noise.MARGIN - 1e-9 for low in floors)
        # Set by exp(x) at the widest declared spread, over twelve prior spreads.
        assert 4.0 < needed < 4.05, needed

    def test_the_reach_is_the_foot_of_the_reference_s_state_window(self):
        # The gap quadrature treats that many prior spreads as the state's range, so
        # a narrower survey would understate what the iterate can reach.
        assert reachable_noise.REACH_IN_SPREADS == 12.0

    def test_the_ridge_floor_is_below_the_pole_and_its_operating_point_is_on_it(self):
        ridge = reachable_noise.RIDGE
        floor = reachable_noise._infimum_over_the_line(ridge)
        assert abs(floor - 0.5) < 1e-6, floor
        assert abs(float(ridge.noise(np.asarray(ridge.prior_mean))) - 1.0) < 1e-12

    def test_no_reachable_noise_means_no_scale(self):
        assert reachable_noise.clearing_scale(0.0) == float("inf")
