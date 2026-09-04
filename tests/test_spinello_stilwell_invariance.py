"""The checks behind `research/spinello_stilwell_rung.md`, routes 1 and 2.

Running the module is checking it: every claim it prints is asserted inside it.
"""

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


def test_it_reports_no_warrant():
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(invariance, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(invariance))


@pytest.mark.parametrize("iterate", [invariance.iterate, scheme.iterate])
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

    def test_the_sin_family_attains_the_pole_exactly(self):
        # 1.5 + 0.5 sin(x) reaches 1.0 where sin(x) = -1. At lambda = 1 the pole is on
        # the family rather than near it, which is what makes the unit choice
        # compulsory rather than prudent.
        floor = reachable_noise._infimum_over_the_line(FAMILIES["sin"])
        assert abs(floor - 1.0) < 1e-6, floor

    def test_the_exponential_cannot_be_cleared_over_the_whole_line(self):
        # Its infimum is zero, so no lambda puts every state clear. This is the one
        # family where a guard is still owed, and route 3 is where the consequence of
        # an iterate escaping the declared window gets measured.
        floor = reachable_noise._infimum_over_the_line(FAMILIES["exponential"])
        assert floor < 1e-9, floor

    def test_one_scale_clears_every_family_over_the_reachable_window(self):
        # The result that makes a single declared unit choice possible.
        surveyed = {**FAMILIES, reachable_noise.RIDGE.key: reachable_noise.RIDGE}
        floors = [
            reachable_noise.reachable_noise(family, spread)[0]
            for family in surveyed.values()
            for spread in SIGMAS
        ]
        needed = max(reachable_noise.clearing_scale(low) for low in floors)
        assert all(needed**2 * low >= reachable_noise.MARGIN - 1e-9 for low in floors)
        assert 2.5 < needed < 2.6, needed

    def test_the_ridge_operating_point_sits_below_the_pole(self):
        floor = reachable_noise._infimum_over_the_line(reachable_noise.RIDGE)
        assert floor < 1.0, floor
