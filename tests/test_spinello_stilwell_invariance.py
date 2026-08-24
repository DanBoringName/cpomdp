"""The rescaling checks behind `research/spinello_stilwell_rung.md` Q1 and Q2.

Running the module is checking it: every claim it prints is asserted inside it.
"""

import pytest

from research.spinello_stilwell import invariance


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
        invariance.iterate(**invariance.CASE, scale=scale)[0]
        for scale in invariance.SCALES
    ]
    assert max(estimates) - min(estimates) < 1e-12


def test_the_iteration_count_is_not_unit_free():
    # The other half: the path does move, which is why the pole is reachable in some
    # units and not others.
    counts = {
        invariance.iterate(**invariance.CASE, scale=scale)[2]
        for scale in invariance.SCALES
    }
    assert len(counts) > 1, counts


def test_it_reports_no_warrant():
    # The package invariant, stated in its `__init__`. A `run_checks` or a `_SOURCE` is
    # the shape that would let a route be collected as though it had decided something.
    assert not hasattr(invariance, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(invariance))


@pytest.mark.parametrize("owed", ["max_iterations", "tolerance"])
def test_the_rung_s_open_decisions_stay_arguments(owed):
    # The budget and the tolerance are undeclared on purpose: ADR-054 routes them to a
    # measurement that has not been made. Baking either in here would settle by default
    # what the record says is open.
    import inspect

    parameter = inspect.signature(invariance.iterate).parameters[owed]
    assert parameter.default is not inspect.Parameter.empty
    assert not hasattr(invariance, owed.upper())
