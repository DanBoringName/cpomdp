"""Route 5: what a budget has to be sized against, and what exhausting one costs.

Route 5 of `research/spinello_stilwell_rung.md`, and the survey the budget declaration
reads from. Three things get measured here.

Q7 says a truncated run returns a wrong covariance as well as a wrong mean, because
`U-bar` is evaluated at the current iterate. That is the case for routing a
non-convergent step to `VOID` rather than reporting it, and it is measured rather than
argued.

A budget is a number, so something has to size it. The convergence survey runs the
modified scheme over the declared families at the declared spreads and reports what the
worst cell needs.

ADR-057 argues the deletion is surgical. The comparison here is what that costs at a
finite budget: same case, same budget, both curvatures, the departure per iterate. Run::

    uv run --no-sync python -m research.spinello_stilwell.budget

Not the rung. No budget is declared here and no warrant is reported. Every number
printed says which budget produced it.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity here and
`spread` is cpomdp's.
"""

import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np

from research.checks.gap_kernel import FAMILIES, SIGMAS, NoiseFamily
from research.spinello_stilwell import scheme
from research.spinello_stilwell.invariance import WorkedCase

__all__ = [
    "BOX_EDGES",
    "BULK_EDGES",
    "OPERATING_CURVATURES",
    "PROBE_BUDGET",
    "PROBE_TOLERANCE",
    "SLOPES",
    "SLOW_CASE",
    "TRUNCATION_GRID",
    "convergence_counts",
    "counts_over_the_operating_range",
    "departure_at_budget",
    "main",
    "modification_cost",
    "noise_at_of",
]

#: What the probes here run to convergence at, printed beside every number they
#: produce. The rung's own budget stays undeclared until an ADR takes it, which is what
#: this module is evidence for rather than a substitute for.
PROBE_BUDGET = 200
PROBE_TOLERANCE = 1e-14

#: A case that takes its time. The prior is tight against a distant reading, so the
#: iterate has a long way to travel and a truncated run is a long way short.
SLOW_CASE: WorkedCase = {
    "observation": 6.0,
    "prior_mean": 0.2,
    "prior_variance": 0.01,
    "base_noise": 1.0,
    "curvature": 1.0,
}

#: The cells the truncation cost is measured over. Curvature and prior variance are the
#: two axes GATE-D4's registered family sweeps, and both change how far the iterate
#: travels, so both change what stopping it early costs.
TRUNCATION_GRID: tuple[WorkedCase, ...] = tuple(
    WorkedCase(
        observation=SLOW_CASE["observation"],
        prior_mean=SLOW_CASE["prior_mean"],
        prior_variance=prior_variance,
        base_noise=SLOW_CASE["base_noise"],
        curvature=curvature,
    )
    for curvature in (0.1, 1.0, 5.0, 20.0)
    for prior_variance in (0.01, 0.25, 1.0)
)


#: The slope of each declared family, written out. `FAMILIES` declares `R` and (35)
#: needs `R'` as well. A central difference cannot stand in here: its own error is
#: around `1e-11` in the iterate, so a run would stop converging at a tolerance far
#: coarser than the one this survey is sizing, and the count would measure the
#: difference rather than the scheme. `_slopes_match_a_difference` is what keeps these
#: honest against the declarations they differentiate.
SLOPES: dict[str, Callable[[float], float]] = {
    "quadratic": lambda state: 2.0 * state,
    "exponential": math.exp,
    "tanh": lambda state: 0.5 * (1.0 - math.tanh(state) ** 2),
    "sin": lambda state: 0.5 * math.cos(state),
    "constant": lambda _: 0.0,
}


#: The curvatures the operating-range sweep covers. `1` is the declared quadratic
#: family and the rest bracket it, since no curvature range is registered anywhere yet
#: and D2 sweeps this axis.
OPERATING_CURVATURES = (0.1, 1.0, 5.0, 20.0, 100.0)

#: Where the reading sits, in predictive standard deviations off the prior mean. The
#: gap sweeps an observation box nine of these wide, so the rung is called at every one
#: of them. The bulk carries the predictive mass and the edge carries almost none, and
#: they are counted apart because they answer different questions: what a declared
#: budget has to cover, and where it has to report `VOID` instead.
BULK_EDGES = (0.0, 1.0, 2.0)
BOX_EDGES = (4.5, 9.0)


def counts_over_the_operating_range(
    tolerance: float, *, log_block: bool = False
) -> dict[float, tuple[int, int]]:
    """The worst iteration count per curvature, in the bulk and out at the box edge.

    The declared spreads give the prior. `R(x) = 1 + kappa x^2` gives the noise, so the
    predictive spread is `sqrt(P + R(mu))` and the reading is placed in units of it.

    Args:
        tolerance: stop when the step falls below this, relative to the prior spread.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        For each curvature, the worst count in the bulk and the worst at the edge.
    """
    worst = {}
    for curvature in OPERATING_CURVATURES:
        counts = {"bulk": 0, "edge": 0}
        for spread in SIGMAS:
            predictive = math.sqrt(spread**2 + 1.0 + curvature)
            for where, offsets in (("bulk", BULK_EDGES), ("edge", BOX_EDGES)):
                for offset in offsets:
                    _, _, taken = scheme.iterate(
                        observation=1.0 + offset * predictive,
                        prior_mean=1.0,
                        prior_variance=spread**2,
                        base_noise=1.0,
                        curvature=curvature,
                        scale=1.0,
                        log_block=log_block,
                        tolerance=tolerance,
                        max_iterations=PROBE_BUDGET,
                    )
                    counts[where] = max(counts[where], taken)
        worst[curvature] = (counts["bulk"], counts["edge"])
    return worst


def noise_at_of(family: NoiseFamily) -> scheme.NoiseAt:
    """A declared family read as `(R(x), R'(x))`, with the slope from `SLOPES`.

    Args:
        family: the declared `R`.

    Returns:
        A callable giving the noise and its slope at a state.
    """
    slope = SLOPES[family.key]

    def noise_at(state: float) -> tuple[float, float]:
        value = float(np.asarray(family.noise(np.asarray(state, dtype=float))))
        return value, slope(state)

    return noise_at


def _slopes_match_a_difference(states: tuple[float, ...]) -> dict[str, float]:
    """The worst gap between each declared slope and a difference of its family.

    Args:
        states: where to compare them.

    Returns:
        The worst absolute gap per family.
    """
    step = 1e-6
    worst = {}
    for key, family in FAMILIES.items():
        gaps = []
        for state in states:
            ahead = float(np.asarray(family.noise(np.asarray(state + step))))
            behind = float(np.asarray(family.noise(np.asarray(state - step))))
            gaps.append(abs(SLOPES[key](state) - (ahead - behind) / (2.0 * step)))
        worst[key] = max(gaps)
    return worst


def departure_at_budget(
    case: WorkedCase, budget: int, *, log_block: bool = False
) -> tuple[float, float]:
    """How far a run cut off at `budget` is from the same run allowed to finish.

    Args:
        case: the model numbers.
        budget: the number of iterations the cut run is allowed.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The relative departure of the mean and of the posterior variance.
    """
    settled_mean, settled_variance, _ = scheme.iterate(
        **case,
        scale=1.0,
        log_block=log_block,
        tolerance=PROBE_TOLERANCE,
        max_iterations=PROBE_BUDGET,
    )
    mean, variance, _ = scheme.iterate(
        **case,
        scale=1.0,
        log_block=log_block,
        tolerance=0.0,
        max_iterations=budget,
    )
    return (
        abs(mean - settled_mean) / abs(settled_mean),
        abs(variance - settled_variance) / settled_variance,
    )


def convergence_counts(*, log_block: bool = False) -> dict[tuple[str, float], int]:
    """Iterations the modified scheme needs on every declared cell.

    The prior mean is the family's own, the reading is put two prior spreads off it,
    and the prior variance is the declared spread squared. What varies between cells is
    the family and the spread, which is the grid route 2 surveyed.

    Args:
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The count for each family and spread.
    """
    counts = {}
    for key, family in FAMILIES.items():
        noise_at = noise_at_of(family)
        for spread in SIGMAS:
            _, _, taken = scheme.iterate_with(
                observation=family.prior_mean + 2.0 * spread,
                prior_mean=family.prior_mean,
                prior_variance=spread**2,
                noise_at=noise_at,
                scale=1.0,
                log_block=log_block,
                tolerance=PROBE_TOLERANCE,
                max_iterations=PROBE_BUDGET,
            )
            counts[key, spread] = taken
    return counts


def modification_cost(case: WorkedCase, budget: int) -> tuple[float, float]:
    """The gap between the printed scheme and the modification at one budget.

    Args:
        case: the model numbers.
        budget: the iterations both runs are allowed.

    Returns:
        The relative gap in the mean and in the posterior variance.
    """
    printed_mean, printed_variance, _ = scheme.iterate(
        **case, scale=1.0, log_block=True, tolerance=0.0, max_iterations=budget
    )
    mean, variance, _ = scheme.iterate(
        **case, scale=1.0, log_block=False, tolerance=0.0, max_iterations=budget
    )
    return (
        abs(printed_mean - mean) / abs(mean),
        abs(printed_variance - variance) / variance,
    )


def main() -> None:
    """Run all three measurements and assert what each one decides."""
    print(f"Route 5. Probe budget {PROBE_BUDGET}, tolerance {PROBE_TOLERANCE:.0e}.")

    print("\nBoth halves move when the budget is cut. Modified scheme, budget one.")
    print(
        f"{'kappa':>8} {'prior var':>11} {'mean departure':>17}"
        f" {'variance departure':>20}"
    )
    truncated = []
    for case in TRUNCATION_GRID:
        mean_gap, variance_gap = departure_at_budget(case, 1)
        truncated.append((case, mean_gap, variance_gap))
        print(
            f"  {case['curvature']:>6} {case['prior_variance']:>11}"
            f" {mean_gap:>17.6g} {variance_gap:>20.6g}"
        )
    # Q7's claim, and the reason a truncated step is `VOID` rather than a number.
    assert all(mean_gap > 0.0 for _, mean_gap, _ in truncated), truncated
    assert all(variance_gap > 0.0 for _, _, variance_gap in truncated), truncated
    # The covariance is not a bystander with an error bounded by the mean's. On the
    # broadest prior at the strongest curvature it is out by more than the mean is, so
    # a caller who trusted the covariance of a truncated run would be worse off than
    # one who trusted its mean.
    worst_variance = max(variance_gap for _, _, variance_gap in truncated)
    dominated = [
        case["curvature"]
        for case, mean_gap, variance_gap in truncated
        if variance_gap > mean_gap
    ]
    print(f"\n  worst variance departure {worst_variance:.4g}")
    print(f"  cells where the variance is out by more than the mean: {len(dominated)}")
    assert worst_variance > 0.4, worst_variance
    assert dominated, truncated

    print("\nBoth settle as the budget grows. Slow case, modified scheme.")
    print(f"{'budget':>8} {'mean departure':>18} {'variance departure':>20}")
    cuts = [
        (budget, *departure_at_budget(SLOW_CASE, budget)) for budget in (1, 2, 3, 5)
    ]
    for budget, mean_gap, variance_gap in cuts:
        print(f"  {budget:>6} {mean_gap:>18.6g} {variance_gap:>20.6g}")
    # The failure is truncation and not the scheme. A departure that stayed flat as the
    # budget grew would mean something else was wrong.
    for wider, tighter in pairwise(cuts):
        assert tighter[1] < wider[1], (wider, tighter)
        assert tighter[2] < wider[2], (wider, tighter)

    print("\nThe declared slopes against a difference of the declarations.")
    agreement = _slopes_match_a_difference((-1.0, 0.0, 0.5, 1.0, 2.0))
    for key, gap in agreement.items():
        print(f"  {FAMILIES[key].name:<24} worst gap {gap:.3e}")
    # A wrong slope would make every count below wrong in a way nothing else catches.
    assert all(gap < 1e-6 for gap in agreement.values()), agreement

    print("\nWhat the declared grid needs, modified scheme, tolerance 1e-14.")
    counts = convergence_counts()
    print(f"{'family':<26} " + " ".join(f"{spread:>6.2f}" for spread in SIGMAS))
    for key, family in FAMILIES.items():
        row = " ".join(f"{counts[key, spread]:>6}" for spread in SIGMAS)
        print(f"  {family.name:<24} {row}")
    survey_worst = max(counts.values())
    print(f"\nWorst cell: {survey_worst} iterations. No cell reached the probe budget.")
    # A budget is read off this. The declaration is an ADR's to make, and what it has
    # to cover is the worst cell here rather than the typical one.
    assert survey_worst < PROBE_BUDGET, survey_worst
    assert max(counts.values()) > min(counts.values()), counts

    print("\nWhat the whole operating range needs, not only the declared families.")
    for tolerance in (1e-14, 1e-12):
        print(f"  tolerance {tolerance:.0e}")
        print(f"  {'kappa':>7} {'worst in the bulk':>19} {'worst at the box edge':>23}")
        worst = counts_over_the_operating_range(tolerance)
        for curvature, (bulk, edge) in worst.items():
            print(f"  {curvature:>7} {bulk:>19} {edge:>23}")
        print(
            f"    bulk {max(bulk for bulk, _ in worst.values())},"
            f" edge {max(edge for _, edge in worst.values())}"
        )
    tight = counts_over_the_operating_range(1e-12)
    # The numbers a budget declaration reads. The declared families understate this by
    # a factor of six, because they are run at a reading two spreads out and the gap
    # runs the rung all the way to its box edge.
    assert max(bulk for bulk, _ in tight.values()) == 23, tight
    assert max(edge for _, edge in tight.values()) == 65, tight
    assert all(edge > bulk for bulk, edge in tight.values()), tight
    # The edge is where the count runs away and it is also where nothing is weighed.
    edge_weight = math.exp(-0.5 * BOX_EDGES[-1] ** 2)
    print(
        f"\n  Predictive weight at the box edge, against the centre: {edge_weight:.2e}"
    )
    assert edge_weight < 1e-17, edge_weight

    print("\nWhat the deletion costs at a finite budget, on the slow case.")
    print(f"{'budget':>8} {'mean gap':>18} {'variance gap':>20}")
    costs = [(budget, *modification_cost(SLOW_CASE, budget)) for budget in (1, 2, 3, 5)]
    for budget, mean_gap, variance_gap in costs:
        print(f"  {budget:>6} {mean_gap:>18.6g} {variance_gap:>20.6g}")
    # ADR-057 argues the deletion is surgical and the argument is symbolic. Here is the
    # number: the two schemes differ at a budget of one and the difference closes as
    # both converge to the same fixed point, which is what "surgical" has to mean.
    assert costs[0][1] > 0.0, costs[0]
    assert costs[-1][1] < costs[0][1], costs
    settled = modification_cost(SLOW_CASE, PROBE_BUDGET)
    print(f"  {PROBE_BUDGET:>6} {settled[0]:>18.6g} {settled[1]:>20.6g}")
    assert settled[0] < 1e-12, settled
    assert settled[1] < 1e-12, settled

    print("\nWhat this decides:")
    print("  - A truncated run is wrong in the mean and in the covariance. The")
    print(f"    covariance is out by up to {worst_variance:.3g} at a budget of one,")
    print(
        f"    and by more than the mean on {len(dominated)} of"
        f" {len(TRUNCATION_GRID)} declared cells."
    )
    print("    `VOID` is the right routing for a step that did not converge.")
    print(
        f"  - The declared grid's worst cell needs {survey_worst} iterations at 1e-14."
    )
    print("    Over the whole operating range at 1e-12 it is 23 in the bulk and 65")
    print("    at the box edge, where the predictive weight is 2.6e-18.")
    print("  - The two curvatures differ only before convergence. Run out, they agree")
    print("    to machine precision, so the deletion moves no fixed point.")


if __name__ == "__main__":
    main()
