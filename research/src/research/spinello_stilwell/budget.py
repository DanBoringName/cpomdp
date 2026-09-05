"""Route 5: what a budget has to be sized against, and what exhausting one costs.

Route 5 of `research/spinello_stilwell_rung.md`, and the survey the budget declaration
reads from. Three things get measured here.

Q7 says a truncated run returns a wrong covariance as well as a wrong mean, because
`U-bar` is evaluated at the current iterate. That is the case for routing a
non-convergent step to `VOID` rather than reporting it, and it is measured rather than
argued.

A budget is a number, so something has to size it. Three sweeps do: the declared
families at a reading two prior spreads out, the curvature axis out to the observation
box's edge, and the declared families over the same offsets. The last is what says
whether a declared cell can exhaust a declared budget.

ADR-057 argues the deletion is surgical. The comparison here is what that costs at a
finite budget: same case, same budget, both curvatures, the departure per iterate. Run::

    uv run --no-sync python -m research.spinello_stilwell.budget

Not the rung. ADR-058 declares the rung's budget off these counts. Nothing here runs to
it, and no warrant is reported. Every number printed says which budget produced it.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity here and
`spread` is cpomdp's.
"""

import math
from collections.abc import Callable
from itertools import pairwise

import numpy as np

from research.checks.gap_kernel import (
    FAMILIES,
    SIGMAS,
    NoiseFamily,
    plugin_noise_of,
)
from research.spinello_stilwell import scheme
from research.spinello_stilwell.invariance import (
    PROBE_BUDGET,
    PROBE_TOLERANCE,
    WorkedCase,
)

__all__ = [
    "BOX_EDGES",
    "BULK_EDGES",
    "DECLARED_BUDGET",
    "OPERATING_CURVATURES",
    "OUTSIDE_THE_BOX",
    "SLOPES",
    "SLOW_CASE",
    "TRUNCATION_GRID",
    "convergence_counts",
    "counts_over_the_declared_families",
    "counts_over_the_operating_range",
    "departure_at_budget",
    "main",
    "modification_cost",
    "noise_at_of",
    "slopes_match_a_difference",
    "three_iterates_at",
]

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
#: difference rather than the scheme. `slopes_match_a_difference` is what keeps these
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

#: ADR-058's budget, read back so the sweeps can report which cells exhaust it. Nothing
#: here runs to it: a probe that stopped at the declared cap would report the cap
#: instead of the count the cap has to be judged against.
DECLARED_BUDGET = 64

#: Where the reading sits, in predictive standard deviations off the prior mean. The
#: gap's observation box runs nine of these out on either side of it
#: (`measure_truncation`'s default multiplier, a half-width), so `9.0` is the edge and
#: `4.5` is half way to it. The bulk carries the predictive mass and the edge carries
#: almost none, and they are counted apart because they answer different questions:
#: what a declared budget has to cover, and where it has to report `VOID` instead.
BULK_EDGES = (0.0, 1.0, 2.0)
BOX_EDGES = (4.5, 9.0)

#: One offset past the registered edge, where a budget stops being the question. Nine
#: is the half-width the truncation study measured at, not a bound anyone declared, so
#: what happens outside it decides whether widening a budget is an escape.
OUTSIDE_THE_BOX = 13.0


def _worst_counts(
    noise_at: scheme.NoiseAt,
    prior_mean: float,
    prior_variance: float,
    plugin_noise: float,
    *,
    tolerance: float,
    log_block: bool,
) -> tuple[int, int]:
    """The worst count over the bulk offsets and over the box-edge ones, for one cell.

    Args:
        noise_at: the declared `R`, read as `(R(x), R'(x))`.
        prior_mean: the predicted mean, which the reading is placed off.
        prior_variance: the predicted variance.
        plugin_noise: `R(prior_mean)`, which sets the predictive spread with it.
        tolerance: stop when the step falls below this, relative to the prior spread.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The worst count in the bulk and the worst at the box edge.
    """
    predictive = math.sqrt(prior_variance + plugin_noise)

    def count_at(offset: float) -> int:
        return scheme.iterate_with(
            observation=prior_mean + offset * predictive,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            noise_at=noise_at,
            scale=1.0,
            log_block=log_block,
            tolerance=tolerance,
            max_iterations=PROBE_BUDGET,
        )[2]

    return (
        max(count_at(offset) for offset in BULK_EDGES),
        max(count_at(offset) for offset in BOX_EDGES),
    )


def counts_over_the_operating_range(
    tolerance: float, *, log_block: bool = False
) -> dict[float, tuple[int, int]]:
    """The worst iteration count per curvature, in the bulk and out at the box edge.

    The declared spreads give the prior and `R(x) = 1 + kappa x^2` gives the noise, so
    this sweeps the curvature axis past the one declared family rather than the family
    axis. `counts_over_the_declared_families` is the other half.

    Args:
        tolerance: stop when the step falls below this, relative to the prior spread.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        For each curvature, the worst count in the bulk and the worst at the edge.
    """
    worst = {}
    for curvature in OPERATING_CURVATURES:
        counts = (0, 0)
        for spread in SIGMAS:
            cell = _worst_counts(
                scheme.quadratic_noise(1.0, curvature),
                1.0,
                spread**2,
                1.0 + curvature,
                tolerance=tolerance,
                log_block=log_block,
            )
            counts = (max(counts[0], cell[0]), max(counts[1], cell[1]))
        worst[curvature] = counts
    return worst


def counts_over_the_declared_families(
    tolerance: float, *, log_block: bool = False
) -> dict[str, tuple[int, int]]:
    """The same sweep over the declared families, which the curvature axis cannot reach.

    `counts_over_the_operating_range` varies `kappa` and never leaves `1 + kappa x^2`.
    A budget has to cover the families GATE-D4 registered, at the readings the gap
    actually calls the rung at, and `exp(x)` at the box edge is nothing the quadratic
    sweep measured.

    Args:
        tolerance: stop when the step falls below this, relative to the prior spread.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        For each family, the worst count in the bulk and the worst at the edge.
    """
    worst = {}
    for key, family in FAMILIES.items():
        noise_at = noise_at_of(family)
        plugin_noise = plugin_noise_of(family)
        counts = (0, 0)
        for spread in SIGMAS:
            cell = _worst_counts(
                noise_at,
                family.prior_mean,
                spread**2,
                plugin_noise,
                tolerance=tolerance,
                log_block=log_block,
            )
            counts = (max(counts[0], cell[0]), max(counts[1], cell[1]))
        worst[key] = counts
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


def three_iterates_at(
    family: NoiseFamily,
    spread: float,
    offset: float,
    iterations: int,
    *,
    log_block: bool = False,
) -> tuple[float, float, float]:
    """Where a run stands after each of three consecutive iterations, with no tolerance.

    A run that has settled gives three equal states. A run in a two-cycle gives the
    first and third equal and the second somewhere else. Three calls of `iterate_with`
    rather than a sequence of its own, because a second copy of (35) is the thing that
    drifts.

    Args:
        family: the declared `R`.
        spread: the prior standard deviation.
        offset: where the reading sits, in predictive standard deviations.
        iterations: the last of the three iteration counts.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The estimate after `iterations - 2`, `iterations - 1` and `iterations` steps.
    """
    predictive = math.sqrt(spread**2 + plugin_noise_of(family))
    states = tuple(
        scheme.iterate_with(
            observation=family.prior_mean + offset * predictive,
            prior_mean=family.prior_mean,
            prior_variance=spread**2,
            noise_at=noise_at_of(family),
            scale=1.0,
            log_block=log_block,
            tolerance=0.0,
            max_iterations=taken,
        )[0]
        for taken in (iterations - 2, iterations - 1, iterations)
    )
    return states[0], states[1], states[2]


def slopes_match_a_difference(states: tuple[float, ...]) -> dict[str, float]:
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
    """Run every measurement and assert what each one decides."""
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
    # Q7's claim, and the reason a truncated step is `VOID` rather than a number. The
    # write-up quotes the range, so both ends of it are pinned.
    assert all(mean_gap > 0.0 for _, mean_gap, _ in truncated), truncated
    assert all(variance_gap > 0.0 for _, _, variance_gap in truncated), truncated
    assert abs(min(row[1] for row in truncated) - 0.00689) < 5e-6, truncated
    assert abs(min(row[2] for row in truncated) - 7.577e-7) < 5e-10, truncated
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
    worst_case, worst_mean, _ = max(truncated, key=lambda row: row[2])
    print(f"\n  worst variance departure {worst_variance:.4g}")
    print(
        f"  at kappa {worst_case['curvature']}, prior variance"
        f" {worst_case['prior_variance']}, where the mean is out by {worst_mean:.3g}"
    )
    print(f"  cells where the variance is out by more than the mean: {len(dominated)}")
    # The prose reports the pair, so both are pinned rather than bounded. The cell is
    # the widest prior at the strongest curvature, which the attribution also claims.
    assert (worst_case["curvature"], worst_case["prior_variance"]) == (20.0, 1.0)
    assert abs(worst_variance - 0.425) < 5e-4, worst_variance
    assert abs(worst_mean - 0.357) < 5e-4, worst_mean
    assert dominated == [20.0], dominated

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
    agreement = slopes_match_a_difference((-1.0, 0.0, 0.5, 1.0, 2.0))
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
    # This reading is two *prior* spreads out, which is inside the bulk. It is the
    # number the sweep below has to be compared against, so it is pinned here.
    assert survey_worst == 10, counts
    assert counts["constant", 0.06] == 2, counts

    print("\nThe curvature axis, at the readings the gap actually calls the rung at.")
    swept = {}
    for tolerance in (1e-14, 1e-12):
        print(f"  tolerance {tolerance:.0e}")
        print(f"  {'kappa':>7} {'worst in the bulk':>19} {'worst at the box edge':>23}")
        swept[tolerance] = counts_over_the_operating_range(tolerance)
        for curvature, (bulk, edge) in swept[tolerance].items():
            print(f"  {curvature:>7} {bulk:>19} {edge:>23}")
        print(
            f"    bulk {max(bulk for bulk, _ in swept[tolerance].values())},"
            f" edge {max(edge for _, edge in swept[tolerance].values())}"
        )
    tight = swept[1e-12]
    # The reading is out at the box edge here and two prior spreads out in the survey
    # above, so the two are not in the same units and the sweep below is what compares
    # against the declared families like for like.
    assert max(bulk for bulk, _ in tight.values()) == 23, tight
    assert max(edge for _, edge in tight.values()) == 65, tight
    assert all(edge > bulk for bulk, edge in tight.values()), tight

    print("\nThe family axis, over the same offsets, at the declared tolerance.")
    print(f"  {'family':<26} {'worst in the bulk':>19} {'worst at the box edge':>23}")
    families = counts_over_the_declared_families(1e-12)
    for key, (bulk, edge) in families.items():
        print(f"  {FAMILIES[key].name:<26} {bulk:>19} {edge:>23}")
    exhausted = [key for key, (_, edge) in families.items() if edge > DECLARED_BUDGET]
    print(
        f"\n  families whose box edge exhausts the declared budget of"
        f" {DECLARED_BUDGET}: {exhausted}"
    )
    # A declared family at a declared spread, not a bracketing curvature. `R` is
    # bounded and oscillates here, so the iterate crosses several periods of it before
    # it settles, and the count at the box edge is double the quadratic sweep's worst.
    assert max(bulk for bulk, _ in families.values()) == 16, families
    assert families["sin"] == (10, 124), families
    assert exhausted == ["sin"], families
    assert all(count < PROBE_BUDGET for cell in families.values() for count in cell)
    print("\nOne offset further out, where a budget is not the question.")
    print(f"  {'family':<26} {'x[n-2]':>16} {'x[n-1]':>16} {'x[n]':>16}")
    cycles = {}
    for key in ("sin", "quadratic"):
        cycles[key] = three_iterates_at(FAMILIES[key], 0.30, OUTSIDE_THE_BOX, 400)
        row = " ".join(f"{state:>16.9f}" for state in cycles[key])
        print(f"  {FAMILIES[key].name:<26} {row}")
    # `1.5 + 0.5 sin(x)` settles into a stable two-cycle at 13 predictive spreads and
    # stays in it for as long as it is run. No budget converts that cell into a number,
    # which is what makes `VOID` the routing and widening the budget not an escape.
    settled, alternate, back = cycles["sin"]
    assert abs(settled - back) < 1e-12, cycles["sin"]
    assert abs(alternate - back) > 1.0, cycles["sin"]
    # The unbounded family, run at the same offset, is on its fixed point by then.
    assert max(cycles["quadratic"]) - min(cycles["quadratic"]) < 1e-12, cycles
    print(
        f"  the bounded family alternates by {abs(alternate - back):.3f} and the"
        f" unbounded one has settled"
    )

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
    assert abs(costs[0][1] - 3.2e-3) < 5e-5, costs[0]
    assert abs(costs[-1][1] - 6.65e-6) < 5e-8, costs[-1]
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
        f"  - Two prior spreads out, the declared grid's worst cell needs"
        f" {survey_worst}"
    )
    print("    iterations at 1e-14. Out at the box edge, at 1e-12, the curvature")
    print("    axis needs 65 and `1.5 + 0.5 sin(x)` needs 124, where the predictive")
    print("    weight is 2.6e-18. A budget of 64 voids that node.")
    print("  - One offset past the box edge the bounded family stops converging at")
    print("    all, into a stable two-cycle, so no budget is a cure for a cell that")
    print("    exhausts one. `VOID` is doing work a wider cap cannot do.")
    print("  - The two curvatures differ only before convergence. Run out, they agree")
    print("    to machine precision, so the deletion moves no fixed point.")


if __name__ == "__main__":
    main()
