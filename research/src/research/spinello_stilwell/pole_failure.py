"""Route 3: the printed curvature either side of the pole, and which side is silent.

Route 3 of `research/spinello_stilwell_rung.md`. Q3 states the failure twice over. From
above, `1/ln noise` runs to plus infinity, so the step collapses and a truncated run
hands back the prediction with a converged run's shape. From below, `1/P + R` passes
through zero and turns indefinite, so the step is unbounded at the crossing and uphill
past it.

ADR-057 deleted the block, so neither side reaches the shipped rung. The printed scheme
still runs here, and this is where the two failures are measured rather than argued.
Run::

    uv run --no-sync python -m research.spinello_stilwell.pole_failure

Not the rung. No guard, no warrant, and every run prints the budget it took.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity here and
`spread` is cpomdp's.
"""

import math
from itertools import pairwise
from typing import NamedTuple

from research.spinello_stilwell import scheme
from research.spinello_stilwell.invariance import (
    PROBE_BUDGET,
    PROBE_TOLERANCE,
    WorkedCase,
)

__all__ = [
    "OFFSETS",
    "RIDGE",
    "Step",
    "case_at",
    "curvature_across_the_pole",
    "main",
    "objective_at",
    "singular_state",
    "step_at",
    "trajectory",
]

#: How close to the pole the sweep gets, as a factor off `noise = 1`. Four decades,
#: because a term going as `1/ln noise` moves slowly enough that one decade proves
#: nothing about the limit.
OFFSETS = (1e-2, 1e-4, 1e-6, 1e-8)

#: Where the prediction is put when a case is built at a chosen distance from the pole.
#: Off the origin, because `R'(0) = 0` on a symmetric ridge kills the deleted block for
#: a reason that has nothing to do with the pole.
_POLE_STATE = 0.5
_POLE_CURVATURE = 1.0
_PRIOR_VARIANCE = 0.04
_OBSERVATION = 1.7

#: The registered ridge, `R0 = 1/2` and `kappa = 1`. Its noise is one at
#: `x = +/- sqrt(1/2)`, which is also its operating point `mu* = sqrt(R0/kappa)`, so the
#: pole sits on the operating point and the crossing below sits inside it. Nothing runs
#: a trajectory from here: what is measured is the step from a state, and the prior
#: stays where the case declares it.
RIDGE: WorkedCase = {
    "observation": _OBSERVATION,
    "prior_mean": math.sqrt(0.5),
    "prior_variance": _PRIOR_VARIANCE,
    "base_noise": 0.5,
    "curvature": 1.0,
}


class Step(NamedTuple):
    """One iteration of (35), with the two numbers that decided it."""

    estimate: float
    noise: float
    curvature: float
    step: float
    objective: float


def case_at(offset: float) -> WorkedCase:
    """A worked case whose prediction sits at `noise = 1 + offset`.

    The base noise is solved for rather than swept, so the distance from the pole is
    the only thing that changes between cases.

    Args:
        offset: how far above the pole to put the prediction. Negative goes below.

    Returns:
        The case, with its noise slope non-zero at the prediction.
    """
    return {
        "observation": _OBSERVATION,
        "prior_mean": _POLE_STATE,
        "prior_variance": _PRIOR_VARIANCE,
        "base_noise": 1.0 + offset - _POLE_CURVATURE * _POLE_STATE**2,
        "curvature": _POLE_CURVATURE,
    }


def objective_at(case: WorkedCase, state: float) -> float:
    """Equation (18) at a state, for the run this case declares."""
    noise, _ = scheme.quadratic_noise(case["base_noise"], case["curvature"])(state)
    return scheme.objective(
        state,
        case["prior_mean"],
        case["prior_variance"],
        noise,
        case["observation"] - state,
    )


def step_at(case: WorkedCase, state: float, *, log_block: bool) -> Step:
    """The step (35) takes from one state, and the two numbers that decided it.

    The state is separate from the case's prior mean on purpose. What the pole does to
    an iterate that has wandered onto it is the question, and the prior stays where it
    was declared while that happens.

    Args:
        case: the model numbers.
        state: where the iterate currently is.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        Where the step lands, with the noise, the curvature and the objective there.
    """
    noise, noise_slope = scheme.quadratic_noise(case["base_noise"], case["curvature"])(
        state
    )
    residual = case["observation"] - state
    prior_precision = 1.0 / case["prior_variance"]
    curvature = scheme.gauss_newton_curvature(
        noise, noise_slope, 1.0, residual, log_block=log_block
    )
    step = (
        prior_precision * (state - case["prior_mean"])
        + scheme.gradient(noise, noise_slope, 1.0, residual)
    ) / (prior_precision + curvature)
    landed = state - step
    return Step(landed, noise, curvature, step, objective_at(case, landed))


def trajectory(
    case: WorkedCase,
    *,
    log_block: bool,
    max_iterations: int,
    tolerance: float,
) -> list[Step]:
    """Every iterate of a run, which is what a stall has to be read off.

    `scheme.iterate` returns the destination. A failure that consists of not moving is
    invisible in a destination and plain in the sequence.

    Args:
        case: the model numbers.
        log_block: keep the `r3` block of (35d). False is the modification.
        max_iterations: give up after this many steps.
        tolerance: stop when the step falls below this, relative to the prior spread.

    Returns:
        One `Step` per iteration taken, in order.
    """
    state = case["prior_mean"]
    taken = []
    for _ in range(max_iterations):
        moved = step_at(case, state, log_block=log_block)
        taken.append(moved)
        state = moved.estimate
        if abs(moved.step) < tolerance * math.sqrt(case["prior_variance"]):
            break
    return taken


def curvature_across_the_pole() -> list[tuple[float, float, float]]:
    """The printed curvature at each offset, above the pole and below it.

    Returns:
        One row per offset: the offset, the curvature above, the curvature below.
    """
    rows = []
    for offset in OFFSETS:
        above = step_at(case_at(offset), _POLE_STATE, log_block=True)
        below = step_at(case_at(-offset), _POLE_STATE, log_block=True)
        rows.append((offset, above.curvature, below.curvature))
    return rows


def singular_state(case: WorkedCase) -> float:
    """The state below the pole where `1/P + R` is zero, by bisection.

    At the origin the noise slope vanishes, so the deleted block does too and the sum
    is the prior precision plus a real square. Approaching the pole from inside, the
    block runs to minus infinity and the sum follows. A crossing sits between them.

    Args:
        case: the model numbers. Its ridge has to reach `noise = 1`.

    Returns:
        The state where the sum changes sign.

    Raises:
        ValueError: if the two ends do not bracket a crossing.
    """
    prior_precision = 1.0 / case["prior_variance"]
    edge = math.sqrt((1.0 - case["base_noise"]) / case["curvature"])

    def total(state: float) -> float:
        return prior_precision + step_at(case, state, log_block=True).curvature

    low, high = 0.0, edge * (1.0 - 1e-12)
    if total(low) <= 0.0 or total(high) >= 0.0:
        raise ValueError(f"no sign change between {low} and {high}")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if total(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def main() -> None:
    """Measure both sides of the failure and assert what each one shows."""
    print(
        f"Route 3, printed scheme. Probe budget {PROBE_BUDGET},"
        f" tolerance {PROBE_TOLERANCE:.0e}."
    )

    print("\nThe printed curvature is not evaluable on the pole itself.")
    try:
        scheme.gauss_newton_curvature(1.0, 1.0, 1.0, 1.2, log_block=True)
    except ZeroDivisionError as failure:
        print(f"  printed: ZeroDivisionError({failure})")
    else:  # pragma: no cover - the assert below is the check
        raise AssertionError("the printed curvature evaluated at noise = 1")
    modified = scheme.gauss_newton_curvature(1.0, 1.0, 1.0, 1.2, log_block=False)
    print(f"  modified: {modified:.6g}")
    assert abs(modified - 2.56) < 5e-3, modified

    print("\nEither side of the pole, at the prediction.")
    print(f"{'offset':>10} {'curvature above':>18} {'curvature below':>18}")
    rows = curvature_across_the_pole()
    for offset, above, below in rows:
        print(f"  {offset:>8.0e} {above:>18.6g} {below:>18.6g}")
    # From above the block is positive and runs to plus infinity, from below it is
    # negative and runs to minus infinity. Both are the same `1/ln noise`.
    assert all(above > 0.0 for _, above, _ in rows), rows
    assert all(below < 0.0 for _, _, below in rows), rows
    for (_, above, below), (_, closer_above, closer_below) in pairwise(rows):
        assert closer_above > above, (above, closer_above)
        assert closer_below < below, (below, closer_below)

    print("\nThe step the first iteration takes, either side.")
    print(f"{'offset':>10} {'step above':>16} {'step below':>16}")
    firsts = []
    for offset in OFFSETS:
        above = step_at(case_at(offset), _POLE_STATE, log_block=True)
        below = step_at(case_at(-offset), _POLE_STATE, log_block=True)
        firsts.append((offset, above.step, below.step))
        print(f"  {offset:>8.0e} {above.step:>16.6g} {below.step:>16.6g}")
    # Close in, the step collapses in proportion to the distance from the pole, on
    # both sides. The sign is what separates them and the size does not show it. The
    # widest offset below is the exception, and it is the crossing measured further
    # down rather than a failure of the pattern.
    for offset, above_step, below_step in firsts[1:]:
        assert abs(above_step) < 10.0 * offset, (offset, above_step)
        assert abs(below_step) < 10.0 * offset, (offset, below_step)
    for (_, wider, _), (_, closer, _) in pairwise(firsts):
        assert abs(closer) < abs(wider), (wider, closer)

    print("\nThe silent side: what a tolerance has to be finer than, above the pole.")
    closest = case_at(OFFSETS[-1])
    spread = math.sqrt(closest["prior_variance"])
    first = step_at(closest, _POLE_STATE, log_block=True)
    silent_above = abs(first.step) / spread
    converged = trajectory(
        closest,
        log_block=True,
        max_iterations=PROBE_BUDGET,
        tolerance=PROBE_TOLERANCE,
    )
    repaired = trajectory(
        closest,
        log_block=False,
        max_iterations=PROBE_BUDGET,
        tolerance=PROBE_TOLERANCE,
    )
    prediction = closest["prior_mean"]
    answer = converged[-1].estimate
    agreement = abs(answer - repaired[-1].estimate)
    print(f"  first printed step {first.step:.6g}, prior spread {spread}")
    print(f"  any relative tolerance above {silent_above:.3g} stops the run at one")
    print(f"  the run stopped there reports {prediction - first.step:.9f}")
    print(f"  run out, both curvatures agree on {answer:.9f} to {agreement:.1e}")
    print(f"  so the reported estimate is wrong by {abs(answer - prediction):.4g}")
    print(
        f"  iterations to get there: printed {len(converged)}, modified {len(repaired)}"
    )
    stalled = trajectory(
        closest,
        log_block=True,
        max_iterations=PROBE_BUDGET,
        tolerance=10.0 * silent_above,
    )
    # The severity-one case. At a tolerance a factor of ten coarser than the collapsed
    # step the run stops on its first iteration, and stopping early is what a converged
    # run does. The number it hands back is the prediction.
    assert len(stalled) == 1, len(stalled)
    assert abs(stalled[-1].estimate - prediction) < 1e-7, stalled[-1].estimate
    # Both estimates are quoted to nine figures, so both are pinned there. The gap
    # between them is the error a caller is handed for stopping on iteration one.
    assert abs(prediction - first.step - 0.500000057) < 5e-10, first.step
    assert abs(answer - 0.549191167) < 5e-10, answer
    assert abs(abs(answer - prediction) - 0.049191) < 5e-7, answer
    # The two curvatures share this fixed point, which is what makes the printed run's
    # extra iterations a cost and not a different answer.
    assert agreement < 1e-15, (answer, repaired[-1].estimate)
    # The distance to the pole sets the threshold, so no declared tolerance is safe for
    # every case. Halving the offset halves the tolerance that would have caught it.
    nearer = case_at(OFFSETS[-1] / 100.0)
    nearer_step = step_at(nearer, _POLE_STATE, log_block=True).step
    assert abs(nearer_step) < 0.02 * abs(first.step), (first.step, nearer_step)
    # The modification has no threshold. Its first step is the ordinary one, so the
    # same tolerance leaves it running.
    assert abs(step_at(closest, _POLE_STATE, log_block=False).step) > 1e-2
    # The printed scheme also pays for the collapse in iterations, which is the
    # per-cycle cost RFC-001 accounts for. Both counts are quoted in the write-up.
    assert (len(converged), len(repaired)) == (24, 12), (converged, repaired)

    print("\nThe loud side: where `1/P + R` crosses zero, on the registered ridge.")
    crossing = singular_state(RIDGE)
    print(f"  crossing at x = {crossing:.12f}")
    # Quoted to twelve figures in the write-up, so pinned to them here. It sits inside
    # the pole, which is what "the sum turns indefinite before the noise reaches one"
    # means.
    assert abs(crossing - 0.694474243706) < 5e-13, crossing
    assert crossing < math.sqrt(0.5), crossing
    print(f"{'distance':>12} {'step inside':>16} {'step outside':>16}")
    approach = []
    for distance in (1e-8, 1e-9, 1e-10):
        inside = step_at(RIDGE, crossing * (1.0 - distance), log_block=True)
        outside = step_at(RIDGE, crossing * (1.0 + distance), log_block=True)
        approach.append((distance, inside.step, outside.step))
        print(f"  {distance:>10.0e} {inside.step:>16.6g} {outside.step:>16.6g}")
    # A sum passing through zero is a step passing through infinity. The two sides
    # point opposite ways and neither is bounded: a tenth of the distance is close to
    # ten times the step.
    for _, inside_step, outside_step in approach:
        assert inside_step * outside_step < 0.0, (inside_step, outside_step)
    for (_, wider, _), (_, closer, _) in pairwise(approach):
        assert abs(closer) > 5.0 * abs(wider), (wider, closer)
    assert abs(abs(approach[-1][1]) - 8.81e6) < 5e3, approach[-1]

    print("\nPast the crossing the step climbs the objective.")
    edge = math.sqrt((1.0 - RIDGE["base_noise"]) / RIDGE["curvature"])
    uphill = 0.5 * (crossing + edge)
    before = objective_at(RIDGE, uphill)
    climbed = step_at(RIDGE, uphill, log_block=True)
    descended = step_at(RIDGE, uphill, log_block=False)
    print(f"  at x = {uphill:.6f}, objective {before:.9f}")
    print(
        f"  printed lands at {climbed.estimate:.6f}, objective {climbed.objective:.9f}"
    )
    print(
        f"  modified lands at {descended.estimate:.6f}, "
        f"objective {descended.objective:.9f}"
    )
    # An indefinite step matrix is not a descent direction, which is the whole of Q3's
    # second half. The modification is a real square there, so it descends.
    assert 1.0 / RIDGE["prior_variance"] + climbed.curvature < 0.0, climbed.curvature
    assert climbed.objective > before, (before, climbed.objective)
    assert descended.objective < before, (before, descended.objective)

    print("\nWhat this measures:")
    print("  - The printed curvature has no value at noise = 1 and no bound near it.")
    print("  - Above the pole the step collapses in proportion to the distance, so")
    print("    every tolerance has a neighbourhood in which the run stops on its")
    print("    first iteration and reports the prediction as converged.")
    print("  - Below the pole the sum passes through zero, so the step is unbounded")
    print("    at the crossing and climbs the objective past it.")
    print("  - The modification has neither failure, which is ADR-057's claim, and")
    print("    both are properties of the printed scheme rather than of the ladder.")


if __name__ == "__main__":
    main()
