"""The three tests step 10 of the hand derivation names, and what each is blind to.

`research/spinello_stilwell_hand_derivation.md` derives the modification: drop the `r3`
block from (35d) and keep (35c), (35e) and the objective verbatim. Step 10 names three
tests for it and says what each misses. This runs them. Run::

    uv run --no-sync python -m research.spinello_stilwell.repair

1. **Unit dependence.** The printed scheme's output moves when the observation is
   rescaled and the modification's does not. At convergence neither moves, so the test
   only bites at a finite budget, which is the single-step rung (36) the ladder declares
   separately. Blind to a fixed point that is wrong in every unit alike.
2. **The fixed-noise reduction.** At `grad_noise = 0` both variants must reproduce the
   ordinary Kalman update, the rung's only external check. Blind to every term that
   carries `grad_noise`.
3. **Curvature agreement well above the pole.** As the noise grows the printed (35d) and
   the modification converge, so the change is confined to where it was argued for.
   Blind to `noise < 1`, which is the regime the whole question is about.

None of the three is the rung, and none reports a warrant. Test 1's form here is not the
form the notes predicted, and `research/spinello_stilwell_rung.md` carries that
correction as a dated entry under Q1.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity throughout
and `spread` is cpomdp's.
"""

from itertools import pairwise

from research.spinello_stilwell import scheme
from research.spinello_stilwell.invariance import (
    AT_THE_PROBE_BUDGET,
    CASE,
    PROBE_BUDGET,
    PROBE_TOLERANCE,
    SCALES,
    WorkedCase,
)

__all__ = [
    "FIXED_NOISE_CASE",
    "GROWING_NOISE",
    "curvature_departure",
    "estimate_spread",
    "fixed_noise_departure",
    "main",
]

FIXED_NOISE_CASE: WorkedCase = {
    "observation": 1.7,
    "prior_mean": 1.0,
    "prior_variance": 0.04,
    "base_noise": 2.0,
    "curvature": 0.0,
}
"""Test 2's case: the worked case with the state dependence switched off.

`curvature = 0` is `grad_noise = 0`. `base_noise` is 2 rather than the worked case's 1
because a noise of exactly 1 sits on the pole, where the printed scheme is undefined
even with no state dependence at all. `pole_is_reachable_at_fixed_noise` measures that
separately rather than hiding it behind a choice of constant.
"""

GROWING_NOISE = (2.0, 10.0, 100.0, 1.0e4, 1.0e6)
"""Test 3's noise values, all above the pole, spanning six decades."""


def estimate_spread(
    *,
    log_block: bool,
    max_iterations: int,
    tolerance: float,
) -> float:
    """How far the estimate moves across the declared unit choices.

    Runs the worked case at every scale in `invariance.SCALES` and returns the span of
    the resulting estimates. Zero is a unit-free filter.

    Args:
        log_block: keep the `r3` block of (35d). False is the modification.
        max_iterations: the budget. One is the single-step rung (36).
        tolerance: the convergence tolerance, zero to spend the budget in full.

    Returns:
        The largest estimate minus the smallest.
    """
    estimates = [
        scheme.iterate(
            **CASE,
            scale=scale,
            log_block=log_block,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )[0]
        for scale in SCALES
    ]
    return max(estimates) - min(estimates)


def fixed_noise_departure(*, log_block: bool, scale: float) -> tuple[float, float]:
    """How far the scheme lands from the Kalman update when the noise is constant.

    The oracle is exact here: the observation map is affine and the noise does not
    depend on the state, so the ordinary Kalman posterior is the Bayesian one.

    Args:
        log_block: keep the `r3` block of (35d). False is the modification.
        scale: `lambda`, the factor the observation is multiplied by.

    Returns:
        The absolute departure in the mean and in the variance.
    """
    estimate, variance, taken = scheme.iterate(
        **FIXED_NOISE_CASE, **AT_THE_PROBE_BUDGET, scale=scale, log_block=log_block
    )
    assert taken < PROBE_BUDGET, (scale, log_block, taken)
    oracle_mean, oracle_variance = scheme.kalman_update(
        FIXED_NOISE_CASE["observation"],
        FIXED_NOISE_CASE["prior_mean"],
        FIXED_NOISE_CASE["prior_variance"],
        FIXED_NOISE_CASE["base_noise"],
    )
    return abs(estimate - oracle_mean), abs(variance - oracle_variance)


def pole_is_reachable_at_fixed_noise() -> bool:
    """Whether the printed scheme is defined on the pole with no state dependence.

    At `noise = 1` the printed fourth term of (35d) is `grad_noise^2 / 0`. With
    `grad_noise = 0` that is zero over zero, so the expression has no value even in the
    one regime where the rung has an oracle to check against.

    Returns:
        True where the printed scheme fails and the modification does not.
    """
    on_the_pole: WorkedCase = {**FIXED_NOISE_CASE, "base_noise": 1.0}
    try:
        scheme.iterate(**on_the_pole, **AT_THE_PROBE_BUDGET, scale=1.0, log_block=True)
    except ZeroDivisionError:
        scheme.iterate(**on_the_pole, **AT_THE_PROBE_BUDGET, scale=1.0, log_block=False)
        return True
    return False


def curvature_departure(noise: float) -> float:
    """The printed (35d) against the modification, relative to the modification.

    The difference is the deleted block alone, `grad_noise^2 / (4 noise^2 ln noise)`, so
    this is that block measured against the real square it sits beside.

    Args:
        noise: the observation-noise variance, `sigma`.

    Returns:
        The printed curvature minus the modified, over the modified. Negative below
        the pole, where the block is.
    """
    printed = scheme.gauss_newton_curvature(noise, 2.0, 1.0, 0.3, log_block=True)
    modified = scheme.gauss_newton_curvature(noise, 2.0, 1.0, 0.3, log_block=False)
    return (printed - modified) / modified


def main() -> None:
    """Run the three tests and assert each."""
    print(
        "=== Test 1: does the reported estimate depend on the observation's units ==="
    )
    converged = {
        name: estimate_spread(
            log_block=block, max_iterations=PROBE_BUDGET, tolerance=PROBE_TOLERANCE
        )
        for name, block in (("printed", True), ("modified", False))
    }
    single = {
        name: estimate_spread(log_block=block, max_iterations=1, tolerance=0.0)
        for name, block in (("printed", True), ("modified", False))
    }
    print(
        f"  converged:    printed {converged['printed']:.3e}"
        f"   modified {converged['modified']:.3e}"
    )
    print(
        f"  budget of 1:  printed {single['printed']:.3e}"
        f"   modified {single['modified']:.3e}"
    )
    # Run to convergence the printed scheme is already unit-free, so the test only
    # separates the two at a finite budget. That budget of one is rung (36).
    assert converged["printed"] < 1e-15, converged
    assert converged["modified"] < 1e-15, converged
    assert single["printed"] > 1e-5, single
    assert single["modified"] < 1e-15, single
    print("  the printed single step is unit-dependent and the modified one is not")

    print("\n=== Test 2: the fixed-noise reduction against the Kalman oracle ===")
    for scale in SCALES:
        for name, block in (("printed", True), ("modified", False)):
            mean_gap, variance_gap = fixed_noise_departure(log_block=block, scale=scale)
            print(
                f"  lambda={scale:<5} {name:<9} mean {mean_gap:.3e}"
                f"   variance {variance_gap:.3e}"
            )
            assert mean_gap < 1e-12, (scale, name, mean_gap)
            assert variance_gap < 1e-12, (scale, name, variance_gap)
    print("  both variants reproduce the Kalman posterior in every unit choice")

    assert pole_is_reachable_at_fixed_noise()
    print("  on the pole the printed scheme is undefined and the modification is not")

    print("\n=== Test 3: how far the modification moves the curvature ===")
    departures = [curvature_departure(noise) for noise in GROWING_NOISE]
    for noise, departure in zip(GROWING_NOISE, departures, strict=True):
        print(f"  noise={noise:<9g} relative difference {departure:+.4e}")
    assert all(later < earlier for earlier, later in pairwise(departures)), departures
    assert departures[-1] < 1e-6, departures[-1]
    print("  the two agree in the limit, so the change is confined to the small noise")

    below = curvature_departure(0.9)
    printed_below = scheme.gauss_newton_curvature(0.9, 2.0, 1.0, 0.3, log_block=True)
    print(
        f"  noise=0.9   relative difference {below:+.4e}"
        f", printed curvature {printed_below:+.4e}"
    )
    # What test 3 is blind to, measured rather than asserted about: below the pole the
    # printed curvature is negative, so the step matrix can lose definiteness there.
    assert below < 0.0, below
    assert printed_below < 0.0, printed_below
    print("  below the pole the printed curvature is negative, which test 3 cannot see")


if __name__ == "__main__":
    main()
