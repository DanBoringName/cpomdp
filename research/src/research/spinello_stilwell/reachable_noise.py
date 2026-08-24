"""Can one unit choice clear the pole for every declared noise family?

Route 2 of `research/spinello_stilwell_rung.md`. The Gauss-Newton matrix of
Spinello-Stilwell (35d) is singular where the observation noise equals one, and
rescaling the observation by `lambda` moves that pole to `noise = lambda^-2` without
moving the converged estimate (`invariance`). So a `lambda` that puts a family's whole
reachable noise on one side of one removes the hazard by construction.

Whether such a `lambda` exists is a property of each declared family, and this measures
it. Run::

    uv run --no-sync python -m research.spinello_stilwell.reachable_noise

The families are read from `research.checks.gap_kernel.FAMILIES` and not extended. That
dict is a registered set, and a survey has no business adding to what it surveys. The
ridge is declared locally instead, since it is the registered operating point and is not
in that dict.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. Everything here says `noise` for the
paper's quantity and `spread` for cpomdp's.
"""

import math

import numpy as np

from research.checks.gap_kernel import FAMILIES, SIGMAS, NoiseFamily

__all__ = [
    "MARGIN",
    "REACH_IN_SPREADS",
    "RIDGE",
    "clearing_scale",
    "main",
    "reachable_noise",
]

#: How far out the prior the reachable set is taken to run, in prior standard
#: deviations. The gap quadrature already sizes its `y` grid at nine, and the
#: Gauss-Newton iterate is not confined to the prior's bulk, so a narrower reach would
#: understate the hazard rather than measure it.
REACH_IN_SPREADS = 9.0

#: How far clear of the pole a unit choice has to put the reachable set. A factor rather
#: than an absolute, since the pole is multiplicative in the rescaling.
MARGIN = 1.2

#: The registered ridge, `R(x) = R0 + kappa x^2` evaluated at `mu* = sqrt(R0/kappa)`.
#: Declared here because it is not in `FAMILIES` and because `R0 = 1/2` is the case the
#: write-up names: the noise at the operating point is `2 R0`, so the pole falls on it.
RIDGE = NoiseFamily(
    name="R0 + kappa x^2 at mu*, R0 = 1/2",
    key="ridge_half",
    noise=lambda x: 0.5 + 1.0 * np.asarray(x, dtype=float) ** 2,
    prior_mean=math.sqrt(0.5 / 1.0),
    unbounded=True,
)


def reachable_noise(
    family: NoiseFamily, spread: float, reach: float = REACH_IN_SPREADS
) -> tuple[float, float]:
    """The smallest and largest noise over the states a run of this family reaches.

    Args:
        family: the declared `R` and its prior mean.
        spread: the prior standard deviation.
        reach: how many prior standard deviations to sweep either side of the mean.

    Returns:
        The minimum and maximum of `R(x)` over that window.
    """
    half_width = reach * spread
    states = np.linspace(
        family.prior_mean - half_width, family.prior_mean + half_width, 20001
    )
    values = np.asarray(family.noise(states), dtype=float)
    return float(values.min()), float(values.max())


def clearing_scale(lowest_noise: float, margin: float = MARGIN) -> float:
    """The smallest `lambda` putting `lowest_noise` a factor `margin` above the pole.

    Rescaling sends `R -> lambda^2 R`, so clearing the pole from above needs
    `lambda^2 * lowest_noise >= margin`.

    Args:
        lowest_noise: the smallest reachable `R`.
        margin: how far above one the rescaled minimum must sit.

    Returns:
        The required `lambda`.
    """
    return math.sqrt(margin / lowest_noise)


def _infimum_over_the_line(family: NoiseFamily) -> float:
    """The greatest lower bound of `R` over the whole state line, by wide sweep.

    A family whose infimum is zero has no `lambda` that clears the pole for every state,
    only one that clears it for a declared box. That distinction is what decides whether
    the rung needs a guard, so it is measured rather than read off the family's
    `unbounded` flag, which describes growth upward and says nothing about the floor.
    """
    states = np.linspace(-40.0, 40.0, 400001)
    return float(np.asarray(family.noise(states), dtype=float).min())


def main() -> None:
    """Survey every declared family and assert what the survey decides."""
    surveyed = {**FAMILIES, RIDGE.key: RIDGE}

    print("Infimum of R over the whole line, and what it implies for a unit choice.")
    print(f"{'family':<28} {'inf R':>12} {'clears at lambda >':>20}")
    floors = {}
    for key, family in surveyed.items():
        floor = _infimum_over_the_line(family)
        floors[key] = floor
        needed = "no lambda" if floor <= 1e-9 else f"{clearing_scale(floor):.4f}"
        print(f"  {family.name:<26} {floor:>12.6g} {needed:>20}")

    # Four of the declared families never dip below one, so any lambda above one puts
    # their whole state line clear. This is a property of how they were declared.
    for key in ("quadratic", "tanh", "sin", "constant"):
        assert floors[key] >= 1.0 - 1e-9, (key, floors[key])
    # `sin` reaches exactly one, so at lambda = 1 the pole is attained rather than
    # merely approached. It is the family that makes the unit choice compulsory rather
    # than prudent.
    assert abs(floors["sin"] - 1.0) < 1e-6, floors["sin"]
    # The ridge at R0 = 1/2 sits below one, which is the case the write-up names.
    assert floors["ridge_half"] < 1.0, floors["ridge_half"]
    # The exponential's floor is zero, so no single lambda clears it everywhere.
    assert floors["exponential"] < 1e-9, floors["exponential"]

    print("\nOver the reachable window only, at each declared prior spread.")
    print(f"{'family':<28} {'spread':>7} {'min R':>11} {'max R':>11} {'lambda':>9}")
    worst_scale = 0.0
    for family in surveyed.values():
        for spread in SIGMAS:
            low, high = reachable_noise(family, spread)
            scale = clearing_scale(low)
            worst_scale = max(worst_scale, scale)
            print(
                f"  {family.name:<26} {spread:>7.2f} {low:>11.5g} {high:>11.5g} "
                f"{scale:>9.4f}"
            )

    print(f"\nOne lambda for every family at every declared spread: {worst_scale:.4f}")
    # A single declared unit choice covers the whole survey, which is what makes this a
    # convention rather than a per-cell tuning.
    for key, family in surveyed.items():
        for spread in SIGMAS:
            low, _ = reachable_noise(family, spread)
            assert worst_scale**2 * low >= MARGIN - 1e-9, (key, spread, low)

    print("\nWhat this decides:")
    print("  - Four families never reach below R = 1, so any lambda > 1 clears them.")
    print("  - `sin` attains R = 1 exactly, so lambda = 1 puts the pole on the family.")
    print("  - The ridge at R0 = 1/2 sits below R = 1 at its operating point.")
    print(
        "  - `exp(x)` has an infimum of zero, so no lambda clears it for every state."
    )
    print("    It is cleared only over a declared box, and the required lambda grows")
    print("    without bound as that box widens. That family, alone, is where a guard")
    print("    would still be needed.")


if __name__ == "__main__":
    main()
