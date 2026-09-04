"""Can one unit choice clear the pole for every declared noise family?

Route 2 of `research/spinello_stilwell_rung.md`. The Gauss-Newton matrix of
Spinello-Stilwell (35d) is singular where the observation noise equals one, and
rescaling the observation by `lambda` moves that pole to `noise = lambda^-2` without
moving the converged estimate (`invariance`). So a `lambda` that puts a family's whole
reachable noise on one side of one removes the hazard by construction.

Whether such a `lambda` exists is a property of each declared family, and this measures
it. ADR-057 went the other way and deleted the block the pole lives in, so the shipped
rungs declare no unit choice. This stays as the measurement of what a units-only repair
would have required, and the printed scheme still runs in `research/`, where routes 3,
5 and 7 need the pole reachable. Run::

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

from research.checks.gap_kernel import FAMILIES, SIGMAS, X_WINDOW_LADDER, NoiseFamily

__all__ = [
    "MARGIN",
    "REACH_IN_SPREADS",
    "RIDGE",
    "clearing_scale",
    "main",
    "reachable_noise",
]

#: How far out the prior the reachable set is taken to run, in prior standard
#: deviations. The gap quadrature's state window starts at the foot of
#: `X_WINDOW_LADDER`, so that is the narrowest reach the reference already treats as
#: the state's range, and a reach narrower than it would understate the hazard rather
#: than measure it.
REACH_IN_SPREADS = X_WINDOW_LADDER[0]

#: How far clear of the pole a unit choice has to put the reachable set. A factor rather
#: than an absolute, since the pole is multiplicative in the rescaling.
MARGIN = 1.2

#: The registered ridge, `R(x) = R0 + kappa x^2`, its prior at `mu* = sqrt(R0/kappa)`.
#: Declared here because it is not in `FAMILIES` and because `R0 = 1/2` is the case the
#: write-up names: the noise at the operating point is `2 R0`, so the pole falls on it,
#: and the floor of the whole line is `R0` itself at `x = 0`.
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
        The required `lambda`, infinite where no reachable noise is above zero.
    """
    if lowest_noise <= 0.0:
        return math.inf
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
    name_width = max(len(family.name) for family in surveyed.values())

    print("Infimum of R over the whole line, and what it implies for a unit choice.")
    print(f"{'family':<{name_width + 2}} {'inf R':>12} {'clears at lambda >':>20}")
    floors = {}
    for key, family in surveyed.items():
        floor = _infimum_over_the_line(family)
        floors[key] = floor
        needed = clearing_scale(floor)
        shown = "no lambda" if math.isinf(needed) else f"{needed:.4f}"
        print(f"  {family.name:<{name_width}} {floor:>12.6g} {shown:>20}")

    # Four of the declared families never dip below one, so a lambda that clears the
    # margin puts their whole state line clear. A property of how they were declared.
    for key in ("quadratic", "tanh", "sin", "constant"):
        assert floors[key] >= 1.0 - 1e-9, (key, floors[key])
    # Two of them reach exactly one: `1 + x^2` at the origin and `1.5 + 0.5 sin(x)`
    # wherever the sine is minus one. At lambda = 1 the pole is on both families rather
    # than near them, so a units-only repair would have had to move the pole rather
    # than widen a margin against it.
    for key in ("quadratic", "sin"):
        assert abs(floors[key] - 1.0) < 1e-6, (key, floors[key])
    # The ridge's whole-line floor is R0 at the origin, below the pole. Its operating
    # point is a different place: there the noise is 2 R0, exactly one for R0 = 1/2.
    assert abs(floors["ridge_half"] - 0.5) < 1e-6, floors["ridge_half"]
    assert abs(float(RIDGE.noise(np.asarray(RIDGE.prior_mean))) - 1.0) < 1e-12
    # The exponential's floor is zero, so no single lambda clears it everywhere.
    assert floors["exponential"] < 1e-9, floors["exponential"]

    print("\nOver the reachable window only, at each declared prior spread.")
    print(
        f"{'family':<{name_width + 2}} {'spread':>7} {'min R':>11} {'max R':>11}"
        f" {'lambda':>9}"
    )
    lows: dict[tuple[str, float], float] = {}
    for key, family in surveyed.items():
        for spread in SIGMAS:
            low, high = reachable_noise(family, spread)
            lows[key, spread] = low
            print(
                f"  {family.name:<{name_width}} {spread:>7.2f} {low:>11.5g}"
                f" {high:>11.5g} {clearing_scale(low):>9.4f}"
            )
    worst_scale = max(clearing_scale(low) for low in lows.values())

    # Which cells of the declared grid reach the pole at lambda = 1. Both priors sit
    # at one. The quadratic's minimum is one unit away, the sine's nearest is at
    # -pi/2, so the window reaches the first at far narrower spreads than the second.
    on_the_pole = {
        key: sum(lows[key, spread] <= 1.0 + 1e-6 for spread in SIGMAS)
        for key in ("quadratic", "sin")
    }
    print(f"\nCells of the declared grid on the pole at lambda = 1: {on_the_pole}")
    assert on_the_pole["quadratic"] == sum(
        REACH_IN_SPREADS * spread >= 1.0 for spread in SIGMAS
    ), on_the_pole
    assert on_the_pole["sin"] == sum(
        REACH_IN_SPREADS * spread >= 1.0 + math.pi / 2 for spread in SIGMAS
    ), on_the_pole
    assert on_the_pole["quadratic"] > on_the_pole["sin"] > 0, on_the_pole

    print(f"\nOne lambda for every family at every declared spread: {worst_scale:.4f}")
    # One scale covers the whole survey, so a units-only repair would have been a
    # convention rather than a per-cell tuning. ADR-057 declares none.
    for (key, spread), low in lows.items():
        assert worst_scale**2 * low >= MARGIN - 1e-9, (key, spread, low)

    print("\nWhat this measures, for the printed scheme:")
    print("  - Four families never reach below R = 1, so any lambda above")
    print(f"    sqrt(MARGIN) = {math.sqrt(MARGIN):.4f} clears them with the margin.")
    print("  - `1 + x^2` and `1.5 + 0.5 sin(x)` attain R = 1 exactly, so lambda = 1")
    print("    puts the pole on both. The quadratic reaches it in more of the grid.")
    print("  - The ridge at R0 = 1/2 has its floor at R0, below the pole, and its")
    print("    operating point on it.")
    print(
        "  - `exp(x)` has an infimum of zero, so no lambda clears it for every state."
    )
    print("    It is cleared only over a declared box, and the required lambda grows")
    print("    without bound as that box widens. That family, alone, is why the units")
    print("    answer was insufficient on its own, and ADR-057 removed the block")
    print("    instead of clearing the pole.")


if __name__ == "__main__":
    main()
