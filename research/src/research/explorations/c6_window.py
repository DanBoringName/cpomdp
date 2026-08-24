"""The fit window under subtraction, as run.

Regenerates every number in `research/c6_window_exploration.md`, which is exploratory
and registers nothing. Run it with `python -m research.explorations.c6_window`.

Two of the quantities printed are validations rather than findings, and both are
asserted rather than only shown. The exact OLS bias is checked against the ratio table
`research/gate_d4_registration.md` publishes for its first-order envelope. The sextic
arm is checked against the scaling identity rescaling `u` by the exponent gives,
`bias(m, L) = m·bias(1, m·L)`, which relates the two arms exactly and is what stops the
sextic one being an untested copy of the quartic one.

The noise term here is a stand-in, `σ_p = A/D`, with `A` fitted so the quartic
optimisation reproduces the registered `D*`. The write-up says what that costs and why
nothing may be declared from it.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

from research.explorations.operating_point import (
    BETA,
    DECADES,
    F_STAR,
    LN10,
    SIGMA_P,
)

__all__ = [
    "BETA",
    "REGISTERED",
    "largest_f",
    "ols_bias",
    "optimum",
]

#: What `research/gate_d4_registration.md` records at `k_min = 10` and `β = 0.05`, as
#: `(D*, f*, σ_p)`. The calibration below is pinned to the first of these.
REGISTERED = (DECADES, F_STAR, SIGMA_P)

#: The registration's ratio of its first-order envelope to the exact integral, at
#: `f = 0.02`, keyed by `D`. Reproducing this is what licenses the integral below.
REGISTERED_RATIOS = {1.0: 1.71, 2.0: 1.27, 3.0: 1.16}


def ols_bias(fraction: float, decades: float, exponent: int) -> float:
    """The exact bias an unmodelled term leaves in an OLS log-log exponent.

    `noise_model._slope_shift` computes the same `Cov(v, ·)/Var(v)` functional on a
    shifted domain. The two stay separate deliberately: each module is the record of
    what its write-up ran, each is verified against its own independent arm, and a
    shared helper would let an edit to one silently move the other's published numbers.

    The correction is `−fraction·e^{exponent·u}` with `u = ln σ − ln σ_max`, so `u = 0`
    is the top edge and the correction is exactly `fraction` of the leading term there.
    The fit runs over `u ∈ [−L, 0]` with `L = decades·ln10`, uniform in `u`.

    Args:
        fraction: the correction's size at the top edge, relative to the leading term.
        decades: the window width in decades of `σ`.
        exponent: the correction's order in `σ` above the leading term, `2` for a
            quartic left in the fit and `4` for a sextic left after subtraction.

    Returns:
        The bias in the fitted exponent, negative for a correction that bends the curve
        below a pure power law.
    """
    if fraction <= 0:
        return 0.0
    width = decades * LN10

    def residual(point: float) -> float:
        return float(np.log1p(-fraction * np.exp(exponent * point)))

    mean_point = -width / 2
    mean_residual = quad(residual, -width, 0, limit=200)[0] / width
    covariance = (
        quad(
            lambda point: (point - mean_point) * (residual(point) - mean_residual),
            -width,
            0,
            limit=200,
        )[0]
        / width
    )
    return covariance / (width**2 / 12)


def largest_f(decades: float, exponent: int, budget: float) -> float:
    """The largest correction the bias budget admits at this window width.

    Args:
        decades: the window width in decades of `σ`.
        exponent: the correction's order, as in `ols_bias`.
        budget: the bias magnitude the fit may carry.

    Returns:
        The fraction at which the exact bias reaches `budget`, or a value just below one
        where the budget is never reached.
    """
    ceiling = 0.999
    if abs(ols_bias(ceiling, decades, exponent)) < budget:
        return ceiling
    return brentq(
        lambda fraction: abs(ols_bias(fraction, decades, exponent)) - budget,
        1e-9,
        ceiling,
        xtol=1e-12,
    )


def _bias_budget(noise_constant: float, decades: float) -> float | None:
    """What the total budget leaves for bias once the noise term is spent.

    `None` where the noise alone meets or exceeds `BETA`, so the objective and the
    post-optimisation read cannot disagree about where that plateau starts.
    """
    noise = noise_constant / decades
    if noise >= BETA:
        return None
    return float(np.sqrt(BETA**2 - noise**2))


def _negative_threshold(
    decades: float, noise_constant: float, exponent: int, power: float
) -> float:
    """Minus `T` up to constants, `T` going as `f**power · 10**(−2D)`."""
    budget = _bias_budget(noise_constant, decades)
    if budget is None:
        return 0.0
    return -(largest_f(decades, exponent, budget) ** power) * 10 ** (-2 * decades)


def optimum(
    noise_constant: float, exponent: int, power: float
) -> tuple[float, float, float]:
    """Where `T` is largest, subject to the total budget on the fitted exponent.

    Args:
        noise_constant: `A` in the stand-in `σ_p = A/D`.
        exponent: the residual correction's order, as in `ols_bias`.
        power: how `T` scales in the correction's size, `1` for the quartic edge and
            `0.5` for the sextic one.

    Returns:
        The optimal `(decades, fraction, noise)`.
    """
    found = minimize_scalar(
        _negative_threshold,
        bounds=(0.05, 4.0),
        args=(noise_constant, exponent, power),
        method="bounded",
        options={"xatol": 1e-6},
    )
    decades = float(found.x)
    budget = _bias_budget(noise_constant, decades)
    if budget is None:
        # The objective is a flat zero on this plateau, so the minimiser's point is
        # arbitrary and no window satisfies the constraint.
        raise ValueError(
            f"the noise term {noise_constant / decades:.4f} already fills the budget "
            f"{BETA} at D = {decades:.4f}, so no window satisfies the constraint"
        )
    return decades, largest_f(decades, exponent, budget), noise_constant / decades


def main() -> None:
    """Print every number the write-up quotes, asserting the two validations."""
    print("the exact bias against the registration's first-order envelope, f = 0.02")
    for decades, published in REGISTERED_RATIOS.items():
        exact = ols_bias(0.02, decades, 2)
        ratio = (-3 * 0.02 / (decades * LN10) ** 2) / exact
        print(f"  D={decades}: computed {ratio:.3f}, registration {published}")
        assert abs(ratio - published) < 0.005, (decades, ratio, published)

    print("\nthe sextic arm: rescaling u by m gives bias(m, L) = m·bias(1, m·L),")
    print("so the m=4 bias at D is twice the m=2 bias at 2D, not equal to it")
    for decades in (1.0, 2.0):
        quartic = ols_bias(0.005, 2 * decades, 2)
        sextic = ols_bias(0.005, decades, 4)
        print(
            f"  m=4 at D={decades} over m=2 at D={2 * decades}: {sextic / quartic:.6f}"
        )
        assert abs(sextic / quartic - 2.0) < 1e-6, (decades, sextic, quartic)

    print("\ncalibrating the stand-in noise term to the registered D*")
    constant = brentq(
        lambda value: optimum(value, 2, 1)[0] - REGISTERED[0], 0.001, 0.049, xtol=1e-10
    )
    quartic = optimum(constant, 2, 1)
    print(f"  A = {constant:.6f}")
    print("  quartic:    D* = {:.4f}, f* = {:.4f}, sigma_p = {:.4f}".format(*quartic))
    print(
        "  registered: D* = {:.4f}, f* = {:.4f}, sigma_p = {:.4f}".format(*REGISTERED)
    )

    print("\nre-optimising against the sextic residual, T going as sqrt(f)")
    sextic = optimum(constant, 4, 0.5)
    print("  sextic:     D* = {:.4f}, f* = {:.4f}, sigma_p = {:.4f}".format(*sextic))
    print(f"  D moves by a factor of {sextic[0] / quartic[0]:.3f}")
    held = -_negative_threshold(REGISTERED[0], constant, 4, 0.5)
    best = -_negative_threshold(sextic[0], constant, 4, 0.5)
    print(f"  holding D at {REGISTERED[0]} costs a factor of {best / held:.3f} on T")

    print("\nsensitivity of that factor to the calibration")
    for scale in (0.85, 1.0, 1.15):
        shifted = optimum(constant * scale, 4, 0.5)[0]
        base = optimum(constant * scale, 2, 1)[0]
        moved = shifted / base
        print(f"  A x {scale:.2f}: D moves by {moved:.3f}")


if __name__ == "__main__":
    main()
