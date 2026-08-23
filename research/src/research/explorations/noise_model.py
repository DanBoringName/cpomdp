"""Does the D2 fit's error on the exponent fall as `1/√N`?

`research/gate_d4_registration.md` models the statistical term as `σ_p ≈ ε√12/(D√N)`,
so it falls with the number of `σ` samples. That is the standard error of a slope under
**independent random** errors on each point.

The error being propagated is `δ_ref`, which the registration's own opening line calls a
*certified discretisation bound*. A bound on a deterministic quantity is not a random
draw: the reference filter evaluated at neighbouring `σ` is wrong in the same direction
for the same reason, so the errors are correlated rather than independent, and averaging
more of them does not average them away.

This module computes what the error on the exponent actually is under three readings of
`δ_ref`, and prints them side by side at the registered operating point. It registers
nothing. Run it with `python -m research.explorations.noise_model`.

Notation follows the registration. `v = ln(σ/σ_min)` runs over `[0, L]` with
`L = D·ln10`. The relative error on the gap is `ε(v) = (1/k)·e^{−2v}`, which is `1/k` at
the bottom edge by the definition of `σ_min` and improves as `σ²` across the window,
the gap growing as `σ²` while the bound does not.
"""

import numpy as np
from scipy.integrate import quad

from research.explorations.operating_point import (
    BETA,
    DECADES,
    K_MIN,
    LN10,
    PUBLISHED_SIGMA_P,
    SIGMA_P,
)

__all__ = [
    "LN10",
    "REGISTERED",
    "fitted_shift",
    "independent_random",
    "independent_random_corrected",
    "systematic_offset",
    "systematic_tracking",
    "systematic_worst_case",
    "unweighted_standard_error",
    "weighted_standard_error",
]


#: The registered operating point: `(k_min, D*, σ_p)`.
REGISTERED = (K_MIN, DECADES, SIGMA_P)


def relative_error(point: float, k: float) -> float:
    """`ε(v)`, the bound's size relative to the gap at `v`."""
    return float(np.exp(-2.0 * point) / k)


def independent_random(k: float, decades: float, samples: int) -> float:
    """The registration's formula as written: `ε√12/(D√N)`.

    Kept in the form the document states so its figures can be reproduced. It is off by
    `ln 10`: the slope's standard error is `σ_y√12/(L√N)` with `L` the window width in
    *nats*, and `D` is in decades. `independent_random_corrected` is the same quantity
    with the conversion applied.

    Args:
        k: how many times the bound the gap clears at the bottom edge.
        decades: window width in decades of `σ`.
        samples: `N`, the number of `σ` values fitted.

    Returns:
        The standard error on the fitted exponent, as the document computes it.
    """
    return float(np.sqrt(12.0) / (k * decades * np.sqrt(samples)))


def independent_random_corrected(k: float, decades: float, samples: int) -> float:
    """The same standard error with the decades-to-nats conversion applied.

    Args:
        k: as above.
        decades: as above.
        samples: as above.

    Returns:
        The standard error on the fitted exponent.
    """
    return float(np.sqrt(12.0) / (k * decades * LN10 * np.sqrt(samples)))


def unweighted_standard_error(k: float, decades: float, samples: int) -> float:
    """The exact unweighted OLS standard error, summed rather than approximated.

    Args:
        k: as above.
        decades: as above.
        samples: as above.

    Returns:
        The standard error on the fitted exponent.
    """
    points = np.linspace(0.0, decades * LN10, samples)
    return float((1.0 / k) / np.sqrt(((points - points.mean()) ** 2).sum()))


def weighted_standard_error(k: float, decades: float, samples: int) -> float:
    """The same under weights `1/ε²`, which is what "heteroscedastic" would mean here.

    Args:
        k: as above.
        decades: as above.
        samples: as above.

    Returns:
        The standard error on the fitted exponent.
    """
    points = np.linspace(0.0, decades * LN10, samples)
    weights = k**2 * np.exp(4.0 * points)
    centre = (weights * points).sum() / weights.sum()
    return float(1.0 / np.sqrt((weights * (points - centre) ** 2).sum()))


def _slope_shift(field, decades: float) -> float:
    """How a deterministic relative-error field moves the fitted exponent.

    An additive error on the gap enters the log fit as `ε(v)` to first order, so the
    ordinary-least-squares slope moves by `Cov(v, ε)/Var(v)`. There is no `N` in it: the
    field is a function, not a sample, and refining the grid converges the integral
    rather than shrinking the answer.

    Args:
        field: `ε` as a function of `v`.
        decades: window width in decades of `σ`.

    Returns:
        The shift in the fitted exponent.
    """
    width = decades * LN10
    mean_point = width / 2
    mean_field = quad(field, 0, width, limit=200)[0] / width
    covariance = (
        quad(
            lambda point: (point - mean_point) * (field(point) - mean_field),
            0,
            width,
            limit=200,
        )[0]
        / width
    )
    return covariance / (width**2 / 12)


def systematic_offset(k: float, decades: float) -> float:
    """A constant absolute error of one bound, the same at every `σ`.

    The plainest deterministic reading: the filter is off by `δ_ref` everywhere. The
    relative error still varies, because the gap it is divided by grows as `σ²`.

    Args:
        k: as above.
        decades: as above.

    Returns:
        The shift in the fitted exponent. Independent of `N`.
    """
    return abs(_slope_shift(lambda point: relative_error(point, k), decades))


def systematic_tracking(k: float, decades: float) -> float:
    """An error proportional to the gap itself, so the *relative* error is flat.

    The benign deterministic reading, and the one that would rescue the registration's
    figure. A flat `ε` has zero covariance with `v`, so it moves the intercept and
    leaves the exponent alone however large it is. Whether the reference filter's error
    behaves this way is a property of the filter, unknown until one exists.

    Args:
        k: as above.
        decades: as above.

    Returns:
        The shift in the fitted exponent, which is zero up to quadrature.
    """
    return abs(_slope_shift(lambda _: 1.0 / k, decades))


def systematic_worst_case(k: float, decades: float) -> float:
    """The largest shift any error field within the bound can produce.

    `Cov(v, ε)` is largest when `ε` sits on its envelope and takes the sign of
    `v − v̄`, so the worst admissible field is that one. This is what the bound alone
    licenses: without knowing the filter's error *shape*, nothing rules it out.

    Args:
        k: as above.
        decades: as above.

    Returns:
        The shift in the fitted exponent. Independent of `N`.
    """
    width = decades * LN10
    mean_point = width / 2

    def weighted(point: float) -> float:
        return abs(point - mean_point) * relative_error(point, k)

    # Split at the kink. `quad` across `|v - v̄|` misses by a few parts in ten thousand,
    # which is small and is still the integrator being asked the wrong question.
    integral = (
        quad(weighted, 0, mean_point, limit=200)[0]
        + quad(weighted, mean_point, width, limit=200)[0]
    )
    return (integral / width) / (width**2 / 12)


def fitted_shift(field, decades: float, samples: int) -> float:
    """The shift measured by actually running the fit, rather than by the integral.

    Builds a pure power law, applies the relative error field, fits `ln gap` against
    `ln σ` by ordinary least squares, and reports how far the slope lands from two.
    Independent of `_slope_shift`: it shares no code with it, uses the exact logarithm
    rather than the first-order `ln(1 + ε) ≈ ε`, and samples rather than integrates.

    The sign is kept. A positive offset flattens the exponent below two; a field that
    changes sign across the window steepens it above.

    Args:
        field: `ε` as a function of `v = ln(σ/σ_min)`.
        decades: as above.
        samples: how many `σ` values to fit.

    Returns:
        The measured shift in the exponent, signed.
    """
    points = np.linspace(0.0, decades * LN10, samples)
    corrected = np.array([1.0 + field(float(point)) for point in points])
    slope = np.polyfit(points, np.log(np.exp(2.0 * points) * corrected), 1)[0]
    return float(slope - 2.0)


def main() -> None:
    """Print the readings, identify the estimator, and check every claim by fitting."""
    k, decades, registered = REGISTERED
    print(f"at the registered point: k = {k:g}, D = {decades}, beta = {BETA}")

    print("\n  which estimator did the registration's sigma_p come from?")
    print("  it says heteroscedastic weights; its numbers say otherwise.")
    for k_i, d_i, value in PUBLISHED_SIGMA_P:
        plain = unweighted_standard_error(k_i, d_i, 60)
        weighted = weighted_standard_error(k_i, d_i, 60)
        print(
            f"    k={k_i:>4g} D={d_i}: published {value:.4f}  "
            f"unweighted(N=60) {plain:.4f}  weighted(N=60) {weighted:.4f}"
        )
    print("    unweighted at N about 60 reproduces all four; weighted needs N = 6.5")
    print("    and has no solution for the fourth. The deterministic readings below")
    print("    are therefore compared against the estimator the numbers actually use.")

    print("\n  the formula as written carries a decades-for-nats slip:")
    print(f"    as written, N=345   {independent_random(k, decades, 345):.6f}")
    fixed = independent_random_corrected(k, decades, 345)
    print(f"    corrected,  N=345   {fixed:.6f}")
    print(f"    exact OLS,  N=345   {unweighted_standard_error(k, decades, 345):.6f}")
    exact_63 = unweighted_standard_error(k, decades, 63)
    print(f"    exact OLS,  N=63    {exact_63:.6f}   (the registered {registered})")

    tracking = systematic_tracking(k, decades)
    offset = systematic_offset(k, decades)
    worst = systematic_worst_case(k, decades)
    print("\n  deterministic readings, which carry no N at all:")
    print(f"    an error tracking the gap, flat in ratio      {tracking:.4f}")
    print(f"    a constant offset of one bound                {offset:.4f}")
    print(f"    the worst field the bound admits              {worst:.4f}")

    print("\n  measured by running the fit, sign kept, no linearisation:")
    for label, field in (
        ("tracking", lambda point: 1.0 / k),
        ("constant offset", lambda point: relative_error(point, k)),
        (
            "worst case",
            lambda point: (
                float(np.sign(point - decades * LN10 / 2)) * relative_error(point, k)
            ),
        ),
    ):
        for samples in (500, 50000):
            shift = fitted_shift(field, decades, samples)
            print(f"    {label:<16} N={samples:>6}: {shift:+.4f}")

    print(f"\n  against the total budget beta = {BETA}, unweighted:")
    for label, value in (
        ("tracking the gap", tracking),
        ("constant offset", offset),
        ("worst case", worst),
    ):
        verdict = "inside" if value < BETA else "OVER BUDGET on its own"
        print(
            f"    {label:<20} {value:.4f} = {value / BETA:5.1%} of beta, {verdict}"
        )

    # Checks. The exact fit must reproduce the integral to the linearisation, must not
    # move with N, and must order the three fields the way the envelope argument says.
    exact_offset = fitted_shift(lambda p: relative_error(p, k), decades, 50000)
    exact_worst = fitted_shift(
        lambda p: float(np.sign(p - decades * LN10 / 2)) * relative_error(p, k),
        decades,
        50000,
    )
    assert exact_offset < 0 < exact_worst, (exact_offset, exact_worst)
    assert abs(abs(exact_offset) - offset) / offset < 0.05
    assert abs(abs(exact_worst) - worst) / worst < 0.05
    assert abs(exact_worst) > worst, "linearising understates the worst field"
    coarse = fitted_shift(lambda p: relative_error(p, k), decades, 50)
    assert abs(abs(coarse) - abs(exact_offset)) < 5e-4, (coarse, exact_offset)
    assert worst >= offset >= tracking
    assert tracking < 1e-12
    print("\n  checks passed: the fit agrees with the integral, does not move with N,")
    print("  and the offset flattens the exponent where the worst field steepens it")


if __name__ == "__main__":
    main()
