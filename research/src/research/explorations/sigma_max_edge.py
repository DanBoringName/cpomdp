"""Where the fit window's upper edge sits once the quartic is subtracted.

The dilute-versus-subtract rule registered on 2026-08-07 fired for **subtract**, so the
quartic is removed from the fit and the leading term it still carries is the sextic.
`σ_max` was defined against the quartic. This works out what the sextic edge is, what
`T` becomes under it, and where the binding cell moves.

It registers nothing and computes no value of `T`. The formula and the binding cell are
what an amendment needs before any evaluation happens, and evaluating first would be the
ordering this programme keeps warning about. Run it with
`python -m research.explorations.sigma_max_edge`.

On the ridge of `d4-family-v1`, with `R₀ = 1` and `μ* = √(R₀/κ)`:

```text
c₂ =  κ/4        c₄ = 3κ(κ − 2)/16        c₆ = −κ(7κ + 9)(13κ − 3)/48
```
"""

import numpy as np

from research.explorations.operating_point import KAPPA_MIN

__all__ = [
    "binding_kappa",
    "quartic_window_factor",
    "sextic_window_factor",
]

#: Where each coefficient vanishes on the ridge, so neither edge is defined there.
QUARTIC_ZERO = 2.0
SEXTIC_ZERO = 3.0 / 13.0


def c2(kappa):
    """The leading coefficient on the ridge."""
    return kappa / 4.0


def c4(kappa):
    """The quartic coefficient on the ridge."""
    return 3.0 * kappa * (kappa - 2.0) / 16.0


def c6(kappa):
    """The sextic coefficient on the ridge."""
    return -kappa * (7.0 * kappa + 9.0) * (13.0 * kappa - 3.0) / 48.0


def quartic_window_factor(kappa):
    """`c₂²/|c₄|`, the κ-dependent part of the window width under the quartic edge.

    With `σ_max² = f·c₂/|c₄|` and `σ_min² = k·δ_ref/c₂`, the ratio of the edges is this
    quantity times `f/(k·δ_ref)`, so minimising the window over `κ` minimises it.

    Args:
        kappa: the curvature dial, on the ridge.

    Returns:
        The factor. Infinite where `c₄` vanishes, which is the edge being undefined
        rather than a division going wrong, so it is returned rather than warned about.
    """
    lower = np.abs(c4(kappa))
    safe = np.where(lower == 0.0, 1.0, lower)
    return np.where(lower == 0.0, np.inf, c2(kappa) ** 2 / safe)


def sextic_window_factor(kappa):
    """`c₂^{3/2}/√|c₆|`, the same under the sextic edge.

    Subtraction leaves the sextic as the leading unmodelled term, so the edge is where
    `|c₆|σ⁶` reaches `f·c₂σ²`, giving `σ_max⁴ = f·c₂/|c₆|`. The edge ratio is then this
    quantity times `√f/(k·δ_ref)`.

    Args:
        kappa: the curvature dial, on the ridge.

    Returns:
        The factor. Infinite where `c₆` vanishes, on the same terms.
    """
    lower = np.sqrt(np.abs(c6(kappa)))
    safe = np.where(lower == 0.0, 1.0, lower)
    return np.where(lower == 0.0, np.inf, c2(kappa) ** 1.5 / safe)


def binding_kappa(factor, lower, upper, samples=200001):
    """The `κ` in `[lower, upper]` that minimises a window factor.

    Args:
        factor: `quartic_window_factor` or `sextic_window_factor`.
        lower: the sweep's floor.
        upper: the sweep's ceiling.
        samples: how finely to scan.

    Returns:
        The minimising `κ` and the factor there.
    """
    grid = np.linspace(lower, upper, samples)
    values = factor(grid)
    values = np.where(np.isfinite(values), values, np.inf)
    index = int(np.argmin(values))
    return float(grid[index]), float(values[index])


def main() -> None:
    """Print the two edges, where each is undefined, and where the binding cell sits."""
    print("the two edge definitions")
    print("  quartic (as registered)  sigma_max^2 = f c2/|c4|")
    print("    T = f * c2^2/|c4| * 10^(-2D)          linear in f")
    print("  sextic  (after subtracting)  sigma_max^4 = f c2/|c6|")
    print("    T = c2^(3/2) * sqrt(f/|c6|) * 10^(-2D)   square root in f")

    print("\nwhere each edge is undefined on the ridge")
    print(f"  c4 = 0 at kappa = {QUARTIC_ZERO}")
    print(f"  c6 = 0 at kappa = 3/13 = {SEXTIC_ZERO:.6f}")
    print("  they never coincide, so one edge is always available")

    print(f"\nthe declared floor is kappa_min = {KAPPA_MIN}, which is below 3/13.")
    print("  does the floor still bind under the sextic edge?")
    for ceiling in (0.2, 0.23, 0.5, 1.0, 2.0, 4.0, 10.0):
        kappa, value = binding_kappa(sextic_window_factor, KAPPA_MIN, ceiling)
        at_floor = np.isclose(kappa, KAPPA_MIN, atol=1e-4)
        verdict = "floor binds" if at_floor else f"binds at kappa = {kappa:.4f}"
        print(f"    sweep [{KAPPA_MIN}, {ceiling:>4g}]: {verdict}")

    print("\n  the same question under the quartic edge, for comparison")
    for ceiling in (0.5, 1.0, 1.9, 4.0):
        kappa, _ = binding_kappa(quartic_window_factor, KAPPA_MIN, ceiling)
        at_floor = np.isclose(kappa, KAPPA_MIN, atol=1e-4)
        verdict = "floor binds" if at_floor else f"binds at kappa = {kappa:.4f}"
        print(f"    sweep [{KAPPA_MIN}, {ceiling:>4g}]: {verdict}")

    print("\n  the sextic factor's shape, to show why")
    for kappa in (0.1, 0.15, 0.2, SEXTIC_ZERO, 0.3, 0.5, 1.0, 2.0, 10.0, 100.0):
        value = sextic_window_factor(np.array([kappa]))[0]
        label = "  <- c6 vanishes" if abs(kappa - SEXTIC_ZERO) < 1e-9 else ""
        print(f"    kappa={kappa:<8g} c2^(3/2)/sqrt|c6| = {value:>10.5g}{label}")

    # Checks. Below the pole at 3/13 the factor rises, so a floor under the pole is the
    # minimiser there. Above the pole it dips and then rises again towards the large-κ
    # limit (1/4)^{3/2}/sqrt(91/48), so it never returns below the floor's value and the
    # floor binds on any sweep, whatever the ceiling.
    below = sextic_window_factor(np.linspace(KAPPA_MIN, SEXTIC_ZERO - 1e-6, 5000))
    assert np.all(np.diff(below) > 0), "the sextic factor should rise up to the pole"
    limit = 0.25**1.5 / np.sqrt(91.0 / 48.0)
    tail = sextic_window_factor(np.array([1e3, 1e4, 1e5]))
    assert np.all(np.diff(tail) > 0), "above the pole it rises towards the limit"
    assert abs(tail[-1] - limit) / limit < 1e-4, (tail[-1], limit)

    # Log-spaced, not linear. A linear grid from the pole to 1e6 steps by about five and
    # skips the dip entirely, which is the only part of the range this claim is about.
    above_grid = np.geomspace(SEXTIC_ZERO * (1 + 1e-9), 1e6, 400001)
    above = sextic_window_factor(above_grid)
    dip, at = float(above.min()), float(above_grid[int(np.argmin(above))])
    floor_value = float(sextic_window_factor(np.array([KAPPA_MIN]))[0])
    print(f"\n  above the pole the factor dips to {dip:.6f} at kappa = {at:.4f}")
    print(
        f"  and rises to {limit:.6f} as kappa grows; the floor's value is "
        f"{floor_value:.6f}"
    )
    assert dip > floor_value, (dip, at, floor_value)
    print("  checks passed: it rises to the pole, dips to a minimum well above the")
    print("  floor's value, and so kappa_min binds whatever ceiling the sweep takes")


if __name__ == "__main__":
    main()
