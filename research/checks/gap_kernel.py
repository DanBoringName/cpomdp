"""The inference-gap quadrature, shared by the checks that measure things about it.

One object, defined once. The averaged inference gap under state-dependent noise::

    E_{y∼p*}[ KL(q ‖ p(x|y)) ]

with `p(x|y) ∝ N(x; μ, σ²)·N(y; x, R(x))` the exact non-Gaussian posterior, `q` the
agent's Gaussian from a Kalman update with the plug-in `R̂ = R(μ)`, and `p*(y)` the exact
predictive from the same joint.

Three conventions are pinned here rather than being rediscovered per caller, since the
scripts that produced the 28 fitted `c₄` cases are gone and prose was all that survived
them. The cases themselves are recorded in ``research/gate_d4_registration.md``, the
RESULT of 2026-08-10: 28 cases across four `R` families (quadratic, exponential, `tanh`,
`sin`). The conventions are:

- **Direction.** Reverse, ``KL(q ‖ p)``, at :func:`gap`. The battery's decomposition
  (``research/fep_falsification_battery.md`` line 21) and the registration
  (``research/gate_d4_registration.md`` line 128) both specify this one. The forward
  direction is a different number and no declared figure refers to it.
- **Where `R` is evaluated.** At the *prior* mean, `R̂ = R(μ)`, frozen across the update,
  at :func:`plugin_posterior`. Never at a post-observation mean.
- **The average over `y`.** Under the true `p*(y)`, at :func:`log_predictive`. The
  agent's plug-in predictive `N(y; μ, σ² + R(μ))` appears nowhere in the average.

Extracted from ``predictive_truncation.py`` when a second check needed the same
quadrature. Callers own the questions; this module owns the integral.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import quad

__all__ = [
    "FAMILIES",
    "SIGMAS",
    "NoiseFamily",
    "Quadrature",
    "VoidReason",
    "assert_positive_noise",
    "core_and_tail",
    "gap",
    "integrate_gap",
    "log_predictive",
    "plugin_noise_of",
    "plugin_posterior",
    "predictive_sd",
]

#: How far past the grid edge a tail integral runs, in absolute `y` units.
TAIL_EXTENT = 40.0

#: `q`'s support, in its own standard deviations. The KL integrand is `q`-weighted, so
#: this window governs the cross term and nothing else.
Q_SUPPORT_SD = 14.0

#: How far below its peak the `p*` inner integrand must fall before the `x` window
#: closes, in nats.
X_WINDOW_DECAY = 75.0

#: Ladder the `x` window expands along, in prior standard deviations. Expansion rather
#: than a fixed multiple because for unbounded `R` the mixture's dominant `x` runs
#: outward as `|y − μ|` grows. Measured, the fixed and expanding windows agree to ten
#: digits on the declared family, since the far tail carries no weight. The ladder is
#: insurance for families with faster-growing `R`, not a correction to a known error.
X_WINDOW_LADDER = (12.0, 24.0, 48.0, 96.0, 192.0, 384.0)

#: Below this, a tail integral may be reporting the quadrature's own noise rather than a
#: measurement. Cells at or under it are VOID, never PASS: a check that reports 3e-15 as
#: a pass is reporting its noise as a certificate.
#:
#: Inherited from the external study, where the width ladder saturated near 3.4e-15.
#: This kernel resolves further. The same ladder reaches 1.6e-21 at 18 sd, so the floor
#: is conservative here and voids cells it could in fact measure. That direction is the
#: safe one. It cannot manufacture a PASS. A refinement test is the operative
#: convergence check, and tightening this constant is a decision for whoever registers
#: the rule change, not a tuning knob.
QUADRATURE_FLOOR = 5e-15

#: Fraction by which a refinement may move a reported figure before it was not
#: converged when it was reported.
CONVERGENCE_BAR = 0.10


class VoidReason(Enum):
    """Why a cell reports no measurement.

    ``NUMERICAL_FLOOR`` — the integral is at or below the quadrature's own noise, or it
    moved under refinement. Either way the number is not a measurement.

    ``NON_POSITIVE_NOISE`` — `R` went non-positive somewhere on the quadrature span. The
    registration records this as a precondition after a linear family silently produced
    the log of a negative number.

    Local to ``research/checks`` rather than added to ``cpomdp.warrant``: a new export
    there needs a ``docs/api`` page before anything can reference it, and these modules
    are not on the main suite. The value goes into ``CheckReport.detail``, so no
    parallel result type is introduced.
    """

    NUMERICAL_FLOOR = "at or below the quadrature floor"
    NON_POSITIVE_NOISE = "R non-positive on the span"


@dataclass(frozen=True)
class NoiseFamily:
    """A declared state-dependent sensor noise `R(x)` with its prior mean.

    Args:
        name: how the family prints, in the notation the registration uses.
        noise: `R` — state-dependent sensor noise, evaluated elementwise on arrays.
        prior_mean: μ — the prior mean the gap is expanded about.
        unbounded: whether `R` grows without bound in `x`. A declared property of a
            declared family, not a measurement.
        crossover: ν* — where the mixture's tail turns from Gaussian to exponential, as
            a function of prior variance. Leading order, so it prints beside a grid edge
            and nothing asserts its value. ``None`` where no estimate is derived.
        reference: external relative truncations at 9 sd, by σ. Keyed to the family
            rather than to its position in a list, since a reference compared against
            the wrong family is a fabricated disagreement.
        log_noise_derivative: `ℓ'(μ)` — the first log-derivative of `R` at the prior
            mean, where it is known in closed form. ``None`` where it is not.
    """

    name: str
    noise: Callable[[np.ndarray], np.ndarray]
    prior_mean: float
    unbounded: bool
    crossover: Callable[[float], float] | None = None
    reference: dict[float, float] | None = None
    log_noise_derivative: float | None = None


def _d4_crossover(prior_variance: float) -> float:
    """ν* for `d4-family-v1`, from balancing the prior cost against the likelihood.

    Args:
        prior_variance: σ² — the prior variance.

    Returns:
        The leading-order `y` offset at which the tail turns exponential.
    """
    sigma = math.sqrt(prior_variance)
    return 2.0 * (prior_variance + 2.0) / sigma  # κ = 1, R(μ) = 2


#: The declared families. The first is `d4-family-v1` as declared in
#: research/gate_d4_registration.md section 1, at κ = 1, R₀ = 1, μ = 1. The rest carry
#: the parametrisations the external reference assumed.
#:
#: ``log_noise_derivative`` is `R'(μ)/R(μ)`, written out rather than differentiated so
#: the `c₂` check has an independent arm. The registration requires `R'(μ) ≠ 0` as a
#: fixture precondition. Every family here satisfies it except ``constant``, which
#: violates it deliberately: that is what makes the gap vanish there.
FAMILIES: dict[str, NoiseFamily] = {
    "quadratic": NoiseFamily(
        name="1 + x²",
        noise=lambda x: 1.0 + x**2,
        prior_mean=1.0,
        unbounded=True,
        crossover=_d4_crossover,
        reference={
            0.06: 1.0e-14,
            0.10: 3.2e-13,
            0.15: 1.9e-11,
            0.20: 4.9e-10,
            0.25: 5.7e-09,
            0.30: 3.5e-08,
        },
        log_noise_derivative=2.0 * 1.0 / (1.0 + 1.0**2),  # 2x/(1+x²) at x = 1
    ),
    "exponential": NoiseFamily(
        name="exp(x)",
        noise=np.exp,
        prior_mean=1.0,
        unbounded=True,
        reference={0.15: 2.7e-11, 0.25: 1.8e-08, 0.30: 1.7e-07},
        log_noise_derivative=1.0,  # exp(x)/exp(x)
    ),
    "tanh": NoiseFamily(
        name="1.5 + 0.5 tanh(x)",
        noise=lambda x: 1.5 + 0.5 * np.tanh(x),
        prior_mean=1.0,
        unbounded=False,
        log_noise_derivative=(
            0.5 * (1.0 - math.tanh(1.0) ** 2) / (1.5 + 0.5 * math.tanh(1.0))
        ),
    ),
    "sin": NoiseFamily(
        name="1.5 + 0.5 sin(x)",
        noise=lambda x: 1.5 + 0.5 * np.sin(x),
        prior_mean=1.0,
        unbounded=False,
        log_noise_derivative=(0.5 * math.cos(1.0) / (1.5 + 0.5 * math.sin(1.0))),
    ),
    #: Fixed `R`, where the Kalman filter *is* the exact Bayesian filter and the gap is
    #: identically zero. The registration records a method that computed exactly this
    #: and was therefore measuring nothing. Kept as a falsifier of the implementation.
    "constant": NoiseFamily(
        name="2 (fixed)",
        noise=lambda x: np.full_like(np.asarray(x, dtype=float), 2.0),
        prior_mean=1.0,
        unbounded=False,
        log_noise_derivative=0.0,
    ),
}

#: Default `σ` grid, matching the external reference table.
SIGMAS: tuple[float, ...] = (0.06, 0.10, 0.15, 0.20, 0.25, 0.30)


@dataclass(frozen=True)
class Quadrature:
    """Tolerances and window sizes one measurement runs at.

    Args:
        epsabs: absolute tolerance handed to the outer and inner integrals.
        epsrel: relative tolerance handed to the same.
        x_window_scale: multiplies the `x` window the expansion ladder settles on.
        limit: subdivision limit per integral.
    """

    epsabs: float = 1e-16
    epsrel: float = 1e-12
    x_window_scale: float = 1.0
    limit: int = 200

    def refined(self) -> Quadrature:
        """The same settings at half the tolerance and double the inner bounds.

        Returns:
            The refinement a convergence check runs at.
        """
        return Quadrature(
            epsabs=self.epsabs / 2.0,
            epsrel=self.epsrel / 2.0,
            x_window_scale=self.x_window_scale * 2.0,
            limit=self.limit * 2,
        )


def _log_gaussian(value: float, mean: float, variance: float) -> float:
    """Log density of a scalar normal.

    Args:
        value: where to evaluate.
        mean: the mean.
        variance: the variance.

    Returns:
        The log density.
    """
    return -0.5 * (math.log(2.0 * math.pi * variance) + (value - mean) ** 2 / variance)


def _log_gaussian_at(
    value: np.ndarray, mean: np.ndarray | float, variance: np.ndarray | float
) -> np.ndarray:
    """The same density over arrays, for the grid scans.

    Args:
        value: where to evaluate.
        mean: the mean.
        variance: the variance.

    Returns:
        The log densities.
    """
    return -0.5 * (np.log(2.0 * np.pi * variance) + (value - mean) ** 2 / variance)


def plugin_noise_of(family: NoiseFamily) -> float:
    """`R̂ = R(μ)`, the plug-in the agent freezes across its update.

    Args:
        family: the declared `R` and its prior mean.

    Returns:
        The sensor noise at the prior mean.
    """
    return float(family.noise(np.asarray(family.prior_mean)))


def plugin_posterior(
    family: NoiseFamily, prior_variance: float, y: float
) -> tuple[float, float]:
    """The agent's Gaussian `q`: a Kalman update with `R` frozen at the prior mean.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        y: the observation.

    Returns:
        The posterior mean and variance of `q`.
    """
    noise = plugin_noise_of(family)  # R̂ = R(μ)
    gain = prior_variance / (prior_variance + noise)
    return family.prior_mean + gain * (y - family.prior_mean), gain * noise


def log_joint(family: NoiseFamily, prior_variance: float, x: float, y: float) -> float:
    """Log of the unnormalised joint `N(x; μ, σ²)·N(y; x, R(x))`.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        x: the latent state.
        y: the observation.

    Returns:
        The log joint density, unnormalised in `x`.
    """
    noise = float(family.noise(np.asarray(x)))
    return _log_gaussian(x, family.prior_mean, prior_variance) + _log_gaussian(
        y, x, noise
    )


def log_joint_at(
    family: NoiseFamily, prior_variance: float, x: np.ndarray, y: float
) -> np.ndarray:
    """The same joint over an `x` grid, for the peak and window scans.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        x: the latent states to evaluate at.
        y: the observation.

    Returns:
        The log joint at each point.
    """
    noise = family.noise(x)
    return _log_gaussian_at(x, family.prior_mean, prior_variance) + _log_gaussian_at(
        np.asarray(float(y)), x, noise
    )


def x_window(
    family: NoiseFamily, prior_variance: float, y: float, settings: Quadrature
) -> float:
    """Half-width in `x` that puts the joint's endpoints far below its peak.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        y: the observation.
        settings: the tolerances and the window scale.

    Returns:
        The half-width, in absolute `x` units.
    """
    sigma = math.sqrt(prior_variance)
    for rungs in X_WINDOW_LADDER:
        half_width = rungs * sigma
        grid = family.prior_mean + np.linspace(-half_width, half_width, 801)
        scan = log_joint_at(family, prior_variance, grid, y)
        peak = float(np.max(scan))
        if max(float(scan[0]), float(scan[-1])) < peak - X_WINDOW_DECAY:
            return half_width * settings.x_window_scale
    return X_WINDOW_LADDER[-1] * sigma * settings.x_window_scale


def assert_positive_noise(family: NoiseFamily, span: tuple[float, float]) -> None:
    """Reject a family whose `R` goes non-positive anywhere on the quadrature span.

    The registration records this as a precondition after a linear `R` silently produced
    the log of a negative number. The span is wide precisely where `σ` is large.

    Args:
        family: the declared `R`.
        span: the closed interval the quadrature covers.

    Raises:
        ValueError: if `R` is non-positive anywhere on the span.
    """
    grid = np.linspace(span[0], span[1], 20001)
    noise = family.noise(grid)
    if bool(np.any(noise <= 0.0)):
        worst = float(grid[int(np.argmin(noise))])
        raise ValueError(
            f"R goes non-positive on [{span[0]:.3g}, {span[1]:.3g}] "
            f"(minimum at x = {worst:.4g}). The quadrature would take the log of a "
            "negative number."
        )


def log_predictive(
    family: NoiseFamily, prior_variance: float, y: float, settings: Quadrature
) -> float:
    """`log p*(y)`, the exact predictive from the joint with the true `R(x)`.

    This is the density the gap is averaged under, and it is the true one: `R(x)` inside
    the integral, not the agent's `R̂`.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        y: the observation.
        settings: the tolerances and the window scale.

    Returns:
        The log predictive density at `y`.
    """
    half_width = x_window(family, prior_variance, y, settings)
    lo, hi = family.prior_mean - half_width, family.prior_mean + half_width
    scan = log_joint_at(family, prior_variance, np.linspace(lo, hi, 801), y)
    peak = float(np.max(scan))
    breakpoints = (y,) if lo < y < hi else ()
    mass, _ = quad(
        lambda x: math.exp(log_joint(family, prior_variance, x, y) - peak),
        lo,
        hi,
        epsabs=settings.epsabs,
        epsrel=settings.epsrel,
        limit=settings.limit,
        points=breakpoints or None,
    )
    return peak + math.log(mass)


def gap(
    family: NoiseFamily,
    prior_variance: float,
    y: float,
    settings: Quadrature,
    log_density: float | None = None,
) -> float:
    """`KL(q ‖ p(x|y))` — reverse direction, agent against exact, at one `y`.

    Split as ``−H[q] − E_q[log w] + log Z`` so the entropy is closed form and only the
    cross term needs the grid.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        y: the observation.
        settings: the tolerances and the window scale.
        log_density: `log p*(y)` if the caller already has it, else ``None``.

    Returns:
        The reverse KL in nats.
    """
    mean_q, variance_q = plugin_posterior(family, prior_variance, y)
    support = Q_SUPPORT_SD * math.sqrt(variance_q) * settings.x_window_scale
    cross, _ = quad(
        lambda x: (
            math.exp(_log_gaussian(x, mean_q, variance_q))
            * log_joint(family, prior_variance, x, y)
        ),
        mean_q - support,
        mean_q + support,
        epsabs=settings.epsabs,
        epsrel=settings.epsrel,
        limit=settings.limit,
    )
    if log_density is None:
        log_density = log_predictive(family, prior_variance, y, settings)
    negative_entropy = -0.5 * (1.0 + math.log(2.0 * math.pi * variance_q))
    return negative_entropy - cross + log_density


def weighted_gap(
    family: NoiseFamily, prior_variance: float, y: float, settings: Quadrature
) -> float:
    """The integrand of the averaged gap: `p*(y)·KL(q ‖ p(·|y))`.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        y: the observation.
        settings: the tolerances and the window scale.

    Returns:
        The integrand at `y`.
    """
    log_density = log_predictive(family, prior_variance, y, settings)
    if log_density < -700.0:  # the density has underflowed; the product with it is zero
        return 0.0
    return math.exp(log_density) * gap(family, prior_variance, y, settings, log_density)


def integrate_gap(
    family: NoiseFamily,
    prior_variance: float,
    span: tuple[float, float],
    settings: Quadrature,
) -> float:
    """The averaged gap over one `y` interval.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        span: the `y` interval.
        settings: the tolerances and the window scale.

    Returns:
        `∫ p*(y)·KL(q ‖ p(·|y)) dy` over the interval.
    """
    value, _ = quad(
        lambda y: weighted_gap(family, prior_variance, y, settings),
        span[0],
        span[1],
        epsabs=settings.epsabs,
        epsrel=settings.epsrel,
        limit=settings.limit,
    )
    return value


def core_and_tail(
    family: NoiseFamily,
    prior_variance: float,
    half_width: float,
    settings: Quadrature,
) -> tuple[float, float]:
    """The gap inside a `y` grid and the mass that grid throws away.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        half_width: L — the grid half-width in `y`.
        settings: the tolerances and the window scale.

    Returns:
        The core integral and the summed two-sided tail.
    """
    centre = family.prior_mean
    core = integrate_gap(
        family, prior_variance, (centre - half_width, centre + half_width), settings
    )
    upper = integrate_gap(
        family,
        prior_variance,
        (centre + half_width, centre + half_width + TAIL_EXTENT),
        settings,
    )
    lower = integrate_gap(
        family,
        prior_variance,
        (centre - half_width - TAIL_EXTENT, centre - half_width),
        settings,
    )
    return core, upper + lower


def predictive_sd(
    family: NoiseFamily, prior_variance: float, span: float, settings: Quadrature
) -> float:
    """`√Var_{p*}[y]`, measured by quadrature rather than assumed.

    Args:
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        span: half-width in `y` to integrate the moments over.
        settings: the tolerances and the window scale.

    Returns:
        The true predictive standard deviation.
    """
    centre = family.prior_mean
    lo, hi = centre - span, centre + span

    def moment(power: int) -> float:
        # Descriptive rather than the measurement under test, and the span is wide
        # against a sharply peaked density. Tightening to the suite's tolerance only
        # buys roundoff warnings on digits nothing reads.
        value, _ = quad(
            lambda y: (
                (y - centre) ** power
                * math.exp(log_predictive(family, prior_variance, y, settings))
            ),
            lo,
            hi,
            epsabs=1e-13,
            epsrel=1e-11,
            limit=settings.limit,
            points=(centre,),
        )
        return value

    mass = moment(0)
    return math.sqrt(moment(2) / mass - (moment(1) / mass) ** 2)
