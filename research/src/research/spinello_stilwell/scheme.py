"""The paper's scalar update, with and without the block the repair deletes.

`research/spinello_stilwell_hand_derivation.md` derives (35c) to (35e), and its step 5
splits the Gauss-Newton curvature by Jacobian row::

    curvature = (dr2)^T (dr2) + (dr3)^T (dr3)

The second block is the fourth printed term of (35d), `grad_noise^2 / (4 noise^2 ln
noise)`. It is the only part of the scheme that is not a real square, and the only part
that moves when the observation is rescaled. `log_block=False` is the scheme with it
removed.

The paper's letter for the Gauss-Newton matrix is the one cpomdp uses for the noise, so
it is never written here. `gauss_newton_curvature` is that matrix, and the docstrings
say "the printed curvature" for the paper's form and "the modified curvature" for the
scheme's.

One implementation, two callers: `invariance` measures what rescaling moves and `repair`
measures what the deletion costs. A second copy of (35) is the thing that drifts.

Not the rung. No guard at the pole, which ADR-057 decides the shipped rung does not
need. The iteration budget is an argument rather than a declaration, because
`research/spinello_stilwell_rung.md` still owes that one.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity throughout
and `spread` is cpomdp's.
"""

import math
from collections.abc import Callable

NoiseAt = Callable[[float], tuple[float, float]]
"""A declared `R` read at a state, returning `(noise, its slope)` in native units."""

__all__ = [
    "NoiseAt",
    "fisher_information",
    "gauss_newton_curvature",
    "gradient",
    "iterate",
    "iterate_with",
    "kalman_update",
    "objective",
    "quadratic_noise",
]


def gradient(
    noise: float,
    noise_slope: float,
    mean_slope: float,
    residual: float,
) -> float:
    """The measurement part of the gradient, equation (35c).

    Args:
        noise: the observation-noise variance at the current iterate, `sigma`.
        noise_slope: its derivative in the state, `grad sigma`.
        mean_slope: the derivative of the observation mean, `grad h`.
        residual: the innovation, `zeta`.

    Returns:
        `s`, which carries no `ln noise` term because the logs cancel out of it.
    """
    return (
        -(residual / noise) * mean_slope
        + (1.0 / (2.0 * noise)) * (1.0 - residual**2 / noise) * noise_slope
    )


def gauss_newton_curvature(
    noise: float,
    noise_slope: float,
    mean_slope: float,
    residual: float,
    *,
    log_block: bool = True,
) -> float:
    """The Gauss-Newton matrix, equation (35d), scalar case.

    The first three printed terms are the `r2` block and collapse to
    `(1/noise) * b * b` with `b = mean_slope + (residual / 2 noise) * noise_slope`, so
    they are a real square for every state. The fourth is the `r3` block, whose sign
    follows `ln noise`.

    Args:
        noise: the observation-noise variance at the current iterate, `sigma`.
        noise_slope: its derivative in the state, `grad sigma`.
        mean_slope: the derivative of the observation mean, `grad h`.
        residual: the innovation, `zeta`.
        log_block: keep the `r3` block. False is the modification of step 6.

    Returns:
        The printed curvature, or the modified one without its fourth term.
    """
    real_square = (
        mean_slope**2 / noise
        + (residual / (2.0 * noise**2)) * 2.0 * mean_slope * noise_slope
        + (residual**2 / (4.0 * noise**3)) * noise_slope**2
    )
    if not log_block:
        return real_square
    return real_square + noise_slope**2 / (4.0 * noise**2 * math.log(noise))


def fisher_information(
    noise: float,
    noise_slope: float,
    mean_slope: float,
) -> float:
    """The information a measurement carries on average, equation (35e).

    Built from `s` alone, so no `ln noise` reaches it and the modification leaves it
    untouched. Both terms are non-negative, so the posterior precision it feeds is
    positive whatever the state.

    Args:
        noise: the observation-noise variance at the current iterate, `sigma`.
        noise_slope: its derivative in the state, `grad sigma`.
        mean_slope: the derivative of the observation mean, `grad h`.

    Returns:
        `U-bar`.
    """
    return mean_slope**2 / noise + noise_slope**2 / (2.0 * noise**2)


def objective(
    estimate: float,
    prior_mean: float,
    prior_variance: float,
    noise: float,
    residual: float,
) -> float:
    """The function the rung minimises, equation (18), up to constants in the state.

    Step 1 of the derivation: prior distance, measurement mismatch, and the Gaussian
    normaliser. The third term is what `R(x)` keeps in play, and it is why a step that
    climbs can be told from one that descends.

    Args:
        estimate: the current iterate.
        prior_mean: the predicted mean.
        prior_variance: the predicted variance.
        noise: the observation-noise variance at the iterate, `sigma`.
        residual: the innovation there, `zeta`.

    Returns:
        `l(x)`, comparable between two iterates of the same run and not otherwise.
    """
    prior_gap = estimate - prior_mean
    return 0.5 * (prior_gap**2 / prior_variance + residual**2 / noise + math.log(noise))


def quadratic_noise(base_noise: float, curvature: float) -> NoiseAt:
    """`R(x) = base_noise + curvature * x^2` and its slope, as one callable.

    Args:
        base_noise: `R0`.
        curvature: `kappa`.

    Returns:
        A callable giving `(R(x), R'(x))`.
    """

    def noise_at(state: float) -> tuple[float, float]:
        return base_noise + curvature * state**2, 2.0 * curvature * state

    return noise_at


def kalman_update(
    observation: float,
    prior_mean: float,
    prior_variance: float,
    noise: float,
) -> tuple[float, float]:
    """The ordinary scalar Kalman update, the oracle for the fixed-noise reduction.

    Exact where the observation map is affine and the noise does not depend on the
    state, which is the one regime in which the rung has an independent answer to
    agree with.

    Args:
        observation: the reading.
        prior_mean: the predicted mean.
        prior_variance: the predicted variance.
        noise: the observation-noise variance, constant in the state.

    Returns:
        The posterior mean and variance.
    """
    gain = prior_variance / (prior_variance + noise)
    return prior_mean + gain * (observation - prior_mean), (1.0 - gain) * prior_variance


def iterate(
    observation: float,
    prior_mean: float,
    prior_variance: float,
    base_noise: float,
    curvature: float,
    scale: float,
    tolerance: float,
    max_iterations: int,
    *,
    log_block: bool = True,
) -> tuple[float, float, int]:
    """Spinello-Stilwell (35) for a scalar linear-mean sensor, in rescaled units.

    `R(x) = base_noise + curvature * x^2` and `h(x) = x`, both rescaled by `scale`.
    The budget and the tolerance have no defaults. A probe runs at some budget and
    says which, and the rung's own stays undeclared (ADR-056). Exhausting the budget
    is reported through the count returned, which then equals `max_iterations`.

    Args:
        observation: the reading, in unrescaled units.
        prior_mean: the predicted mean.
        prior_variance: the predicted variance.
        base_noise: `R0`.
        curvature: `kappa`.
        scale: `lambda`, the factor the observation is multiplied by.
        tolerance: stop when the step falls below this, relative to the prior spread.
        max_iterations: give up after this many steps.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The estimate in unrescaled state units, the posterior variance, and the number
        of iterations taken. A count equal to `max_iterations` is a run that did not
        converge, and the caller decides what that means.
    """
    return iterate_with(
        observation,
        prior_mean,
        prior_variance,
        quadratic_noise(base_noise, curvature),
        scale,
        tolerance,
        max_iterations,
        log_block=log_block,
    )


def iterate_with(
    observation: float,
    prior_mean: float,
    prior_variance: float,
    noise_at: NoiseAt,
    scale: float,
    tolerance: float,
    max_iterations: int,
    *,
    log_block: bool = True,
) -> tuple[float, float, int]:
    """(35) for any declared `R`, which is what a survey over several families needs.

    The quadratic case reaches this through `iterate`. Routes 3 and 5 reach it with
    the other declared families, which have no closed-form iterate count between them.

    Args:
        observation: the reading, in unrescaled units.
        prior_mean: the predicted mean.
        prior_variance: the predicted variance.
        noise_at: the declared `R`, read as `(R(x), R'(x))` in native units.
        scale: `lambda`, the factor the observation is multiplied by.
        tolerance: stop when the step falls below this, relative to the prior spread.
        max_iterations: give up after this many steps.
        log_block: keep the `r3` block of (35d). False is the modification.

    Returns:
        The estimate in unrescaled state units, the posterior variance, and the number
        of iterations taken.
    """
    reading = scale * observation
    prior_precision = 1.0 / prior_variance
    estimate = prior_mean
    taken = 0
    while taken < max_iterations:
        taken += 1
        native_noise, native_slope = noise_at(estimate)
        noise = scale**2 * native_noise
        mean_slope = scale
        noise_slope = scale**2 * native_slope
        residual = reading - scale * estimate

        step = (
            prior_precision * (estimate - prior_mean)
            + gradient(noise, noise_slope, mean_slope, residual)
        ) / (
            prior_precision
            + gauss_newton_curvature(
                noise, noise_slope, mean_slope, residual, log_block=log_block
            )
        )
        estimate -= step
        if abs(step) < tolerance * math.sqrt(prior_variance):
            break

    native_noise, native_slope = noise_at(estimate)
    fisher = fisher_information(scale**2 * native_noise, scale**2 * native_slope, scale)
    return estimate, 1.0 / (prior_precision + fisher), taken
