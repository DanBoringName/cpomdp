"""The paper's scalar update, with and without the block the repair deletes.

`research/spinello_stilwell_hand_derivation.md` derives (35c) to (35e), and its step 5
splits the Gauss-Newton curvature by Jacobian row::

    R = (dr2)^T (dr2) + (dr3)^T (dr3)

The second block is the fourth printed term of (35d), `grad_noise^2 / (4 noise^2 ln
noise)`. It is the only part of the scheme that is not a real square, and the only part
that moves when the observation is rescaled. `log_block=False` is the scheme with it
removed.

One implementation, two callers: `invariance` measures what rescaling moves and `repair`
measures what the deletion costs. A second copy of (35) is the thing that drifts.

Not the rung. No guard at the pole. The iteration budget is an argument rather than
a declaration, because `research/spinello_stilwell_rung.md` still owes both decisions.

**Care with the letter sigma.** cpomdp writes `sigma` for the prior spread and the paper
writes it for the observation-noise variance. `noise` is the paper's quantity throughout
and `spread` is cpomdp's.
"""

import math

__all__ = [
    "fisher_information",
    "gauss_newton_curvature",
    "gradient",
    "iterate",
    "kalman_update",
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
        `R` as printed, or `R_mod` without its fourth term.
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
    scale: float = 1.0,
    tolerance: float = 1e-14,
    max_iterations: int = 200,
    *,
    log_block: bool = True,
) -> tuple[float, float, int]:
    """Spinello-Stilwell (35) for a scalar linear-mean sensor, in rescaled units.

    `R(x) = base_noise + curvature * x^2` and `h(x) = x`, both rescaled by `scale`.

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
        The converged estimate in unrescaled state units, the posterior variance, and
        the number of iterations taken.
    """
    reading = scale * observation
    prior_precision = 1.0 / prior_variance
    estimate = prior_mean
    taken = 0
    while taken < max_iterations:
        taken += 1
        noise = scale**2 * (base_noise + curvature * estimate**2)
        mean_slope = scale
        noise_slope = scale**2 * 2.0 * curvature * estimate
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

    noise = scale**2 * (base_noise + curvature * estimate**2)
    noise_slope = scale**2 * 2.0 * curvature * estimate
    fisher = fisher_information(noise, noise_slope, scale)
    return estimate, 1.0 / (prior_precision + fisher), taken
