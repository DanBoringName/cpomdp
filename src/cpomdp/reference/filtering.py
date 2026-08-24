"""Exact Bayesian filtering on a quadrature grid.

One step of Bayes as it is actually written: multiply the prior by the likelihood,
divide by what that integrates to. On a grid both operations are elementwise, and the
normalising integral that has no closed form in general is a weighted sum.

The intractability of exact Bayes is a statement about dimension. A tensor grid costs
``k^d`` nodes, which is hopeless by ``d = 10`` and free at ``d = 1``. This module is
an instrument rather than a deployable filter: it runs on a low-dimensional latent to
measure what a Gaussian filter gets wrong on the same model, and it is never on an
agent's per-cycle path.
"""

from numpy.typing import ArrayLike

from cpomdp.reference.likelihood import ObservationLikelihood
from cpomdp.reference.quadrature import GridDensity

__all__ = ["condition"]


def condition(
    prior: GridDensity,
    likelihood: ObservationLikelihood,
    observation: ArrayLike,
) -> GridDensity:
    """The measurement update: ``p(x | y) ∝ p(x) · p(y | x)``, normalised.

    Exact up to the grid. The only error is the quadrature's, which is discretisation
    and truncation, both of which shrink under refinement. No step of this assumes
    the posterior has any particular shape, which is the difference from a Gaussian
    filter, whose error under a state-dependent ``R(x)`` is structural and shrinks
    under nothing.

    The result is normalised, since an unnormalised posterior fed back in as the next
    step's prior drifts out of floating-point range within a few steps. Nothing is
    lost by it: ``posterior.log_normaliser - prior.log_normaliser`` is the log of
    what the product integrated to, which is the log evidence ``log p(y)`` whenever
    the prior itself integrates to one.

    Args:
        prior: the belief before the observation, on the grid the posterior will use.
        likelihood: evaluated once at every node of ``prior.grid``.
        observation: the reading to condition on.

    Returns:
        The posterior on the same grid, integrating to one.
    """
    log_likelihood = likelihood.log_likelihood(observation, prior.grid.nodes)
    return GridDensity(
        prior.grid,
        prior.log_density + log_likelihood,
        prior.log_normaliser,
    ).normalise()
