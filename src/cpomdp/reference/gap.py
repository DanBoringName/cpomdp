"""The averaged inference gap, ``E_{y∼p*}[ KL(q ‖ p(x|y)) ]``.

What a Gaussian filter loses by being Gaussian, in nats, averaged over the readings
the true model actually produces. Zero when the approximation is exact, which under a
fixed ``R`` it is, and positive under a state-dependent ``R(x)`` for reasons no
resolution removes.

Three conventions are pinned, and they are the same three
``research.checks.gap_kernel`` pins. They belong to the quantity rather than to either
implementation, and a fork between the two would be two different numbers under one
name:

- **Direction is reverse**, ``KL(q ‖ p)``, the agent against the exact posterior. The
  forward direction is a different number and no declared figure refers to it.
- **Where the approximation reads its noise is the rule's business**, not this
  module's. The plug-in rung freezes ``R̂ = R(μ)`` at the prior mean across the
  update. That decision lives in whatever is passed as ``approximate_posterior``.
- **The average is under the true ``p*(y)``**, never the agent's own predictive.
  ``p*`` falls out of the same product the exact posterior comes from, so it costs
  nothing extra and cannot drift from it.

That last point is why the observation grid needs the same care as the state grid.
``p*`` under an unbounded ``R(x)`` is a scale mixture, so its tails are exponential
and a half-width of "k standard deviations" sizes a Gaussian tail against one that is
not (``research.checks.predictive_truncation`` measures where a given ``k`` fails).
The box is declared by the caller and what fell outside it is reported.

Two engines compute this quantity today. ADR-052 records why, and issue #102 tracks
cutting it back to one once the ``p*`` work settles where that code lives.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.reference.likelihood import ObservationLikelihood
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid

__all__ = ["InferenceGap", "averaged_inference_gap"]

# What an approximate filter does with one reading: prior in, Gaussian belief out, on
# the prior's own grid. The rule ladder's rungs are the implementations of this.
ApproximatePosterior = Callable[[GridDensity, ArrayLike], GridDensity]


@jax.jit
def _weigh_one_observation(
    prior: GridDensity,
    approximation: GridDensity,
    log_likelihood: Float64[Array, "N"],
) -> tuple[Float64[Array, ""], Float64[Array, ""]]:
    """``log p*(y)`` and ``KL(q ‖ p(x|y))`` from one reading's log-likelihood.

    Both come off the same unnormalised product, so the predictive and the posterior
    cannot disagree about what the model says.

    Compiled because the sweep calls it once per observation node and every operation
    in it is a small array op. Unjitted, the dispatch dominates: the divergence alone
    is three separate passes over the state grid, and the loop spends its time
    launching them rather than computing. The shapes and the lattice are constant
    across the sweep, so this traces once.
    """
    joint = GridDensity(prior.grid, prior.log_density + log_likelihood)
    return joint.log_mass, approximation.kl_to(joint)


@dataclass(frozen=True)
class InferenceGap:
    """A measured gap, with what a reader needs to judge it.

    Attributes:
        value: the averaged gap in nats, the expectation under the captured part of
            ``p*``. Normalised by ``predictive_mass``, so it is the conditional
            expectation given a reading inside the box rather than an undercount.
        predictive_mass: what ``p*(y)`` integrated to over the declared observation
            box. One when the box caught everything. Below one when it did not, and
            then ``value`` speaks for a conditional the caller did not ask for.
        divergences: ``KL(q ‖ p(x|y))`` at each observation node, in the grid's node
            order. Where the gap concentrates in ``y`` is a property of the gap, not
            a by-product, and it is what a tail-dominated result looks like.
        observation_grid: the grid the average was taken on, carrying the box and the
            resolution that produced these numbers.
    """

    value: float
    predictive_mass: float
    divergences: Float64[Array, "K"]
    observation_grid: QuadratureGrid


def averaged_inference_gap(
    prior: GridDensity,
    likelihood: ObservationLikelihood,
    approximate_posterior: ApproximatePosterior,
    observation_grid: QuadratureGrid,
) -> InferenceGap:
    """Measure ``E_{y∼p*}[ KL(q ‖ p(x|y)) ]`` over a declared observation box.

    One pass per observation node. Each builds the unnormalised product
    ``p(x)·p(y|x)`` once and reads both things off it: what it integrates to is
    ``p*(y)``, and normalising it gives the exact posterior. The predictive and the
    posterior therefore come from the same object and cannot disagree.

    Costs ``K`` likelihood evaluations over ``N`` nodes, which is linear in both. It
    is cheap beside a single prediction, which is quadratic in ``N``.

    Args:
        prior: the true state density before the reading. Normalised internally, so
            ``p*`` is a density rather than a scaled one.
        likelihood: the true observation likelihood. The same one produces ``p*`` and
            the exact posterior, since the gap is about the approximation only.
        approximate_posterior: the rule under test, called as
            ``rule(prior, observation)`` and returning its Gaussian belief on the
            prior's grid.
        observation_grid: the box and resolution the average is taken over. Its
            dimension is the observation's, not the state's.

    Returns:
        The gap and its diagnostics.

    Raises:
        ValueError: if a rule returns a belief on a lattice other than the prior's.
            Two densities on different lattices have no divergence between them.
    """
    prior = prior.normalise()
    states = prior.grid.nodes

    log_predictive = []
    divergences = []
    for observation in observation_grid.nodes:
        approximation = approximate_posterior(prior, observation)
        if not approximation.grid.same_lattice_as(prior.grid):
            raise ValueError(
                "approximate_posterior must return a belief on the prior's lattice, "
                f"got {approximation.grid!r} against {prior.grid!r}"
            )
        log_mass, divergence = _weigh_one_observation(
            prior,
            approximation,
            likelihood.log_likelihood(observation, states),
        )
        log_predictive.append(log_mass)
        divergences.append(divergence)

    predictive = GridDensity(observation_grid, jnp.stack(log_predictive))
    stacked = jnp.stack(divergences)
    return InferenceGap(
        value=float(predictive.expectation(stacked)),
        predictive_mass=float(jnp.exp(predictive.log_mass)),
        divergences=stacked,
        observation_grid=observation_grid,
    )
