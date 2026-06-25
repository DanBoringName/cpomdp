"""Canonical-form Gaussian messages: the FFG's wire payload.

A message on an edge of the factor graph is a Gaussian potential carried in
canonical (information) form, ``(precision, potential) = (Λ, h)`` with
``Λ = Σ⁻¹`` (precision matrix) and ``h = Σ⁻¹μ`` (potential / information
vector). Two properties make this the storage form (DECISIONS.md ADR-012):

- **Factor product is addition.** Multiplying Gaussian potentials — what a
  factor node does when it combines its incoming messages — is
  ``(Λ1, h1) + (Λ2, h2) = (Λ1+Λ2, h1+h2)``. No inversion, ever, on this path.
- **Moment form (mean, covariance) is a view**, computed only at readout
  (``to_moment``) or when a block of variables must be eliminated
  (``marginalize`` — the one place an inversion is intrinsic to the
  operation: it inverts only the eliminated block, never the whole ``Λ``).
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance, validate_finite


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CanonicalGaussian:
    """A Gaussian message in canonical (information) form, ``(Λ, h)``.

    - ``precision`` -- Λ = Σ⁻¹. An n x n matrix. Symmetric positive-*semi*-
      definite (not necessarily definite: a single, not-yet-combined factor
      message can be rank-deficient — e.g. a factor that doesn't yet
      constrain every direction. Only the *combined* product of all of a
      variable's incoming messages needs to be definite, and only at the
      point something reads its moment form).
    - ``potential`` -- h = Σ⁻¹μ. A 1-D vector of length n.

    Construct directly from canonical parameters. There is no
    ``from_moment``/moment-form constructor in v0.4 (open question, parked in
    the build plan) — moment form is purely a readout view via
    ``to_moment``.
    """

    precision: Float64[Array, "n, n"]
    potential: Float64[Array, "n"]

    def __init__(self, precision: ArrayLike, potential: ArrayLike):
        object.__setattr__(self, "precision", jnp.asarray(precision, dtype=float))
        object.__setattr__(self, "potential", jnp.asarray(potential, dtype=float))
        self._validate()

    def _validate(self):
        if self.potential.ndim != 1:
            raise ValueError(
                f"potential must be 1-D vector, got shape {self.potential.shape}"
            )
        validate_finite(self.potential, "potential")
        validate_covariance(self.precision, "precision", require_definite=False)
        n = self.potential.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match a {n}-D potential, "
                f"got shape {self.precision.shape}"
            )

    @property
    def ndim(self) -> int:
        """Dimensionality of the message — the length of the potential vector."""
        return self.precision.shape[0]

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float64[Array, "n"], Float64[Array, "n n"]], None]:
        """Leaves for JAX: ``(precision, potential)``, no static aux data."""
        return (self.precision, self.potential), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: None,
        children: tuple[Float64[Array, "n"], Float64[Array, "n n"]],
    ) -> "CanonicalGaussian":
        """Rebuild from leaves without validating — the leaves may be tracers."""
        precision, potential = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "precision", precision)
        object.__setattr__(obj, "potential", potential)
        return obj
