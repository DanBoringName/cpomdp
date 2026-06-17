"""Shared construction-time validators (internal)."""

import jax.numpy as jnp
from jaxtyping import Array, Float64

__all__ = ["validate_covariance"]


def validate_covariance(cov: Float64[Array, "n n"], name: str) -> None:
    """Square (2-D, n x n) + symmetric check.

    Shared by Belief.cov, dynamics_noise and sensor_noise — all three are
    covariance matrices with the same invariants. Positive-semi-definiteness is
    deliberately NOT checked here: it's enforced at the trust boundary (user
    input), not on every construction. See DECISIONS.md ADR-002.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {cov.shape}")
    if not jnp.allclose(cov, cov.T):
        raise ValueError(f"{name} must be symmetric.")
