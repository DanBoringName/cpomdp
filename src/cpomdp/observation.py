"""Observation models: how a hidden state produces a sensor reading.

The ``ObservationModel`` protocol is the seam the EFE core asks for a local
linear-Gaussian ``(C, R)`` about a state. ``FixedSensor`` is the constant case
(the v0.2 default); state-dependent sensors arrive in v0.3.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance

__all__ = ["FixedSensor", "ObservationModel"]


@runtime_checkable
class ObservationModel(Protocol):
    """How a hidden state produces an observation, as a local linear-Gaussian map.

    The EFE core never assumes a fixed sensor matrix; it asks the observation
    model to linearize itself about a state ``x``, getting back the local
    ``(C, R)`` (the observation Jacobian and the noise covariance there). For a
    fixed sensor these are constant; for a state-dependent sensor they vary.

    Linearise is the english spelling, but literature dictates a z.
    """

    sensor_model: Float64[Array, "m n"]  # C
    sensor_noise: Float64[Array, "m m"]  # R
    is_fixed: bool

    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Local ``(C, R)`` about state ``x``."""
        ...


@dataclass(frozen=True, init=False)
class FixedSensor:
    """A sensor whose (C, R) never change with state — the v0.2 default.

    ``linearize`` returns the same stored matrices for every ``x``: a fixed
    linear sensor *is* its own linear approximation everywhere. This is the
    regime where EFE's epistemic term is constant and collapses to LQR
    (DECISIONS.md ADR-003).
    """

    sensor_model: Float64[Array, "m n"]  # C
    sensor_noise: Float64[Array, "m m"]  # R
    is_fixed = True

    def __init__(self, sensor_model: ArrayLike, sensor_noise: ArrayLike) -> None:
        object.__setattr__(self, "sensor_model", jnp.asarray(sensor_model, dtype=float))
        object.__setattr__(self, "sensor_noise", jnp.asarray(sensor_noise, dtype=float))
        self._validate()

    def linearize(
        self, x: ArrayLike
    ) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """Return the stored ``(C, R)`` unchanged — the same for every ``x``."""
        return self.sensor_model, self.sensor_noise

    def _validate(self) -> None:
        if self.sensor_model.ndim != 2:
            raise ValueError(
                f"sensor_model must be a 2-D (m x n) matrix, "
                f"got shape {self.sensor_model.shape}"
            )
        validate_covariance(self.sensor_noise, "sensor_noise")
        m = self.sensor_model.shape[0]
        if self.sensor_noise.shape != (m, m):
            raise ValueError(
                f"sensor_noise must be {m}x{m} to match the {m}-D observation, "
                f"got shape {self.sensor_noise.shape}"
            )
