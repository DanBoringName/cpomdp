from cpomdp._validation import validate_covariance
from jaxtyping import Float64
from jax import Array
import jax.numpy as jnp
import jax
from numpy.typing import ArrayLike
from dataclasses import dataclass

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class GaussianObservation:

    sensor_model: Float64[Array, "m, n"]
    sensor_noise: Float64[Array, "m, m"]

    def __init__(self, sensor_model: ArrayLike, sensor_noise: ArrayLike):
        object.__setattr__(self, "sensor_model", jnp.asarray(sensor_model, dtype=float))
        object.__setattr__(self, "sensor_noise", jnp.asarray(sensor_noise, dtype=float))
        self._validate()

    def _validate(self):
        if self.sensor_model.ndim != 2:
            raise ValueError(
                f"sensor_model must be 2-D vector, got shape {self.sensor_model.shape}"
            )
        validate_covariance(self.sensor_model, "sensor_model", require_definite=True)


class GaussianTransition: