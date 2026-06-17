"""FixedSensor: the constant-(C, R) observation model."""

import jax.numpy as jnp
import pytest

from cpomdp.observation import FixedSensor, ObservationModel


def test_fixed_sensor_linearize_returns_stored_matrices():
    C = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.5]])
    sensor = FixedSensor(C, R)

    C_out, R_out = sensor.linearize(jnp.array([3.0, 7.0]))

    assert jnp.allclose(C_out, C)
    assert jnp.allclose(R_out, R)


def test_fixed_sensor_is_state_independent():
    sensor = FixedSensor(jnp.array([[1.0, 0.0]]), jnp.array([[0.5]]))

    C0, R0 = sensor.linearize(jnp.array([0.0, 0.0]))
    C1, R1 = sensor.linearize(jnp.array([99.0, -99.0]))

    assert jnp.allclose(C0, C1)
    assert jnp.allclose(R0, R1)
    assert sensor.is_fixed is True


def test_fixed_sensor_satisfies_protocol():
    sensor = FixedSensor(jnp.array([[1.0]]), jnp.array([[1.0]]))

    assert isinstance(sensor, ObservationModel)


def test_fixed_sensor_rejects_non_2d_sensor_model():
    with pytest.raises(ValueError, match="2-D"):
        FixedSensor(jnp.array([1.0, 0.0]), jnp.array([[0.5]]))


def test_fixed_sensor_rejects_mismatched_dims():
    C = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # m=2
    R = jnp.array([[0.5]])  # m=1
    with pytest.raises(ValueError, match="match the 2-D observation"):
        FixedSensor(C, R)
