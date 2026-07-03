"""Node-targeted epistemic value for the FFG EFE (issue #26, Phase A).

The epistemic term generalises from whole-state info gain (the observation-space
½(ln det S − ln det R) the kernel computes today) to info gain about a *chosen* latent's
marginal, in state space: ½(ln det Σ⁺[target] − ln det Σ_post[target]). This file pins
the kernel — first the full-state anchor (it must reproduce the observation-space
number), then the node-restricted case.
"""

import jax.numpy as jnp
import numpy as np

from cpomdp.efe import _state_info_gain, expected_free_energy
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_full_state_info_gain_equals_observation_space_form():
    # target = the WHOLE state: the state-space info gain must reproduce today's
    # observation-space epistemic ½(ln det S − ln det R), with S = C·Σ⁺·Cᵀ + R.
    rng = np.random.default_rng(0)
    n, m = 4, 2
    sigma_pred = jnp.asarray(_spd(rng, n))  # Σ⁺, the predicted joint covariance
    sensor_model = jnp.asarray(rng.standard_normal((m, n)))  # C
    sensor_noise = jnp.asarray(_spd(rng, m))  # R

    # The observation-space number the current kernel computes.
    s = sensor_model @ sigma_pred @ sensor_model.T + sensor_noise  # S
    _, logdet_s = np.linalg.slogdet(s)
    _, logdet_r = np.linalg.slogdet(sensor_noise)
    expected = 0.5 * (logdet_s - logdet_r)

    # The new state-space route, restricted to the whole state (target = all indices).
    got = _state_info_gain(sigma_pred, sensor_model, sensor_noise, target=range(n))
    np.testing.assert_allclose(float(got), expected, atol=1e-10)


def test_node_info_gain_matches_independent_moment_update():
    # Restrict target to ONE node's block, and check against Σ_post computed the *other*
    # way — the Kalman moment update Σ_post = Σ⁺ − K·C·Σ⁺ (the function uses info form).
    rng = np.random.default_rng(1)
    n, m = 5, 2
    sigma_pred = _spd(rng, n)  # Σ⁺
    sensor_model = rng.standard_normal((m, n))  # C
    sensor_noise = _spd(rng, m)  # R
    target = [1, 2]  # a node occupying state indices 1..2

    # Independent (moment-form) posterior covariance, then the block log-det drop.
    s = sensor_model @ sigma_pred @ sensor_model.T + sensor_noise
    gain = sigma_pred @ sensor_model.T @ np.linalg.inv(s)  # K
    sigma_post = sigma_pred - gain @ sensor_model @ sigma_pred
    block = np.ix_(target, target)
    _, logdet_pred = np.linalg.slogdet(sigma_pred[block])
    _, logdet_post = np.linalg.slogdet(sigma_post[block])
    expected = 0.5 * (logdet_pred - logdet_post)

    got = _state_info_gain(
        jnp.asarray(sigma_pred),
        jnp.asarray(sensor_model),
        jnp.asarray(sensor_noise),
        target=target,
    )
    np.testing.assert_allclose(float(got), expected, atol=1e-9)


def test_full_state_gain_matches_expected_free_energy_epistemic():
    # The production tie: on a real model, the whole-state info gain must reproduce the
    # epistemic that `expected_free_energy` actually returns (Σ⁺ = A·Σ·Aᵀ + Q).
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    sensor_model = np.array([[1.0, 0.0]])  # C
    sensor_noise = np.array([[0.5]])  # R
    model = LinearGaussianModel(
        dynamics=dynamics,
        sensor_model=sensor_model,
        dynamics_noise=dynamics_noise,
        sensor_noise=sensor_noise,
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control=[[0.0], [1.0]],
    )
    belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
    action = jnp.array([0.5])
    preference = Preference(goal=[1.0], precision=[[2.0]])

    _, parts = expected_free_energy(model, belief, action, preference)

    sigma = np.asarray(belief.cov)
    sigma_pred = dynamics @ sigma @ dynamics.T + dynamics_noise  # Σ⁺
    got = _state_info_gain(
        jnp.asarray(sigma_pred),
        jnp.asarray(sensor_model),
        jnp.asarray(sensor_noise),
        target=range(2),
    )
    np.testing.assert_allclose(float(got), float(parts["epistemic"]), atol=1e-9)
