"""Tier-1 linear-Gaussian factor nodes (v0.4 Phase 2, DECISIONS.md ADR-012).

Oracle-first: these pin ``GaussianObservation`` and ``GaussianTransition`` before
they exist, so the file is RED until ``cpomdp.ffg.factors.linear_gaussian`` lands.
Each oracle is an independent path — the moment-form measurement update and the
moment-form predict, computed in plain NumPy — never the canonical-form math
under test.

The decomposition these factors implement (one Kalman step = Phase 1 ops):

- ``GaussianObservation(C, R).message(y)`` -> ``CanonicalGaussian`` on x, the
  information form ``(CᵀR⁻¹C, CᵀR⁻¹y)``. The measurement *update* is then
  ``belief + message`` (``CanonicalGaussian.__add__``).
- ``GaussianTransition(A, Q).predict(message, control_term)`` -> the *predict*
  step: build the joint over ``[x, x']``, fold the message into the x block,
  marginalize x out. Moment form: ``AΣAᵀ+Q``, ``Aμ+b``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.ffg.factors.linear_gaussian import GaussianObservation, GaussianTransition
from cpomdp.ffg.message import CanonicalGaussian


def _spd(rng, n):
    """A random n x n symmetric positive-definite matrix (NumPy, independent)."""
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _belief_as_canonical(mean, cov):
    """Moment-form (mean, cov) -> its canonical message (NumPy, independent)."""
    precision = np.linalg.inv(cov)
    return CanonicalGaussian(precision, precision @ mean)


# --- Observation factor: the measurement update -------------------------------


class TestGaussianObservation:
    def test_stores_coerced_arrays(self):
        fac = GaussianObservation([[1.0, 0.0]], [[2.0]])
        assert isinstance(fac.sensor_model, jax.Array)
        np.testing.assert_array_equal(fac.sensor_model, [[1.0, 0.0]])
        np.testing.assert_array_equal(fac.sensor_noise, [[2.0]])

    def test_rejects_singular_sensor_noise(self):
        # R is inverted in the message, so a singular R is rejected at construction.
        with pytest.raises(ValueError, match="positive-definite"):
            GaussianObservation([[1.0]], [[0.0]])

    def test_rejects_sensor_noise_shape_mismatch(self):
        # C is 1xn (m=1) but R is 2x2 — R must be m x m.
        with pytest.raises(ValueError, match="match"):
            GaussianObservation([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])

    def test_message_is_information_form_of_likelihood(self):
        rng = np.random.default_rng(0)
        n, m = 3, 2
        C = rng.standard_normal((m, n))
        R = _spd(rng, m)
        y = rng.standard_normal(m)
        msg = GaussianObservation(C, R).message(y)
        Rinv = np.linalg.inv(R)
        np.testing.assert_allclose(msg.precision, C.T @ Rinv @ C, atol=1e-10)
        np.testing.assert_allclose(msg.potential, C.T @ Rinv @ y, atol=1e-10)

    @pytest.mark.parametrize(("n", "m"), [(1, 1), (2, 1), (3, 2), (4, 3)])
    def test_update_matches_moment_form_measurement_update(self, n, m):
        # Oracle: the standard Kalman *update* (no prediction), moment form, NumPy.
        rng = np.random.default_rng(n * 10 + m)
        C = rng.standard_normal((m, n))
        R = _spd(rng, m)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        y = rng.standard_normal(m)
        gain = cov @ C.T @ np.linalg.inv(C @ cov @ C.T + R)
        mean_post = mean + gain @ (y - C @ mean)
        cov_post = (np.eye(n) - gain @ C) @ cov

        post = _belief_as_canonical(mean, cov) + GaussianObservation(C, R).message(y)
        out_mean, out_cov = post.to_moment()
        np.testing.assert_allclose(out_mean, mean_post, atol=1e-8)
        np.testing.assert_allclose(out_cov, cov_post, atol=1e-8)

    def test_jit_and_grad_through_message(self):
        fac = GaussianObservation([[1.0, 0.5], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        y = jnp.array([1.0, -1.0])
        eager = fac.message(y).potential
        jitted = jax.jit(lambda yy: fac.message(yy).potential)(y)
        np.testing.assert_allclose(jitted, eager, atol=1e-12)
        grad = jax.grad(lambda yy: fac.message(yy).potential.sum())(y)
        assert bool(jnp.all(jnp.isfinite(grad)))


# --- Transition factor: the predict step --------------------------------------


class TestGaussianTransition:
    def test_stores_coerced_arrays(self):
        fac = GaussianTransition([[1.0]], [[2.0]])
        assert isinstance(fac.dynamics, jax.Array)
        np.testing.assert_array_equal(fac.dynamics, [[1.0]])

    def test_rejects_singular_process_noise(self):
        # Q is inverted in the joint, so a singular Q is rejected at construction.
        with pytest.raises(ValueError, match="positive-definite"):
            GaussianTransition([[1.0]], [[0.0]])

    def test_rejects_nonsquare_dynamics(self):
        with pytest.raises(ValueError, match="square"):
            GaussianTransition([[1.0, 0.0]], [[1.0]])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_predict_matches_moment_form(self, n):
        # Oracle: moment-form predict, NumPy. cov_pred = AΣAᵀ+Q, mean_pred = Aμ.
        rng = np.random.default_rng(100 + n)
        A = rng.standard_normal((n, n))
        Q = _spd(rng, n)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        cov_pred = A @ cov @ A.T + Q
        mean_pred = A @ mean

        pred = GaussianTransition(A, Q).predict(_belief_as_canonical(mean, cov))
        out_mean, out_cov = pred.to_moment()
        np.testing.assert_allclose(out_mean, mean_pred, atol=1e-7)
        np.testing.assert_allclose(out_cov, cov_pred, atol=1e-7)

    def test_predict_applies_control_term(self):
        # The control shifts the predicted mean by b; the covariance is unchanged.
        rng = np.random.default_rng(7)
        n = 2
        A = rng.standard_normal((n, n))
        Q = _spd(rng, n)
        mean = rng.standard_normal(n)
        cov = _spd(rng, n)
        b = rng.standard_normal(n)
        mean_pred = A @ mean + b
        cov_pred = A @ cov @ A.T + Q

        pred = GaussianTransition(A, Q).predict(
            _belief_as_canonical(mean, cov), control_term=b
        )
        out_mean, out_cov = pred.to_moment()
        np.testing.assert_allclose(out_mean, mean_pred, atol=1e-7)
        np.testing.assert_allclose(out_cov, cov_pred, atol=1e-7)

    def test_jit_and_grad_through_predict(self):
        fac = GaussianTransition([[1.0, 0.1], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        msg = CanonicalGaussian([[2.0, 0.0], [0.0, 2.0]], [1.0, 0.0])
        eager = fac.predict(msg).potential
        jitted = jax.jit(
            lambda h: fac.predict(CanonicalGaussian(msg.precision, h)).potential
        )(msg.potential)
        np.testing.assert_allclose(jitted, eager, atol=1e-10)
        grad = jax.grad(lambda b: fac.predict(msg, control_term=b).potential.sum())(
            jnp.zeros(2)
        )
        assert bool(jnp.all(jnp.isfinite(grad)))