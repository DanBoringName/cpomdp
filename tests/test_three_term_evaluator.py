"""The two divergences a cell is scored by, and the shape that keeps them honest."""

from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.observation import CallableSensor
from cpomdp.scoring import (
    _SERIES_BELOW,
    Decomposition,
    _excess_over_log,
    gaussian_kl,
    misspecification_step,
    observation_predictive,
)
from cpomdp.types import Belief, LinearGaussianModel


def test_the_type_carries_the_two_divergences_and_nothing_else():
    # Standing prohibition 1: never obtain a term by subtracting H(p*). No entropy
    # field, no estimator slot, no total. Adding one is a deliberate edit here first.
    assert [f.name for f in fields(Decomposition)] == [
        "misspecification",
        "inference_gap",
    ]


# --- the divergence both terms are built from ----------------------------------------


def _naive_gaussian_kl(mean_a, cov_a, mean_b, cov_b):
    """The textbook form, subtractions included, as the oracle."""
    n = len(mean_a)
    precision_b = np.linalg.inv(cov_b)
    shift = mean_a - mean_b
    return 0.5 * (
        np.trace(precision_b @ cov_a)
        - n
        + np.linalg.slogdet(cov_b)[1]
        - np.linalg.slogdet(cov_a)[1]
        + shift @ precision_b @ shift
    )


MEAN_A = np.array([0.3, -1.2, 2.0])
COV_A = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
MEAN_B = np.array([0.0, -1.0, 2.5])
COV_B = np.array([[1.5, -0.1, 0.0], [-0.1, 1.3, 0.4], [0.0, 0.4, 0.9]])


def test_identical_gaussians_diverge_by_exactly_zero():
    assert gaussian_kl(MEAN_A, COV_A, MEAN_A, COV_A) == 0.0


def test_the_scalar_closed_form():
    mean_a, var_a, mean_b, var_b = 0.4, 0.8, -0.1, 1.7
    expected = (
        0.5 * np.log(var_b / var_a)
        + (var_a + (mean_a - mean_b) ** 2) / (2 * var_b)
        - 0.5
    )
    assert gaussian_kl([mean_a], [[var_a]], [mean_b], [[var_b]]) == pytest.approx(
        expected, abs=1e-14
    )


def test_matches_the_textbook_form_where_that_form_is_accurate():
    assert gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B) == pytest.approx(
        _naive_gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B), abs=1e-12
    )


def test_the_divergence_is_directed():
    forward = gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B)
    reverse = gaussian_kl(MEAN_B, COV_B, MEAN_A, COV_A)
    assert abs(forward - reverse) > 1e-3


def test_an_affine_change_of_coordinates_leaves_it_unchanged():
    # The reparameterisation invariance the ledger relies on to call nats a scale-free
    # unit. Both Gaussians move through the same map, so the divergence must not.
    transform = np.array([[3.0, 1.0, 0.0], [0.0, -2.0, 0.5], [1.0, 0.0, 4.0]])
    offset = np.array([10.0, -3.0, 0.25])
    before = gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B)
    after = gaussian_kl(
        transform @ MEAN_A + offset,
        transform @ COV_A @ transform.T,
        transform @ MEAN_B + offset,
        transform @ COV_B @ transform.T,
    )
    assert after == pytest.approx(before, rel=1e-12)


def test_near_equality_reads_small_and_never_negative():
    # Two Gaussians a relative 1e-13 apart in covariance diverge by ~n·(1e-13)²/4. The
    # textbook subtraction lands within rounding of zero, of either sign, which is
    # exactly the reading a 1e-12 bar cannot interpret.
    nearly = COV_A * (1.0 + 1e-13)
    measured = gaussian_kl(MEAN_A, nearly, MEAN_A, COV_A)
    assert 0.0 <= measured < 1e-24


def test_a_degenerate_covariance_is_refused_by_name():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="cov_b"):
        gaussian_kl([0.0, 0.0], np.eye(2), [0.0, 0.0], singular)


def test_a_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape"):
        gaussian_kl([0.0, 0.0], np.eye(2), [0.0], np.eye(1))


def test_the_series_and_the_direct_form_agree_at_the_switch():
    # Two evaluations of one function. Either side of the crossover they must return
    # the same number, or the divergence would step where the branch changes.
    for excess in (-_SERIES_BELOW, _SERIES_BELOW):
        for side in (excess * (1 - 1e-9), excess * (1 + 1e-9)):
            direct = side - np.log1p(side)
            assert float(_excess_over_log(np.array([side]))[0]) == pytest.approx(
                direct, rel=1e-9
            )


# --- the misspecification term ------------------------------------------------------


DT = 0.1


def _model(*, observation_noise: float = 1e-2) -> LinearGaussianModel:
    """A double integrator, built afresh on every call so p* and p share nothing."""
    return LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[observation_noise]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [DT]],
    )


BELIEF = Belief(mean=[0.4, -0.2], cov=[[0.5, 0.1], [0.1, 0.3]])
ACTION = np.array([0.7])


def test_the_predictive_is_the_belief_pushed_through_dynamics_then_sensor():
    model = _model()
    dynamics, control, sensor = (
        np.array([[1.0, DT], [0.0, 1.0]]),
        np.array([[0.0], [DT]]),
        np.array([[1.0, 0.0]]),
    )
    mean_pred = dynamics @ np.asarray(BELIEF.mean) + control @ ACTION
    cov_pred = dynamics @ np.asarray(BELIEF.cov) @ dynamics.T + 1e-4 * np.eye(2)
    mean, cov = observation_predictive(model, BELIEF, ACTION)
    np.testing.assert_allclose(mean, sensor @ mean_pred, rtol=1e-14)
    np.testing.assert_allclose(cov, sensor @ cov_pred @ sensor.T + 1e-2, rtol=1e-14)


def test_a_model_with_a_control_matrix_needs_an_action():
    with pytest.raises(ValueError, match="action"):
        observation_predictive(_model(), BELIEF, None)


def test_a_state_dependent_sensor_is_refused_by_name():
    model = LinearGaussianModel(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=[[0.0], [DT]],
        observation_model=CallableSensor(
            observation_matrix=[[1.0, 0.0]],
            noise_fn=lambda x, _p: jnp.array([[1e-2 * (1 + x[0] ** 2)]]),
            noise_params=(),
        ),
    )
    with pytest.raises(ValueError, match="observation_model"):
        observation_predictive(model, BELIEF, ACTION)


def test_two_separately_built_equal_models_have_no_misspecification():
    # Exactly zero, not merely small. Two builds from the same numbers run the same
    # arithmetic, and the divergence of identical Gaussians is 0.0 by construction.
    assert misspecification_step(_model(), BELIEF, _model(), BELIEF, ACTION) == 0.0


def test_a_perturbed_sensor_noise_moves_the_term_by_the_scalar_closed_form():
    truth, model = _model(), _model(observation_noise=1.5e-2)
    _, true_cov = observation_predictive(truth, BELIEF, ACTION)
    true_var = float(true_cov[0, 0])
    model_var = true_var - 1e-2 + 1.5e-2
    expected = 0.5 * (np.log(model_var / true_var) + true_var / model_var - 1.0)
    assert misspecification_step(truth, BELIEF, model, BELIEF, ACTION) == pytest.approx(
        expected, rel=1e-12
    )


def test_the_term_reads_the_beliefs_it_is_handed_and_not_the_prior():
    # The two sides may sit at different beliefs, as they will once each exact filter
    # has folded its own view of the history. Moving one belief moves the term.
    truth, model = _model(), _model()
    shifted = Belief(mean=[0.9, -0.2], cov=BELIEF.cov)
    assert misspecification_step(truth, BELIEF, model, shifted, ACTION) > 0.0
