"""The two divergences a cell is scored by, and the shape that keeps them honest."""

from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.degraded import WrongFixedRBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.observation import CallableSensor
from cpomdp.reference.gap import averaged_inference_gap
from cpomdp.reference.likelihood import FixedNoiseLikelihood
from cpomdp.reference.quadrature import GridDensity, QuadratureGrid
from cpomdp.scoring import (
    _SERIES_BELOW,
    Decomposition,
    _excess_over_log,
    gaussian_kl,
    inference_gap_step,
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


@pytest.mark.parametrize(
    "cov",
    [COV_A, np.array([[0.5, 0.1], [0.1, 0.3]]), np.array([[2.0, -0.7], [-0.7, 0.9]])],
)
def test_identical_gaussians_diverge_by_exactly_zero(cov):
    # Including correlated 2x2 cases, where a pivoting solve of the factor against
    # itself leaves a residue that squares to 1e-32.
    mean = np.zeros(cov.shape[0])
    assert gaussian_kl(mean, cov, mean, cov) == 0.0


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


# --- the inference gap --------------------------------------------------------------


def _scalar_model(true_noise: float) -> LinearGaussianModel:
    """A static scalar state under a fixed noise, so one step is the whole story."""
    return LinearGaussianModel(
        dynamics_matrix=[[1.0]],
        observation_matrix=[[1.0]],
        dynamics_noise=[[0.0]],
        observation_noise=[[true_noise]],
        prior=Belief(mean=[0.3], cov=[[0.8]]),
    )


def _averaged_gaussian_gap(true_noise, plugin_noise, prior_var=0.8):
    """The closed form `test_reference_gap` holds the grid engine to, restated here."""
    gain = prior_var / (prior_var + true_noise)
    plugin_gain = prior_var / (prior_var + plugin_noise)
    exact_var = (1.0 - gain) * prior_var
    approx_var = (1.0 - plugin_gain) * prior_var
    innovation_var = prior_var + true_noise
    return (
        0.5 * np.log(exact_var / approx_var)
        + (approx_var + (plugin_gain - gain) ** 2 * innovation_var) / (2 * exact_var)
        - 0.5
    )


def _step(backend, belief, action=None):
    return lambda y: backend.infer_states(y, belief, action)


def test_the_exact_rule_has_no_gap_at_all():
    model = _model()
    exact = KalmanBackend(model)
    true_mean, true_cov = observation_predictive(model, BELIEF, ACTION)
    measured = inference_gap_step(
        _step(exact, BELIEF, ACTION), _step(exact, BELIEF, ACTION), true_mean, true_cov
    )
    assert measured == 0.0


@pytest.mark.parametrize("magnitude", [-0.5, 0.6, 3.0])
def test_a_wrong_fixed_noise_matches_the_scalar_closed_form(magnitude):
    true_noise = 0.5
    model = _scalar_model(true_noise)
    true_mean, true_cov = observation_predictive(model, model.prior, None)
    measured = inference_gap_step(
        _step(WrongFixedRBackend(model, magnitude=magnitude), model.prior),
        _step(KalmanBackend(model), model.prior),
        true_mean,
        true_cov,
    )
    assert measured == pytest.approx(
        _averaged_gaussian_gap(true_noise, true_noise * (1 + magnitude)), rel=1e-12
    )


def test_the_closed_form_agrees_with_the_grid_engine():
    # The other engine, on the one case both can reach (ADR-053). The grid integrates
    # KL(q ‖ p(x|y)) against p*(y) numerically and knows nothing about gains, so its
    # agreement is evidence about the closed form rather than about the probing.
    true_noise, magnitude = 0.5, 0.6
    model = _scalar_model(true_noise)
    agent = WrongFixedRBackend(model, magnitude=magnitude)
    states = QuadratureGrid(lower=[-14.0], upper=[14.0], counts=[2801])
    observations = QuadratureGrid(lower=[-16.0], upper=[16.0], counts=[401])

    def gaussian_on(grid, mean, var):
        x = np.asarray(grid.nodes)[:, 0]
        return GridDensity(
            grid, -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)
        )

    def rule(prior, observation):
        belief = agent.infer_states(np.asarray(observation), model.prior, None)
        return gaussian_on(prior.grid, float(belief.mean[0]), float(belief.cov[0, 0]))

    on_grid = averaged_inference_gap(
        gaussian_on(states, 0.3, 0.8),
        FixedNoiseLikelihood([[1.0]], observation_noise=[[true_noise]]),
        rule,
        observations,
    )
    true_mean, true_cov = observation_predictive(model, model.prior, None)
    closed = inference_gap_step(
        _step(agent, model.prior),
        _step(KalmanBackend(model), model.prior),
        true_mean,
        true_cov,
    )
    assert closed == pytest.approx(on_grid.value, rel=1e-6)


def test_the_two_dimensional_case_matches_the_textbook_expectation():
    # Gains by formula rather than by probing, and the average by the textbook trace,
    # so both halves of the closed form are checked against something that shares
    # nothing with them.
    model, magnitude = _model(), 0.5
    dynamics, sensor = np.array([[1.0, DT], [0.0, 1.0]]), np.array([[1.0, 0.0]])
    noise = np.array([[1e-2]])
    cov_pred = dynamics @ np.asarray(BELIEF.cov) @ dynamics.T + 1e-4 * np.eye(2)
    innovation_cov = sensor @ cov_pred @ sensor.T + noise  # S
    exact_gain = cov_pred @ sensor.T @ np.linalg.inv(innovation_cov)  # K
    wrong_gain = (
        cov_pred
        @ sensor.T
        @ np.linalg.inv(sensor @ cov_pred @ sensor.T + noise * (1 + magnitude))
    )
    exact_cov = (np.eye(2) - exact_gain @ sensor) @ cov_pred
    wrong_cov = (np.eye(2) - wrong_gain @ sensor) @ cov_pred
    drift = wrong_gain - exact_gain
    precision = np.linalg.inv(exact_cov)
    expected = 0.5 * (
        np.trace(precision @ wrong_cov)
        - 2
        + np.linalg.slogdet(exact_cov)[1]
        - np.linalg.slogdet(wrong_cov)[1]
        + np.trace(precision @ drift @ innovation_cov @ drift.T)
    )

    true_mean, true_cov = observation_predictive(model, BELIEF, ACTION)
    measured = inference_gap_step(
        _step(WrongFixedRBackend(model, magnitude=magnitude), BELIEF, ACTION),
        _step(KalmanBackend(model), BELIEF, ACTION),
        true_mean,
        true_cov,
    )
    assert measured == pytest.approx(expected, rel=1e-12)


def test_an_update_that_is_not_affine_is_refused():
    def bent(y):
        return Belief(mean=[float(y[0]) ** 2, 0.0], cov=np.eye(2))

    exact = _step(KalmanBackend(_model()), BELIEF, ACTION)
    true_mean, true_cov = observation_predictive(_model(), BELIEF, ACTION)
    with pytest.raises(ValueError, match="agent_update is not affine"):
        inference_gap_step(bent, exact, true_mean, true_cov)


def test_an_update_whose_covariance_reads_the_observation_is_refused():
    def widening(y):
        return Belief(mean=[float(y[0]), 0.0], cov=np.eye(2) * (1 + float(y[0]) ** 2))

    exact = _step(KalmanBackend(_model()), BELIEF, ACTION)
    true_mean, true_cov = observation_predictive(_model(), BELIEF, ACTION)
    with pytest.raises(ValueError, match="covariance depends on the reading"):
        inference_gap_step(widening, exact, true_mean, true_cov)
