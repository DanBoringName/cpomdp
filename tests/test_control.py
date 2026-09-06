import numpy as np
import pytest
import scipy.linalg

from cpomdp.control import LQRController, finite_horizon_lqr
from cpomdp.types import Belief, LinearGaussianModel

# Inside this module the terse letters a/b/qc/rc are local scalars for the
# Riccati/gain hand-math below, NOT the role-named public API
# (dynamics/control/goal_precision/effort_penalty). The library deliberately spells
# those out to avoid the Q/R collision (ADR-003); here, where we're transcribing
# the textbook DARE formula to check it line-for-line, the letters keep the
# matrix algebra readable and carry no API meaning.

# A double-integrator point mass: state = [position, velocity], a force moves the
# velocity, velocity moves the position. dt small enough to be well-conditioned.
# This is both a controllable system (so a steady-state gain exists) and exactly
# the plant the 2-D reaching demo will use, so the oracle here guards the demo too.
DT = 0.1
DYNAMICS = [[1.0, DT], [0.0, 1.0]]
CONTROL = [[0.0], [DT]]
GOAL_PRECISION = [[1.0, 0.0], [0.0, 1.0]]
EFFORT_PENALTY = [[0.1]]


def _point_mass_model():
    # The noise/sensor fields are required to build a model but don't enter the
    # LQR solve at all — control selection reads only dynamics + control.
    return LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=[[1.0, 0.0]],  # observe position
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def _scipy_gain(dynamics_matrix, control_matrix, goal_precision, effort_penalty):
    """L∞ via scipy's Schur-based DARE solver — the independent oracle.

    scipy returns the cost-to-go P, not the gain, so we derive the gain with the
    same closing formula LQRController uses: L∞ = (Rc + BᵀPB)⁻¹(BᵀPA). Because
    scipy reaches P by Schur decomposition rather than value iteration, a
    transpose or orientation bug in our loop can't survive in both.
    """
    a = np.asarray(dynamics_matrix, dtype=float)
    b = np.asarray(control_matrix, dtype=float)
    qc = np.asarray(goal_precision, dtype=float)
    rc = np.asarray(effort_penalty, dtype=float)
    p = scipy.linalg.solve_discrete_are(a, b, qc, rc)
    return np.linalg.solve(rc + b.T @ p @ b, b.T @ p @ a)


class TestLQRGain:
    def test_gain_matches_scipy_dare(self):
        # The core oracle: our hand-rolled fixed-point iteration must agree with
        # scipy's independent Schur solve. Disagreement = the bug is ours.
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        expected = _scipy_gain(DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY)
        np.testing.assert_allclose(controller.gain, expected, atol=1e-8)

    def test_gain_has_shape_p_by_n(self):
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        # one action, two states
        assert controller.gain.shape == (1, 2)

    def test_closed_loop_is_stable(self):
        # The real test that the gain is right in sign AND magnitude: the
        # closed-loop dynamics (A - B·L∞) must be stable, i.e. every eigenvalue
        # strictly inside the unit circle. A sign-flipped gain would push the
        # eigenvalues out and this would fail loudly.
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        a = np.asarray(DYNAMICS)
        b = np.asarray(CONTROL)
        closed_loop = a - b @ controller.gain
        eigvals = np.linalg.eigvals(closed_loop)
        assert np.all(np.abs(eigvals) < 1.0)


class TestLQRAction:
    def test_action_pushes_toward_goal(self):
        # From the origin with a target at position +1, the force must be
        # positive (accelerate toward the goal). A dropped minus sign flips this.
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0]))
        assert action[0] > 0

    def test_zero_error_gives_zero_action(self):
        # Sitting exactly on an equilibrium goal, the controller asks for nothing.
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        action = controller.action(mean=np.array([1.0, 0.0]), goal=np.array([1.0, 0.0]))
        np.testing.assert_allclose(action, [0.0], atol=1e-12)

    def test_rejects_wrong_shape_goal(self):
        controller = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        )
        with pytest.raises(ValueError, match="goal"):
            controller.action(mean=np.array([0.0, 0.0]), goal=np.array([1.0, 0.0, 0.0]))


class TestLQRValidation:
    def test_rejects_model_without_control(self):
        model = LinearGaussianModel(
            dynamics_matrix=DYNAMICS,
            observation_matrix=[[1.0, 0.0]],
            dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
            observation_noise=[[1e-2]],
            prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        )  # no control matrix
        with pytest.raises(ValueError, match="control matrix"):
            LQRController(
                model, goal_precision=GOAL_PRECISION, effort_penalty=EFFORT_PENALTY
            )

    def test_rejects_wrong_goal_precision_shape(self):
        with pytest.raises(ValueError, match="goal_precision"):
            LQRController(
                _point_mass_model(),
                goal_precision=[[1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_wrong_effort_penalty_shape(self):
        with pytest.raises(ValueError, match="effort_penalty"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[1.0, 0.0], [0.0, 1.0]],
            )

    def test_rejects_asymmetric_goal_precision(self):
        # An off-diagonal typo: symmetric on shape, asymmetric in value. Without
        # the check this silently yields a non-symmetric cost-to-go and a wrong
        # gain — the hardest failure to trace in a control loop.
        with pytest.raises(ValueError, match="symmetric"):
            LQRController(
                _point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_rejects_indefinite_effort_penalty(self):
        # effort_penalty is inverted against in the gain solve; a zero (singular)
        # cost must fail loudly, not blow up mid-recursion.
        with pytest.raises(ValueError, match="positive-definite"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=[[0.0]],
            )

    def test_rejects_negative_semidefinite_goal_precision(self):
        with pytest.raises(ValueError, match="positive-semi-definite"):
            LQRController(
                _point_mass_model(),
                goal_precision=[[-1.0, 0.0], [0.0, 1.0]],
                effort_penalty=EFFORT_PENALTY,
            )

    def test_raises_when_not_converged(self):
        # max_iter too small to reach the fixed point -> a loud failure, not a
        # silently-wrong frozen gain. Mirrors the Kalman steady-state guard.
        with pytest.raises(RuntimeError, match="converge"):
            LQRController(
                _point_mass_model(),
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
                max_iter=1,
            )


# --- the finite-horizon schedule ------------------------------------------------------


def _brute_force_plan(dynamics, control, goal_precision, effort_penalty, horizon, x0):
    """The H-step open-loop optimum as a stacked least-squares problem.

    Stacks x_1..x_H = Φ·x0 + Γ·U with a cost on every arrived-at state and every
    action, and solves the normal equations for U. No Riccati recursion anywhere,
    so agreement with the schedule is evidence about the schedule.
    """
    a = np.asarray(dynamics, dtype=float)
    b = np.asarray(control, dtype=float)
    n, p = b.shape
    powers = [np.linalg.matrix_power(a, k) for k in range(horizon + 1)]
    phi = np.vstack([powers[k] for k in range(1, horizon + 1)])
    gamma = np.zeros((horizon * n, horizon * p))
    for i in range(horizon):
        for j in range(i + 1):
            gamma[i * n : (i + 1) * n, j * p : (j + 1) * p] = powers[i - j] @ b
    stage = np.kron(np.eye(horizon), np.asarray(goal_precision, dtype=float))
    effort = np.kron(np.eye(horizon), np.asarray(effort_penalty, dtype=float))
    hessian = gamma.T @ stage @ gamma + effort
    linear = gamma.T @ stage @ phi @ x0
    actions = -np.linalg.solve(hessian, linear)
    cost = x0 @ phi.T @ stage @ phi @ x0 - linear @ np.linalg.solve(hessian, linear)
    return actions[:p], cost


def _schedule(horizon):
    return finite_horizon_lqr(
        _point_mass_model(),
        goal_precision=GOAL_PRECISION,
        effort_penalty=EFFORT_PENALTY,
        horizon=horizon,
    )


class TestFiniteHorizonSchedule:
    def test_one_gain_per_step_and_one_cost_per_remaining_horizon(self):
        schedule = _schedule(6)
        assert schedule.horizon == 6
        assert schedule.gains.shape == (6, 1, 2)
        assert schedule.cost_to_go.shape == (7, 2, 2)
        # Nothing is charged after the last action: the terminal cost is zero.
        np.testing.assert_array_equal(schedule.cost_to_go[0], np.zeros((2, 2)))
        np.testing.assert_array_equal(schedule.first_gain, schedule.gains[0])

    def test_the_one_step_gain_is_the_static_regulator(self):
        a, b = np.asarray(DYNAMICS), np.asarray(CONTROL)
        qc, rc = np.asarray(GOAL_PRECISION), np.asarray(EFFORT_PENALTY)
        expected = np.linalg.solve(rc + b.T @ qc @ b, b.T @ qc @ a)
        np.testing.assert_allclose(_schedule(1).first_gain, expected, rtol=1e-14)

    @pytest.mark.parametrize("horizon", [1, 2, 5, 12])
    def test_the_first_gain_is_the_brute_force_optimum_first_action(self, horizon):
        x0 = np.array([0.8, -0.3])
        expected_action, expected_cost = _brute_force_plan(
            DYNAMICS, CONTROL, GOAL_PRECISION, EFFORT_PENALTY, horizon, x0
        )
        schedule = _schedule(horizon)
        np.testing.assert_allclose(
            -schedule.first_gain @ x0, expected_action, rtol=1e-11, atol=1e-13
        )
        np.testing.assert_allclose(
            x0 @ schedule.cost_to_go[horizon] @ x0, expected_cost, rtol=1e-11
        )

    def test_later_gains_are_the_shorter_schedules_first_gains(self):
        # The gain applied with j steps remaining depends on j alone, so the tail of
        # a long schedule is a shorter schedule. gains[k] has H − k steps remaining.
        long = _schedule(8)
        for horizon in (1, 3, 8):
            np.testing.assert_array_equal(
                long.gains[8 - horizon], _schedule(horizon).first_gain
            )

    def test_the_first_gain_converges_to_the_steady_state_gain_and_is_not_it(self):
        steady = LQRController(
            _point_mass_model(),
            goal_precision=GOAL_PRECISION,
            effort_penalty=EFFORT_PENALTY,
        ).gain
        mismatch = [
            float(np.abs(_schedule(h).first_gain - steady).max()) for h in (1, 5, 20)
        ]
        # Shrinks with H and is not zero at any of them: an unmatched comparison
        # reads this as an error that fades with the horizon.
        assert mismatch[0] > mismatch[1] > mismatch[2] > 1e-6
        np.testing.assert_allclose(_schedule(400).first_gain, steady, atol=1e-10)

    def test_a_horizon_below_one_is_refused(self):
        with pytest.raises(ValueError, match="horizon"):
            _schedule(0)

    def test_the_costs_are_validated_as_the_controller_validates_them(self):
        with pytest.raises(ValueError, match="symmetric"):
            finite_horizon_lqr(
                _point_mass_model(),
                goal_precision=[[1.0, 0.5], [-0.5, 1.0]],
                effort_penalty=EFFORT_PENALTY,
                horizon=3,
            )
        with pytest.raises(ValueError, match="control matrix"):
            finite_horizon_lqr(
                LinearGaussianModel(
                    dynamics_matrix=DYNAMICS,
                    observation_matrix=[[1.0, 0.0]],
                    dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
                    observation_noise=[[1e-2]],
                    prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
                ),
                goal_precision=GOAL_PRECISION,
                effort_penalty=EFFORT_PENALTY,
                horizon=3,
            )
