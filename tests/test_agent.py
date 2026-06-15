import numpy as np
import pytest

from cpomdp.agent import Agent
from cpomdp.types import Belief, LinearGaussianModel

# The double-integrator point mass from test_control: state = [position, velocity],
# a force moves the velocity, velocity moves the position. Observe position only,
# so the filter must infer velocity through the off-diagonal coupling while the
# controller drives both — a real LQG loop, not a toy.
DT = 0.1
DYNAMICS = np.array([[1.0, DT], [0.0, 1.0]])
CONTROL = np.array([[0.0], [DT]])
SENSOR_MODEL = np.array([[1.0, 0.0]])
GOAL = np.array([1.0, 0.0])  # reach position 1, at rest


def _reaching_model() -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=CONTROL,
    )


# A 2D point mass: state = [px, py, vx, vy], with two independent force inputs
# (fx pushes vx, fy pushes vy) and position-only sensing. This is the p=2
# multi-actuator stress case: the controller gain is now (2, 4), so a
# transposed or axis-swapped gain — invisible at p=1 — would steer the wrong
# axis. The asymmetric goal (reach px=+1, py=-1) means a swap can't pass by
# coincidence: the two channels must stay distinct end-to-end.
DYNAMICS_2D = np.array(
    [
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
CONTROL_2D = np.array([[0.0, 0.0], [0.0, 0.0], [DT, 0.0], [0.0, DT]])
SENSOR_MODEL_2D = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
GOAL_2D = np.array([1.0, -1.0, 0.0, 0.0])  # reach (+1, -1), at rest


def _reaching_model_2d() -> LinearGaussianModel:
    return LinearGaussianModel(
        dynamics=DYNAMICS_2D,
        sensor_model=SENSOR_MODEL_2D,
        dynamics_noise=np.eye(4) * 1e-6,
        sensor_noise=np.eye(2) * 1e-2,
        prior=Belief(mean=[0.0, 0.0, 0.0, 0.0], cov=np.eye(4)),
        control=CONTROL_2D,
    )


def _run_closed_loop(agent, true_state, steps, rng=None):
    """Drive the true plant with the agent's own actions; return the final state.

    Each step is the full perceive -> act -> advance cycle. The agent sees only
    `obs` (a noisy position reading), never `true_state` directly. If `rng` is
    given, sensor and process noise are injected — otherwise the plant is clean.

    The plant matrices are read off the agent's own model, so this drives a
    p=1 single-force point mass and a p=2 multi-actuator plant identically.
    """
    A, B, C = agent.model.dynamics, agent.model.control, agent.model.sensor_model
    true_state = np.asarray(true_state, dtype=float)
    for _ in range(steps):
        obs = C @ true_state
        if rng is not None:
            obs = obs + rng.normal(0.0, 0.1, size=obs.shape)
        agent.infer_states(obs)  # perceive
        action = agent.sample_action()  # act
        true_state = A @ true_state + B @ action  # advance the plant
        if rng is not None:
            true_state = true_state + rng.normal(0.0, 1e-3, size=true_state.shape)
    return true_state


def test_goal_is_equilibrium() -> None:
    np.testing.assert_allclose(DYNAMICS @ GOAL, GOAL)
    np.testing.assert_allclose(DYNAMICS_2D @ GOAL_2D, GOAL_2D)


def test_reaches_goal_noiseless() -> None:
    agent = Agent(_reaching_model(), goal=GOAL)
    final_state = _run_closed_loop(
        agent, [0.0, 0.0], steps=200
    )  # <-- the helper returns it
    np.testing.assert_allclose(final_state, GOAL, atol=1e-2)
    np.testing.assert_allclose(
        agent.belief.mean, final_state, atol=1e-2
    )  # Check model thinks it is where it is


def test_reaches_goal_under_noise() -> None:
    agent = Agent(_reaching_model(), goal=GOAL)
    final_state = _run_closed_loop(
        agent, [0.0, 0.0], steps=200, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(agent.belief.mean, final_state, atol=0.1)


def test_reaches_goal_multi_actuator() -> None:
    # The end-to-end p=2 check: two independent actuators, position-only
    # sensing. Drives the full agent -> controller -> gain stack with a
    # non-scalar action and an asymmetric goal, so a mis-oriented (transposed
    # or axis-swapped) gain would steer the wrong axis and never converge.
    agent = Agent(_reaching_model_2d(), goal=GOAL_2D)
    final_state = _run_closed_loop(agent, [0.0, 0.0, 0.0, 0.0], steps=300)

    assert agent.sample_action().shape == (2,)  # genuinely a 2-actuator action
    np.testing.assert_allclose(final_state, GOAL_2D, atol=1e-2)
    np.testing.assert_allclose(agent.belief.mean, final_state, atol=1e-2)


def test_perceive_only_agent_cannot_act():
    # A goal-less agent on a control-free model: a pure tracker.
    model = LinearGaussianModel(
        dynamics=[[1.0, DT], [0.0, 1.0]],
        sensor_model=SENSOR_MODEL,
        dynamics_noise=[[1e-6, 0.0], [0.0, 1e-6]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        # no control= → no action channel
    )
    agent = Agent(model)  # no goal

    # perceive works: folding in an observation returns a Belief and sharpens it
    belief = agent.infer_states([0.5])

    assert belief.cov[0, 0] < model.prior.cov[0, 0]  # uncertainty dropped

    # but it cannot act — no goal to steer toward
    with pytest.raises(ValueError, match="goal"):
        agent.sample_action()
