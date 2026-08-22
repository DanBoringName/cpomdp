import contextlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cpomdp.backends.kalman import KalmanBackend
from cpomdp.harness import ScoredAgent, World
from cpomdp.observation import CallableSensor
from cpomdp.types import Belief, LinearGaussianModel

# The double integrator from test_agent: state = [position, velocity], a force moves
# the velocity, velocity moves the position, position is observed. Two *separately
# constructed* instances stand in for p* and p — the harness never derives one from
# the other, so the tests never do either.
DT = 0.1
DYNAMICS = np.array([[1.0, DT], [0.0, 1.0]])
CONTROL = np.array([[0.0], [DT]])
SENSOR = np.array([[1.0, 0.0]])


def _model(*, damping: float = 1.0) -> LinearGaussianModel:
    """A double integrator whose velocity decays at ``damping`` per step.

    ``damping=1.0`` is the undamped truth. Anything else is a misspecified model, and
    the misspecification lives in one entry so a cell can name what it perturbed.
    """
    dynamics = np.array([[1.0, DT], [0.0, damping]])
    return LinearGaussianModel(
        dynamics_matrix=dynamics,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control_matrix=CONTROL,
    )


def _reachable(root: object, *, limit: int = 20_000) -> set[int]:
    """Every object id reachable from ``root`` by attribute and container traversal.

    Follows closure cells and a bound method's ``__self__`` as well as attributes, so a
    captured reference counts as reaching. Arrays are terminal: they hold numbers, not
    references onward, and iterating a traced array is a mistake in its own right. The
    limit is a runaway guard; a cycle is already handled by ``seen``.
    """
    seen: set[int] = set()
    queue = [root]
    while queue and len(seen) < limit:
        obj = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, jax.Array | np.ndarray | str | bytes | int | float | bool):
            continue
        if isinstance(obj, dict):
            queue.extend(obj.keys())
            queue.extend(obj.values())
            continue
        if isinstance(obj, list | tuple | set | frozenset):
            queue.extend(obj)
            continue
        queue.extend(vars(obj).values() if hasattr(obj, "__dict__") else ())
        queue.extend(
            getattr(obj, name)
            for name in getattr(obj, "__slots__", ())
            if hasattr(obj, name)
        )
        if (bound := getattr(obj, "__self__", None)) is not None:
            queue.append(bound)
        for cell in getattr(obj, "__closure__", None) or ():
            with contextlib.suppress(ValueError):  # empty cell: a recursive definition
                queue.append(cell.cell_contents)
    return seen


# --- the detector, before the seam it is pointed at ---------------------------------
#
# The seam tests below assert a negative, so they also pass when the walk finds
# nothing at all. These plant each way a reference can be held and require it found.


@pytest.mark.parametrize(
    "leak",
    [
        pytest.param(lambda world: world, id="attribute"),
        pytest.param(lambda world: [world], id="inside a list"),
        pytest.param(lambda world: {"w": world}, id="inside a dict"),
        pytest.param(lambda world: world.step, id="bound method"),
        pytest.param(lambda world: lambda: world, id="closure cell"),
    ],
)
def test_the_walk_finds_a_planted_reference(leak):
    world = World(_model())
    agent = ScoredAgent(_model())
    agent._leak = leak(world)  # ty: ignore[unresolved-attribute]
    assert id(world) in _reachable(agent)


def test_the_walk_finds_a_planted_parameter():
    truth = _model()
    agent = ScoredAgent(_model())
    agent._leak = truth.dynamics_matrix  # ty: ignore[unresolved-attribute]
    assert id(truth.dynamics_matrix) in _reachable(agent)


def test_the_walk_terminates_on_a_cycle():
    left: dict = {}
    right = {"back": left}
    left["on"] = right
    assert id(right) in _reachable(left)


# --- the seam: no path from the agent to the world's parameters -------------------


def test_world_exposes_no_generative_model():
    world = World(_model())
    exposed = [getattr(world, name) for name in dir(world) if not name.startswith("_")]
    assert not [x for x in exposed if isinstance(x, LinearGaussianModel)]


def test_world_exposes_no_generative_parameter():
    truth = _model()
    world = World(truth)
    parameters = {
        id(truth.dynamics_matrix),
        id(truth.observation_matrix),
        id(truth.dynamics_noise),
        id(truth.observation_noise),
        id(truth.control_matrix),
        id(truth.prior),
    }
    exposed = {
        id(getattr(world, name)) for name in dir(world) if not name.startswith("_")
    }
    assert not parameters & exposed


def test_scored_agent_accepts_no_world():
    parameters = inspect.signature(ScoredAgent.__init__).parameters
    annotations = {str(p.annotation) for p in parameters.values()}
    assert "world" not in parameters
    assert not [a for a in annotations if "World" in a]


def test_the_world_is_not_reachable_from_the_agent():
    world = World(_model())
    agent = ScoredAgent(_model(damping=0.8))
    key = jax.random.PRNGKey(0)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        agent.driven_step(world.step([1.0], subkey), [1.0])

    assert id(world) not in _reachable(agent)


def test_the_worlds_parameters_are_not_reachable_from_the_agent():
    truth = _model()
    world = World(truth)
    agent = ScoredAgent(_model(damping=0.8))
    agent.driven_step(world.step([1.0], jax.random.PRNGKey(0)), [1.0])

    reached = _reachable(agent)
    assert id(truth) not in reached
    assert id(truth.dynamics_matrix) not in reached


def test_the_agent_keeps_its_own_model():
    agent = ScoredAgent(_model(damping=0.8))
    assert float(agent.model.dynamics_matrix[1, 1]) == pytest.approx(0.8)


# --- exogenous action: the agent never chooses -------------------------------------


def test_scored_agent_cannot_select_actions():
    agent = ScoredAgent(_model())
    assert not hasattr(agent, "sample_action")
    assert not hasattr(agent, "infer_policies")


def test_a_driven_step_requires_an_action():
    agent = ScoredAgent(_model())
    with pytest.raises(TypeError):
        agent.driven_step([0.0])  # ty: ignore[missing-argument]


def test_the_driven_action_reaches_the_predict_step():
    left, right = ScoredAgent(_model()), ScoredAgent(_model())
    observation = [0.5]
    pushed = left.driven_step(observation, [+1.0])
    pulled = right.driven_step(observation, [-1.0])
    assert not np.allclose(np.asarray(pushed.mean), np.asarray(pulled.mean))


def test_the_belief_advances_and_is_replaced_not_mutated():
    agent = ScoredAgent(_model())
    before = agent.belief
    after = agent.driven_step([0.5], [0.0])
    assert after is agent.belief
    assert before is not after
    assert float(before.cov[0, 0]) > float(after.cov[0, 0])


def test_the_agent_starts_at_its_own_prior():
    agent = ScoredAgent(_model())
    assert np.allclose(np.asarray(agent.belief.mean), [0.0, 0.0])


def test_a_backend_is_accepted_and_carries_its_own_model():
    model = _model(damping=0.8)
    agent = ScoredAgent(backend=KalmanBackend(model))
    assert agent.model is model


def test_a_model_and_a_backend_that_disagree_do_not_construct():
    with pytest.raises(ValueError, match="one model"):
        ScoredAgent(_model(), backend=KalmanBackend(_model(damping=0.8)))


def test_no_model_and_no_backend_does_not_construct():
    with pytest.raises(ValueError, match="model or a backend"):
        ScoredAgent()


# --- the world simulates p*, and only p* --------------------------------------------


def test_world_step_returns_one_observation():
    world = World(_model())
    observation = world.step([0.0], jax.random.PRNGKey(0))
    assert observation.shape == (1,)


def test_world_step_advances_the_true_state():
    world = World(_model(), initial_state=[0.0, 1.0])
    key = jax.random.PRNGKey(0)
    first = world.step([0.0], key)
    second = world.step([0.0], key)  # same key: the difference is the state, not noise
    assert float(second[0]) > float(first[0])


def test_the_same_key_gives_the_same_observation():
    left = World(_model(), initial_state=[0.0, 1.0])
    right = World(_model(), initial_state=[0.0, 1.0])
    key = jax.random.PRNGKey(7)
    assert float(left.step([0.5], key)[0]) == float(right.step([0.5], key)[0])


def test_the_initial_state_defaults_to_the_prior_mean():
    world = World(_model())
    assert np.allclose(np.asarray(world.state), [0.0, 0.0])


def test_the_true_state_is_readable_because_the_world_is_not_the_agent():
    world = World(_model(), initial_state=[1.0, 0.0])
    assert np.allclose(np.asarray(world.state), [1.0, 0.0])


def test_an_initial_state_of_the_wrong_shape_does_not_construct():
    with pytest.raises(ValueError, match="1-D vector of length 2"):
        World(_model(), initial_state=[1.0, 0.0, 0.0])


def test_the_world_refuses_a_state_dependent_sensor():
    def sensor(mean, params):
        return jnp.array([[1e-2]]) + jnp.sum(jnp.asarray(mean) ** 2)

    model = LinearGaussianModel(
        dynamics_matrix=DYNAMICS,
        observation_matrix=SENSOR,
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=np.eye(2)),
        control_matrix=CONTROL,
        observation_model=CallableSensor(SENSOR, sensor, ()),
    )
    with pytest.raises(NotImplementedError, match="evaluation point"):
        World(model)
