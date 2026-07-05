"""FFG-aware EFE action selector (issue #26, Phase B4).

The selector vmaps the FFG EFE step over the candidate action grid and argmins. Its
anchor: on a single-node model (no structural couplings) with the whole-state target, it
must pick the same action as the existing ``EFESelector`` on the equivalent flat model.
"""

import numpy as np

from cpomdp.agent import Agent
from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.ffg.factors.linear_gaussian import (
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.selection import (
    EFESelector,
    FfgEfeSelector,
    ObservationGoal,
    Preference,
)
from cpomdp.types import Belief, LinearGaussianModel


def test_ffg_selector_reduces_to_efe_selector_single_node():
    # One node, no couplings, whole-state target: the FFG selector must pick the same
    # grid action as EFESelector on the equivalent flat LinearGaussianModel.
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    sensor_model = np.array([[1.0, 0.0]])  # C
    sensor_noise = np.array([[0.5]])  # R
    control = np.array([[1.0], [0.0]])  # B — drives observed state[0], so G is non-flat

    graph = CouplingGraph(
        root=0,
        dims=(2,),
        couplings=(),
        observations={0: GaussianObservation(sensor_model, sensor_noise)},
    )
    backend = CouplingGraphBackend(
        graph, (GaussianTransition(dynamics, dynamics_noise),), control=control
    )
    model = LinearGaussianModel(
        dynamics=dynamics,
        sensor_model=sensor_model,
        dynamics_noise=dynamics_noise,
        sensor_noise=sensor_noise,
        prior=Belief(mean=np.zeros(2), cov=np.eye(2)),
        control=control,
    )

    belief = Belief(mean=[0.3, -0.2], cov=[[0.7, 0.1], [0.1, 0.4]])
    preference = Preference(goal=[1.0], precision=[[2.0]])
    bounds, k = (-2.0, 2.0), 21

    ffg = FfgEfeSelector(backend, target=range(2), n_candidates=k, action_bounds=bounds)
    efe = EFESelector(model, n_candidates=k, action_bounds=bounds)

    np.testing.assert_allclose(
        np.asarray(ffg.select(belief, preference)),
        np.asarray(efe.select(belief, preference)),
        atol=1e-7,
    )


def _two_node_backend():
    # CheA(0) -> CheY(1); observe CheY; the action drives CheA (the root).
    graph = CouplingGraph(
        root=0,
        dims=(1, 1),
        couplings=(Coupling(0, 1, GaussianCoupling([[0.8]], [[0.05]]), 1.0),),
        observations={1: GaussianObservation([[1.0]], [[0.1]])},
    )
    transitions = (
        GaussianTransition([[0.7]], [[0.1]]),
        GaussianTransition([[0.5]], [[0.08]]),
    )
    return CouplingGraphBackend(graph, transitions, control=[[1.0], [0.0]])


def test_agent_on_ffg_backend_acts_via_ffg_selector():
    # B5 (issue #26): a user drives a branching FFG as an Agent. info_target aims the
    # epistemic at a chosen node; the Agent routes to the FFG EFE selector, and the
    # perceive -> act cycle completes.
    backend = _two_node_backend()
    agent = Agent(
        objective=ObservationGoal([0.5], (-2.0, 2.0), info_target=0),  # info about CheA
        backend=backend,
    )
    agent.infer_states([0.3])  # observe CheY
    action = agent.sample_action()

    assert np.asarray(action).shape == (1,)
    assert isinstance(agent._selector, FfgEfeSelector)


def test_agent_ffg_whole_state_when_info_target_none():
    # info_target=None on the FFG backend still routes to the FFG selector, aimed at the
    # whole state (the default epistemic).
    backend = _two_node_backend()
    agent = Agent(
        objective=ObservationGoal([0.5], (-2.0, 2.0)),  # no info_target
        backend=backend,
    )
    assert isinstance(agent._selector, FfgEfeSelector)
    assert tuple(agent._selector._target) == (0, 1)  # whole joint state


def test_info_target_on_flat_backend_raises():
    # info_target aims at an FFG node; it is meaningless without a branching backend.
    model = LinearGaussianModel(
        dynamics=[[1.0]],
        sensor_model=[[1.0]],
        dynamics_noise=[[0.1]],
        sensor_noise=[[0.2]],
        prior=Belief(mean=[0.0], cov=[[1.0]]),
        control=[[1.0]],
    )
    with np.testing.assert_raises(ValueError):
        Agent(model, ObservationGoal([0.5], (-2.0, 2.0), info_target=0))


def test_ffg_agent_matches_kalman_efe_agent_end_to_end():
    # B6 gate (issue #26): the whole FFG EFE Agent path — single node, info_target=None,
    # so the epistemic is whole-state — must reproduce a KalmanBackend + EFESelector
    # Agent step for step. The non-negotiable "v0.3 behaviour is not broken" check.
    dynamics = np.array([[1.0, 0.1], [0.0, 1.0]])  # A
    dynamics_noise = np.array([[0.1, 0.0], [0.0, 0.1]])  # Q
    sensor_model = np.array([[1.0, 0.0]])  # C
    sensor_noise = np.array([[0.5]])  # R
    control = np.array([[1.0], [0.0]])  # B — drives the observed state[0]
    prior = Belief(mean=np.zeros(2), cov=np.eye(2))
    model = LinearGaussianModel(
        dynamics=dynamics,
        sensor_model=sensor_model,
        dynamics_noise=dynamics_noise,
        sensor_noise=sensor_noise,
        prior=prior,
        control=control,
    )
    graph = CouplingGraph(
        root=0,
        dims=(2,),
        couplings=(),
        observations={0: GaussianObservation(sensor_model, sensor_noise)},
    )
    backend = CouplingGraphBackend(
        graph, (GaussianTransition(dynamics, dynamics_noise),), control=control
    )
    bounds, k = (-2.0, 2.0), 21

    ffg_agent = Agent(
        objective=ObservationGoal([0.5], bounds, n_candidates=k),
        backend=backend,
    )
    ref_agent = Agent(
        model,
        ObservationGoal([0.5], bounds, n_candidates=k),
        backend=KalmanBackend(model),
        selector=EFESelector(model, n_candidates=k, action_bounds=bounds),
    )

    for y in ([0.3], [-0.1], [0.2]):
        ffg_agent.infer_states(y)
        ref_agent.infer_states(y)
        np.testing.assert_allclose(
            np.asarray(ffg_agent.sample_action()),
            np.asarray(ref_agent.sample_action()),
            atol=1e-7,
        )
