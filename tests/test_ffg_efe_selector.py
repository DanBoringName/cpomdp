"""FFG-aware EFE action selector (issue #26, Phase B4).

The selector vmaps the FFG EFE step over the candidate action grid and argmins. Its
anchor: on a single-node model (no structural couplings) with the whole-state target, it
must pick the same action as the existing ``EFESelector`` on the equivalent flat model.
"""

import numpy as np

from cpomdp.backends.coupling import CouplingGraphBackend
from cpomdp.ffg.factors.linear_gaussian import GaussianObservation, GaussianTransition
from cpomdp.ffg.graph import CouplingGraph
from cpomdp.selection import EFESelector, FfgEfeSelector, Preference
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
