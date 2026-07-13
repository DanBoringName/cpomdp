"""cpomdp — continuous active inference for Python.

The continuous-state sibling of pymdp. The public API is the stateful
:class:`Agent` façade over a :class:`LinearGaussianModel`, driven in the same
perceive → act loop pymdp users know::

    from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

    agent = Agent(model, StateGoal(target))
    belief = agent.infer_states(observation)   # perceive
    action = agent.sample_action()             # act

Swap the inference engine via the ``backend=`` argument; :class:`KalmanBackend`
is the default and :class:`InferenceBackend` is the protocol custom backends
implement. The optional RxInfer oracle lives behind the ``rxinfer`` extra —
import it explicitly from ``cpomdp.backends.rxinfer`` so the core stays
Julia-free.

For a *branching* model, declare a :class:`CouplingGraph` whose nodes carry
observation factors (:class:`GaussianObservation`,
:class:`CallableGaussianObservation`) and whose :class:`Coupling` edges carry a
:class:`GaussianCoupling`, then run it through a :class:`CouplingGraphBackend`
with a per-node :class:`GaussianTransition`. A state-dependent ``R(x)`` on a
coupled node cannot be flattened to a fixed linear-Gaussian model; asking it to
raises :class:`IncompatibleLinearizationError` (ADR-019, ADR-020). The
:class:`Agent` dispatches an :class:`FfgEfeSelector` — the FFG peer of
:class:`EFESelector` — for such a backend; pass ``selector=`` to override it.
"""

import jax

from cpomdp.agent import Agent
from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.coupling import (
    CouplingGraphBackend,
    IncompatibleLinearizationError,
)
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import CallableProcessNoise, DynamicsNoise
from cpomdp.efe import expected_free_energy
from cpomdp.ffg.factors.linear_gaussian import (
    CallableGaussianObservation,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
)
from cpomdp.ffg.graph import Coupling, CouplingGraph
from cpomdp.observation import CallableSensor, FixedSensor, ObservationModel
from cpomdp.selection import (
    ActionSelector,
    EFESelector,
    FfgEfeSelector,
    LQRSelector,
    ObservationGoal,
    Preference,
    StateGoal,
)
from cpomdp.structure import ModelStructure
from cpomdp.types import Belief, LinearGaussianModel

# Float64 throughout — the oracle matches to 1e-9 and JAX defaults to float32.
# Process-global by necessity; see ADR-004.
jax.config.update("jax_enable_x64", True)

__version__ = "0.4.1"

__all__ = [
    "ActionSelector",
    "Agent",
    "Belief",
    "CallableGaussianObservation",
    "CallableProcessNoise",
    "CallableSensor",
    "Coupling",
    "CouplingGraph",
    "CouplingGraphBackend",
    "DynamicsNoise",
    "EFESelector",
    "FfgEfeSelector",
    "FixedSensor",
    "GaussianCoupling",
    "GaussianObservation",
    "GaussianTransition",
    "IncompatibleLinearizationError",
    "InferenceBackend",
    "KalmanBackend",
    "LQRSelector",
    "LinearGaussianModel",
    "ModelStructure",
    "ObservationGoal",
    "ObservationModel",
    "Preference",
    "StateGoal",
    "expected_free_energy",
]
