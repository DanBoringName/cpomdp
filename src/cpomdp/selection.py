"""Action selection: the seam between a belief+preference and an action."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance
from cpomdp.control import LQRController
from cpomdp.types import Belief

__all__ = ["ActionSelector", "LQRSelector", "Preference"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Preference:
    """What the agent wants: a goal and how sharply it is preferred.

    Single-mode for v0.3 — one Gaussian preference. The disjunctive *mixture*
    case (visit one of several goals) is RFC-002, deferred; this type is the seam
    that a mixture ``Preference`` plugs into.

    ``precision`` is unused by ``LQRSelector`` (it is baked into the controller's
    Riccati solve at construction); it is carried here for the EFE pragmatic term
    added in Phase 1A.
    """

    goal: Float64[Array, "n"]
    precision: Float64[Array, "n n"]

    def __init__(self, goal: ArrayLike, precision: ArrayLike | None = None) -> None:
        goal = jnp.asarray(goal, dtype=float)
        object.__setattr__(self, "goal", goal)
        n = goal.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(n) if precision is None else jnp.asarray(precision, dtype=float),
        )
        self._validate()

    def _validate(self) -> None:
        if self.goal.ndim != 1:
            raise ValueError(f"goal must be a 1-D vector, got shape {self.goal.shape}")
        validate_covariance(self.precision, "precision")
        n = self.goal.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match the {n}-D goal, "
                f"got shape {self.precision.shape}"
            )

    def tree_flatten(self):
        """Leaves: (goal, precision); no static aux. Lets jit/vmap take a Preference."""
        return (self.goal, self.precision), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild without re-validating — the leaves may be tracers."""
        goal, precision = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "goal", goal)
        object.__setattr__(obj, "precision", precision)
        return obj


@runtime_checkable
class ActionSelector(Protocol):
    """Chooses an action from a belief and a preference.

    The abstraction wall for action selection: ``LQRSelector`` is the fixed-sensor
    case (EFE collapses to LQR, ADR-003); ``EFESelector`` arrives in v0.3 for
    state-dependent sensing. ``Agent`` depends only on this, never on a concrete
    selector.
    """

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The action to take given the current ``belief`` and ``preference``."""
        ...


@dataclass(frozen=True)
class LQRSelector:
    """Adapts an LQRController to the ActionSelector interface.

    A thin wrapper: it owns no control logic, it just forwards to the
    front-loaded controller, unpacking the belief's mean and the preference's
    goal. EFE collapses to LQR under a fixed sensor (ADR-003), so for that regime
    this *is* the action selector.
    """

    controller: LQRController

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """Forward to the controller: ``action(belief.mean, preference.goal)``.

        No control logic of its own — the Riccati solve was front-loaded into the
        controller at construction. This just unpacks the belief and preference
        into the controller's ``(mean, goal)`` signature.
        """
        return self.controller.action(belief.mean, preference.goal)
