"""Action selection: the seam between a belief+preference and an action."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp._validation import validate_covariance
from cpomdp.control import LQRController
from cpomdp.efe import expected_free_energy
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "ActionSelector",
    "EFESelector",
    "LQRSelector",
    "ObservationGoal",
    "Preference",
    "StateGoal",
]


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


class EFESelector:
    """Greedy (H=1) EFE action selection over a front-loaded candidate grid.

    Per-cycle cost = exactly ``n_candidates`` kernel evaluations (attributable work,
    CLAUDE.md / RFC-001). Myopic by design: one-step EFE, not horizon-aware — the
    H-step rollout is the deferred Phase 3 seam.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        n_candidates: int,
        action_bounds: tuple[float, float],
    ) -> None:
        if model.control is None:
            raise ValueError(
                "EFESelector needs a model with a control matrix; an action has no "
                "effect on a control-free (pure-tracking) model."
            )
        self._model = model
        lo, hi = action_bounds
        # p=1 (the tests/corridor): a column of candidate actions, front-loaded once.
        # p>1 (the 2-D figure) is a meshgrid — WIP.
        self._candidates = jnp.linspace(lo, hi, n_candidates)[:, None]

    def select(self, belief: Belief, preference: Preference) -> Float64[Array, "p"]:
        """The candidate action minimising one-step ``G`` over the front-loaded grid.

        One ``vmap`` of the EFE kernel across the candidates, then ``argmin`` — the
        whole per-cycle cost. At H=1 the chosen candidate *is* the action (no
        first-of-a-sequence slicing; that is the deferred horizon seam).
        """
        g = jax.vmap(
            lambda a: expected_free_energy(self._model, belief, a, preference)[0]
        )(self._candidates)
        return self._candidates[jnp.argmin(g)]

    @property
    def n_candidates(self) -> int:
        """The per-cycle EFE-evaluation count — attributable work (RFC-001)."""
        return self._candidates.shape[0]


@dataclass(frozen=True, init=False)
class StateGoal:
    """A state-space objective: reach a target state (the LQR / fixed-sensor regime).

    The complete spec for the state-tracking path - the target plus the LQR cost
    weights it implies. ``precision`` is LQR's state weight Q; ``effort`` is its
    action weight R, left None here because the action dimension p isn't known
    until the Agent pairs this with a model (the Agent fills the identity). The
    Agent dispatches a StateGoal to an LQRSelector. Not a pytree - construction-
    time only; the Agent extracts a Preference for the selector.
    """

    target: Float64[Array, "n"]
    precision: Float64[Array, "n n"]
    effort: Float64[Array, "p p"] | None

    def __init__(self, target: ArrayLike, *, precision=None, effort=None) -> None:
        target = jnp.asarray(target, dtype=float)
        object.__setattr__(self, "target", target)
        n = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(n) if precision is None else jnp.asarray(precision, dtype=float),
        )
        object.__setattr__(
            self,
            "effort",
            None if effort is None else jnp.asarray(effort, dtype=float),
        )
        self._validate()

    def _validate(self) -> None:
        if self.target.ndim != 1:
            raise ValueError(
                f"target must be a 1-D vector, got shape {self.target.shape}"
            )
        validate_covariance(self.precision, "precision")
        n = self.target.shape[0]
        if self.precision.shape != (n, n):
            raise ValueError(
                f"precision must be {n}x{n} to match the {n}-D target, "
                f"got shape {self.precision.shape}"
            )


@dataclass(frozen=True, init=False)
class ObservationGoal:
    """An observation-space objective: prefer to observe a target (the EFE regime).

    The complete spec for the information-seeking path - the preferred observation,
    how sharply it is preferred (``precision``), and the action-search config the
    EFESelector front-loads: ``action_bounds`` is the action box, ``n_candidates``
    its resolution. The Agent dispatches an ObservationGoal to an EFESelector. Not
    a pytree - construction-time only; the Agent extracts a Preference.
    """

    target: Float64[Array, "m"]
    precision: Float64[Array, "m m"]
    action_bounds: tuple[float, float]
    n_candidates: int

    def __init__(
        self, target, action_bounds, *, precision=None, n_candidates=21
    ) -> None:
        target = jnp.asarray(target, dtype=float)
        object.__setattr__(self, "target", target)
        m = target.shape[0]
        object.__setattr__(
            self,
            "precision",
            jnp.eye(m) if precision is None else jnp.asarray(precision, dtype=float),
        )
        object.__setattr__(self, "action_bounds", action_bounds)
        object.__setattr__(self, "n_candidates", n_candidates)
        self._validate()

    def _validate(self) -> None:
        if self.target.ndim != 1:
            raise ValueError(
                f"target must be a 1-D vector, got shape {self.target.shape}"
            )
        validate_covariance(self.precision, "precision")
        m = self.target.shape[0]
        if self.precision.shape != (m, m):
            raise ValueError(
                f"precision must be {m}x{m} to match the {m}-D observation goal, "
                f"got shape {self.precision.shape}"
            )
        lo, hi = self.action_bounds
        if not lo < hi:
            raise ValueError(
                f"action_bounds must be (lo, hi) with lo < hi, got {self.action_bounds}"
            )
        if self.n_candidates < 2:
            raise ValueError(
                f"n_candidates must be at least 2 to search, got {self.n_candidates}"
            )
