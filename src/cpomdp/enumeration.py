"""Exhaustive EFE search over a declared finite action set (ADR-030, ADR-031).

The enumerated counterpart of ``EFESelector``'s continuous grid, and a deliberately
*different object* so the two cannot be confused. ``EFESelector`` samples a continuous
action box and tiles each sample into a **constant-action** policy — a sample of a
continuum, warrant ``CORROBORATED`` (Prover 3a: corroborates, never decides).
``EnumeratedEfeSearch`` enumerates **every** length-H sequence of a finite, declared,
versioned ``FiniteActionSet`` — all ``|A|^H`` of them, *varying* sequences included — so
"no policy in this set flips" is **decided**, not sampled: warrant ``PROVED`` (Prover
3b). A ``CompletenessCertificate`` records ``expected = |A|^H`` against the ``visited``
count and asserts they match (ADR-030), so the decisive warrant is earned, not assumed.

The set is versioned because it is a modelling commitment: an action added after results
are seen must show up in the diff, not be discovered by a reviewer. The ``PROVED``
warrant is scoped to the *declared* set — whether that set is fine enough is a separate
question a refinement-stability check answers, not this certificate.

Internal seam: imported from ``cpomdp.enumeration``, not re-exported at the top level.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float64

from cpomdp.efe import policy_efe, policy_efe_ffg
from cpomdp.types import Belief, LinearGaussianModel

if TYPE_CHECKING:
    # Duck-typed at runtime (read for .goal/.precision via the efe kernels, and the
    # backend's predict/observation methods), so no runtime dependency on selection or
    # the backends package — keeps selection/backends -> enumeration the only edges.
    from cpomdp.backends.base import EfeBackend
    from cpomdp.selection import Preference

__all__ = [
    "CompletenessCertificate",
    "EnumeratedEfeSearch",
    "EnumeratedSearchResult",
    "FiniteActionSet",
    "IncompleteEnumerationError",
    "OpenLoopSelector",
    "RecedingHorizonSelector",
    "SearchWarrant",
]


class SearchWarrant(Enum):
    """How well a search's ``no policy flips`` claim is warranted (standing rule 6).

    ``PROVED`` — a finite set enumerated in full **decides** its universal (Prover 3b).
    ``CORROBORATED`` — a grid sample of a continuum only **corroborates** it (Prover
    3a); the optimum could sit between the sampled points. The two must print in
    different vocabulary, never both as a bare ``PASS``.
    """

    PROVED = "PROVED"
    CORROBORATED = "CORROBORATED"


class IncompleteEnumerationError(Exception):
    """Fewer than ``|A|^H`` policies enumerated: a sample, not a decision."""


@dataclass(frozen=True, init=False)
class FiniteActionSet:
    """A finite, declared, **versioned** set of actions to enumerate over.

    A distinct type from the continuous grid's ``(lo, hi, n_candidates)`` config, so the
    enumerated (``PROVED``) and sampled (``CORROBORATED``) families cannot be mixed up
    at a call site. Construction-time only (not a pytree).

    Args:
        actions: the declared actions, shape ``(A, p)`` — ``A`` actions of dimension
            ``p``. At least two (a set of one is no choice to search).
        version: a non-empty tag, so an action added later shows up in the diff.
    """

    actions: Float64[Array, "A p"]
    version: str

    def __init__(self, actions, *, version: str) -> None:
        object.__setattr__(self, "actions", jnp.asarray(actions, dtype=float))
        object.__setattr__(self, "version", version)
        self._validate()

    def _validate(self) -> None:
        if self.actions.ndim != 2:
            raise ValueError(
                f"actions must be a 2-D (A, p) array, got shape {self.actions.shape}"
            )
        if self.actions.shape[0] < 2:
            raise ValueError(
                f"a FiniteActionSet needs at least 2 actions to search, got "
                f"{self.actions.shape[0]}"
            )
        if not isinstance(self.version, str) or not self.version:
            raise ValueError(
                "version must be a non-empty string — the set is declared and versioned"
            )

    @property
    def size(self) -> int:
        """The action count ``A`` (the base of ``|A|^H``)."""
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        """The action dimension ``p``."""
        return int(self.actions.shape[1])


@dataclass(frozen=True)
class CompletenessCertificate:
    """Evidence an enumeration was exhaustive: ``expected`` vs ``visited`` (ADR-030).

    ``expected = |A|^H`` is computed independently from the set size and horizon;
    ``visited`` is the number of policies actually enumerated. Equal ⇒ the search
    decided its universal, so it carries ``PROVED``. Printed in the warrant's own
    vocabulary, never a bare ``PASS``.
    """

    expected: int
    visited: int
    warrant: SearchWarrant

    @property
    def complete(self) -> bool:
        """Whether every expected policy was visited."""
        return self.expected == self.visited

    def __str__(self) -> str:
        """The certificate as a one-line warrant string in its own vocabulary."""
        return (
            f"{self.warrant.value} (finite set, |A|^H = {self.expected}, "
            f"visited {self.visited})"
        )


class EnumeratedSearchResult(NamedTuple):
    """The outcome of one exhaustive search.

    Attributes:
        best_policy: the argmin sequence, shape ``(H, p)``.
        g: the full EFE vector over every enumerated policy, shape ``(|A|^H,)`` — G-D's
            dual reporting at H > 1 for free.
    """

    best_policy: Float64[Array, "H p"]
    g: Float64[Array, "N"]


class _PolicyScorer(Protocol):
    """Scores one policy to its summed EFE ``G`` — the seam that swaps flat for FFG.

    The enumerated search owns the enumeration, the certificate, and the argmin; *how* a
    policy is scored is injected here (ADR-034). ``action_dim`` lets the search validate
    the declared action set against the scorer's control dimension.
    """

    action_dim: int

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]: ...


@dataclass(frozen=True)
class _FlatScorer:
    """Scores with ``policy_efe`` on a flat ``LinearGaussianModel``."""

    model: LinearGaussianModel

    @property
    def action_dim(self) -> int:
        return self.model.n_controls

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]:
        return policy_efe(self.model, belief, policy, preference)[0]


@dataclass(frozen=True)
class _FfgScorer:
    """Scores with ``policy_efe_ffg`` on an FFG backend, aimed at ``target``."""

    backend: "EfeBackend"
    target: tuple[int, ...]

    @property
    def action_dim(self) -> int:
        return self.backend.model.n_controls

    def score(
        self, belief: Belief, policy: Float64[Array, "H p"], preference: "Preference"
    ) -> Float64[Array, ""]:
        return policy_efe_ffg(
            self.backend, belief, policy, preference, target=self.target
        )[0]


class EnumeratedEfeSearch:
    """Exhaustive EFE search over ``A^H`` for a declared finite action set (ADR-031).

    Front-loads (ADR-002) the full ``|A|^H`` enumeration of *varying* sequences at
    construction, with its completeness certificate; ``evaluate`` scores them all and
    returns the argmin policy and the full ``G`` vector. Because the set is finite and
    fully enumerated the search **decides** its universal — warrant ``PROVED``, distinct
    from ``EFESelector``'s ``CORROBORATED`` grid.

    Scoring is a strategy (ADR-034): the default constructor scores on a flat
    ``LinearGaussianModel`` (``policy_efe``); ``over_backend`` scores on an FFG backend
    (``policy_efe_ffg``, aimed at a node's block) so the same exhaustive search runs the
    coupled-tree crossover model a flat model cannot express. The enumeration,
    the certificate, and the cost are shared — only the per-policy scoring differs.

    Unlike ``EFESelector`` this supports ``p >= 1`` and *varying* sequences: it can
    express a detour-then-exploit policy the constant-action family cannot.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        action_set: FiniteActionSet,
        *,
        horizon: int,
    ) -> None:
        if model.control is None:
            raise ValueError(
                "EnumeratedEfeSearch needs a model with a control matrix; an "
                "action has no effect on a control-free (pure-tracking) model."
            )
        self._setup(_FlatScorer(model), action_set, horizon)

    @classmethod
    def over_backend(
        cls,
        backend: "EfeBackend",
        action_set: FiniteActionSet,
        *,
        target: Sequence[int],
        horizon: int,
    ) -> "EnumeratedEfeSearch":
        """Exhaustive search that scores on an FFG ``backend`` via ``policy_efe_ffg``.

        The FFG counterpart of the flat constructor: it enumerates the same ``A^H`` set
        with the same completeness certificate, but scores each policy on a coupling
        graph with the epistemic aimed at ``target`` — a node's block (via
        ``backend.block``) or the whole state. This runs the coupled-tree model the flat
        constructor cannot express.
        """
        if backend.model.control is None:
            raise ValueError(
                "EnumeratedEfeSearch.over_backend needs a backend with a control "
                "matrix; an action has no effect without one."
            )
        search = cls.__new__(cls)
        search._setup(_FfgScorer(backend, tuple(target)), action_set, horizon)
        return search

    def _setup(
        self, scorer: _PolicyScorer, action_set: FiniteActionSet, horizon: int
    ) -> None:
        """Shared construction: validate, front-load ``A^H``, assert the certificate."""
        if action_set.action_dim != scorer.action_dim:
            raise ValueError(
                f"the action set is p={action_set.action_dim} but the control is "
                f"p={scorer.action_dim}; the declared actions must match the action "
                "dimension."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self._scorer = scorer
        self._action_set = action_set
        self._horizon = int(horizon)
        # Front-load the enumeration: one row per A^H sequence, built once at
        # construction (host-side itertools; never in the hot path).
        index_grid = np.array(
            list(itertools.product(range(action_set.size), repeat=self._horizon))
        )  # (N, H)
        self._policies = action_set.actions[jnp.asarray(index_grid)]  # (N, H, p)

        expected = action_set.size**self._horizon
        visited = int(self._policies.shape[0])
        if visited != expected:  # ADR-030: the certificate is asserted, not assumed
            raise IncompleteEnumerationError(
                f"enumeration visited {visited} of an expected {expected} = |A|^H "
                "policies; the search would be a sample, not a decision."
            )
        self._certificate = CompletenessCertificate(
            expected=expected, visited=visited, warrant=SearchWarrant.PROVED
        )

    def evaluate(
        self, belief: Belief, preference: "Preference"
    ) -> EnumeratedSearchResult:
        """Score every enumerated policy; return the argmin and the full ``G`` vector.

        One ``vmap`` of the scorer over the front-loaded ``|A|^H`` policies, then a
        NaN-safe ``argmin`` (a NaN-scoring policy must not silently win). The scorer
        holds the flat model or the FFG backend as a closure constant, so the ``vmap``
        maps only over the policies.
        """
        g = jax.vmap(lambda pol: self._scorer.score(belief, pol, preference))(
            self._policies
        )
        best = jnp.argmin(jnp.where(jnp.isnan(g), jnp.inf, g))
        return EnumeratedSearchResult(best_policy=self._policies[best], g=g)

    @property
    def policies(self) -> Float64[Array, "N H p"]:
        """The enumerated ``A^H`` policies, shape ``(|A|^H, H, p)`` (front-loaded)."""
        return self._policies

    @property
    def n_policies(self) -> int:
        """The enumerated policy count ``|A|^H`` — attributable work (RFC-001)."""
        return int(self._policies.shape[0])

    @property
    def horizon(self) -> int:
        """The sequence length H."""
        return self._horizon

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = ``|A|^H * H`` — the honest exponential cost."""
        return self.n_policies * self._horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — a finite set enumerated in full decides its universal."""
        return SearchWarrant.PROVED

    @property
    def certificate(self) -> CompletenessCertificate:
        """The front-loaded completeness certificate (ADR-030)."""
        return self._certificate


class RecedingHorizonSelector:
    """Receding-horizon driver over ``EnumeratedEfeSearch`` (an ``ActionSelector``).

    Every ``select`` re-runs the full ``A^H`` search on the current belief and returns
    the first action of the best sequence. The belief drives every choice, so the plan
    is recomputed each step. Kept separate from ``OpenLoopSelector`` because item E's
    matched-horizon bracket depends on which mode is running (ADR-031).
    """

    def __init__(self, search: EnumeratedEfeSearch) -> None:
        self._search = search

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Re-plan on ``belief`` and return the first action of the best sequence."""
        return self._search.evaluate(belief, preference).best_policy[0]

    @property
    def horizon(self) -> int:
        """The planning horizon H."""
        return self._search.horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — inherited from the enumerated search."""
        return self._search.warrant

    @property
    def certificate(self) -> CompletenessCertificate:
        """The search's completeness certificate (ADR-030)."""
        return self._search.certificate

    @property
    def cost_per_plan(self) -> int:
        """Step-evals for one search = ``|A|^H * H`` (RFC-001 attributable work)."""
        return self._search.cost_per_cycle

    @property
    def replan_interval(self) -> int:
        """Cycles between re-plans. 1: this driver plans every step."""
        return 1

    @property
    def cost_per_cycle(self) -> int:
        """Per-cycle step-evals = ``|A|^H * H``. It plans every cycle."""
        return self.cost_per_plan // self.replan_interval


class OpenLoopSelector:
    """Open-loop driver over ``EnumeratedEfeSearch`` (an ``ActionSelector``).

    Plans once on the belief of the first ``select``, commits to the whole best
    sequence, and returns its actions in order while ignoring the interim beliefs. It
    re-plans only when the committed sequence is exhausted, or after ``reset``. Ignoring
    the interim beliefs is what makes it open-loop, and the distinction from
    ``RecedingHorizonSelector`` is item E's matched-horizon bracket (ADR-031).

    Stateful: it holds the committed plan and a step counter. The ``Agent`` drives it
    eagerly, one ``select`` per step, so the mutation is safe.
    """

    def __init__(self, search: EnumeratedEfeSearch) -> None:
        self._search = search
        self._plan: Float64[Array, "H p"] | None = None
        self._step = 0

    def select(self, belief: Belief, preference: "Preference") -> Float64[Array, "p"]:
        """Return the next committed action, re-planning on ``belief`` when spent."""
        if self._plan is None or self._step >= self._search.horizon:
            self._plan = self._search.evaluate(belief, preference).best_policy
            self._step = 0
        action = self._plan[self._step]
        self._step += 1
        return action

    def reset(self) -> None:
        """Drop the committed plan so the next ``select`` re-plans from scratch."""
        self._plan = None
        self._step = 0

    @property
    def horizon(self) -> int:
        """The planning horizon H, which is also the re-plan interval."""
        return self._search.horizon

    @property
    def warrant(self) -> SearchWarrant:
        """``PROVED`` — inherited from the enumerated search."""
        return self._search.warrant

    @property
    def certificate(self) -> CompletenessCertificate:
        """The search's completeness certificate (ADR-030)."""
        return self._search.certificate

    @property
    def cost_per_plan(self) -> int:
        """Step-evals for one search = ``|A|^H * H`` (RFC-001 attributable work)."""
        return self._search.cost_per_cycle

    @property
    def replan_interval(self) -> int:
        """Cycles between re-plans = H. It commits to the whole sequence."""
        return self._search.horizon

    @property
    def cost_per_cycle(self) -> int:
        """Amortised per-cycle step-evals = ``|A|^H`` (one ``|A|^H * H`` plan per H)."""
        return self.cost_per_plan // self.replan_interval
