"""Two declared axes: what a model gets wrong, and what its filter gets wrong.

``ModelSpec`` is a model's parameters, held as inert arrays. ``build`` turns them into a
``LinearGaussianModel``, a fresh one on every call, sharing no array with the last. Two
things built from one spec are equal in value and separate in memory.

``Perturbation`` is a named change to one parameter, carried as data rather than as a
function: the parameter it touches and how far, relative to what the spec declares.
``CORRECT`` touches nothing.

``InferenceRule`` is the same idea for the filter. It names one of four kinds, from
exact through to a filter reading only the diagonal of its own covariance, and builds a
backend over a model without altering the model.

``ConstructorSet`` and ``InferenceSet`` collect each axis under a version, so a change
to either is a change to a line of source.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.degraded import DiagonalCovarianceBackend, WrongFixedRBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.dynamics import DynamicsNoise
from cpomdp.observation import ObservationModel
from cpomdp.structure import ModelStructure
from cpomdp.types import Belief, LinearGaussianModel

__all__ = [
    "CORRECT",
    "EXACT_INFERENCE",
    "ConstructorSet",
    "InferenceKind",
    "InferenceRule",
    "InferenceSet",
    "ModelSpec",
    "Perturbation",
]

_UNVERSIONED = (
    "version must be a non-empty string — the {subject} is declared and versioned"
)

# The parameters a perturbation may scale. The prior is not among them: it is a
# ``Belief`` rather than a matrix, and scaling a mean of zeros changes nothing while
# scaling a covariance changes the filter's transient rather than its model.
PERTURBABLE = (
    "dynamics_matrix",
    "observation_matrix",
    "dynamics_noise",
    "observation_noise",
    "control_matrix",
)


@dataclass(frozen=True)
class Perturbation:
    """A named, relative change to one declared parameter.

    Applied as ``value * (1 + magnitude)``, so ``0.0`` is the identity and ``-0.5``
    halves the parameter. One knob, so a cell can be named by what it perturbed and by
    how far.

    Args:
        name: The cell's label, non-empty and unique within a set.
        parameter: Which parameter to scale, one of ``PERTURBABLE``. ``None`` perturbs
            nothing, which is what ``CORRECT`` is.
        magnitude: The relative change. Must be ``0.0`` when ``parameter`` is ``None``.
    """

    name: str
    parameter: str | None
    magnitude: float

    def __post_init__(self) -> None:
        """Reject a nameless entry, an unknown parameter, and a magnitude on nothing."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a Perturbation needs a name, so a cell can be labelled")
        if self.parameter is None:
            if self.magnitude != 0.0:
                raise ValueError(
                    f"{self.name!r} perturbs nothing but names a magnitude of "
                    f"{self.magnitude}; pass a parameter or pass 0.0"
                )
            return
        if self.parameter not in PERTURBABLE:
            raise ValueError(
                f"{self.parameter!r} is not a perturbable parameter; declared: "
                f"{list(PERTURBABLE)}"
            )


CORRECT = Perturbation(name="correct", parameter=None, magnitude=0.0)
"""The cell that changes nothing, where a built model matches the spec exactly."""


@dataclass(frozen=True)
class ModelSpec:
    """The declared parameters a model is built from, and the version they carry.

    Held as read-only numpy arrays, so the spec itself never enters a computation.
    ``build`` is the only route from here to a ``LinearGaussianModel``, and it copies,
    so nothing built from a spec can reach back into it or into a sibling.

    ``observation_model``, ``dynamics_noise_model`` and ``structure`` pass through
    unchanged. They are objects rather than arrays, no perturbation names them, and a
    model that carries one carries the same one every build.

    Args:
        dynamics_matrix: State to next state — A, shape ``(n, n)``.
        observation_matrix: State to reading — C, shape ``(m, n)``.
        dynamics_noise: Process-noise covariance — Q, shape ``(n, n)``.
        observation_noise: Reading-noise covariance — R, shape ``(m, m)``.
        prior_mean: Centre of the initial belief, shape ``(n,)``.
        prior_cov: Covariance of the initial belief, shape ``(n, n)``.
        control_matrix: Action to state — B, shape ``(n, p)``. ``None`` for a model
            that takes no action.
        observation_model: A state-dependent sensor, passed through.
        dynamics_noise_model: State-dependent process noise, passed through.
        structure: A declared factor / blanket partition, passed through.
        version: A non-empty tag, so a parameter changed later shows up in the diff.
    """

    dynamics_matrix: np.ndarray
    observation_matrix: np.ndarray
    dynamics_noise: np.ndarray
    observation_noise: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    version: str
    control_matrix: np.ndarray | None = None
    observation_model: ObservationModel | None = None
    dynamics_noise_model: DynamicsNoise | None = None
    structure: ModelStructure | None = None

    def __init__(
        self,
        *,
        dynamics_matrix: ArrayLike,
        observation_matrix: ArrayLike,
        dynamics_noise: ArrayLike,
        observation_noise: ArrayLike,
        prior_mean: ArrayLike,
        prior_cov: ArrayLike,
        version: str,
        control_matrix: ArrayLike | None = None,
        observation_model: ObservationModel | None = None,
        dynamics_noise_model: DynamicsNoise | None = None,
        structure: ModelStructure | None = None,
    ) -> None:
        for name, value in (
            ("dynamics_matrix", dynamics_matrix),
            ("observation_matrix", observation_matrix),
            ("dynamics_noise", dynamics_noise),
            ("observation_noise", observation_noise),
            ("prior_mean", prior_mean),
            ("prior_cov", prior_cov),
        ):
            object.__setattr__(self, name, _frozen(value))
        object.__setattr__(
            self,
            "control_matrix",
            None if control_matrix is None else _frozen(control_matrix),
        )
        object.__setattr__(self, "observation_model", observation_model)
        object.__setattr__(self, "dynamics_noise_model", dynamics_noise_model)
        object.__setattr__(self, "structure", structure)
        object.__setattr__(self, "version", version)
        if not isinstance(version, str) or not version:
            raise ValueError(_UNVERSIONED.format(subject="spec"))

    def build(self, perturbation: Perturbation = CORRECT) -> LinearGaussianModel:
        """A model built from these parameters, with ``perturbation`` applied.

        Every array is copied, so two models built from one spec share no leaf and
        neither can be reached from the other.

        Args:
            perturbation: The named change to apply. Defaults to ``CORRECT``, which
                reproduces the declared parameters.

        Returns:
            A ``LinearGaussianModel``, validated by its own constructor.

        Raises:
            ValueError: If the perturbation names ``control_matrix`` on a spec that
                declares none, or if the result fails the model's own validation, as a
                covariance scaled to zero or below does.
        """
        parameter = perturbation.parameter
        if parameter == "control_matrix" and self.control_matrix is None:
            raise ValueError(
                f"{perturbation.name!r} perturbs control_matrix, and this spec "
                f"declares no control_matrix to perturb"
            )
        scale = 1.0 + perturbation.magnitude

        def value(name: str) -> np.ndarray | None:
            declared = getattr(self, name)
            if declared is None:
                return None
            return declared * scale if name == parameter else declared

        return LinearGaussianModel(
            dynamics_matrix=jnp.array(value("dynamics_matrix")),
            observation_matrix=jnp.array(value("observation_matrix")),
            dynamics_noise=jnp.array(value("dynamics_noise")),
            observation_noise=jnp.array(value("observation_noise")),
            prior=Belief(
                mean=jnp.array(self.prior_mean), cov=jnp.array(self.prior_cov)
            ),
            control_matrix=(
                None
                if self.control_matrix is None
                else jnp.array(value("control_matrix"))
            ),
            observation_model=self.observation_model,
            dynamics_noise_model=self.dynamics_noise_model,
            structure=self.structure,
        )


@dataclass(frozen=True)
class ConstructorSet:
    """A declared, versioned set of perturbations to run a spec through.

    Names are unique, so a cell can be identified by its label alone. At least one
    member perturbs nothing, so the set always contains the cell where the built model
    matches the spec.

    Args:
        perturbations: The declared members, in the order they are reported.
        version: A non-empty tag, so a member added later shows up in the diff.
    """

    perturbations: tuple[Perturbation, ...]
    version: str

    def __post_init__(self) -> None:
        """Reject an unversioned, empty, duplicated or wholly perturbed set."""
        _validate_declared(
            self.perturbations,
            self.version,
            subject="set",
            contains=lambda p: p.parameter is None,
            requirement=(
                "no member of this set is unperturbed, so no cell builds the model the "
                "spec declares; include CORRECT"
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The declared names, in declaration order."""
        return tuple(p.name for p in self.perturbations)

    @property
    def size(self) -> int:
        """The member count."""
        return len(self.perturbations)

    def build_all(self, spec: ModelSpec) -> tuple[tuple[str, LinearGaussianModel], ...]:
        """Every member applied to ``spec``, paired with the name that produced it.

        Args:
            spec: The parameters each member is applied to.

        Returns:
            One ``(name, model)`` pair per member, in declaration order. Every model is
            its own object and shares no array with any other.
        """
        return tuple((p.name, spec.build(p)) for p in self.perturbations)


class InferenceKind(Enum):
    """What a filter does differently from exact inference under the model.

    ``EXACT`` is the per-step Kalman filter, which for a fixed-noise linear-Gaussian
    model is the Bayesian posterior. The other three each drop something: the per-step
    gain, the declared observation noise, or the covariance's off-diagonal terms.
    """

    EXACT = "exact"
    FROZEN_GAIN = "FrozenGain"
    WRONG_FIXED_R = "WrongFixedR"
    DIAGONAL_COVARIANCE_ONLY = "DiagonalCovarianceOnly"


@dataclass(frozen=True)
class InferenceRule:
    """A named way of inferring under a model, built into a backend on demand.

    The model is never altered. ``WrongFixedR`` reads a scaled observation noise and is
    the only kind with a magnitude to read; the rest are the kind alone.

    Args:
        name: The cell's label, non-empty and unique within a set.
        kind: Which of the four declared kinds this cell is.
        magnitude: Relative change to the filter's observation noise. Must be ``0.0``
            unless ``kind`` is ``WRONG_FIXED_R``.
    """

    name: str
    kind: InferenceKind
    magnitude: float = 0.0

    def __post_init__(self) -> None:
        """Reject a nameless rule, and a magnitude on a kind with nothing to scale."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("an InferenceRule needs a name, so a cell can be labelled")
        if self.magnitude != 0.0 and self.kind is not InferenceKind.WRONG_FIXED_R:
            raise ValueError(
                f"{self.kind.value} reads no magnitude, and {self.name!r} names "
                f"{self.magnitude}; only WrongFixedR scales anything"
            )

    def build(self, model: LinearGaussianModel) -> InferenceBackend:
        """The backend this cell infers with, over ``model``.

        Args:
            model: The model the agent is scored under. Every backend built here
                reports it as ``backend.model``, degraded or not.

        Returns:
            An inference backend.

        Raises:
            ValueError: If the kind cannot be built over this model — a frozen gain on
                state-dependent noise, or a magnitude leaving no valid covariance.
        """
        if self.kind is InferenceKind.EXACT:
            return KalmanBackend(model)
        if self.kind is InferenceKind.FROZEN_GAIN:
            return KalmanBackend(model, steady_state=True)
        if self.kind is InferenceKind.WRONG_FIXED_R:
            return WrongFixedRBackend(model, magnitude=self.magnitude)
        if self.kind is InferenceKind.DIAGONAL_COVARIANCE_ONLY:
            return DiagonalCovarianceBackend(KalmanBackend(model))
        # Not a fallthrough: a kind added without a branch here would otherwise be
        # built as whichever branch came last, and report a cell it never ran.
        raise ValueError(f"{self.kind} names no way of building a backend")


EXACT_INFERENCE = InferenceRule(name="exact", kind=InferenceKind.EXACT)
"""The cell that infers exactly, against which the other cells are a degradation."""


@dataclass(frozen=True)
class InferenceSet:
    """A declared, versioned set of ways to infer, to run one model through.

    Names are unique. At least one member is exact, so the set always contains the cell
    a degradation is measured against.

    Args:
        rules: The declared members, in the order they are reported.
        version: A non-empty tag, so a member added later shows up in the diff.
    """

    rules: tuple[InferenceRule, ...]
    version: str

    def __post_init__(self) -> None:
        """Reject an unversioned, empty, duplicated or wholly degraded set."""
        _validate_declared(
            self.rules,
            self.version,
            subject="set",
            contains=lambda r: r.kind is InferenceKind.EXACT,
            requirement=(
                "no member of this set infers exactly, so no cell is the one the "
                "others are degraded from; include EXACT_INFERENCE"
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The declared names, in declaration order."""
        return tuple(rule.name for rule in self.rules)

    @property
    def size(self) -> int:
        """The member count."""
        return len(self.rules)

    def build_all(
        self, model: LinearGaussianModel
    ) -> tuple[tuple[str, InferenceBackend], ...]:
        """Every member built over ``model``, paired with the name that produced it.

        Args:
            model: The model each member infers under.

        Returns:
            One ``(name, backend)`` pair per member, in declaration order. Every backend
            reports ``model`` as its own.
        """
        return tuple((rule.name, rule.build(model)) for rule in self.rules)


class _Named(Protocol):
    """Anything a declared set holds: it has a name, and that name identifies it."""

    name: str


_Member = TypeVar("_Member", bound=_Named)


def _validate_declared(
    members: tuple[_Member, ...],
    version: str,
    *,
    subject: str,
    contains: Callable[[_Member], bool],
    requirement: str,
) -> None:
    """The checks both declared sets share: versioned, non-empty, unique, complete."""
    if not isinstance(version, str) or not version:
        raise ValueError(_UNVERSIONED.format(subject=subject))
    if not members:
        raise ValueError(f"a declared {subject} needs at least one member")
    names = [member.name for member in members]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate name(s) {duplicates}; a cell is identified by its name alone"
        )
    if not [member for member in members if contains(member)]:
        raise ValueError(requirement)


def _frozen(value: ArrayLike) -> np.ndarray:
    """A read-only float copy, so a caller's array cannot change a spec after it."""
    array = np.array(value, dtype=float)
    array.setflags(write=False)
    return array
