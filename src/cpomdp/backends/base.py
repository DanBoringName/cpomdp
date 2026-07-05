"""The ``InferenceBackend`` protocol and the shared per-step input validator."""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["EfeBackend", "InferenceBackend", "validate_step_inputs"]


@runtime_checkable
class InferenceBackend(Protocol):
    """A swappable inference engine for a ``LinearGaussianModel``.

    A backend is *built from a model*: any expensive, data-independent work
    (front-loading — see DECISIONS.md ADR-002) happens at construction, so the
    per-step ``infer_states`` stays cheap. Each call advances the belief one
    recursive filter step: the current belief goes in as the ``prior`` and the
    updated belief comes back as the posterior.

    The Protocol is structural: any class with a matching ``infer_states`` (and the
    ``model`` it was built from) is a backend, with no shared base class. This is the
    abstraction wall — the native Kalman fast path and the RxInfer oracle are
    interchangeable behind it, and neither's implementation (JAX, juliacall, …) leaks
    into this signature.
    """

    # The model this backend was built from. A backend is *built from a model*, so it
    # can always hand it back — the Agent derives its own ``model`` from here.
    model: LinearGaussianModel

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief by one filter step: ``prior`` in, posterior out.

        Given the current belief (``prior``) and a new ``observation`` (plus the
        ``action`` just taken, if the model has a control matrix), return the
        updated belief.
        """
        ...


@runtime_checkable
class EfeBackend(InferenceBackend, Protocol):
    """An ``InferenceBackend`` that also supports *factored* EFE action selection.

    A node-structured backend: beyond ``infer_states`` it exposes the pre-observation
    predicted joint (``predicted_belief`` → Σ⁺), the real observation model
    (``observation_model`` → C, R), and the per-node ``block`` the epistemic term can
    target (issue #26). ``FfgEfeSelector`` and the Agent's EFE dispatch depend on *this*
    abstraction, never the concrete backend class — so action selection stays above the
    backend layer (Dependency Inversion).
    """

    dims: tuple[int, ...]
    n_total: int

    @property
    def observation_model(self) -> tuple[Float64[Array, "m n"], Float64[Array, "m m"]]:
        """The real sensor ``(C, R)`` embedded in the joint state (the EFE reads it)."""
        ...

    def predicted_belief(
        self, prior: Belief, action: ArrayLike | None = None
    ) -> Belief:
        """The joint after dynamics + couplings, before observing (Σ⁺, μ⁺)."""
        ...

    def block(self, node: int) -> range:
        """The joint-state indices node ``node`` occupies (``info_target`` → block)."""
        ...

    def observation_noise_at(self, mean: ArrayLike) -> Float64[Array, "m m"]:
        """The stacked observation noise R at a predicted mean (fixed → constant)."""
        ...


def validate_step_inputs(
    model: LinearGaussianModel,
    observation: ArrayLike,
    prior: Belief,
    action: ArrayLike | None,
) -> tuple[Float64[Array, "m"], Float64[Array, "p"] | None]:
    """Coerce and shape-check one step's inputs at the trust boundary.

    ``LinearGaussianModel`` validates the model once at construction; this gives
    the per-step runtime data the same care, since that is where library users
    actually slip. Living here (not in any one backend) keeps the native fast
    path and the RxInfer oracle validating *identically* — an oracle that
    accepted inputs the fast path rejected would not be a trustworthy oracle.

    The shape checks also close the silent-broadcast trap: a length-1
    observation would otherwise broadcast against the ``m``-D prediction error
    and yield a confident *wrong* belief rather than an error.

    Args:
        model: The model being filtered under — supplies the expected dims.
        observation: Raw sensor reading; any array-like, coerced to float.
        prior: The incoming belief, checked against the model's state dim.
        action: Raw action; any array-like or ``None``. Coerced only when the
            model has a control matrix.

    Returns:
        The coerced ``(observation, action)``. ``action`` is ``None`` for a
        control-free model, otherwise a float array of shape ``(p,)``.

    Raises:
        ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not
            over the ``n``-D state, the model needs an action but got ``None``,
            or ``action`` is not shape ``(p,)``.
    """
    observation = jnp.asarray(observation, dtype=float)
    m = model.n_observations
    if observation.shape != (m,):
        raise ValueError(
            f"observation must be a 1-D vector of length {m} "
            f"(the observation dimension), got shape {observation.shape}"
        )

    if prior.ndim != model.n_states:
        raise ValueError(
            f"prior must be a belief over the {model.n_states}-D state, "
            f"got a {prior.ndim}-D belief"
        )

    if model.control is None:
        return observation, None

    if action is None:
        raise ValueError(
            "this model has a control matrix; infer_states requires an action"
        )
    action = jnp.asarray(action, dtype=float)
    p = model.n_controls
    if action.shape != (p,):
        raise ValueError(
            f"action must be a 1-D vector of length {p} "
            f"(the action dimension), got shape {action.shape}"
        )
    return observation, action
