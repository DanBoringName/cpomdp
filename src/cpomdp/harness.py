"""The world and the agent as separate objects, with no reference between them.

``World`` runs the process that produces observations. ``ScoredAgent`` runs the model
it filters with. Neither holds the other: ``World`` has no accessor returning its model
or any parameter of it, and ``ScoredAgent`` has no constructor slot for a ``World``.
Nothing here carries a reference from either one to the other.

Actions arrive from outside. ``driven_step`` takes the action to predict with, so one
sequence can drive several agents. There is no selector here and no ``sample_action``.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float64
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.backends.kalman import KalmanBackend
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["ScoredAgent", "World"]


class World:
    """The process an agent is scored against: true state in, observations out.

    Holds a ``LinearGaussianModel`` privately and advances one true state under it.
    Each ``step`` applies an action, draws the process noise, and returns a noisy
    observation of the state it arrived at. There is no ``model`` attribute and no
    accessor returning a parameter of it.

    ``state`` is readable, so a run can score against the true trajectory.

    A state-dependent ``R(x)`` or ``Q(x)`` is refused at construction. Sampling either
    needs an evaluation point, the departed state or the arrived-at one, and the two
    give different trajectories.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        initial_state: ArrayLike | None = None,
    ) -> None:
        """Build a world running ``model``, starting at ``initial_state``.

        Args:
            model: The process that generates observations — p*. Kept private; the
                world exposes no route back to it.
            initial_state: The true state to start from, shape ``(n,)``. Defaults to
                the model prior's mean.

        Raises:
            NotImplementedError: If the model carries a state-dependent ``R(x)`` or
                ``Q(x)``.
            ValueError: If ``initial_state`` is not a 1-D vector of length ``n``.
        """
        if model.observation_model is not None and not model.observation_model.is_fixed:
            raise NotImplementedError(
                "a state-dependent R(x) has no settled evaluation point here: the "
                "filter reads it at the predicted mean, and a world that knows its "
                "own state exactly does not. Pass a fixed observation_noise."
            )
        if model.dynamics_noise_model is not None:
            raise NotImplementedError(
                "a state-dependent Q(x) has no settled evaluation point here: the "
                "departed state and the arrived-at state give different trajectories. "
                "Pass a fixed dynamics_noise."
            )
        self._model = model
        if initial_state is None:
            state = model.prior.mean
        else:
            state = jnp.asarray(initial_state, dtype=float)
            if state.shape != (model.n_states,):
                raise ValueError(
                    f"initial_state must be a 1-D vector of length "
                    f"{model.n_states} (the state dimension), got shape {state.shape}"
                )
        self._state = state

    @property
    def state(self) -> Float64[Array, "n"]:
        """The current true state, shape ``(n,)``."""
        return self._state

    @property
    def n_states(self) -> int:
        """Dimension of the true state (n)."""
        return self._model.n_states

    @property
    def n_observations(self) -> int:
        """Dimension of an observation (m)."""
        return self._model.n_observations

    @property
    def n_controls(self) -> int:
        """Dimension of an action (p); 0 if the process takes no action."""
        return self._model.n_controls

    def step(self, action: ArrayLike | None, key: Array) -> Float64[Array, "m"]:
        """Advance the true state by one step and observe it.

        The action drives the transition out of the current state; the observation
        reads the state it arrives at. Both noise draws come from ``key``, so two
        worlds given the same key and the same start produce the same reading.

        Args:
            action: The action applied over this step, shape ``(p,)``. ``None`` for a
                process with no control matrix.
            key: A JAX PRNG key, split internally into a process draw and a sensor
                draw. Pass a fresh key per step or the trajectory repeats.

        Returns:
            One observation, shape ``(m,)``.

        Raises:
            ValueError: If the process has a control matrix and ``action`` is ``None``
                or not shape ``(p,)``.
        """
        model = self._model
        key_dynamics, key_observation = jax.random.split(key)

        mean_next = model.dynamics_matrix @ self._state
        if model.control_matrix is not None:
            if action is None:
                raise ValueError(
                    "this world has a control matrix; step requires an action"
                )
            action = jnp.asarray(action, dtype=float)
            if action.shape != (model.n_controls,):
                raise ValueError(
                    f"action must be a 1-D vector of length {model.n_controls} "
                    f"(the action dimension), got shape {action.shape}"
                )
            mean_next = mean_next + model.control_matrix @ action

        # svd rather than the cholesky default: dynamics_noise is only required
        # positive *semi*-definite, so a noiseless state dimension is legal here.
        self._state = jax.random.multivariate_normal(
            key_dynamics, mean_next, model.dynamics_noise, method="svd"
        )
        return jax.random.multivariate_normal(
            key_observation,
            model.observation_matrix @ self._state,
            model.observation_noise,
        )


class ScoredAgent:
    """An agent that filters under its own model, driven by actions from outside.

    Owns a ``LinearGaussianModel`` and the belief it carries forward. Each
    ``driven_step`` folds one observation in, predicting with the action it is handed.
    The agent neither chooses nor remembers actions, and has no selector and no
    ``sample_action``.

    Pass a model to filter with a per-step ``KalmanBackend``, or pass a backend to
    filter with something else. Passing both is refused unless they carry the same
    model.
    """

    def __init__(
        self,
        model: LinearGaussianModel | None = None,
        *,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Build an agent filtering under ``model``, or under ``backend``'s model.

        Args:
            model: The agent's own generative model — p. Optional if a ``backend`` is
                given, since a backend carries the model it was built from.
            backend: The inference engine. Defaults to a per-step ``KalmanBackend``
                over ``model``; pass another to degrade or replace the inference
                without touching the model.

        Raises:
            ValueError: If neither a model nor a backend is given, or if both are
                given and they are not the same model.
        """
        if backend is None:
            if model is None:
                raise ValueError(
                    "ScoredAgent needs a model or a backend to filter with."
                )
            backend = KalmanBackend(model)
        elif model is None:
            model = backend.model
        elif model is not backend.model:
            raise ValueError(
                "a ScoredAgent is scored under one model, and the model passed is not "
                "the one the backend was built from. Pass whichever is intended, or "
                "build the backend from that model."
            )
        self.model = model
        self.belief = model.prior
        self._backend = backend

    def driven_step(self, observation: ArrayLike, action: ArrayLike) -> Belief:
        """Fold one observation in, predicting with the action supplied.

        The current belief goes in as the prior and the posterior comes back out and is
        stored, which is the recursive filter advanced one step. The belief is replaced
        rather than mutated.

        Args:
            observation: The reading for this step, shape ``(m,)``.
            action: The action applied since the previous observation, shape ``(p,)``.

        Returns:
            The updated belief, also stored on ``self.belief``.

        Raises:
            ValueError: On a shape mismatch in ``observation`` or ``action``, enforced
                by the backend's ``validate_step_inputs``.
        """
        self.belief = self._backend.infer_states(observation, self.belief, action)
        return self.belief
