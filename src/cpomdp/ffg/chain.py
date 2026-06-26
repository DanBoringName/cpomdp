"""FFG chain inference backend: ``predict ∘ update ∘ to_moment`` = a Kalman step.

The keystone of v0.4 Phase 2 (ADR-012): wiring the Phase-1/Phase-2 message algebra
into the ``InferenceBackend`` protocol and showing that, on a *chain* topology, it
reproduces the existing Kalman path. Gaussian belief propagation on a linear chain
*is* the Kalman filter — so this backend is interchangeable with ``KalmanBackend``
behind the same seam, and the gate (``tests/test_ffg_chain.py``) holds it to
numerical identity against that path.

One step decomposes into the owned algebra exactly as the factor docstrings promise::

    prior (moment) ──┐
                     ▼  to canonical: Λ₀ = Σ⁻¹, h₀ = Λ₀μ
        GaussianTransition.predict(prior_msg, control_term)   # predict  (x → x')
                     ▼
        predicted_msg + GaussianObservation.message(y)        # update   (+ = product)
                     ▼  to_moment: Σ = Λ⁻¹, μ = Λ⁻¹h
                posterior (moment)

Scope (tier-1, fixed matrices). Both factors invert their noise covariance, so this
backend handles only the plain fixed-matrix linear-Gaussian model: state-dependent
``R(x)``/``Q(x)`` (the ``observation``/``process_noise`` fields) are out of scope and
rejected at construction, and a deterministic (``Q = 0``) transition has no
information form and is rejected by the transition factor. These are the documented
divergences from moment-form Kalman, harmless for the chain the gate exercises.

Energy note (RFC-001). Bridging a moment-form protocol (``Belief`` in, ``Belief``
out) to info-form internals costs two inversions per step that the native Kalman
path does not pay: ``Σ⁻¹`` to lift the incoming prior into canonical form, and
``Λ⁻¹`` to read the posterior back out. The *factors* themselves are front-loaded —
built once at construction from the fixed model matrices, never per step — so the
loop body stays the four cheap algebra ops above. This backend is the correctness
demonstration, not the production hot path; the extra inversions are the price of
the protocol bridge, flagged here rather than hidden.
"""

import jax.numpy as jnp
from numpy.typing import ArrayLike

from cpomdp.backends.base import validate_step_inputs
from cpomdp.ffg.factors.linear_gaussian import GaussianObservation, GaussianTransition
from cpomdp.ffg.message import CanonicalGaussian
from cpomdp.types import Belief, LinearGaussianModel

__all__ = ["ChainBackend"]


class ChainBackend:
    """FFG message-passing inference on a linear-Gaussian *chain*.

    Implements the ``InferenceBackend`` protocol via the canonical-form message
    algebra (``CanonicalGaussian`` + the tier-1 factors), not the moment-form
    Kalman recursion. Constructed from a model, then advances a belief one step at
    a time (prior in, posterior out); see the module docstring for the per-step
    decomposition and the scope/energy notes.

    Args:
        model: The linear-Gaussian generative model to filter under. Must be the
            plain fixed-matrix kind — a model carrying a state-dependent
            ``observation`` (``R(x)``) or ``process_noise`` (``Q(x)``) is out of
            scope for the tier-1 chain and rejected here. ``dynamics_noise`` (Q)
            must be positive-*definite* (the information form inverts it).
    """

    def __init__(self, model: LinearGaussianModel) -> None:
        """Validate scope and front-load the two fixed factor nodes.

        Build the ``GaussianObservation`` (from C, R) and ``GaussianTransition``
        (from A, Q) *once* here — they are data-independent, so constructing them
        per step would burn compute the regime doesn't need (RFC-001). Reject the
        out-of-scope models up front: noise that is *state-dependent* — test it with
        the ``is_fixed`` flag, not ``is None`` (an ``observation`` can be present but
        fixed, e.g. a ``FixedSensor``), mirroring ``KalmanBackend``::

            sensor_fixed  = model.observation   is None or model.observation.is_fixed
            process_fixed = model.process_noise is None or model.process_noise.is_fixed

        and raise a ``ValueError`` naming "state-dependent" if either is False. (This
        Phase-2 rejection is temporary — Phase 2.5 lifts it via a linearize-at-μ⁻
        plug-in, ADR-012 amendment 2026-06-26.) A singular/indefinite Q surfaces as
        the transition factor's own "positive-definite" error when that factor is
        built.

        Args:
            model: see the class docstring.

        Raises:
            ValueError: If the model carries state-dependent ``R(x)``/``Q(x)``, or
                if Q is not positive-definite (raised by ``GaussianTransition``).
        """
        self.model = model
        sensor_fixed = model.observation is None or model.observation.is_fixed
        process_fixed = model.process_noise is None or model.process_noise.is_fixed
        if not (sensor_fixed and process_fixed):
            raise ValueError(
                "ChainBackend (tier-1) needs fixed sensor and process noise; a "
                "state-dependent R(x) or Q(x) is not yet representable on the FFG "
                "chain (Phase 2.5) — use KalmanBackend for state-dependent noise."
            )
        self._transition = GaussianTransition(
            model.dynamics, model.dynamics_noise
        )  # A, Q
        self._observation = GaussianObservation(
            model.sensor_model, model.sensor_noise
        )  # C, R

    def infer_states(
        self,
        observation: ArrayLike,
        prior: Belief,
        action: ArrayLike | None = None,
    ) -> Belief:
        """Advance the belief by one filter step via the FFG algebra.

        Validate the per-step inputs at the trust boundary with the shared
        ``validate_step_inputs`` (identical checks to ``KalmanBackend`` — same seam,
        same errors), form the control shift ``b = control @ action`` (zero when the
        model is uncontrolled), then run the module-docstring pipeline: lift the
        prior into canonical form, ``predict`` through the transition factor, add the
        observation factor's ``message(y)`` (the measurement update), and
        ``to_moment`` the result back into a ``Belief``.

        Args:
            observation: The latest sensor reading, shape ``(m,)``.
            prior: The current belief, this step's previous posterior. Never mutated.
            action: The action just taken, shape ``(p,)``. Required iff the model has
                a control matrix; pass ``None`` for pure filtering.

        Returns:
            The posterior belief — a new ``Belief``; the prior is left untouched.

        Raises:
            ValueError: If ``observation`` is not shape ``(m,)``, ``prior`` is not a
                belief over the model's ``n``-D state, the model has a control matrix
                but ``action`` is ``None``, or ``action`` is not shape ``(p,)``.
        """
        observation, action = validate_step_inputs(
            self.model, observation, prior, action
        )
        model = self.model
        control = model.control
        if control is None:
            control_term = jnp.zeros(model.n_states)
        else:
            # validate_step_inputs guarantees a non-None action when control exists
            assert action is not None
            control_term = control @ action

        prior_precision = jnp.linalg.inv(prior.cov)  # Λ₀ = Σ⁻¹
        prior_msg = CanonicalGaussian._unchecked(
            prior_precision, prior_precision @ prior.mean
        )  # h₀ = Λ₀μ; invariant-preserving lift of a validated Belief — no re-validate

        predicted = self._transition.predict(prior_msg, control_term)
        posterior_msg = predicted + self._observation.message(observation)

        mean_post, cov_post = posterior_msg.to_moment()  # Σ = Λ⁻¹, μ = Λ⁻¹h

        return Belief(mean=mean_post, cov=cov_post)
