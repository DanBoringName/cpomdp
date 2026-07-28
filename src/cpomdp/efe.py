"""One-step Expected Free Energy (EFE) for the linear-Gaussian regime.

This module computes ``G(a)`` — the Expected Free Energy of taking action ``a``
from the current belief — and its decomposition into a *pragmatic* (goal-seeking)
and an *epistemic* (information-seeking) part. Minimising ``G`` over actions is
how an agent chooses what to do.

================================================================================
THE DECISION THIS FILE ENCODES  (see DECISIONS.md ADR-005)
================================================================================
There is *no single agreed formula* for EFE in the active-inference literature —
the pragmatic term in particular has at least three forms in circulation, and
sources disagree on signs and on whether risk is a cross-entropy or a KL. One
route is chosen here deliberately, and the two questions that most affect what
the objective *does* are settled rather than assumed: the cross-entropy and
KL-risk groupings are the same objective (the derivation is below, and both forms
are reproduced by the tests), and the sign convention follows from that. What
stays genuinely open is flagged ``# FRAGILE(lit):`` — the preference domain, and
the fork between the full pragmatic term and a mean-only one.

--------------------------------------------------------------------------------
THE LOCKED DEFINITION  (decomposition (b): cross-entropy − info-gain)
--------------------------------------------------------------------------------
Given belief ``(μ, Σ)``, action ``a``, model ``(A, B, Q)`` with sensor ``(C, R)``,
and an OBSERVATION-space preference ``(g, Λ)`` (goal observation ``g``, precision
``Λ``):

    predict:    μ⁺ = A·μ + B·a            Σ⁺ = A·Σ·Aᵀ + Q
    sense:      (C, R) = observation.linearize(μ⁺)
                o⁺ = C·μ⁺                  S = C·Σ⁺·Cᵀ + R          # predicted-obs cov
    pragmatic:  ½·(o⁺ − g)ᵀ·Λ·(o⁺ − g)  +  ½·tr(Λ·S)
    epistemic:  ½·(ln det S − ln det R)            # = I(state; obs) ≥ 0, info gain
    G = pragmatic − epistemic                      # minimise: low cost, high info

``S`` is computed once and feeds BOTH terms — there is no n×n work and no Σ_post
or Kalman gain in the one-step EFE (those are only needed for belief propagation
in the H-step rollout, ``policy_efe``). The epistemic identity
``½(ln det Σ⁺ − ln det Σ_post) = ½ ln(det S / det R)`` lets us stay in m×m.

--------------------------------------------------------------------------------
THE FRAGILE CHOICES  (grep: ``FRAGILE(lit)``)
--------------------------------------------------------------------------------
1. Preference domain = OBSERVATIONS, not states. Canonical pymdp/Friston puts
   preferences over outcomes, so this is the faithful choice — but it diverges
   from ADR-003's collapse argument, which is written in state space, and it does
   NOT match the state-space ``goal`` the LQR path currently uses. Reconciling the
   two consumers of ``Preference`` (state-space LQR vs obs-space EFE) is an OPEN
   design point; for C = I (fully observed) they coincide.
2. Pragmatic = FULL form (mean + ½tr(ΛS)), i.e. cross-entropy −E_Q[ln P(o)] up to a
   fixed constant. Cross-entropy paired with −info-gain (as here) and KL-risk
   paired with +ambiguity are the SAME objective — not a behavioural fork; the two
   groupings differ by a constant, and each carries one half of the same entropy
   gap. The genuine literature fork is FULL vs *mean-only* (drop the ½tr(ΛS) term →
   an agent blind to predicted-observation variance/ambiguity). The *forbidden mix*
   (KL-risk pragmatic − info-gain) double-counts H[Q(o)]: a bug, not an option.
3. Epistemic = STATE information gain (salience), not parameter information gain
   (novelty). Only the state info-gain I(state; obs) is computed; parameter/novelty
   EFE is out of scope.
4. The sensor is linearized at μ⁺ (the predicted mean). A fixed sensor is its own
   linearization everywhere, so the point is immaterial there — but for a
   state-dependent ``R(x)`` it is the modelling commitment the whole epistemic term
   rests on: the action moves μ⁺, μ⁺ chooses R, and R sets the information. It is a
   first-order rule, and a plug-in one: it drops the correction for R's curvature
   across the belief's spread.
5. Sign convention: G is MINIMISED; ``pragmatic`` is a cost (lower better) and
   ``epistemic`` is a value (higher better), so G = pragmatic − epistemic.

NOT IMPLEMENTED (named seams): the *mean-only* pragmatic (drops ½tr(ΛS); the real
literature alternative — an ambiguity-blind agent); parameter/novelty info gain.
The KL-risk grouping is NOT a separate option: paired correctly (+ambiguity) it is
this same G; paired with −info-gain it double-counts H[Q(o)] (a bug to avoid).

--------------------------------------------------------------------------------
THE DATA FLOW  (top → bottom: what goes in → what comes out)
--------------------------------------------------------------------------------
    IN ── model=(A,control,Q,sensor)  belief=(μ,Σ)  action=a  preference=(g,Λ)
      │
      ▼    GUARD      control is None?  ──►  raise ValueError
      │
      ▼    PREDICT    μ⁺ = A·μ + control·a     (action enters HERE only)
      │               Σ⁺ = A·Σ·Aᵀ + Q          (this step's Σ⁺ does not see the action)
      │
      ▼    SENSE      (C, R) = linearize(μ⁺)   or fixed (model.C, model.R)
      │               o⁺ = C·μ⁺
      │               S  = C·Σ⁺·Cᵀ + R         (predicted-obs cov; computed ONCE)
      │
      ├──► PRAGMATIC  ½·(o⁺−g)ᵀ·Λ·(o⁺−g) + ½·tr(Λ·S)    (cost,  lower better)
      ├──► EPISTEMIC  ½·(ln det S − ln det R)            (value, higher better)
      │
      ▼    G = pragmatic − epistemic
      │
      ▼    OUT        return (G, {"pragmatic": …, "epistemic": …})

"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float64

from cpomdp.types import Belief, LinearGaussianModel

if TYPE_CHECKING:
    # Type-only import: the kernel reads preference.goal/.precision by duck-typing,
    # so it does NOT depend on selection at runtime. This keeps the dependency
    # one-way (selectors -> kernel) and lets EFESelector import this module.
    from cpomdp.selection import Preference

__all__ = ["expected_free_energy", "policy_efe"]


@dataclass(frozen=True)
class _EfeStep:
    """One EFE step: the public split plus the moments the rollout propagates.

    ``expected_free_energy`` exposes only ``g`` + ``{pragmatic, epistemic}``; the
    H-step rollout (``policy_efe``, B2) additionally needs ``mu_pred``/``sigma_pred``/
    ``s`` (μ⁺, Σ⁺, S) to propagate the belief — but NOT ``C`` (it fetches its own where
    it propagates). Consumed locally; it never crosses a ``jit``/``vmap``/``scan``
    boundary, so it needs no pytree registration.
    """

    g: Float64[Array, ""]
    pragmatic: Float64[Array, ""]
    epistemic: Float64[Array, ""]
    mu_pred: Float64[Array, "n"]
    sigma_pred: Float64[Array, "n n"]
    s: Float64[Array, "m m"]


def expected_free_energy(
    model: LinearGaussianModel,
    belief: Belief,
    action: Float64[Array, "p"],
    preference: "Preference",
) -> tuple[Float64[Array, ""], dict[str, Float64[Array, ""]]]:
    """Expected Free Energy of taking ``action`` from ``belief``, and its split.

    Computes ``G = pragmatic − epistemic`` for the locked linear-Gaussian
    definition documented at the top of this module. Pure ``jnp``, so it composes
    under ``jit``/``vmap``/``grad`` — in particular ``vmap``/``grad`` over a batch
    of candidate ``action`` vectors (with ``model``/``belief``/``preference`` held
    fixed), which is how ``EFESelector`` will search.

    Args:
        model: The generative model. Must have a control matrix (an action has no
            meaning without one). Its ``observation`` supplies the local ``(C, R)``;
            ``None`` means the fixed sensor ``(sensor_model, sensor_noise)``.
        belief: The current belief ``(μ, Σ)``.
        action: The candidate action ``a``, shape ``(p,)``.
        preference: The goal as an OBSERVATION-space ``Preference`` — ``goal`` is a
            preferred observation ``g`` (shape ``(m,)``) and ``precision`` is ``Λ``
            (shape ``(m, m)``). See FRAGILE(lit) #1 in the module docstring.

    Returns:
        ``(G, {"pragmatic": ..., "epistemic": ...})`` — the scalar EFE and its two
        non-negative components. Lower ``G`` is preferred.

    Raises:
        ValueError: If the model has no control matrix.
    """
    if model.control is None:
        raise ValueError(
            "expected_free_energy needs a model with a control matrix; an action "
            "has no effect on a control-free (pure-tracking) model."
        )
    control = model.control  # narrowed to Array by the guard above
    action = jnp.asarray(action, dtype=float)
    step = _efe_step(
        model,
        belief.mean,
        belief.cov,
        control,
        action,
        preference.goal,
        preference.precision,
    )
    return step.g, {"pragmatic": step.pragmatic, "epistemic": step.epistemic}


def policy_efe(
    model: LinearGaussianModel,
    belief: Belief,
    policy: Float64[Array, "H p"],
    preference: "Preference",
) -> tuple[Float64[Array, ""], dict[str, Float64[Array, ""]]]:
    """Summed EFE of a horizon-H ``policy`` from ``belief`` (the rollout seam).

    Internal — ``EFESelector`` searches over it. A ``lax.scan`` over the ``policy``
    rows sums each step's ``G`` while propagating the belief predict-only between
    steps (the mean follows the prediction; the covariance contracts by the Kalman
    update, reusing the moments ``_efe_step`` returns). ``H`` is ``policy.shape[0]``;
    at ``H = 1`` it reduces exactly to ``expected_free_energy``. Composes under
    ``jit`` / ``vmap`` / ``grad``.

    Returns:
        ``(G, {"pragmatic": ..., "epistemic": ...})`` — the summed EFE and its
        summed components over the horizon.
    """
    if model.control is None:
        raise ValueError(
            "policy_efe needs a model with a control matrix; an action has no "
            "effect on a control-free (pure-tracking) model."
        )
    control = model.control
    goal, precision = preference.goal, preference.precision
    policy = jnp.asarray(policy, dtype=float)

    def step(carry, action):
        mu, sigma = carry
        r = _efe_step(model, mu, sigma, control, action, goal, precision)
        # predict-only propagation: rollout fetches its OWN C (the one eval the
        # one-step wrapper never pays), reusing the (Σ⁺, S) r already returned.
        c = (
            model.C
            if model.observation is None
            else model.observation.linearize(r.mu_pred)[0]
        )
        p_xo = r.sigma_pred @ c.T
        sigma_post = r.sigma_pred - p_xo @ jnp.linalg.solve(r.s, p_xo.T)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        return (r.mu_pred, sigma_post), (r.g, r.pragmatic, r.epistemic)

    _, (gs, prags, epis) = lax.scan(step, (belief.mean, belief.cov), policy)
    return jnp.sum(gs), {"pragmatic": jnp.sum(prags), "epistemic": jnp.sum(epis)}


def _logdet_pd(matrix: Float64[Array, "k k"]) -> Float64[Array, ""]:
    """``ln det`` of a covariance, or NaN when it is not positive definite.

    Every determinant the epistemic term takes is of a covariance, so a non-positive
    ``slogdet`` sign means the caller handed over something that is not one — a
    degenerate ``R(x)`` at a reachable state, say. Returning NaN rather than the real
    ``ln|det|`` keeps a meaningless value out of the objective: it propagates to ``G``
    and *loses* the nan-safe argmin at the selection boundary instead of winning it
    with a plausible-looking number.

    The test is a Cholesky factorisation, which succeeds with a strictly positive
    diagonal exactly when the matrix is positive definite. A determinant sign would
    be cheaper but would pass any matrix with an even number of negative eigenvalues
    — ``diag(-1, -2)`` has a positive determinant and is not a covariance.
    """
    chol = jnp.linalg.cholesky(matrix)
    diag = jnp.diagonal(chol)
    definite = jnp.all(diag > 0)
    # The inner guard keeps log() off a non-positive diagonal, so the NaN that comes
    # back is the one this function chose rather than one the logarithm produced.
    safe = jnp.where(definite, diag, 1.0)
    return jnp.where(definite, 2.0 * jnp.sum(jnp.log(safe)), jnp.nan)


def _state_info_gain(
    sigma_pred: Float64[Array, "n n"],
    sensor_model: Float64[Array, "m n"],
    sensor_noise: Float64[Array, "m m"],
    target: Sequence[int],
) -> Float64[Array, ""]:
    """Information gain about a target block of the state from one observation.

    The entropy drop of the target latent's marginal — how much observing sharpens the
    belief about *those* state indices (issue #26). Unlike the whole-state
    observation-space shortcut in ``_efe_step`` (``½(ln det S − ln det R)``), this forms
    the posterior covariance explicitly so a sub-block can be read out::

        Σ_post = (Σ⁺⁻¹ + Cᵀ·R⁻¹·C)⁻¹   # prior info + observation info, inverted back
        gain   = ½·(ln det Σ⁺[target] − ln det Σ_post[target])

    With ``target`` the whole state this equals the observation-space epistemic (via
    Sylvester's identity); restricting ``target`` to a node's indices gives info gain
    about that latent — the factored epistemics ADR-014 finding #3 needs.

    Args:
        sigma_pred: Σ⁺, the predicted joint state covariance (n x n).
        sensor_model: C, the observation matrix (m x n).
        sensor_noise: R, the observation noise covariance (m x m).
        target: the state indices whose marginal the info gain is about.

    Returns:
        The scalar information gain (nats), ≥ 0 for nested target/observation; NaN
        where ``R`` is not positive definite, matching ``_efe_step``.
    """
    prior_info = jnp.linalg.inv(sigma_pred)  # Σ⁺⁻¹
    obs_info = sensor_model.T @ jnp.linalg.inv(sensor_noise) @ sensor_model  # Cᵀ R⁻¹ C
    sigma_post = jnp.linalg.inv(prior_info + obs_info)

    idx = jnp.asarray(list(target))
    gain = 0.5 * (
        _logdet_pd(sigma_pred[jnp.ix_(idx, idx)])
        - _logdet_pd(sigma_post[jnp.ix_(idx, idx)])
    )
    # A non-positive-definite R still inverts to something finite above, so the two
    # blocks can both come back positive definite and the gain look reasonable. The
    # noise has to be checked on its own for the guard to mean anything.
    return jnp.where(jnp.isnan(_logdet_pd(sensor_noise)), jnp.nan, gain)


def _ffg_efe_step(
    mu_plus: Float64[Array, "n"],
    sigma_plus: Float64[Array, "n n"],
    sensor_model: Float64[Array, "m n"],
    sensor_noise: Float64[Array, "m m"],
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
    target: Sequence[int],
) -> tuple[Float64[Array, ""], dict[str, Float64[Array, ""]]]:
    """One EFE step over the FFG's predicted joint, with a node-targeted epistemic.

    The FFG counterpart of ``_efe_step`` (issue #26). It takes the predicted joint
    moments directly — ``μ⁺``/``Σ⁺`` from ``CouplingGraphBackend.predicted_belief``,
    which already fold in the structural couplings — rather than recomputing
    ``Σ⁺ = A·Σ·Aᵀ + Q`` from a flat model (that route mistakes the couplings for
    observations). The ``pragmatic`` term is identical to ``_efe_step``
    (observation-space cross-entropy); the ``epistemic`` term is the only change — info
    gain about the ``target`` latent's marginal (``_state_info_gain``) instead of the
    whole-state observation-space determinant.

    With no couplings and ``target`` the whole state, this reproduces ``_efe_step`` /
    ``expected_free_energy`` exactly; a node's block targets the epistemic at that
    latent — the factored analogue of the T-Maze cue (ADR-014 finding #3). Pure
    ``jnp``, so it rides ``jit``/``vmap`` over a grid of candidate actions (which vary
    ``μ⁺``, while ``Σ⁺`` at this one step does not see the action (ADR-003)). What the
    action does reach is ``R`` at ``μ⁺``, and through it the posterior.

    Args:
        mu_plus: μ⁺, the predicted joint mean under the candidate action (n-D).
        sigma_plus: Σ⁺, the predicted joint covariance (n x n) — couplings included.
        sensor_model: C, the real observation matrix over the joint state (m x n).
        sensor_noise: R, the observation noise covariance (m x m).
        goal: g, the preferred observation (m-D).
        precision: Λ, how sharply the goal is preferred (m x m).
        target: the state indices whose info gain is the epistemic value (a node's
            block, or the whole state).

    Returns:
        ``(G, {"pragmatic": ..., "epistemic": ...})`` — the scalar EFE ``G = pragmatic −
        epistemic`` (minimised) and its two parts.
    """
    o_pred = sensor_model @ mu_plus  # o⁺ = C·μ⁺
    s = sensor_model @ sigma_plus @ sensor_model.T + sensor_noise  # S = C·Σ⁺·Cᵀ + R

    residual = o_pred - goal
    pragmatic = 0.5 * residual @ precision @ residual + 0.5 * jnp.trace(precision @ s)

    epistemic = _state_info_gain(sigma_plus, sensor_model, sensor_noise, target)

    return pragmatic - epistemic, {"pragmatic": pragmatic, "epistemic": epistemic}


def _efe_step(
    model: LinearGaussianModel,
    mu: Float64[Array, "n"],
    sigma: Float64[Array, "n n"],
    control: Float64[Array, "n p"],
    action: Float64[Array, "p"],
    goal: Float64[Array, "m"],
    precision: Float64[Array, "m m"],
) -> _EfeStep:
    # --- predict: push the belief one step through the dynamics under `action` ---
    # Mirrors the covariance predict in kalman._gain_and_posterior_cov (cov_pred);
    # NB within THIS step the action moves only the mean, so Σ⁺ here does not see it —
    # the whole reason the epistemic term collapses under a fixed sensor (ADR-003). The
    # independence is local: a state-dependent R(x) or Q(x) makes the noise, and so the
    # posterior this step hands the next one, depend on where the action put the mean.
    # Over a horizon the covariance does move with the policy; only the one-step
    # predicted covariance is blind to it.
    mu_pred = model.A @ mu + control @ action
    process_q = (
        model.Q
        if model.process_noise is None
        else model.process_noise.noise_at(mu_pred)
    )

    sigma_pred = model.A @ sigma @ model.A.T + process_q  # Σ⁺ = AΣAᵀ + process_q

    # --- sense: predicted-observation moments (o⁺, S) + conditional noise R at μ⁺ ---
    # The sensor owns its moment-matching (D1): the kernel never reconstructs o⁺/S.
    # FRAGILE(lit) #4: everything is evaluated at μ⁺. Immaterial for a fixed sensor,
    # which is its own linearization everywhere; for a state-dependent R(x) this point
    # is what the action reaches, and so what the epistemic term ends up measuring.
    if model.observation is None:
        # FAST PATH — a bare matvec/matmul, byte-identical to Phase 1A. Kept inline
        # (no method dispatch) so the fixed-sensor hot path stays lean.
        sensor_model, sensor_noise = model.C, model.R
        o_pred = sensor_model @ mu_pred
        pred_obs_cov = sensor_model @ sigma_pred @ sensor_model.T + sensor_noise
    else:
        # Linear sensors return exact (C·μ⁺, C·Σ⁺·Cᵀ+R, R); NonlinearSensor (2.5)
        # returns its 2nd-order moments. S feeds the pragmatic term, R the epistemic.
        o_pred, pred_obs_cov, sensor_noise = model.observation.gaussianize(
            mu_pred, sigma_pred
        )

    # --- pragmatic: expected negative log-preference (cross-entropy form) ---
    # FRAGILE(lit) #1: `preference` is read in OBSERVATION space (g over o, Λ over o).
    # FRAGILE(lit) #2: cross-entropy form = mean term + ½tr(ΛS). The ½tr(ΛS) piece is
    # the variance penalty that distinguishes this from the mean-only form; against the
    # KL-risk form it differs only by −½ln det S, which the epistemic term restores.
    residual = o_pred - goal
    pragmatic_mean = 0.5 * residual @ precision @ residual
    pragmatic_var = 0.5 * jnp.trace(precision @ pred_obs_cov)
    pragmatic = pragmatic_mean + pragmatic_var

    # --- epistemic: state information gain I(state; obs) = ½ ln(det S / det R) ---
    # FRAGILE(lit) #3: this is *salience* (state info gain), not *novelty* (parameter
    # info gain). `_logdet_pd` carries the positive-definiteness guard, so a degenerate
    # R(x) at a reachable state yields NaN rather than a plausible-but-wrong finite
    # value; `_state_info_gain` guards the same way, so both epistemic routes agree.
    epistemic = 0.5 * (_logdet_pd(pred_obs_cov) - _logdet_pd(sensor_noise))

    # FRAGILE(lit) #5: G = pragmatic − epistemic (minimise). Pairing cross-entropy
    # with −info-gain is decomposition (b); it is self-consistent (no double-count).
    g = pragmatic - epistemic
    return _EfeStep(
        g=g,
        pragmatic=pragmatic,
        epistemic=epistemic,
        mu_pred=mu_pred,
        sigma_pred=sigma_pred,
        s=pred_obs_cov,
    )
