"""Checkable conditions on a model's sensor, over the states an action can reach.

Whether a state-dependent ``R(x)`` actually buys an agent anything is not a property of
the noise function alone — it depends on where the control can put the predicted mean.
A noise that varies only in a corner no policy reaches is, for every purpose the filter
and the objective have, a constant. The conditions here are the ones that separate the
two cases, and each is reported against a *sampled reachable set*: the predicted means
of a set of candidate actions from a shared belief.

Four conditions, in the order they bite:

``full_row_rank``
    ``C`` carries no redundant channels, so ``C Cᵀ`` is invertible. Matching a
    posterior covariance then identifies the noise uniquely; without it, only the
    noise's action on the row space of ``C`` is identified, and covariances off that
    subspace are free.

``definite``
    ``R(x) ≻ 0`` at every sampled mean. The epistemic term inverts ``R`` and takes its
    log-determinant, so anywhere this fails the term is not defined — the kernel returns
    NaN there rather than a plausible-looking number.

``non_constant``
    ``R`` takes different values across the sampled means. This is what makes the
    posterior covariance, and so the Kalman gain, depend on the action.

``epistemic_varies``
    The per-step epistemic value differs across those means. Stronger than
    ``non_constant``: the epistemic term is a log-determinant, a single scalar summary,
    and a covariance can move along a contour of it without the scalar noticing. For a
    scalar observation the two coincide, since the map is then strictly monotone in the
    variance. ``loewner_comparable`` reports the cheap sufficient condition — noise
    covariances ordered by ``≺`` always separate the determinant.

Sampling is what it is: these answer "over the actions I asked about", not "over every
reachable state". A negative is evidence, not proof.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import jax
import numpy as np
from jaxtyping import Float64
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    # Annotation-only: every diagnostic here reads its argument's fields by duck typing,
    # so diagnostics depends on none of these at runtime (they already depend on nothing
    # here). Keeps the dependency one-way.
    from cpomdp.efe import PolicyEfeTrace
    from cpomdp.types import Belief, LinearGaussianModel


class ProbeBackend(Protocol):
    """The three members ``probe_model`` reads off a graph backend.

    Structural rather than a name, because three members is what the function requires.
    Any backend growing them works, and the diagnostic keeps its one-way dependency on
    the backends package. ``CouplingGraphBackend`` satisfies it.
    """

    @property
    def observation_model(self) -> tuple[jax.Array, jax.Array]:
        """``(C, R)`` over the joint state."""
        ...

    def predicted_belief(
        self, prior: "Belief", action: ArrayLike | None = None
    ) -> "Belief":
        """The belief one prediction step ahead under ``action``."""
        ...

    def observation_noise_at(self, mean: ArrayLike) -> Float64[jax.Array, "m m"]:
        """The stacked ``R`` linearized at a predicted mean."""
        ...


__all__ = [
    "ProbeBackend",
    "RolloutConditioning",
    "SensorReport",
    "epistemic_value",
    "is_positive_definite",
    "loewner_order",
    "probe_model",
    "rollout_conditioning",
]


def is_positive_definite(matrix: ArrayLike) -> bool:
    """Whether ``matrix`` is symmetric positive definite.

    A Cholesky factorisation, which succeeds exactly on the positive-definite
    matrices — rather than a determinant sign, which passes anything with an even
    number of negative eigenvalues.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.allclose(m, m.T, atol=1e-12):
        return False
    try:
        np.linalg.cholesky(m)
    except np.linalg.LinAlgError:
        return False
    return True


def epistemic_value(
    predicted_cov: ArrayLike, sensor_model: ArrayLike, sensor_noise: ArrayLike
) -> float:
    """The per-step epistemic value ``½(ln det S − ln det R)``, in nats.

    The mutual information between state and observation under the predictive joint,
    with ``S = C Σ⁻ Cᵀ + R`` the innovation covariance. Computed here on the host for
    reporting; the objective's own copy lives in ``cpomdp.efe``.

    Returns NaN where ``R`` or ``S`` is not positive definite, so an undefined value
    stays visibly undefined.
    """
    cov = np.asarray(predicted_cov, dtype=float)
    c = np.asarray(sensor_model, dtype=float)
    r = np.asarray(sensor_noise, dtype=float)
    s = c @ cov @ c.T + r
    if not (is_positive_definite(r) and is_positive_definite(s)):
        return float("nan")
    return 0.5 * float(np.linalg.slogdet(s)[1] - np.linalg.slogdet(r)[1])


def loewner_order(a: ArrayLike, b: ArrayLike) -> str:
    """Compare two covariances in the Loewner order.

    Returns ``"a<b"`` when ``b − a`` is positive definite, ``"b<a"`` for the reverse,
    ``"equal"`` when they agree, and ``"incomparable"`` when neither difference is
    definite. A strictly comparable pair is enough to separate the epistemic value, so
    only incomparable pairs need the determinant consulted.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if np.allclose(x, y, atol=1e-12):
        return "equal"
    if is_positive_definite(y - x):
        return "a<b"
    if is_positive_definite(x - y):
        return "b<a"
    return "incomparable"


@dataclass(frozen=True)
class SensorReport:
    """What the sampled reachable set says about a model's sensor.

    Attributes:
        n_samples: how many predicted means were probed.
        rank: the rank of ``C``.
        n_observations: the number of observation channels, ``C``'s row count.
        full_row_rank: whether ``C`` has no redundant channels.
        definite: whether ``R`` was positive definite at every sampled mean.
        indefinite_at: the sampled means where it was not.
        non_constant: whether ``R`` took more than one value across the samples.
        noise_spread: the largest pairwise distance between the sampled ``R``.
        epistemic_varies: whether the epistemic value differed across the samples.
        epistemic_range: its smallest and largest sampled values, in nats.
        loewner_comparable: whether the extreme pair of sampled ``R`` is strictly
            ordered — the cheap sufficient condition for the epistemic value to move.
    """

    n_samples: int
    rank: int
    n_observations: int
    full_row_rank: bool
    definite: bool
    indefinite_at: tuple[tuple[float, ...], ...]
    non_constant: bool
    noise_spread: float
    epistemic_varies: bool
    epistemic_range: tuple[float, float]
    loewner_comparable: bool

    @property
    def flattens(self) -> bool:
        """Whether the sampled evidence says the sensor is a constant in disguise.

        True when ``R`` never moved across the sampled means: the covariance recursion
        then never consults the action, so a fixed noise schedule reproduces the agent
        and the epistemic term cannot tell two policies apart.
        """
        return not self.non_constant

    def summary(self) -> str:
        """A few lines a human can read, one per condition."""
        lo, hi = self.epistemic_range
        mark = {True: "yes", False: "NO "}
        return "\n".join(
            [
                f"sampled {self.n_samples} predicted mean(s)",
                f"  full row rank C   {mark[self.full_row_rank]}  "
                f"(rank {self.rank} of {self.n_observations})",
                f"  R positive def.   {mark[self.definite]}  "
                f"({len(self.indefinite_at)} failing sample(s))",
                f"  R non-constant    {mark[self.non_constant]}  "
                f"(spread {self.noise_spread:.3e})",
                f"  epistemic varies  {mark[self.epistemic_varies]}  "
                f"({lo:.6f} .. {hi:.6f} nats)",
                f"  Loewner-ordered   {mark[self.loewner_comparable]}",
            ]
        )

    def __str__(self) -> str:
        """The same text ``summary`` returns."""
        return self.summary()


def _linearizations(
    observation, sensor_model: np.ndarray, means: np.ndarray
) -> list[np.ndarray]:
    """Each mean's noise covariance, read through whatever the caller supplied."""
    if observation is None:  # a fixed sensor carries no per-state noise
        return []
    return [np.asarray(observation.linearize(mu)[1], dtype=float) for mu in means]


def probe_model(
    model: "LinearGaussianModel | ProbeBackend",
    belief: "Belief",
    actions: Sequence[ArrayLike],
    *,
    tol: float = 1e-12,
) -> SensorReport:
    """Probe a model's sensor over the predicted means a set of actions reaches.

    Each action is pushed through the prediction step to get ``(μ⁻, Σ⁻)``; the sensor is
    linearized at each ``μ⁻`` and the four conditions are evaluated over the resulting
    sample. This is the reachable set the conditions are about — vary ``actions`` to
    widen it.

    Args:
        model: a ``LinearGaussianModel``, or a backend exposing
            ``predicted_belief`` / ``observation_noise_at`` / ``observation_model``
            (``CouplingGraphBackend`` does).
        belief: the belief to predict from — the shared prior every action starts at.
        actions: the candidate actions to sample.
        tol: below this, two noise covariances or two epistemic values count as equal.

    Returns:
        A ``SensorReport``.

    Raises:
        ValueError: If ``actions`` is empty.
    """
    if len(actions) == 0:
        raise ValueError("probe_model needs at least one action to predict under.")

    if hasattr(model, "predicted_belief"):  # a graph backend
        backend = cast("ProbeBackend", model)
        sensor_model = np.asarray(backend.observation_model[0], dtype=float)
        predicted = [backend.predicted_belief(belief, np.asarray(a)) for a in actions]
        means = np.asarray([np.asarray(p.mean, dtype=float) for p in predicted])
        covs = [np.asarray(p.cov, dtype=float) for p in predicted]
        noises = [
            np.asarray(backend.observation_noise_at(mu), dtype=float) for mu in means
        ]
    else:  # a flat LinearGaussianModel
        sensor_model = np.asarray(model.sensor_model, dtype=float)
        dynamics = np.asarray(model.dynamics, dtype=float)
        prior_mean = np.asarray(belief.mean, dtype=float)
        prior_cov = np.asarray(belief.cov, dtype=float)
        # A control-free model goes nowhere under any action, which is itself an answer:
        # every sample lands on the same predicted mean and the noise cannot move.
        if model.control is None:
            means = np.asarray([dynamics @ prior_mean for _ in actions])
        else:
            control = np.asarray(model.control, dtype=float)
            means = np.asarray(
                [
                    dynamics @ prior_mean + control @ np.asarray(a, dtype=float).ravel()
                    for a in actions
                ]
            )
        cov = dynamics @ prior_cov @ dynamics.T + np.asarray(
            model.dynamics_noise, dtype=float
        )
        covs = [cov] * len(actions)
        fixed = np.asarray(model.sensor_noise, dtype=float)
        noises = _linearizations(model.observation, sensor_model, means) or [
            fixed
        ] * len(actions)

    rank = int(np.linalg.matrix_rank(sensor_model))
    n_obs = int(sensor_model.shape[0])

    indefinite = tuple(
        tuple(float(v) for v in mu)
        for mu, r in zip(means, noises, strict=True)
        if not is_positive_definite(r)
    )

    spread = 0.0
    for i in range(len(noises)):
        for j in range(i + 1, len(noises)):
            spread = max(spread, float(np.max(np.abs(noises[i] - noises[j]))))

    epistemics = [
        epistemic_value(cov, sensor_model, r)
        for cov, r in zip(covs, noises, strict=True)
    ]
    finite = [e for e in epistemics if np.isfinite(e)]
    lo, hi = (min(finite), max(finite)) if finite else (float("nan"), float("nan"))

    # The extreme pair is the one most likely to be ordered; a strictly ordered pair is
    # enough on its own, so there is no need to walk every combination.
    comparable = False
    if finite and len(noises) > 1:
        i_lo = int(np.argmin([e if np.isfinite(e) else np.inf for e in epistemics]))
        i_hi = int(np.argmax([e if np.isfinite(e) else -np.inf for e in epistemics]))
        comparable = loewner_order(noises[i_lo], noises[i_hi]) in ("a<b", "b<a")

    return SensorReport(
        n_samples=len(actions),
        rank=rank,
        n_observations=n_obs,
        full_row_rank=rank == n_obs,
        definite=not indefinite,
        indefinite_at=indefinite,
        non_constant=spread > tol,
        noise_spread=spread,
        epistemic_varies=bool(finite) and (hi - lo) > tol,
        epistemic_range=(lo, hi),
        loewner_comparable=comparable,
    )


@dataclass(frozen=True)
class RolloutConditioning:
    """Per-step conditioning of an H-step rollout, computed on the host.

    Every array field is length-H, indexed by rollout step. The condition numbers and
    the smallest eigenvalue are what a monotonically contracting ``Σ_post`` degrades
    first; ``all_positive_definite`` folds a per-step positive-definiteness check over
    ``Σ⁺``, ``S`` and ``Σ_post`` into one flag, so a matrix that has gone non-PD
    anywhere in the horizon is visible without inspecting every step.

    Attributes:
        cond_sigma_pred: 2-norm condition number of each predicted covariance ``Σ⁺``.
        cond_s: condition number of each innovation covariance ``S``.
        cond_sigma_post: condition number of each contracted covariance ``Σ_post``.
        min_eig_sigma_post: smallest eigenvalue of each ``Σ_post``.
        all_positive_definite: whether every ``Σ⁺``, ``S`` and ``Σ_post`` is PD.
    """

    cond_sigma_pred: np.ndarray
    cond_s: np.ndarray
    cond_sigma_post: np.ndarray
    min_eig_sigma_post: np.ndarray
    all_positive_definite: bool


def rollout_conditioning(trace: "PolicyEfeTrace") -> RolloutConditioning:
    """Per-step condition numbers and PD flags of a rollout trace, on the host.

    Reads the per-step matrices ``policy_efe_trace`` already returns and applies the
    condition-number discipline to the rollout. The assertions a caller builds on top —
    a ceiling on the condition numbers, a floor under ``min_eig_sigma_post`` — belong
    here on the host, outside any ``jit``/``vmap``, which is the only place they can
    raise. Pure NumPy.

    Args:
        trace: a ``PolicyEfeTrace`` — read for its ``sigma_pred``
            (``Σ⁺``), ``s`` (``S``) and ``sigma_post`` (``Σ_post``) columns.

    Returns:
        A ``RolloutConditioning``.
    """
    sigma_pred = np.asarray(trace.sigma_pred, dtype=float)
    s = np.asarray(trace.s, dtype=float)
    sigma_post = np.asarray(trace.sigma_post, dtype=float)

    return RolloutConditioning(
        cond_sigma_pred=np.array([np.linalg.cond(m) for m in sigma_pred]),
        cond_s=np.array([np.linalg.cond(m) for m in s]),
        cond_sigma_post=np.array([np.linalg.cond(m) for m in sigma_post]),
        min_eig_sigma_post=np.array([np.linalg.eigvalsh(m).min() for m in sigma_post]),
        all_positive_definite=all(
            is_positive_definite(m)
            for column in (sigma_pred, s, sigma_post)
            for m in column
        ),
    )
