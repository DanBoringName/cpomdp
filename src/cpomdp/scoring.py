"""Scoring a declared cross of models against a declared cross of ways to infer.

The two axes live in ``cpomdp.constructors``: one for what a model gets
wrong about the process, one for what its filter gets wrong about the model. A run
crosses them. Every separation reported from such a run is a comparison between two of
its cells, which is a decision only if the cross was visited in full.

So the cross carries a certificate. `|model axis| x |inference axis|` is a product, and
a product needs its factors the way a policy tree needs its base and exponent: 12 is
``3 x 4`` and ``2 x 6``, so a bare count names neither which axes were crossed nor which
version of each was declared. `ProductCompletenessCertificate` carries both, which is
what makes two runs over different crosses tellable apart (standing prohibition 9).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from cpomdp.backends.base import InferenceBackend
from cpomdp.constructors import ConstructorSet, InferenceSet, ModelSpec
from cpomdp.enumeration import IncompleteEnumerationError
from cpomdp.types import Belief, LinearGaussianModel
from warrantlib import AxisDeclaration, ProductCompletenessCertificate, Warrant

__all__ = [
    "INFERENCE_AXIS",
    "MODEL_AXIS",
    "ConstructorCross",
    "CrossCell",
    "Decomposition",
    "build_cross",
    "gaussian_kl",
    "misspecification_step",
    "observation_predictive",
]

#: What each axis is called on the certificate. Declared here rather than written at the
#: construction site, so the two runs a ledger joins cannot disagree about the names.
MODEL_AXIS = "model"
INFERENCE_AXIS = "inference"


@dataclass(frozen=True)
class Decomposition:
    """The two divergences of the three-term accounting, floor excluded.

    ``E[F] = H(p*) + misspecification + inference gap``. The floor is a property of
    ``p*`` and no business of a scored cell, so there is no entropy field, no estimator
    slot and no total. A term reached by subtracting ``H(p*)`` cannot be represented
    here (standing prohibition 1).

    Args:
        misspecification: ``D_KL[p*(y|u) ‖ p(y|u)]`` in nats, zero iff the cell's model
            is ``p*``.
        inference_gap: ``E_{y∼p*}[ D_KL[q ‖ p(x|y)] ]`` in nats, zero iff the cell
            infers exactly under its own model.
    """

    misspecification: float
    inference_gap: float


@dataclass(frozen=True)
class CrossCell:
    """One cell: a model built one way, inferred another.

    ``model`` is the thing the model-axis parts name, so it carries no ``model_``
    prefix of its own. The inference axis has no such centre, so both its parts take
    the prefix.

    Args:
        model_name: The perturbation that built ``model``, as the model axis
            declared it.
        inference_name: The rule that built ``inference_backend``.
        model: The model this cell is scored under.
        inference_backend: The filter this cell infers with. It reports ``model`` as
            its own, degraded or not.
    """

    model_name: str
    inference_name: str
    model: LinearGaussianModel
    inference_backend: InferenceBackend


@dataclass(frozen=True)
class ConstructorCross:
    """Every cell of the two axes, and the evidence that all of them were built.

    Args:
        cells: The cells, model-major in declaration order, so a row is one model
            inferred every declared way and a column is one way of inferring across
            every declared model.
        certificate: That the product of the declared axis sizes was reached.
    """

    cells: tuple[CrossCell, ...]
    certificate: ProductCompletenessCertificate


def build_cross(
    spec: ModelSpec, models: ConstructorSet, rules: InferenceSet
) -> ConstructorCross:
    """Build every cell of the two axes, and certify that none was missed.

    One model per row, shared across that row's rules. The axes are independent: a
    perturbation is one model inferred several ways, and rebuilding it per cell would
    put a difference on the inference axis that belongs to neither.

    Args:
        spec: The parameters the model axis perturbs.
        models: The declared model axis.
        rules: The declared inference axis.

    Returns:
        The cells and their certificate.

    Raises:
        IncompleteEnumerationError: if fewer cells were built than the two axes
            declare. The count is carried through the loop rather than read off the
            result, so a cell lost on the way out is a failure here instead of a
            shorter run in which every remaining cell still passes.
        ValueError: if a cell cannot be built. A rule may refuse the model its row
            supplies, as a frozen gain refuses state-dependent noise, and a
            perturbation may leave no valid covariance. The cell that refused is named,
            since the message the axes raise names only what it was asked to build.
        RuntimeError: if a frozen gain's steady-state recursion does not converge, from
            the same place and named the same way.
    """
    expected = models.size * rules.size
    cells: list[CrossCell] = []
    visited = 0
    for model_name, model in models.build_all(spec):
        for inference_name, backend in _built_over(model, rules, model_name):
            cells.append(
                CrossCell(
                    model_name=model_name,
                    inference_name=inference_name,
                    model=model,
                    inference_backend=backend,
                )
            )
            visited += 1

    if visited != expected:  # ADR-030: the certificate is asserted, not assumed
        # Rendered by a certificate rather than by hand, so the failure and the
        # evidence describe the cross in one voice. CORROBORATED because a shortfall
        # is what it says it is; a PROVED one would refuse to construct here.
        sampled = _certificate(models, rules, expected=expected, visited=visited)
        raise IncompleteEnumerationError(
            f"the cross visited {visited} of an expected {expected} cells over "
            f"{sampled.set_description}; the run would be a sample, not a decision."
        )
    return ConstructorCross(
        cells=tuple(cells),
        certificate=_certificate(
            models, rules, expected=expected, visited=visited, warrant=Warrant.PROVED
        ),
    )


def _built_over(
    model: LinearGaussianModel, rules: InferenceSet, model_name: str
) -> tuple[tuple[str, InferenceBackend], ...]:
    """One row's backends, with the cell named on anything the row refuses.

    `InferenceSet.build_all` names the rule that refused and cannot name the row, since
    it was handed a model and not the perturbation that produced one. On a cross that
    is half the address.
    """
    try:
        return rules.build_all(model)
    except (ValueError, RuntimeError) as refusal:
        raise type(refusal)(
            f"the cross could not build a cell on row {model_name!r}: {refusal}"
        ) from refusal


def _certificate(
    models: ConstructorSet,
    rules: InferenceSet,
    *,
    expected: int,
    visited: int,
    warrant: Warrant = Warrant.CORROBORATED,
) -> ProductCompletenessCertificate:
    """The cross's certificate, with both axes declared as the sets declared them."""
    return ProductCompletenessCertificate(
        expected=expected,
        visited=visited,
        warrant=warrant,
        axes=(
            AxisDeclaration(name=MODEL_AXIS, size=models.size, version=models.version),
            AxisDeclaration(
                name=INFERENCE_AXIS, size=rules.size, version=rules.version
            ),
        ),
    )


def gaussian_kl(
    mean_a: ArrayLike, cov_a: ArrayLike, mean_b: ArrayLike, cov_b: ArrayLike
) -> float:
    """``D_KL[N(mean_a, cov_a) ‖ N(mean_b, cov_b)]`` in nats, as non-negative parts.

    The textbook form subtracts ``n`` from a trace and one log-determinant from
    another. Near equality those are differences of like-sized numbers, so a
    divergence that is truly zero reads as rounding noise of either sign, and a
    reading below ``1e-12`` then says nothing about which of two Gaussians moved
    (``research/warrant_ledger.md`` section 4). Here every part is a square or a
    ``λ − 1 − ln λ`` over the eigenvalues of ``cov_b⁻¹ · cov_a``, each of which is
    non-negative on its own. Nothing is subtracted, so a small result is a small
    divergence rather than a cancellation.

    Args:
        mean_a: The divergence's first argument's mean, shape ``(n,)``.
        cov_a: Its covariance, shape ``(n, n)``, positive definite.
        mean_b: The second argument's mean, shape ``(n,)``.
        cov_b: Its covariance, shape ``(n, n)``, positive definite.

    Returns:
        The divergence, ``0.0`` exactly when the two Gaussians are identical.

    Raises:
        ValueError: On a shape mismatch, or if either covariance is not positive
            definite. A degenerate Gaussian has no density for the divergence to read.
    """
    mean_a, mean_b = np.asarray(mean_a, dtype=float), np.asarray(mean_b, dtype=float)
    cov_a, cov_b = np.asarray(cov_a, dtype=float), np.asarray(cov_b, dtype=float)
    if mean_a.ndim != 1 or mean_a.shape != mean_b.shape:
        raise ValueError(
            f"the two means must be vectors of one length, got shapes "
            f"{mean_a.shape} and {mean_b.shape}"
        )
    expected = (mean_a.shape[0], mean_a.shape[0])
    if cov_a.shape != expected or cov_b.shape != expected:
        raise ValueError(
            f"the two covariances must be {expected} to match the means, got shapes "
            f"{cov_a.shape} and {cov_b.shape}"
        )
    factor_a = _cholesky_or_refuse(cov_a, "cov_a")
    factor_b = _cholesky_or_refuse(cov_b, "cov_b")

    # Singular values of L_b⁻¹ L_a squared are the eigenvalues of cov_b⁻¹ cov_a.
    ratios = np.linalg.svd(np.linalg.solve(factor_b, factor_a), compute_uv=False) ** 2
    shape_term = _excess_over_log(ratios - 1.0)
    shift = np.linalg.solve(factor_b, mean_a - mean_b)
    return float(0.5 * (shape_term.sum() + shift @ shift))


# Below this, ``x − log1p(x)`` is itself a cancellation: the two agree to nearly every
# digit, and the series has no such subtraction.
_SERIES_BELOW = 1e-4


def _excess_over_log(excess: np.ndarray) -> np.ndarray:
    """``x − ln(1 + x)`` elementwise, non-negative for ``x > −1``, without cancelling.

    Near zero the direct form subtracts two numbers that agree to almost every digit.
    The series ``x²/2 − x³/3 + x⁴/4 − x⁵/5 + x⁶/6`` evaluates the same function there
    with a truncation error below ``x⁷/7``, which at the switch is ``1e-29``.
    """
    small = np.abs(excess) < _SERIES_BELOW
    x = np.where(small, excess, 0.0)
    series = x**2 / 2 - x**3 / 3 + x**4 / 4 - x**5 / 5 + x**6 / 6
    return np.where(small, series, excess - np.log1p(excess))


def _cholesky_or_refuse(cov: np.ndarray, name: str) -> np.ndarray:
    """The lower Cholesky factor, or a ``ValueError`` naming the argument."""
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as failure:
        raise ValueError(
            f"{name} must be positive definite; a degenerate Gaussian has no density "
            f"for a divergence to read"
        ) from failure


def observation_predictive(
    model: LinearGaussianModel, belief: Belief, action: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray]:
    """The mean and covariance of ``p(y | u)`` one step ahead under ``model``.

    The belief is pushed through the dynamics and then through the sensor::

        mean = C · (A · μ + B · u)
        cov  = C · (A · Σ · Aᵀ + Q) · Cᵀ + R

    Gaussian because everything in the model is, which is why this asks for fixed
    noise. Under a state-dependent ``R(x)`` or ``Q(x)`` the predictive is a scale
    mixture and has no covariance that describes it.

    Args:
        model: The model doing the predicting.
        belief: The state belief before the step, over the model's ``n`` states.
        action: The action applied over the step, shape ``(p,)``. ``None`` for a model
            without a control matrix.

    Returns:
        ``(mean, cov)`` of the predicted observation, shapes ``(m,)`` and ``(m, m)``.

    Raises:
        ValueError: If the model carries state-dependent noise, if the model has a
            control matrix and ``action`` is ``None``, or if the belief is not over
            the model's state.
    """
    _require_fixed_noise(model)
    mean = np.asarray(belief.mean, dtype=float)
    cov = np.asarray(belief.cov, dtype=float)
    if mean.shape != (model.n_states,):
        raise ValueError(
            f"the belief is over {mean.shape[0]} states and the model has "
            f"{model.n_states}"
        )
    dynamics_matrix = np.asarray(model.dynamics_matrix, dtype=float)  # A
    observation_matrix = np.asarray(model.observation_matrix, dtype=float)  # C
    mean_pred = dynamics_matrix @ mean
    if model.control_matrix is not None:
        if action is None:
            raise ValueError(
                "this model has a control matrix; the step needs an action"
            )
        mean_pred = mean_pred + np.asarray(
            model.control_matrix, dtype=float
        ) @ np.asarray(action, dtype=float)
    cov_pred = dynamics_matrix @ cov @ dynamics_matrix.T + np.asarray(
        model.dynamics_noise, dtype=float
    )
    return (
        observation_matrix @ mean_pred,
        observation_matrix @ cov_pred @ observation_matrix.T
        + np.asarray(model.observation_noise, dtype=float),
    )


def misspecification_step(
    truth: LinearGaussianModel,
    truth_belief: Belief,
    model: LinearGaussianModel,
    model_belief: Belief,
    action: ArrayLike | None,
) -> float:
    """``D_KL[p*(y|u) ‖ p(y|u)]`` for one step, in nats.

    Each side predicts from its own exact belief, so the term reads what the two
    models say about the next reading and nothing about how either is being filtered.
    A cell that infers badly under the true model scores zero here at every step,
    which is what keeps the two axes separable (battery C2, C3).

    Args:
        truth: ``p*``, the process the observations come from.
        truth_belief: The exact filter's belief under ``truth`` before this step.
        model: ``p``, the cell's model.
        model_belief: The exact filter's belief under ``model`` before this step.
        action: The action applied over the step, common to both.

    Returns:
        The divergence. ``0.0`` exactly when the two models agree in value and their
        beliefs do.
    """
    true_mean, true_cov = observation_predictive(truth, truth_belief, action)
    model_mean, model_cov = observation_predictive(model, model_belief, action)
    return gaussian_kl(true_mean, true_cov, model_mean, model_cov)


def _require_fixed_noise(model: LinearGaussianModel) -> None:
    """Refuse a model whose noise varies with the state.

    The closed forms here are Gaussian ones. Under ``R(x)`` the exact posterior and
    the predictive are not Gaussian, and scoring them needs the reference filter
    rather than a formula.
    """
    for slot in ("observation_model", "dynamics_noise_model"):
        carried = getattr(model, slot)
        if carried is not None and not carried.is_fixed:
            raise ValueError(
                f"the model's {slot} is state-dependent; the closed-form terms here "
                f"are for fixed noise only"
            )
