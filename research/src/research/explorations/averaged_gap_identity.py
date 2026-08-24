"""Is the Gaussian averaged inference gap a closed form, and over what hypotheses?

`E_{y∼p*}[ KL(q ‖ p(x|y)) ]` is what R6 is a claim about. Under a fixed `R` both `q`
and the exact posterior are Gaussian, and the whole functional collapses to algebra.
That collapse is the candidate for a symbolic `PROVED` check, so the form has to be
established before anything asserts it, not after.

Three independent routes, because a formula confirmed by the code that produced it is
confirmed by nothing:

- **Quadrature.** The scalar candidate against `scipy.integrate.quad` over the true
  predictive, at parameter triples spanning four decades in each of `sigma^2`,
  `R_true` and `R_plug`.
- **Monte Carlo.** The multivariate generalisation against sampled innovations, for a
  general observation matrix `C` including a non-square one, which is what decides
  whether the identity is scalar-only or general.
- **Symbolic.** `sympy`, for exactness and for the two structural properties a
  numerical check cannot establish: that the expression vanishes identically at
  `R_plug = R_true`, and that it is not merely small there.

The answer, recorded because it bounds what may be claimed: the identity holds for a
general `C` and any state dimension, and it is exact rather than asymptotic. Its
non-negativity is *not* established here. That is Gibbs' inequality, a pen-and-paper
theorem, and `sympy` returns `None` when asked, so a check claiming non-negativity
would be claiming the wrong prover.

Run it::

    uv run --no-sync python -m research.explorations.averaged_gap_identity
"""

import numpy as np
import sympy as sp
from scipy.integrate import quad

__all__ = [
    "MULTIVARIATE_CASES",
    "averaged_gap",
    "averaged_gap_by_quadrature",
    "averaged_gap_general",
    "main",
    "symbolic_gap",
]


def averaged_gap(
    true_noise: float, plugin_noise: float, prior_variance: float
) -> float:
    """The scalar candidate: `E_y[KL(q ‖ p)]` for a unit observation matrix.

    Both posteriors are Gaussian with the same prior and different plugged-in noise,
    so they differ only in gain and variance. The mean term is quadratic in `y - mu`
    and its expectation under the true predictive is the innovation variance.
    """
    gain = prior_variance / (prior_variance + true_noise)
    plugin_gain = prior_variance / (prior_variance + plugin_noise)
    exact_variance = (1.0 - gain) * prior_variance
    approximate_variance = (1.0 - plugin_gain) * prior_variance
    innovation_variance = prior_variance + true_noise  # S
    return (
        0.5 * np.log(exact_variance / approximate_variance)
        + (approximate_variance + (plugin_gain - gain) ** 2 * innovation_variance)
        / (2.0 * exact_variance)
        - 0.5
    )


def averaged_gap_by_quadrature(
    true_noise: float,
    plugin_noise: float,
    prior_variance: float,
    prior_mean: float = 0.37,
) -> float:
    """The same quantity by direct quadrature over `y`, sharing no algebra with it."""
    innovation_variance = prior_variance + true_noise
    gain = prior_variance / innovation_variance
    plugin_gain = prior_variance / (prior_variance + plugin_noise)
    exact_variance = (1.0 - gain) * prior_variance
    approximate_variance = (1.0 - plugin_gain) * prior_variance

    def integrand(y: float) -> float:
        exact_mean = prior_mean + gain * (y - prior_mean)
        approximate_mean = prior_mean + plugin_gain * (y - prior_mean)
        divergence = (
            0.5 * np.log(exact_variance / approximate_variance)
            + (approximate_variance + (approximate_mean - exact_mean) ** 2)
            / (2.0 * exact_variance)
            - 0.5
        )
        density = np.exp(-0.5 * (y - prior_mean) ** 2 / innovation_variance) / np.sqrt(
            2.0 * np.pi * innovation_variance
        )
        return divergence * density

    half_width = 40.0 * np.sqrt(innovation_variance)
    value, _ = quad(
        integrand,
        prior_mean - half_width,
        prior_mean + half_width,
        epsabs=1e-14,
        epsrel=1e-14,
        limit=800,
    )
    return value


def averaged_gap_general(
    prior_cov: np.ndarray,
    observation_matrix: np.ndarray,
    true_noise: np.ndarray,
    plugin_noise: np.ndarray,
) -> float:
    """The multivariate candidate, for a general (possibly non-square) `C`.

    The mean term becomes `tr(Sigma_p^-1 (K' - K) S (K' - K)^T)`, since the innovation
    has covariance `S` under the true predictive.
    """
    n = prior_cov.shape[0]
    innovation_cov = observation_matrix @ prior_cov @ observation_matrix.T + true_noise
    gain = prior_cov @ observation_matrix.T @ np.linalg.inv(innovation_cov)
    plugin_gain = (
        prior_cov
        @ observation_matrix.T
        @ np.linalg.inv(
            observation_matrix @ prior_cov @ observation_matrix.T + plugin_noise
        )
    )
    exact_cov = (np.eye(n) - gain @ observation_matrix) @ prior_cov
    approximate_cov = (np.eye(n) - plugin_gain @ observation_matrix) @ prior_cov
    exact_precision = np.linalg.inv(exact_cov)
    gain_difference = plugin_gain - gain
    return 0.5 * (
        np.log(np.linalg.det(exact_cov) / np.linalg.det(approximate_cov))
        + np.trace(exact_precision @ approximate_cov)
        + np.trace(
            exact_precision @ gain_difference @ innovation_cov @ gain_difference.T
        )
        - n
    )


def _general_by_monte_carlo(
    prior_cov: np.ndarray,
    observation_matrix: np.ndarray,
    true_noise: np.ndarray,
    plugin_noise: np.ndarray,
    prior_mean: np.ndarray,
    draws: int = 4_000_000,
    seed: int = 1,
) -> tuple[float, float]:
    """The same by sampling innovations, with the estimate's standard error."""
    n = prior_cov.shape[0]
    innovation_cov = observation_matrix @ prior_cov @ observation_matrix.T + true_noise
    gain = prior_cov @ observation_matrix.T @ np.linalg.inv(innovation_cov)
    plugin_gain = (
        prior_cov
        @ observation_matrix.T
        @ np.linalg.inv(
            observation_matrix @ prior_cov @ observation_matrix.T + plugin_noise
        )
    )
    exact_cov = (np.eye(n) - gain @ observation_matrix) @ prior_cov
    approximate_cov = (np.eye(n) - plugin_gain @ observation_matrix) @ prior_cov
    exact_precision = np.linalg.inv(exact_cov)

    rng = np.random.default_rng(seed)
    innovations = rng.multivariate_normal(
        np.zeros(observation_matrix.shape[0]), innovation_cov, size=draws
    )
    mean_offsets = innovations @ (plugin_gain - gain).T
    quadratic = np.einsum("di,ij,dj->d", mean_offsets, exact_precision, mean_offsets)
    constant = 0.5 * (
        np.log(np.linalg.det(exact_cov) / np.linalg.det(approximate_cov))
        + np.trace(exact_precision @ approximate_cov)
        - n
    )
    return constant + 0.5 * quadratic.mean(), 0.5 * quadratic.std() / np.sqrt(draws)


# Three shapes of observation matrix, because a scalar-only identity and a general one
# are different claims and only the second would licence a general `PROVED` row.
MULTIVARIATE_CASES: tuple[
    tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...
] = (
    (
        "square C, correlated R",
        np.array([[1.4, 0.6], [0.6, 0.9]]),
        np.array([[1.0, 0.5], [0.0, 1.0]]),
        np.array([[0.5, 0.15], [0.15, 0.8]]),
        np.array([[1.2, -0.2], [-0.2, 0.4]]),
    ),
    (
        "partial observation, 1x2 C",
        np.array([[1.4, 0.6], [0.6, 0.9]]),
        np.array([[1.0, 0.0]]),
        np.array([[0.3]]),
        np.array([[0.9]]),
    ),
    (
        "scaled non-axis-aligned C",
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        np.array([[2.5, -1.0]]),
        np.array([[0.7]]),
        np.array([[0.2]]),
    ),
)


def symbolic_gap() -> sp.Expr:
    """The scalar candidate as a `sympy` expression in `(sigma^2, R_true, R_plug)`."""
    prior_variance, true_noise, plugin_noise = sp.symbols(
        "sigma2 R_true R_plug", positive=True
    )
    innovation_variance = prior_variance + true_noise
    gain = prior_variance / innovation_variance
    plugin_gain = prior_variance / (prior_variance + plugin_noise)
    exact_variance = (1 - gain) * prior_variance
    approximate_variance = (1 - plugin_gain) * prior_variance
    return (
        sp.Rational(1, 2) * sp.log(exact_variance / approximate_variance)
        + (approximate_variance + (plugin_gain - gain) ** 2 * innovation_variance)
        / (2 * exact_variance)
        - sp.Rational(1, 2)
    )


def main() -> None:
    """Run all three routes and assert what each establishes."""
    print("=== scalar candidate against scipy quadrature ===")
    rng = np.random.default_rng(0)
    triples = [
        (
            float(10 ** rng.uniform(-2, 2)),
            float(10 ** rng.uniform(-2, 2)),
            float(10 ** rng.uniform(-2, 2)),
        )
        for _ in range(40)
    ]
    triples += [
        (1.0, 1.0, 1.0),
        (1.0, 0.5, 0.5),
        (1e-3, 1.0, 1e3),
        (1e3, 1e-3, 1e-3),
        (2.0, 0.5, 0.25),
    ]
    worst = 0.0
    for prior_variance, true_noise, plugin_noise in triples:
        closed = averaged_gap(true_noise, plugin_noise, prior_variance)
        integrated = averaged_gap_by_quadrature(
            true_noise, plugin_noise, prior_variance
        )
        worst = max(worst, abs(closed - integrated) / max(abs(integrated), 1e-300))
    print(f"  {len(triples)} triples, worst relative disagreement {worst:.3e}")
    assert worst < 1e-11, worst

    print("\n=== multivariate generalisation against Monte Carlo ===")
    for (
        name,
        prior_cov,
        observation_matrix,
        true_noise,
        plugin_noise,
    ) in MULTIVARIATE_CASES:
        closed = averaged_gap_general(
            prior_cov, observation_matrix, true_noise, plugin_noise
        )
        sampled, standard_error = _general_by_monte_carlo(
            prior_cov, observation_matrix, true_noise, plugin_noise, np.zeros(2)
        )
        print(
            f"  {name:28s} closed {closed:.9f}  sampled {sampled:.9f} "
            f"+/- {3 * standard_error:.1e}"
        )
        assert abs(closed - sampled) < 4 * standard_error + 1e-12, name

    print("\n=== symbolic ===")
    expression = symbolic_gap()
    true_noise, plugin_noise = sp.symbols("R_true R_plug", positive=True)
    at_the_truth = sp.simplify(expression.subs(plugin_noise, true_noise))
    print(f"  vanishes identically at R_plug = R_true: {at_the_truth == 0}")
    assert at_the_truth == 0
    non_negative = sp.ask(sp.Q.nonnegative(sp.simplify(sp.expand(expression))))
    print(f"  sympy decides non-negativity: {non_negative}")
    # Recorded, not worked around. Non-negativity is Gibbs' inequality, prover 1, and a
    # symbolic check claiming it would be claiming the wrong prover.
    assert non_negative is None

    print("\n  the identity is exact, and general in C and in the state dimension;")
    print("  its non-negativity is a theorem and is not established here.")


if __name__ == "__main__":
    main()
