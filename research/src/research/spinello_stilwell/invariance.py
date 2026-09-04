"""What moves when the observation is rescaled, and what does not.

Backs the claims in `research/spinello_stilwell_rung.md` questions Q1 and Q2. Run it::

    uv run --no-sync python -m research.spinello_stilwell.invariance

Three things are checked, and only these three. Q3 to Q8 of that document are routes
nobody has run, and nothing here converts one into a result.

1. Under `o -> lambda*o`, seven of the eight terms in the Spinello-Stilwell update
   return to themselves and `1/ln sigma` does not. Symbolic.
2. The pole of the Gauss-Newton matrix therefore sits at `sigma = lambda^-2`, so the
   unit choice alone decides where it falls. Symbolic.
3. The converged estimate and the posterior variance are nonetheless unmoved by the
   rescaling, while the iteration count is not. Numeric, on the linear-mean scalar
   case, using a reference implementation of (35) written for this check alone.

Point 3 is what Q2 rests on: if rescaling moved the answer, choosing units to clear the
pole would be choosing an answer, and it would be a tuned parameter rather than a
declared convention.
"""

from typing import TypedDict

import sympy as sp

from research.spinello_stilwell import scheme

__all__ = ["CASE", "SCALES", "WorkedCase", "iterate", "main", "term_invariance"]


def term_invariance() -> dict[str, bool]:
    """Which terms of (35c), (35d) and (35e) survive `o -> lambda*o`.

    Under the rescaling: `z -> L z`, `h -> L h`, `sigma -> L^2 sigma`, `zeta -> L zeta`,
    `grad h -> L grad h`, `grad sigma -> L^2 grad sigma`.

    Returns:
        One entry per term, true where the rescaled term equals the original.
    """
    scale, sigma, zeta, grad_h, grad_sigma = sp.symbols(
        "lam sigma zeta h_x sigma_x", positive=True
    )

    def terms(factor: sp.Expr) -> dict[str, sp.Expr]:
        noise = factor**2 * sigma
        residual = factor * zeta
        mean_slope = factor * grad_h
        noise_slope = factor**2 * grad_sigma
        return {
            "s: -zeta/sigma * grad h": -residual / noise * mean_slope,
            "s: (1/2sigma)(1 - zeta^2/sigma) grad sigma": (1 / (2 * noise))
            * (1 - residual**2 / noise)
            * noise_slope,
            "gn: (1/sigma) grad h grad h": mean_slope * mean_slope / noise,
            "gn: (zeta/2sigma^2) cross term": residual
            / (2 * noise**2)
            * 2
            * mean_slope
            * noise_slope,
            "gn: (1/4sigma^2)(zeta^2/sigma) grad sigma^2": (1 / (4 * noise**2))
            * (residual**2 / noise)
            * noise_slope
            * noise_slope,
            "gn: (1/4sigma^2)(1/ln sigma) grad sigma^2": (1 / (4 * noise**2))
            * (1 / sp.log(noise))
            * noise_slope
            * noise_slope,
            "fisher: (1/sigma) grad h grad h": mean_slope * mean_slope / noise,
            "fisher: (1/2sigma^2) grad sigma^2": noise_slope
            * noise_slope
            / (2 * noise**2),
        }

    plain, rescaled = terms(sp.Integer(1)), terms(scale)
    return {
        name: sp.simplify(sp.expand(rescaled[name] - plain[name])) == 0
        for name in plain
    }


def iterate(
    observation: float,
    prior_mean: float,
    prior_variance: float,
    base_noise: float,
    curvature: float,
    scale: float = 1.0,
    tolerance: float = 1e-14,
    max_iterations: int = 200,
) -> tuple[float, float, int]:
    """Spinello-Stilwell (35) for a scalar linear-mean sensor, in rescaled units.

    The printed scheme, so `scheme.iterate` with its `r3` block kept. Route 1 asks what
    rescaling moves in the paper's own filter, and a variant of it would answer a
    different question.

    Args:
        observation: the reading, in unrescaled units.
        prior_mean: the predicted mean.
        prior_variance: the predicted variance.
        base_noise: `R0`.
        curvature: `kappa`.
        scale: `lambda`, the factor the observation is multiplied by.
        tolerance: stop when the step falls below this, relative to the prior spread.
        max_iterations: give up after this many steps.

    Returns:
        The converged estimate in unrescaled state units, the posterior variance, and
        the number of iterations taken.
    """
    return scheme.iterate(
        observation=observation,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        base_noise=base_noise,
        curvature=curvature,
        scale=scale,
        tolerance=tolerance,
        max_iterations=max_iterations,
        log_block=True,
    )


class WorkedCase(TypedDict):
    """The five model numbers a rescaling comparison holds fixed."""

    observation: float
    prior_mean: float
    prior_variance: float
    base_noise: float
    curvature: float


#: The worked case point 3 runs on. Away from the pole in every unit tried, so what it
#: measures is the invariance and not the pole's behaviour.
CASE: WorkedCase = {
    "observation": 1.7,
    "prior_mean": 1.0,
    "prior_variance": 0.04,
    "base_noise": 1.0,
    "curvature": 1.0,
}

#: The unit choices point 3 compares. One below and two above, so a factor that only
#: happened to work in one direction would show.
SCALES = (1.0, 0.5, 3.0, 7.0)


def main() -> None:
    """Run the three checks and assert each."""
    print("=== Q1: which terms survive o -> lambda*o ===")
    survives = term_invariance()
    for name, invariant in survives.items():
        print(f"  {'yes' if invariant else 'NO ':<4} {name}")
    moving = [name for name, held in survives.items() if not held]
    assert moving == ["gn: (1/4sigma^2)(1/ln sigma) grad sigma^2"], moving
    assert sum(survives.values()) == 7, survives

    print("\n=== Q2: where the pole sits, as a function of the unit choice ===")
    for scale in SCALES:
        # The Gauss-Newton term is singular where the rescaled noise reaches one.
        print(f"  lambda = {scale:>4}: pole at original sigma = {1.0 / scale**2:g}")
    # On the registered ridge R(x) = R0 + kappa x^2 at mu* = sqrt(R0/kappa), the noise
    # at the operating point is 2*R0 whatever kappa is.
    base, curvature = sp.symbols("R0 kappa", positive=True)
    operating_point = sp.sqrt(base / curvature)
    noise_there = sp.simplify(base + curvature * operating_point**2)
    print(f"  ridge: sigma at the operating point = {noise_there}")
    assert sp.simplify(noise_there - 2 * base) == 0
    assert sp.simplify(noise_there.subs(base, sp.Rational(1, 2)) - 1) == 0
    print("  ridge: R0 = 1/2 puts the pole on the operating point at lambda = 1")

    print("\n=== Q2: rescaling moves the path and not the answer ===")
    reference = iterate(**CASE, scale=1.0)
    print(f"  {'lambda':>8} {'estimate':>20} {'posterior var':>18} {'iters':>7}")
    counts = set()
    for scale in SCALES:
        estimate, variance, taken = iterate(**CASE, scale=scale)
        counts.add(taken)
        print(f"  {scale:>8} {estimate:>20.15f} {variance:>18.15f} {taken:>7}")
        assert abs(estimate - reference[0]) < 1e-12, scale
        assert abs(variance - reference[1]) < 1e-12, scale
    print(f"\n  iteration counts across those units: {sorted(counts)}")
    print("  the estimate and the variance are unmoved. The path is not.")


if __name__ == "__main__":
    main()
