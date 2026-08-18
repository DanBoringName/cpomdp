"""The gap's expansion coefficients, symbolically. `c₂` and nothing beyond it.

The inference gap at a fixed observation is a cumulant difference::

    KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]

Expanded in prior spread and averaged over the innovation, its leading coefficient is
`c₂`. The registration derived that coefficient independently, in closed form, as
`(R'(μ)/2R(μ))²`. This module derives it again from the series and checks the two
against each other. That is the `EXACT` licence exactly: agreement of two independently
computed closed forms.

**`ORDER` is 2, and it is a module constant.** The quartic is the next piece of work and
it is not started here. Nothing in this module fits, extracts or guesses a `c₄`. A
number produced before the derivation lands becomes the thing the derivation is then
checked against, and the ledger ends up carrying a fit under a Prover 1 label.
`gap_expansion --c4` is the refutation route, and it reads a residual exponent rather
than producing a candidate.

**The predictive is the leading-order `N(0, R̄)`.** Correct at `σ²` and wrong at `σ⁴`,
where the exact `ν = σz₁ + √R̄·e^{δ/2}·z₂` stops collapsing to it. The replacement
belongs to the work that needs it. Doing it early would put an unchecked object under a
checked result.

Run it::

    uv run --no-sync python -m research.checks.gap_series --check
    uv run --no-sync python -m research.checks.gap_series

Symbolic throughout, on free `l₁..l₄` and a symbolic `R̄`. The construction lives in
``series_kernel``; ``log_ratio_series`` pins its structure; this module asks it for
coefficients.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from functools import cache

import sympy
from research.checks.series_kernel import (
    EXPANSION_SOURCE,
    L1,
    L2,
    L3,
    L4,
    NU,
    RBAR,
    SIGMA,
    cumulants,
    exp_series,
    gaussian_expectation,
    log_ratio_in_sigma,
    predictive_expectation,
    report_condition,
    report_identity,
    truncate,
)

from cpomdp.warrant import CheckReport, Outcome, check_summary

__all__ = [
    "averaged_gap",
    "fixed_observation_gap",
    "gap_from_definition",
    "run_checks",
]

#: The working order in `σ`. Everything below is written against it rather than against
#: the number 2, so the quartic work is this constant plus new stages.
ORDER = 2

#: Where the closed form `c₂` was derived independently of this series.
REGISTRATION_SOURCE = (
    "research/gate_d4_registration.md, RESULT 2026-08-07: c2 = (R'(mu)/2R(mu))^2"
)

#: Where the cumulant statement of the gap is hand derived.
CUMULANT_SOURCE = (
    "research/c4_hand_derivation.md, Step 4 "
    "(the gap as half the variance at leading order)"
)

#: A free positive scale, for the covariance check. `R → a·R` leaves every
#: log-derivative alone, since `log(aR) = log a + log R`, and moves `R̄` alone.
SCALE = sympy.Symbol("a", positive=True)


def _log_one_plus(small: sympy.Expr, order: int) -> sympy.Expr:
    """`log(1 + u)` for a `u` of positive order in `σ`, truncated.

    Args:
        small: `u`, carrying no `σ⁰` term.
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The logarithm's expansion.
    """
    power = sympy.Integer(1)
    total = sympy.Integer(0)
    for step in range(1, order + 1):
        power = truncate(power * small, order)
        total += (-1) ** (step + 1) * power / step
    return truncate(total, order)


# Cached on the same terms as the kernel's expansions, keyed on the tilt and the working
# order. The tilt symbols are built with identical assumptions at both call sites, so
# sympy hands back the same object and the cache hits across them.
@cache
def cumulant_generating(tilt: sympy.Expr, order: int) -> sympy.Expr:
    """`Λ(t) = log E_q[e^{tW}]`, expanded in `σ`.

    Args:
        tilt: `t`, the exponential tilt. Symbolic where a derivative in it is wanted.
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The cumulant generating function's expansion.
    """
    log_ratio = log_ratio_in_sigma(order)
    tilted = exp_series(truncate(tilt * log_ratio, order), order)
    return _log_one_plus(truncate(gaussian_expectation(tilted) - 1, order), order)


@cache
def gap_from_definition(order: int) -> sympy.Expr:
    """The gap at a fixed observation, straight from `log E_q[e^W] − E_q[W]`.

    The definition, with no cumulant identity used. It is the arm the variance form is
    checked against, so it must not be derived from it.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `KL(q ‖ p(·|y))` as a polynomial in `σ`, still carrying `ν`.
    """
    tilt = sympy.Symbol("t", real=True)
    generating = cumulant_generating(tilt, order)
    return truncate(
        generating.subs(tilt, 1) - sympy.diff(generating, tilt).subs(tilt, 0), order
    )


@cache
def fixed_observation_gap(order: int) -> sympy.Expr:
    """The gap at a fixed observation, as half the variance of `W`.

    The leading-order statement: `KL = ½·Var_q(W) + O(κ₃)`. Built from the kernel's
    cumulant recursion rather than from the definition above, so the two stay
    independent.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `½·Var_q(W)` as a polynomial in `σ`, still carrying `ν`.
    """
    return truncate(cumulants(log_ratio_in_sigma(order), 2, order)[2] / 2, order)


@cache
def averaged_gap(order: int) -> sympy.Expr:
    """The gap averaged over the innovation, at leading order in the predictive.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `E_{p*}[KL]` as a polynomial in `σ`, free of `ν`.
    """
    return predictive_expectation(fixed_observation_gap(order), order)


def check_gap_is_half_the_variance() -> list[CheckReport]:
    """C1: the definition and the variance form agree at `σ²`.

    Two derivations of the same object, neither asserted from the other. One expands
    `log E_q[e^W] − E_q[W]` directly. The other takes `κ₂/2` from the cumulant
    recursion. They part company at the third cumulant, which is why the claim is
    scoped to `σ²` and not stated generally.

    **What the agreement does not cover.** Neither arm calls the other, and that is all
    the independence claimed. Both read the same `W` from `log_ratio_in_sigma` and both
    take their expectations through the same `gaussian_expectation`. An error in either
    shared piece moves the two arms together and this check stays green. It separates
    the cumulant route from the generating-function route, not the kernel they share.

    Returns:
        The report, as a one-item list.
    """
    definition = gap_from_definition(ORDER)
    variance = fixed_observation_gap(ORDER)
    return [
        report_identity(
            name=f"C1 gap is half the variance at σ^{ORDER}",
            claim=f"log E_q[e^W] − E_q[W] = ½·Var_q(W) through σ^{ORDER}",
            correspondence=CUMULANT_SOURCE,
            residual=definition - variance,
            shown=(
                f"definition = {sympy.factor(definition)}, "
                f"½·Var = {sympy.factor(variance)}"
            ),
        )
    ]


def check_fixed_observation_coefficient() -> list[CheckReport]:
    """C2: the `σ²` coefficient at fixed `y`, its sign, and its `ν`-dependence.

    `l₁²(R̄ − ν²)²/(8R̄²)`. Non-negative because it is a square over a positive, which
    is a statement about the expression rather than a sample of its values. Still a
    function of `ν`, which records that the average has not happened yet: a coefficient
    that had already lost `ν` would be `c₂` arriving one step early and unnoticed.

    Returns:
        The value report, the sign report and the dependence report.
    """
    coefficient = sympy.expand(fixed_observation_gap(ORDER).coeff(SIGMA, 2))
    claim = L1**2 * (RBAR - NU**2) ** 2 / (8 * RBAR**2)
    as_square = (L1 * (RBAR - NU**2)) ** 2 / (8 * RBAR**2)
    return [
        report_identity(
            name="C2 fixed-y σ² coefficient",
            claim="[σ²] KL(y) = l₁²(R̄ − ν²)²/(8R̄²)",
            correspondence=EXPANSION_SOURCE,
            residual=coefficient - claim,
            shown=f"{sympy.factor(coefficient)}",
        ),
        report_identity(
            name="C2 non-negative by construction",
            claim="the coefficient is a square over 8R̄², and R̄ is declared positive",
            correspondence=EXPANSION_SOURCE,
            residual=coefficient - as_square,
            shown=f"(l₁(R̄ − ν²))²/(8R̄²), with R̄ positive: {RBAR.is_positive}",
        ),
        report_condition(
            name="C2 still depends on ν",
            claim="the y-average has not happened: the coefficient still varies with ν",
            correspondence=EXPANSION_SOURCE,
            holds=sympy.simplify(sympy.diff(coefficient, NU)) != 0,
            shown=f"d/dν = {sympy.factor(sympy.diff(coefficient, NU))}",
        ),
    ]


def check_innovation_average() -> list[CheckReport]:
    """C3: `E[(ν²/R̄ − 1)²] = 2` under `ν ~ N(0, R̄)`.

    The whole content of the `y`-average at this order, isolated from the gap so that a
    failure lands on the moment arithmetic rather than on the expansion.

    Returns:
        The report, as a one-item list.
    """
    integrand = (NU**2 / RBAR - 1) ** 2
    averaged = predictive_expectation(integrand, ORDER)
    return [
        report_identity(
            name="C3 innovation average",
            claim="E[(ν²/R̄ − 1)²] = 2 under ν ~ N(0, R̄)",
            correspondence=(
                "the fourth and second moments of a centred Gaussian, which "
                "series_kernel checks against the Gaussian integral"
            ),
            residual=averaged - 2,
            shown=f"E[(ν²/R̄ − 1)²] = {averaged}",
        )
    ]


def check_c2_against_the_registration() -> list[CheckReport]:
    """C4: `c₂ = l₁²/4`, against the closed form derived before this series existed.

    The registration has `c₂ = (R'(μ)/2R(μ))²`, and `R'/R` is `l₁` by definition, so the
    two forms are the same statement reached by different routes. That is what `EXACT`
    licenses: agreement of two independently computed closed forms, not a numerical
    coincidence at a tolerance.

    Returns:
        The value report and the `ν`-freedom report.
    """
    coefficient = sympy.expand(averaged_gap(ORDER).coeff(SIGMA, 2))
    return [
        report_identity(
            name="C4 c₂ closed form",
            claim="c₂ = l₁²/4, which is (R'(μ)/2R(μ))² since l₁ = R'(μ)/R(μ)",
            correspondence=REGISTRATION_SOURCE,
            residual=coefficient - L1**2 / 4,
            shown=f"c₂ = {coefficient}",
        ),
        report_condition(
            name="C4 c₂ is free of ν",
            claim="the averaged coefficient carries no innovation",
            correspondence=REGISTRATION_SOURCE,
            holds=NU not in coefficient.free_symbols,
            shown=f"free symbols: {sorted(str(s) for s in coefficient.free_symbols)}",
        ),
    ]


def check_direction_independence() -> list[CheckReport]:
    """C5: reverse and forward KL agree at `σ²`, and the claim stops there.

    With `Λ(t) = log E_q[e^{tW}]`, the reverse direction is `Λ(1) − Λ'(0)` and the
    forward one is `Λ'(1) − Λ(1)`. Both are `½κ₂` at leading order, so the choice of
    direction cannot be what produces `c₂`.

    **Not asserted beyond `σ²`.** They diverge from the third cumulant on, and that
    divergence is the whole reason the pinned conventions matter at `σ⁴`. A check that
    quietly held at higher order would be recording a coincidence of truncation.

    Returns:
        The agreement report and the scope report.
    """
    tilt = sympy.Symbol("t", real=True)
    generating = cumulant_generating(tilt, ORDER)
    reverse = truncate(
        generating.subs(tilt, 1) - sympy.diff(generating, tilt).subs(tilt, 0), ORDER
    )
    forward = truncate(
        sympy.diff(generating, tilt).subs(tilt, 1) - generating.subs(tilt, 1), ORDER
    )
    return [
        report_identity(
            name=f"C5 direction independence at σ^{ORDER}",
            claim="reverse KL = forward KL through σ², so c₂ is direction-free",
            correspondence=CUMULANT_SOURCE,
            residual=reverse - forward,
            shown=f"reverse = {sympy.factor(reverse)}",
        ),
        report_condition(
            name="C5 scoped to σ²",
            claim="the agreement is asserted at σ² only, since κ₃ separates them",
            correspondence=CUMULANT_SOURCE,
            holds=ORDER == 2,
            shown=f"ORDER = {ORDER}",
        ),
    ]


def check_scale_covariance(
    expression: sympy.Expr, name: str, coefficient_order: int = 2
) -> list[CheckReport]:
    """C6: the gap is unchanged under `R → a·R`, for a free positive `a`.

    `log(aR) = log a + log R`, so every log-derivative is untouched and `R̄` moves
    alone.
    The innovation moves with it, `ν ~ N(0, R̄)` becoming `N(0, aR̄)`, so `ν → √a·ν` is
    part of the same substitution rather than an extra assumption. What survives is the
    dimensionless `ν²/R̄`, which is why no term may carry a bare `R̄`.

    Written against a supplied expression rather than the `σ²` one, so the quartic
    inherits the guard instead of needing a second copy of it.

    Args:
        expression: the gap expression to test.
        name: what to call it in the report.
        coefficient_order: which power of `σ` to read the coefficient from.

    Returns:
        The invariance report.
    """
    coefficient = sympy.expand(expression.coeff(SIGMA, coefficient_order))
    scaled = coefficient.subs(
        {RBAR: SCALE * RBAR, NU: sympy.sqrt(SCALE) * NU}, simultaneous=True
    )
    return [
        report_identity(
            name=f"C6 scale covariance [{name}]",
            claim=f"[σ^{coefficient_order}] is unchanged under R → aR with ν → √a·ν",
            correspondence=EXPANSION_SOURCE,
            residual=sympy.simplify(scaled - coefficient),
            shown=f"under R → aR: {sympy.factor(sympy.simplify(scaled))}",
        )
    ]


def check_exponential_family() -> list[CheckReport]:
    """C7: for `R = A·e^{bx}`, `c₂` reduces to `b²/4`.

    Every log-derivative above the first vanishes, so `l₂ = l₃ = l₄ = 0` and the only
    surviving parameter is `b`. Trivial at `σ²`, which is the point of wiring it now: at
    `σ⁴` the registration's basis predicts `c₄ = α·b⁴ + ζ·b²/R(μ)`, so exactly two of
    seven terms may survive and this reduction becomes a sharp check on which ones.

    Returns:
        The report, as a one-item list.
    """
    rate = sympy.Symbol("b", real=True)
    flattened = {L1: rate, L2: 0, L3: 0, L4: 0}
    coefficient = sympy.expand(averaged_gap(ORDER).coeff(SIGMA, 2)).subs(flattened)
    return [
        report_identity(
            name="C7 exponential family",
            claim="for R = A·e^{bx}, c₂ = b²/4",
            correspondence=(
                "research/gate_d4_registration.md, section 2: for R = A·e^{bx} every "
                "log-derivative above the first vanishes"
            ),
            residual=coefficient - rate**2 / 4,
            shown=f"c₂|(l₂=l₃=l₄=0) = {coefficient}",
        )
    ]


def run_checks() -> list[CheckReport]:
    """Run every coefficient check, in the order the derivation needs them.

    Returns:
        Every check's report.
    """
    stages: Sequence[Callable[[], list[CheckReport]]] = (
        check_gap_is_half_the_variance,
        check_fixed_observation_coefficient,
        check_innovation_average,
        check_c2_against_the_registration,
        check_direction_independence,
        check_exponential_family,
    )
    reports = [report for stage in stages for report in stage()]
    reports += check_scale_covariance(fixed_observation_gap(ORDER), "fixed y")
    reports += check_scale_covariance(averaged_gap(ORDER), "averaged")
    return reports


def _print_setup() -> None:
    """Print the coefficients the checks are about, for the bare run."""
    fixed = fixed_observation_gap(ORDER)
    averaged = averaged_gap(ORDER)
    print(f"R̄ symbolic, l₁..l₄ free, ORDER = {ORDER}. No family chosen, no numbers.\n")
    print(f"W(σ)        = {log_ratio_in_sigma(ORDER)}")
    print(f"KL(y)       = {sympy.factor(fixed)}")
    print(f"[σ²] KL(y)  = {sympy.factor(fixed.coeff(SIGMA, 2))}")
    print(f"E_p*[KL]    = {sympy.factor(averaged)}")
    print(f"c₂          = {sympy.expand(averaged.coeff(SIGMA, 2))}")
    print(
        "\nNo quartic here. c₄ is the next piece of work, and a number produced "
        "\nbefore its derivation would become the thing the derivation is checked "
        "\nagainst."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every identity holds, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="The gap's expansion coefficients, symbolically. c₂ and no further."
    )
    parser.add_argument("--check", action="store_true", help="run the check suite")
    arguments = parser.parse_args(argv)

    if not arguments.check:
        _print_setup()
        return 0

    reports = run_checks()
    for report in reports:
        print(report)
    print(f"\n{check_summary(reports)}")
    if any(report.outcome is Outcome.FIRED for report in reports):
        print(
            "\nA fired check means the fault is in the kernel or the construction, "
            "not in the target. Fix it there; do not adjust c₂."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
