"""The gap's expansion coefficients, symbolically: `c₂` and `c₄`.

The inference gap at a fixed observation is a cumulant difference::

    KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]  =  κ₂/2 + κ₃/6 + κ₄/24 + …

Expanded in prior spread and averaged over the innovation, its coefficients are `c₂` and
`c₄`. The registration derived `c₂` independently, in closed form, as `(R'(μ)/2R(μ))²`,
and this module derives it again from the series. That is the `EXACT` licence exactly:
agreement of two independently computed closed forms.

`c₄` had no closed form before this module produced one. It had a **fit**, at
`COMPUTED`,
and a registered seven-term basis with two coefficients reported consistent with zero
and five conjectured fractions. Those are predictions, dated earlier than this file, and
the checks below test the derivation against them rather than describing what it found.

**The derivation ran before its registration.** `research/gate_d4_registration.md`
section 7 discloses the sequence at its head. The content is unaffected and the
scheduling is not, and a reader is entitled to judge them separately.

**Nothing here is fitted.** No floats and no numeric value for `R̄`. `ℓ₁..ℓ₄` are free
symbols throughout the derivation, so there is no quantity a measured number could be
substituted into. That is what makes the agreement with the fit evidence rather than
circularity, and it is checkable by reading the module.

One check does choose a family, and it is a consequence rather than an input. C7 sets
`ℓ₂ = ℓ₃ = ℓ₄ = 0` to specialise the *derived* general form to `R = A·e^{bx}`. The
substitution happens after the coefficient exists, never before it, so it cannot supply
the answer it checks.

**The predictive is the exact one.** `ν = σz₁ + √R̄·e^{δ/2}·z₂` is the generative
process written out, not a model of it. Collapsing it to `N(0, R̄)` leaves `c₂` alone
and changes `c₄` by a factor of several, which C9 measures. `predictive_truncation` had
already established why: `p*` is a scale mixture with exponential tails.

Run it::

    uv run --no-sync python -m research.checks.gap_series --check
    uv run --no-sync python -m research.checks.gap_series

The construction lives in ``series_kernel``; ``log_ratio_series`` pins its structure;
this module asks it for coefficients.
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
    Source,
    cumulants,
    exact_predictive_expectation,
    exp_series,
    gaussian_expectation,
    log_ratio_in_sigma,
    predictive_expectation,
    report_condition,
    report_identity,
    truncate,
)
from warrantlib import CheckReport, Outcome, Provenance, check_summary

__all__ = [
    "averaged_gap",
    "basis_coefficients",
    "cumulant_terms",
    "gap_from_definition",
    "half_variance",
    "quartic_coefficient",
    "run_checks",
]

#: The working order in `σ`. Everything below is written against it rather than
#: against a literal, so a further order is this constant plus new stages.
ORDER = 4

#: The commit these checks were first measured at.
_MEASURED_REF = "1888ad4"

#: Where the closed form `c₂` was derived independently of this series. Registered
#: 2026-08-07, nine days before this suite measured against it.
REGISTRATION_SOURCE = Source(
    correspondence=(
        "research/gate_d4_registration.md, RESULT 2026-08-07: c2 = (R'(mu)/2R(mu))^2"
    ),
    provenance=Provenance(
        registered_at="a76cf1b",
        measured_at=_MEASURED_REF,
        registered="RESULT 2026-08-07, c₂ in closed form",
    ),
)

#: Where the seven-term dimensional basis and its parity argument are registered. Same
#: registration commit as the closed form, and equally ahead of the measurement.
BASIS_SOURCE = Source(
    correspondence=(
        "research/gate_d4_registration.md, section 2: the seven-term dimensional "
        "basis, with a parity argument for completeness"
    ),
    provenance=Provenance(
        registered_at="a76cf1b",
        measured_at=_MEASURED_REF,
        registered="section 2, the seven-term dimensional basis and its parity "
        "argument",
    ),
)

#: Where the cumulant statement of the gap is hand derived. The derivation landed
#: 2026-08-17, a day after this suite measured against it, so the ordering runs
#: backwards and is recorded that way rather than smoothed over.
CUMULANT_SOURCE = Source(
    correspondence=(
        "research/c4_hand_derivation.md, Step 4 "
        "(the gap as half the variance at leading order)"
    ),
    provenance=Provenance(
        registered_at="99e3c34",
        measured_at=_MEASURED_REF,
        registered="Step 4, the gap as half the variance at leading order",
    ),
)

#: The Gaussian fourth and second moments, checked in `series_kernel` against the
#: Gaussian integral. Self-backing, so the two refs are one.
INNOVATION_MOMENT_SOURCE = Source(
    correspondence=(
        "the fourth and second moments of a centred Gaussian, which series_kernel "
        "checks against the Gaussian integral"
    ),
    provenance=Provenance(
        registered_at=_MEASURED_REF,
        measured_at=_MEASURED_REF,
        registered="the innovation moments, stated and checked in one commit",
    ),
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
    power: sympy.Expr = sympy.Integer(1)
    total: sympy.Expr = sympy.Integer(0)
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
    tilted = exp_series(truncate(tilt * log_ratio_in_sigma(order), order), order)
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
def half_variance(order: int) -> sympy.Expr:
    """`½·Var_q(W)`, the gap's leading-order form.

    Equal to the gap at `σ²` and **not** at `σ⁴`, where `κ₃/6` enters. Built from the
    kernel's cumulant recursion rather than from the definition above, so the two stay
    independent where they are compared.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `½·Var_q(W)` as a polynomial in `σ`, still carrying `ν`.
    """
    return truncate(cumulants(log_ratio_in_sigma(order), 2, order)[2] / 2, order)


def cumulant_terms(order: int, upto: int = 4) -> dict[int, sympy.Expr]:
    """The gap's cumulant series term by term: `κ_n/n!` for `n = 2 … upto`.

    `KL = Σ_{n≥2} κ_n/n!`, so this is the accounting that says which cumulants reach a
    given order and which contribute nothing. At `σ⁴` only `κ₂` and `κ₃` are live.
    `κ₄`'s lowest term is `σ⁴·κ₄(W₁)`, and `W₁` is linear in `z`, hence Gaussian, hence
    has no fourth cumulant.

    Args:
        order: the highest power of `σ` to keep, inclusive.
        upto: the highest cumulant to include.

    Returns:
        The contributions, keyed by cumulant index.
    """
    found = cumulants(log_ratio_in_sigma(order), upto, order)
    return {
        index: truncate(found[index] / sympy.factorial(index), order)
        for index in range(2, upto + 1)
    }


@cache
def averaged_gap(order: int) -> sympy.Expr:
    """The gap averaged over the innovation, under the **exact** predictive.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `E_{p*}[KL]` as a polynomial in `σ`, free of `ν`.
    """
    return exact_predictive_expectation(gap_from_definition(order), order)


@cache
def leading_order_averaged_gap(order: int) -> sympy.Expr:
    """The same average under a leading-order `N(0, R̄)` predictive.

    Kept so the difference can be measured. It is the object the registration warns
    against, and it is wrong at `σ⁴`.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `E[KL]` under the collapsed predictive.
    """
    return predictive_expectation(gap_from_definition(order), order)


@cache
def quartic_coefficient() -> sympy.Expr:
    """`c₄`, the averaged gap's `σ⁴` coefficient.

    Returns:
        The coefficient, in free `l₁..l₄` and a symbolic `R̄`.
    """
    return sympy.expand(averaged_gap(ORDER).coeff(SIGMA, 4))


@cache
def basis_coefficients() -> tuple[dict[str, sympy.Expr], sympy.Expr]:
    """`c₄` resolved onto the registration's seven-term dimensional basis.

    The basis is not re-derived here. The registration fixed it on 2026-08-07 from a
    dimensional argument with a parity check for completeness, and a second derivation
    in code would be a second thing to keep in sync. This resolves onto it and reports
    the remainder, which is the part a derivation can get wrong.

    Returns:
        The coefficient of each basis term, and what is left after removing all seven.
    """
    inverse = sympy.Symbol("u", positive=True)  # 1/R̄, so the basis is polynomial
    quartic = sympy.expand(quartic_coefficient().subs(RBAR, 1 / inverse))
    monomials = {
        "l'^4": L1**4,
        "l'^2 l''": L1**2 * L2,
        "l''^2": L2**2,
        "l' l'''": L1 * L3,
        "l''''": L4,
        "l'^2 / R": L1**2 * inverse,
        "l'' / R": L2 * inverse,
    }
    polynomial = sympy.Poly(quartic, L1, L2, L3, L4, inverse)
    found = {
        name: polynomial.coeff_monomial(monomial)
        for name, monomial in monomials.items()
    }
    rebuilt = sum(
        (found[name] * monomial for name, monomial in monomials.items()),
        sympy.Integer(0),
    )
    return found, sympy.expand(quartic - rebuilt)


def check_gap_is_half_the_variance() -> list[CheckReport]:
    """C1: the definition and the variance form agree at `σ²`, and part at `σ⁴`.

    Two derivations of the same object, neither asserted from the other. One expands
    `log E_q[e^W] − E_q[W]` directly. The other takes `κ₂/2` from the cumulant
    recursion. At `σ²` they agree. At `σ⁴` they must not, since `κ₃/6` enters, and a
    version of this check that quietly held at both orders would be recording a
    truncation coincidence.

    **What the agreement does not cover.** Neither arm calls the other, and that is all
    the independence claimed. Both read the same `W` from `log_ratio_in_sigma` and both
    take their expectations through the same `gaussian_expectation`. An error in either
    shared piece moves the two arms together and this check stays green. It separates
    the cumulant route from the generating-function route, not the kernel they share.

    Returns:
        The agreement report, the divergence report and the cumulant accounting.
    """
    definition = gap_from_definition(ORDER)
    terms = cumulant_terms(ORDER)
    difference = truncate(definition - terms[2], ORDER)
    return [
        report_identity(
            name="C1 gap is half the variance at σ²",
            claim="log E_q[e^W] − E_q[W] = ½·Var_q(W) at σ²",
            source=CUMULANT_SOURCE,
            residual=difference.coeff(SIGMA, 2),
            shown=f"[σ²] difference = {sympy.simplify(difference.coeff(SIGMA, 2))}",
        ),
        report_condition(
            name="C1 they part at σ⁴",
            claim="½·Var_q(W) is not the gap at σ⁴, because κ₃ enters",
            source=CUMULANT_SOURCE,
            holds=sympy.simplify(difference.coeff(SIGMA, 4)) != 0,
            shown=f"[σ⁴] difference = {sympy.factor(difference.coeff(SIGMA, 4))}",
        ),
        report_identity(
            name="C1 cumulant accounting at σ⁴",
            claim="the gap is κ₂/2 + κ₃/6 through σ⁴, with κ₄ contributing nothing",
            source=CUMULANT_SOURCE,
            residual=definition - truncate(terms[2] + terms[3], ORDER),
            shown=f"[σ⁴] κ₄/24 = {sympy.simplify(terms[4].coeff(SIGMA, 4))}",
        ),
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
    coefficient = sympy.expand(gap_from_definition(ORDER).coeff(SIGMA, 2))
    claim = L1**2 * (RBAR - NU**2) ** 2 / (8 * RBAR**2)
    as_square = (L1 * (RBAR - NU**2)) ** 2 / (8 * RBAR**2)
    return [
        report_identity(
            name="C2 fixed-y σ² coefficient",
            claim="[σ²] KL(y) = l₁²(R̄ − ν²)²/(8R̄²)",
            source=EXPANSION_SOURCE,
            residual=coefficient - claim,
            shown=f"{sympy.factor(coefficient)}",
        ),
        report_identity(
            name="C2 non-negative by construction",
            claim="the coefficient is a square over 8R̄², and R̄ is declared positive",
            source=EXPANSION_SOURCE,
            residual=coefficient - as_square,
            shown=f"(l₁(R̄ − ν²))²/(8R̄²), with R̄ positive: {RBAR.is_positive}",
        ),
        report_condition(
            name="C2 still depends on ν",
            claim="the y-average has not happened: the coefficient still varies with ν",
            source=EXPANSION_SOURCE,
            holds=sympy.simplify(sympy.diff(coefficient, NU)) != 0,
            shown=f"d/dν = {sympy.factor(sympy.diff(coefficient, NU))}",
        ),
    ]


def check_innovation_average() -> list[CheckReport]:
    """C3: `E[(ν²/R̄ − 1)²] = 2` under `ν ~ N(0, R̄)`.

    The whole content of the `y`-average at `σ²`, isolated from the gap so that a
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
            source=INNOVATION_MOMENT_SOURCE,
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
            source=REGISTRATION_SOURCE,
            residual=coefficient - L1**2 / 4,
            shown=f"c₂ = {coefficient}",
        ),
        report_condition(
            name="C4 c₂ is free of ν",
            claim="the averaged coefficient carries no innovation",
            source=REGISTRATION_SOURCE,
            holds=NU not in coefficient.free_symbols,
            shown=f"free symbols: {sorted(str(s) for s in coefficient.free_symbols)}",
        ),
    ]


def check_direction_independence() -> list[CheckReport]:
    """C5: reverse and forward KL agree at `σ²` and part at `σ⁴`.

    With `Λ(t) = log E_q[e^{tW}]`, the reverse direction is `Λ(1) − Λ'(0)` and the
    forward one is `Λ'(1) − Λ(1)`. Both are `½κ₂` at leading order, so the choice of
    direction cannot be what produced `c₂`.

    At `σ⁴` they must differ, since `κ₃` enters them with opposite weight. That is why
    the pinned conventions are load-bearing for `c₄` and were not for `c₂`, and
    asserting the divergence is what stops the `σ²` scoping from looking arbitrary.

    Returns:
        The agreement report and the divergence report.
    """
    tilt = sympy.Symbol("t", real=True)
    generating = cumulant_generating(tilt, ORDER)
    reverse = truncate(
        generating.subs(tilt, 1) - sympy.diff(generating, tilt).subs(tilt, 0), ORDER
    )
    forward = truncate(
        sympy.diff(generating, tilt).subs(tilt, 1) - generating.subs(tilt, 1), ORDER
    )
    difference = truncate(reverse - forward, ORDER)
    return [
        report_identity(
            name="C5 direction independence at σ²",
            claim="reverse KL = forward KL at σ², so c₂ is direction-free",
            source=CUMULANT_SOURCE,
            residual=difference.coeff(SIGMA, 2),
            shown=f"[σ²] reverse = {sympy.factor(reverse.coeff(SIGMA, 2))}",
        ),
        report_condition(
            name="C5 the directions part at σ⁴",
            claim="reverse and forward KL differ at σ⁴, so c₄ is direction-dependent",
            source=CUMULANT_SOURCE,
            holds=sympy.simplify(difference.coeff(SIGMA, 4)) != 0,
            shown=f"[σ⁴] difference = {sympy.factor(difference.coeff(SIGMA, 4))}",
        ),
    ]


def check_exact_predictive_is_required() -> list[CheckReport]:
    """C9: the leading-order predictive gets `c₂` right and `c₄` wrong.

    ``predictive_truncation`` established that `p*` is a scale mixture with exponential
    tails, so no Gaussian stands in for it at any variance. This is where that becomes a
    number: collapsing the predictive to `N(0, R̄)` leaves `c₂` untouched and moves `c₄`
    by a factor of several, so the choice decides the coefficient rather than refining
    it.

    Returns:
        The `σ²` agreement report and the `σ⁴` divergence report.
    """
    exact = averaged_gap(ORDER)
    collapsed = leading_order_averaged_gap(ORDER)
    parted = sympy.simplify(exact.coeff(SIGMA, 4) - collapsed.coeff(SIGMA, 4))
    return [
        report_identity(
            name="C9 predictives agree at σ²",
            claim="collapsing the predictive to N(0, R̄) leaves c₂ unchanged",
            source=EXPANSION_SOURCE,
            residual=exact.coeff(SIGMA, 2) - collapsed.coeff(SIGMA, 2),
            shown=f"c₂ = {sympy.expand(exact.coeff(SIGMA, 2))} either way",
        ),
        report_condition(
            name="C9 predictives part at σ⁴",
            claim="the collapsed predictive gives a different c₄, so nesting sets it",
            source=EXPANSION_SOURCE,
            holds=parted != 0,
            shown=f"exact − collapsed = {sympy.factor(parted)}",
        ),
    ]


def check_quartic_basis() -> list[CheckReport]:
    """C8: `c₄` lies in the declared seven-term span, with the two predicted zeros.

    The registration fixed the basis by dimensions before any of this existed and argued
    its completeness by parity. RESULT 2026-08-10 then reported `ℓ''''` and `ℓ''/R` as
    consistent with zero from a fit, and conjectured five simple fractions without
    establishing them. All of that is registered earlier than this module, so these are
    predictions being tested rather than observations being described.

    Returns:
        The span report, one report per predicted zero, and one per fraction.
    """
    found, remainder = basis_coefficients()
    fractions = {
        "l'^4": sympy.Rational(7, 16),
        "l'^2 l''": sympy.Rational(-1, 4),
        "l''^2": sympy.Rational(1, 8),
        "l' l'''": sympy.Rational(1, 4),
        "l'^2 / R": sympy.Rational(-3, 4),
    }
    reports = [
        report_identity(
            name="C8 c₄ lies in the declared basis",
            claim="c₄ is a combination of the seven registered terms, no remainder",
            source=BASIS_SOURCE,
            residual=remainder,
            shown=f"remainder after the seven terms = {remainder}",
        )
    ]
    reports += [
        report_identity(
            name=f"C8 predicted zero [{term}]",
            claim=f"the {term} coefficient is exactly zero",
            source=EXPANSION_SOURCE,
            residual=found[term],
            shown=f"{term}: {found[term]}",
        )
        for term in ("l''''", "l'' / R")
    ]
    reports += [
        report_identity(
            name=f"C8 fraction [{term}]",
            claim=f"the {term} coefficient is {value}",
            source=EXPANSION_SOURCE,
            residual=found[term] - value,
            shown=f"{term}: {found[term]}",
        )
        for term, value in fractions.items()
    ]
    return reports


def check_scale_covariance(
    expression: sympy.Expr, name: str, coefficient_order: int = 2
) -> list[CheckReport]:
    """C6: how a coefficient transforms under `R → a·R`, for a free positive `a`.

    `log(aR) = log a + log R`, so every log-derivative is untouched and `R̄` moves
    alone. The innovation moves with it, `ν ~ N(0, R̄)` becoming `N(0, aR̄)`, so `ν →
    √a·ν` is part of the same substitution rather than an extra assumption.

    Every monomial is `ℓ^p · ν^{2m} · R̄^{−k}`, so the substitution multiplies it by
    `a^{m−k}`. That exponent is the whole content, and it is what the check reports:

    - **No monomial may have `m > k`.** A positive deficit is a bare `R̄` in the
      numerator, or a `ν²` with no noise to divide it. Either is dimensionally
      impossible, so this is the leg that can fire.
    - `m = k` throughout means invariance. The fixed-`y` `σ²` coefficient is that
      case, being a function of `ν²/R̄` alone.
    - A `−1` appears once the averaged `σ⁴` coefficient is reached, from the `ℓ'²/R`
      term the registration's dimensional argument requires. Demanding invariance
      there would be wrong rather than strict.

    Written against a supplied expression so a further order inherits the guard rather
    than needing a second copy of it.

    Args:
        expression: the gap expression to test.
        name: what to call it in the report.
        coefficient_order: which power of `σ` to read the coefficient from.

    Returns:
        The deficit report and the transformation report.
    """
    inverse = sympy.Symbol("u", positive=True)  # 1/R̄, to read its power off a monomial
    coefficient = sympy.expand(expression.coeff(SIGMA, coefficient_order))
    polynomial = sympy.Poly(
        sympy.expand(coefficient.subs(RBAR, 1 / inverse)), NU, inverse
    )
    deficits = {
        sympy.Rational(int(powers[0]), 2) - int(powers[1])
        for powers, _ in polynomial.terms()
    }
    grouped = sum(
        (
            SCALE ** (sympy.Rational(int(powers[0]), 2) - int(powers[1]))
            * coeff
            * NU ** int(powers[0])
            * RBAR ** -int(powers[1])
            for powers, coeff in polynomial.terms()
        ),
        sympy.Integer(0),
    )
    scaled = coefficient.subs(
        {RBAR: SCALE * RBAR, NU: sympy.sqrt(SCALE) * NU}, simultaneous=True
    )
    return [
        report_condition(
            name=f"C6 no bare R̄ [{name}]",
            claim="every monomial pairs each ν² with at least one 1/R̄",
            source=BASIS_SOURCE,
            holds=max(deficits) <= 0,
            shown=f"deficits m − k present: {sorted(deficits)}",
        ),
        report_identity(
            name=f"C6 scale covariance [{name}]",
            claim="under R → aR with ν → √a·ν each monomial picks up a^(m−k)",
            source=EXPANSION_SOURCE,
            residual=sympy.simplify(scaled - grouped),
            shown=f"under R → aR: {sympy.factor(sympy.simplify(scaled))}",
        ),
    ]


def check_exponential_family() -> list[CheckReport]:
    """C7: for `R = A·e^{bx}`, exactly two of the seven basis terms survive.

    Every log-derivative above the first vanishes, so `l₂ = l₃ = l₄ = 0`. The
    registration predicted `c₄ = α·b⁴ + ζ·b²/R(μ)` from that, before either coefficient
    was known. The derivation supplies `α = 7/16` and `ζ = −3/4`, under the reverse
    direction that `ORDER`-level agreement does not extend to `σ⁴`.

    Returns:
        The `c₂` report and the `c₄` report.
    """
    rate = sympy.Symbol("b", real=True)
    flattened = {L1: rate, L2: 0, L3: 0, L4: 0}
    quadratic = sympy.expand(averaged_gap(ORDER).coeff(SIGMA, 2)).subs(flattened)
    quartic = sympy.expand(quartic_coefficient().subs(flattened))
    claimed = sympy.Rational(7, 16) * rate**4 - 3 * rate**2 / (4 * RBAR)
    return [
        report_identity(
            name="C7 exponential family c₂",
            claim="for R = A·e^{bx}, c₂ = b²/4",
            source=BASIS_SOURCE,
            residual=quadratic - rate**2 / 4,
            shown=f"c₂|(l₂=l₃=l₄=0) = {quadratic}",
        ),
        report_identity(
            name="C7 exponential family c₄",
            claim=(
                "for R = A·e^{bx}, c₄ = 7b⁴/16 − 3b²/(4R̄) under reverse KL, "
                "exactly two terms"
            ),
            source=BASIS_SOURCE,
            residual=quartic - claimed,
            shown=f"c₄|(l₂=l₃=l₄=0) = {sympy.factor(quartic)}",
        ),
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
        check_exact_predictive_is_required,
        check_quartic_basis,
        check_exponential_family,
    )
    reports = [report for stage in stages for report in stage()]
    reports += check_scale_covariance(gap_from_definition(ORDER), "fixed y, σ²")
    reports += check_scale_covariance(averaged_gap(ORDER), "averaged, σ²")
    reports += check_scale_covariance(averaged_gap(ORDER), "averaged, σ⁴", 4)
    return reports


def _print_setup() -> None:
    """Print the coefficients the checks are about, for the bare run."""
    fixed = gap_from_definition(ORDER)
    averaged = averaged_gap(ORDER)
    found, remainder = basis_coefficients()
    print(f"R̄ symbolic, l₁..l₄ free, ORDER = {ORDER}. No family chosen, no numbers.\n")
    print(f"[σ²] KL(y)  = {sympy.factor(fixed.coeff(SIGMA, 2))}")
    print(f"c₂          = {sympy.expand(averaged.coeff(SIGMA, 2))}")
    print(f"c₄          = {sympy.factor(quartic_coefficient())}\n")
    print("c₄ on the registration's seven-term basis:")
    for term, value in found.items():
        print(f"   {term:>10s}   {value}")
    print(f"   {'remainder':>10s}   {remainder}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every identity holds, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="The gap's expansion coefficients, symbolically: c₂ and c₄."
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
            "not in the target. Fix it there; do not adjust a coefficient."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
