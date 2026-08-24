"""The gap's expansion coefficients, symbolically: `c₂`, `c₄` and `c₆`.

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

**Nothing here is fitted.** No floats and no numeric value for `R̄`. `ℓ₁..ℓ₆` are free
symbols throughout the derivation, so there is no quantity a measured number could be
substituted into. That is what makes the agreement with the fit evidence rather than
circularity, and it is checkable by reading the module.

Two checks do choose a family, and in both the choice is a consequence rather than an
input. C7 sets `ℓ₂ = ℓ₃ = ℓ₄ = 0` to specialise the *derived* general form to
`R = A·e^{bx}`. C14 substitutes `R = R₀ + κx²` at `μ* = √(R₀/κ)` to reach the ridge the
registration's `σ_max` edge is defined on. Both substitutions happen after the
coefficient exists, never before it, so neither can supply the answer it checks.

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
from sympy.utilities.iterables import partitions

from research.checks.series_kernel import (
    DERIVATIVE_ORDER,
    EXPANSION_SOURCE,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
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
    "dimensional_basis",
    "gap_from_definition",
    "half_variance",
    "quartic_coefficient",
    "resolve_onto_basis",
    "ridge_specialisation",
    "run_checks",
    "sextic_basis",
    "sextic_coefficient",
    "sextic_resolution",
]

#: The working order in `σ`. Everything below is written against it rather than
#: against a literal, so a further order is this constant plus new stages.
ORDER = 4

#: The order the sextic checks work at. Held apart from `ORDER` so the quartic checks
#: keep both their expansions and their cost, and only what needs `σ⁶` pays for it.
SEXTIC_ORDER = 6

#: The commit these checks were first measured at.
_MEASURED_REF = "1888ad4"

#: The commit at which the sextic checks below first ran. `5ce7695` made the expansion
#: computable and is not where anything measured it, which is what this has to name.
_SEXTIC_MEASURED_REF = "88afd7c"

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

#: The basis size the counting rule must produce, and the number of its coefficients
#: that come out non-zero. Both reach `research/gate_d4_registration.md`'s prose, so
#: both are asserted rather than only printed.
SEXTIC_BASIS_SIZE = 18
SEXTIC_LIVE_TERMS = 15

#: The three basis coefficients that vanish, and why each is expected to. Only the first
#: was predicted; the other two are reported rather than explained.
SEXTIC_ZERO_TERMS = {
    "l6": "l₆ entering W only linearly and the gap carrying no first cumulant",
    "l4/R^1": "reported rather than predicted",
    "l2/R^2": "reported rather than predicted",
}

#: The commit at which the ridge specialisation below first ran.
_RIDGE_MEASURED_REF = "471ba44"

#: Where the ridge is registered: the family, the operating point, and the two
#: coefficients already in closed form when the document landed. `c₆` is *not* there,
#: which is why the sextic report below carries its own source rather than this one.
RIDGE_SOURCE = Source(
    correspondence=(
        "research/c6_hand_derivation.md, 'Why the sextic is wanted': the ridge "
        "R(x) = R₀ + κx² at μ* = √(R₀/κ), with R̄ = 2R₀, c₂ and c₄ specialised"
    ),
    provenance=Provenance(
        registered_at="018ccc7",
        measured_at=_RIDGE_MEASURED_REF,
        registered="the ridge R(x) = R₀ + κx² at μ*, with c₂ = κ/(4R₀) and "
        "c₄ = 3κ(κ − 2)/(16R₀²)",
    ),
)

#: `c₆` on the ridge had no registration to be measured against. It was transcribed
#: into `sigma_max_edge.py` and three documents from a derivation nobody committed,
#: which is the practice ADR-050 rules out. Registering and measuring at one commit is
#: what actually happened here, and the marker is what stops it reading as a prediction.
RIDGE_SEXTIC_SOURCE = Source(
    correspondence=(
        "research/src/research/explorations/sigma_max_edge.py, c6(): the closed form "
        "the σ_max edge, the binding cell and the tail limit are all computed from"
    ),
    provenance=Provenance(
        registered_at=_RIDGE_MEASURED_REF,
        measured_at=_RIDGE_MEASURED_REF,
        registered="c₆ = −κ(7κ + 9)(13κ − 3)/(48R₀³) on the ridge, derived here for "
        "the first time rather than predicted ahead of it",
    ),
)

#: Where the sextic basis and its counting rule are derived, ahead of any run of it.
SEXTIC_BASIS_SOURCE = Source(
    correspondence=(
        "research/c6_hand_derivation.md, section 'The dimensional basis': the "
        "eighteen-term basis, its counting rule, and the parity check"
    ),
    provenance=Provenance(
        registered_at="018ccc7",
        measured_at=_SEXTIC_MEASURED_REF,
        registered="the dimensional basis, its counting rule and the eighteen terms",
    ),
)

#: Where the reach of each cumulant is derived. The amendment supersedes the table the
#: same document opened with, and says so rather than replacing it. Its ref is the
#: amendment's own commit, not the document's: `018ccc7` carries the superseded table
#: and a reviewer sent there would find the claim this cites contradicted.
SEXTIC_CUMULANT_SOURCE = Source(
    correspondence=(
        "research/c6_hand_derivation.md, AMENDMENT 2026-08-23: a joint cumulant of m "
        "factors reaches σ^N only if m ≤ N/2 + 1"
    ),
    provenance=Provenance(
        registered_at="2f39903",
        measured_at=_SEXTIC_MEASURED_REF,
        registered="the connectivity bound on which cumulants reach an order",
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


@cache
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
        The contributions, keyed by cumulant index. Cached on `(order, upto)`, both
        checks that ask for the sextic recursion asking for it identically.
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


def resolve_onto_basis(
    target: sympy.Expr,
    monomials: dict[str, sympy.Expr],
    variables: tuple[sympy.Symbol, ...],
) -> tuple[dict[str, sympy.Expr], sympy.Expr]:
    """`target` resolved onto `monomials`, with whatever is left over.

    The quartic and the sextic resolutions both come through here, so the two
    registered remainders are computed by one method rather than by two copies that
    could drift apart.

    Args:
        target: the coefficient to resolve, polynomial in `variables`.
        monomials: the basis, keyed by a readable name.
        variables: the generators, in a fixed order.

    Returns:
        The coefficient of each basis term, and what is left after removing them all.
    """
    polynomial = sympy.Poly(sympy.expand(target), *variables)
    found = {
        name: polynomial.coeff_monomial(monomial)
        for name, monomial in monomials.items()
    }
    rebuilt = sum(
        (found[name] * monomial for name, monomial in monomials.items()),
        sympy.Integer(0),
    )
    return found, sympy.expand(target - rebuilt)


def quartic_basis_by_hand(inverse: sympy.Symbol) -> dict[str, sympy.Expr]:
    """The registration's seven `c₄` monomials, as it lists them.

    Written out rather than generated. The registration fixed them on 2026-08-07,
    and a second derivation in code would be a second thing to keep in sync.
    `check_counting_rule_against_the_known_orders` compares the generated basis
    against this one, a comparison that would be circular if both came from the
    rule.

    Args:
        inverse: the symbol standing for `1/R̄`, so the terms stay polynomial.

    Returns:
        Each monomial, keyed by the registration's name for it.
    """
    return {
        "l'^4": L1**4,
        "l'^2 l''": L1**2 * L2,
        "l''^2": L2**2,
        "l' l'''": L1 * L3,
        "l''''": L4,
        "l'^2 / R": L1**2 * inverse,
        "l'' / R": L2 * inverse,
    }


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
    return resolve_onto_basis(
        quartic, quartic_basis_by_hand(inverse), (L1, L2, L3, L4, inverse)
    )


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
            check_id="gap_series.gap_is_half_the_variance_at_sigma2",
            claim="log E_q[e^W] − E_q[W] = ½·Var_q(W) at σ²",
            source=CUMULANT_SOURCE,
            residual=difference.coeff(SIGMA, 2),
            shown=f"[σ²] difference = {sympy.simplify(difference.coeff(SIGMA, 2))}",
        ),
        report_condition(
            name="C1 they part at σ⁴",
            check_id="gap_series.gap_and_half_variance_part_at_sigma4",
            claim="½·Var_q(W) is not the gap at σ⁴, because κ₃ enters",
            source=CUMULANT_SOURCE,
            holds=sympy.simplify(difference.coeff(SIGMA, 4)) != 0,
            shown=f"[σ⁴] difference = {sympy.factor(difference.coeff(SIGMA, 4))}",
        ),
        report_identity(
            name="C1 cumulant accounting at σ⁴",
            check_id="gap_series.cumulant_accounting_at_sigma4",
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
            check_id="gap_series.fixed_y_sigma2_coefficient",
            claim="[σ²] KL(y) = l₁²(R̄ − ν²)²/(8R̄²)",
            source=EXPANSION_SOURCE,
            residual=coefficient - claim,
            shown=f"{sympy.factor(coefficient)}",
        ),
        report_identity(
            name="C2 non-negative by construction",
            check_id="gap_series.fixed_y_coefficient_is_non_negative",
            claim="the coefficient is a square over 8R̄², and R̄ is declared positive",
            source=EXPANSION_SOURCE,
            residual=coefficient - as_square,
            shown=f"(l₁(R̄ − ν²))²/(8R̄²), with R̄ positive: {RBAR.is_positive}",
        ),
        report_condition(
            name="C2 still depends on ν",
            check_id="gap_series.fixed_y_coefficient_depends_on_nu",
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
            check_id="gap_series.innovation_average",
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
            check_id="gap_series.c2_closed_form",
            claim="c₂ = l₁²/4, which is (R'(μ)/2R(μ))² since l₁ = R'(μ)/R(μ)",
            source=REGISTRATION_SOURCE,
            residual=coefficient - L1**2 / 4,
            shown=f"c₂ = {coefficient}",
        ),
        report_condition(
            name="C4 c₂ is free of ν",
            check_id="gap_series.c2_is_free_of_nu",
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
            check_id="gap_series.direction_independence_at_sigma2",
            claim="reverse KL = forward KL at σ², so c₂ is direction-free",
            source=CUMULANT_SOURCE,
            residual=difference.coeff(SIGMA, 2),
            shown=f"[σ²] reverse = {sympy.factor(reverse.coeff(SIGMA, 2))}",
        ),
        report_condition(
            name="C5 the directions part at σ⁴",
            check_id="gap_series.directions_part_at_sigma4",
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
            check_id="gap_series.predictives_agree_at_sigma2",
            claim="collapsing the predictive to N(0, R̄) leaves c₂ unchanged",
            source=EXPANSION_SOURCE,
            residual=exact.coeff(SIGMA, 2) - collapsed.coeff(SIGMA, 2),
            shown=f"c₂ = {sympy.expand(exact.coeff(SIGMA, 2))} either way",
        ),
        report_condition(
            name="C9 predictives part at σ⁴",
            check_id="gap_series.predictives_part_at_sigma4",
            claim="the collapsed predictive gives a different c₄, so nesting sets it",
            source=EXPANSION_SOURCE,
            holds=parted != 0,
            shown=f"exact − collapsed = {sympy.factor(parted)}",
        ),
    ]


#: A basis term as it prints, and the same term as a key. `check_id` admits letters,
#: digits and underscores alone, so the primes and the solidus cannot travel into an
#: id, and deriving one would give punctuation a reader cannot map back to a term.
_TERM_IDS = {
    "l''''": "l4",
    "l'' / R": "l2_over_r",
    "l'^4": "l1_pow4",
    "l'^2 l''": "l1_sq_l2",
    "l''^2": "l2_sq",
    "l' l'''": "l1_l3",
    "l'^2 / R": "l1_sq_over_r",
}


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
            check_id="gap_series.c4_lies_in_the_declared_basis",
            claim="c₄ is a combination of the seven registered terms, no remainder",
            source=BASIS_SOURCE,
            residual=remainder,
            shown=f"remainder after the seven terms = {remainder}",
        )
    ]
    reports += [
        report_identity(
            name=f"C8 predicted zero [{term}]",
            check_id=f"gap_series.predicted_zero_{_TERM_IDS[term]}",
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
            check_id=f"gap_series.fraction_{_TERM_IDS[term]}",
            claim=f"the {term} coefficient is {value}",
            source=EXPANSION_SOURCE,
            residual=found[term] - value,
            shown=f"{term}: {found[term]}",
        )
        for term, value in fractions.items()
    ]
    return reports


def check_scale_covariance(
    expression: sympy.Expr, name: str, slug: str, coefficient_order: int = 2
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
        slug: the same subject as a key, joined into each report's `check_id`. Supplied
            rather than derived from `name`, which carries maths glyphs and prose.
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
            check_id=f"gap_series.no_bare_r_bar_{slug}",
            claim="every monomial pairs each ν² with at least one 1/R̄",
            source=BASIS_SOURCE,
            holds=max(deficits) <= 0,
            shown=f"deficits m − k present: {sorted(deficits)}",
        ),
        report_identity(
            name=f"C6 scale covariance [{name}]",
            check_id=f"gap_series.scale_covariance_{slug}",
            claim="under R → aR with ν → √a·ν each monomial picks up a^(m−k)",
            source=EXPANSION_SOURCE,
            residual=sympy.simplify(scaled - grouped),
            shown=f"under R → aR: {sympy.factor(sympy.simplify(scaled))}",
        ),
    ]


@cache
def sextic_coefficient() -> sympy.Expr:
    """`c₆`, the averaged gap's `σ⁶` coefficient.

    Returns:
        The coefficient, in free `l₁..l₆` and a symbolic `R̄`.
    """
    return sympy.expand(averaged_gap(SEXTIC_ORDER).coeff(SIGMA, SEXTIC_ORDER))


@cache
def dimensional_basis(order: int) -> dict[str, sympy.Expr]:
    """The monomials `c_order` may be built from, generated rather than listed.

    Dimension fixes the shape. With `σ ~ L` and `l_n ~ L^{-n}`, a dimensionless gap
    makes `c_order ~ L^{-order}`, so a term `(∏ l_{n_i})·R̄^{-k}` needs
    `Σ n_i = order − 2k`. One monomial per partition of that, for each `k`.

    `k` stops one short of `order/2`: at that value the constraint reads `Σn = 0`, a
    bare inverse power of `R̄` carrying no log-derivative. A constant `R` has every
    `l_n = 0` and an identically zero gap, so a term surviving there cannot appear.

    Args:
        order: the coefficient's order in `σ`, even and at least two.

    Returns:
        Each basis term, keyed by a readable name.

    Raises:
        ValueError: if `order` is odd, below two, or above what `l₁..l₆` can express.
    """
    if order < 2 or order % 2 or order > DERIVATIVE_ORDER:
        raise ValueError(
            f"dimensional_basis({order}) needs an even order between 2 and "
            f"{DERIVATIVE_ORDER}: the odd coefficients vanish and the carried "
            f"log-derivatives stop at l{DERIVATIVE_ORDER}"
        )
    carried = {1: L1, 2: L2, 3: L3, 4: L4, 5: L5, 6: L6}
    basis: dict[str, sympy.Expr] = {}
    for inverse_power in range(order // 2):
        for part in partitions(order - 2 * inverse_power):
            name = "".join(
                f"l{index}^{count}" if count > 1 else f"l{index}"
                for index, count in sorted(part.items())
            )
            term: sympy.Expr = RBAR ** (-inverse_power)
            for index, count in part.items():
                term *= carried[index] ** count
            basis[name + (f"/R^{inverse_power}" if inverse_power else "")] = term
    return basis


@cache
def sextic_basis() -> dict[str, sympy.Expr]:
    """The monomials `c₆` may be built from.

    Returns:
        Each basis term, keyed by a readable name.
    """
    return dimensional_basis(SEXTIC_ORDER)


@cache
def sextic_resolution() -> tuple[dict[str, sympy.Expr], sympy.Expr]:
    """`c₆` resolved onto the eighteen-term basis, with whatever is left over.

    The basis is derived in `research/c6_hand_derivation.md` and generated here from its
    counting rule. Resolving is what a derivation can get wrong, so the remainder is
    returned rather than assumed away.

    Returns:
        The coefficient of each basis term, and what is left after removing all of
        them.
    """
    inverse = sympy.Symbol("u", positive=True)  # 1/R̄, so the basis is polynomial
    basis = {
        name: sympy.expand(term.subs(RBAR, 1 / inverse))
        for name, term in sextic_basis().items()
    }
    target = sympy.expand(sextic_coefficient().subs(RBAR, 1 / inverse))
    return resolve_onto_basis(target, basis, (L1, L2, L3, L4, L5, L6, inverse))


def _slug(term: str) -> str:
    """A basis term's name as a check-id segment.

    `check_id` takes a dotted key of plain identifiers, and a basis name carries `^`
    and `/`. This maps them to underscores rather than inventing a second naming scheme.

    Args:
        term: the basis term's name, as `sextic_basis` keys it.

    Returns:
        The segment.
    """
    return term.replace("^", "").replace("/", "_over_")


def check_sextic_basis() -> list[CheckReport]:
    """C10: `c₆` lies in the span of the eighteen-term basis, with nothing left over.

    A non-zero remainder refutes either the basis or the expansion. It is reported as
    the residual rather than absorbed into a nineteenth term.
    """
    weights, remainder = sextic_resolution()
    live = {name: weight for name, weight in weights.items() if weight != 0}
    reports = [
        report_identity(
            name=f"C10 the basis has {SEXTIC_BASIS_SIZE} terms",
            check_id="gap_series.sextic_basis_size",
            claim=(
                f"the counting rule generates {SEXTIC_BASIS_SIZE} monomials, which is "
                "what the registration's prose says c₆ was resolved onto"
            ),
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.Integer(len(weights) - SEXTIC_BASIS_SIZE),
            shown=f"the rule generated {len(weights)} terms",
        ),
        report_identity(
            name="C10 sextic basis spans c₆",
            check_id="gap_series.sextic_basis_spans",
            claim="c₆ resolves onto the eighteen-term basis with no remainder",
            source=SEXTIC_BASIS_SOURCE,
            residual=remainder,
            shown=f"{len(live)} of {len(weights)} basis coefficients are non-zero",
        ),
        report_identity(
            name=f"C10 {SEXTIC_LIVE_TERMS} coefficients are non-zero",
            check_id="gap_series.sextic_live_terms",
            claim=(
                f"{SEXTIC_LIVE_TERMS} of the {SEXTIC_BASIS_SIZE} basis coefficients "
                "are non-zero, the count the registration publishes"
            ),
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.Integer(len(live) - SEXTIC_LIVE_TERMS),
            shown=f"{len(live)} non-zero",
        ),
    ]
    # The registration names three coefficients that vanish. Only `l₆` was predicted;
    # the other two are reported. All three are asserted, since all three are in prose.
    for term, why in SEXTIC_ZERO_TERMS.items():
        reports.append(
            report_identity(
                name=f"C10 [{term}] c₆ = 0",
                check_id=f"gap_series.sextic_{_slug(term)}_absent",
                claim=f"the {term} coefficient of c₆ is zero, {why}",
                source=(
                    SEXTIC_CUMULANT_SOURCE if term == "l6" else SEXTIC_BASIS_SOURCE
                ),
                # Indexed rather than `.get`: a SEXTIC_ZERO_TERMS key that stops
                # matching `dimensional_basis` would otherwise report PROVED at
                # residual 0, and the check-id and manifest entry are keyed off
                # the same dict, so nothing else would fire either.
                residual=weights[term],
                shown=f"[{term}] c₆ = {weights[term]}",
            )
        )
    return reports


def check_counting_rule_against_the_known_orders() -> list[CheckReport]:
    """C14: the rule reproduces the bases already registered, without being told them.

    `research/gate_d4_registration.md` fixed `c₄`'s basis at seven terms on 2026-08-07
    from a dimensional argument, and `c₂` occupies two. The rule that generated `c₆`'s
    eighteen is asked for both, and its answer is compared against the seven this module
    already lists by hand for the quartic. A rule that reproduces neither is a rule
    fitted to the one case it was written for.
    """
    inverse = sympy.Symbol("u", positive=True)
    by_hand = quartic_basis_by_hand(inverse)
    generated = dimensional_basis(ORDER)
    hand_terms = {
        sympy.expand(term.subs(inverse, 1 / RBAR)) for term in by_hand.values()
    }
    rule_terms = {sympy.expand(term) for term in generated.values()}
    return [
        report_identity(
            name="C14 the rule gives c₂ two terms",
            check_id="gap_series.counting_rule_at_order_two",
            claim="the counting rule generates two monomials at σ², as c₂ occupies",
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.Integer(len(dimensional_basis(2)) - 2),
            shown=f"order 2: {sorted(dimensional_basis(2))}",
        ),
        report_identity(
            name="C14 the rule gives c₄ seven terms",
            check_id="gap_series.counting_rule_at_order_four",
            claim="the counting rule generates the registration's seven at σ⁴",
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.Integer(len(generated) - len(by_hand)),
            shown=f"order 4: {len(generated)} generated, {len(by_hand)} listed",
        ),
        report_identity(
            name="C14 the rule's c₄ basis is the listed one",
            check_id="gap_series.counting_rule_matches_the_quartic_basis",
            claim="the generated monomials at σ⁴ are the seven listed by hand",
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.Integer(len(hand_terms ^ rule_terms)),
            shown=f"symmetric difference: {sorted(map(str, hand_terms ^ rule_terms))}",
        ),
    ]


def check_sextic_cumulant_reach() -> list[CheckReport]:
    """C11: which cumulants reach `σ⁶`, against the connectivity bound.

    `κ_m` reaches `σ^N` only if `m ≤ N/2 + 1`, so at `σ⁶` the fifth and sixth cumulants
    contribute nothing. The document this cites opened with a table saying otherwise for
    the fifth, and its amendment is what is checked here.
    """
    terms = cumulant_terms(SEXTIC_ORDER, upto=SEXTIC_ORDER)
    return [
        report_identity(
            name=f"C11 κ{index} does not reach σ⁶",
            check_id=f"gap_series.sextic_cumulant_{index}_absent",
            claim=(
                f"the σ⁶ coefficient of κ{index}/{index}! is zero, the connectivity "
                f"bound admitting only m ≤ 4"
            ),
            source=SEXTIC_CUMULANT_SOURCE,
            residual=sympy.expand(terms[index]).coeff(SIGMA, SEXTIC_ORDER),
            shown=f"[σ⁶] κ{index}/{index}! = "
            f"{sympy.expand(terms[index]).coeff(SIGMA, SEXTIC_ORDER)}",
        )
        for index in (5, 6)
    ]


def check_sextic_arms_agree() -> list[CheckReport]:
    """C13: the two routes to the gap agree at `σ⁶`.

    One expands `log E_q[e^W] − E_q[W]` through the generating function and uses no
    cumulant identity. The other sums `κ_n/n!` from the cumulant recursion. Neither
    calls the other, and both are averaged under the same exact predictive, so the
    agreement separates the two routes rather than the kernel beneath them.

    This is what licenses `EXACT` for the sextic. Two independently computed closed
    forms agreeing is the licence; a tolerance being met is not.
    """
    generating = averaged_gap(SEXTIC_ORDER)
    recursion = exact_predictive_expectation(
        truncate(
            sum(
                cumulant_terms(SEXTIC_ORDER, upto=SEXTIC_ORDER).values(),
                sympy.Integer(0),
            ),
            SEXTIC_ORDER,
        ),
        SEXTIC_ORDER,
    )
    return [
        report_identity(
            name=f"C13 arms agree at σ^{power}",
            check_id=f"gap_series.sextic_arms_agree_sigma{power}",
            claim=(
                f"the generating function and the cumulant recursion give the same "
                f"σ^{power} coefficient"
            ),
            source=SEXTIC_CUMULANT_SOURCE,
            residual=sympy.expand(
                generating.coeff(SIGMA, power) - recursion.coeff(SIGMA, power)
            ),
            shown=f"[σ^{power}] both routes agree",
        )
        for power in (2, 4, 6)
    ]


def check_odd_orders_vanish() -> list[CheckReport]:
    """C12: the averaged gap carries no odd power of `σ` below `σ⁶`."""
    gap = averaged_gap(SEXTIC_ORDER)
    return [
        report_identity(
            name=f"C12 σ^{power} coefficient vanishes",
            check_id=f"gap_series.odd_order_{power}_vanishes",
            claim=f"the averaged gap has no σ^{power} term",
            source=SEXTIC_BASIS_SOURCE,
            residual=sympy.expand(gap.coeff(SIGMA, power)),
            shown=f"[σ^{power}] E[KL] = {sympy.expand(gap.coeff(SIGMA, power))}",
        )
        for power in (3, 5)
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
            check_id="gap_series.exponential_family_c2",
            claim="for R = A·e^{bx}, c₂ = b²/4",
            source=BASIS_SOURCE,
            residual=quadratic - rate**2 / 4,
            shown=f"c₂|(l₂=l₃=l₄=0) = {quadratic}",
        ),
        report_identity(
            name="C7 exponential family c₄",
            check_id="gap_series.exponential_family_c4",
            claim=(
                "for R = A·e^{bx}, c₄ = 7b⁴/16 − 3b²/(4R̄) under reverse KL, "
                "exactly two terms"
            ),
            source=BASIS_SOURCE,
            residual=quartic - claimed,
            shown=f"c₄|(l₂=l₃=l₄=0) = {sympy.factor(quartic)}",
        ),
    ]


@cache
def ridge_specialisation() -> dict[str, sympy.Expr]:
    """`c₂`, `c₄` and `c₆` on the ridge of `d4-family-v1`.

    Specialises the *derived* general forms to `R(x) = R₀ + κx²` at `μ* = √(R₀/κ)`.
    The substitution happens after each coefficient exists, exactly as C7 does for the
    exponential family, so it cannot supply the answer it is checked against.

    Returns:
        Each coefficient, keyed `c2`, `c4`, `c6`, in `κ` and `R₀`.
    """
    curvature, floor, position = sympy.symbols("kappa R_0 x", positive=True)
    reading = floor + curvature * position**2
    operating_point = sympy.sqrt(floor / curvature)
    carried = {
        symbol: sympy.diff(sympy.log(reading), position, order).subs(
            position, operating_point
        )
        for order, symbol in enumerate((L1, L2, L3, L4, L5, L6), start=1)
    }
    carried[RBAR] = reading.subs(position, operating_point)
    return {
        "c2": sympy.simplify(
            sympy.expand(averaged_gap(ORDER).coeff(SIGMA, 2)).subs(carried)
        ),
        "c4": sympy.simplify(quartic_coefficient().subs(carried)),
        "c6": sympy.simplify(sextic_coefficient().subs(carried)),
    }


def check_ridge_specialisation() -> list[CheckReport]:
    """C14: the ridge closed forms, against the registration and against the code.

    `c₂` and `c₄` were in closed form on the ridge before this ran, so those two are
    predictions and are reported that way. `c₆` was not: it reached the registration
    as a hand transcription in `sigma_max_edge.py` with no derivation committed
    anywhere. This is that derivation, and every number downstream of it — the σ_max
    edge, the binding cell, the −529/24 at κ = 2, the 3/13 root and the tail limit —
    is computed from the expression checked here.

    Returns:
        One report per coefficient.
    """
    curvature, floor = sympy.symbols("kappa R_0", positive=True)
    derived = ridge_specialisation()
    return [
        report_identity(
            name="C14 ridge c₂",
            check_id="gap_series.ridge_c2",
            claim="on the ridge, c₂ = κ/(4R₀)",
            source=RIDGE_SOURCE,
            residual=derived["c2"] - curvature / (4 * floor),
            shown=f"c₂|ridge = {sympy.factor(derived['c2'])}",
        ),
        report_identity(
            name="C14 ridge c₄",
            check_id="gap_series.ridge_c4",
            claim="on the ridge, c₄ = 3κ(κ − 2)/(16R₀²), zero at κ = 2",
            source=RIDGE_SOURCE,
            residual=derived["c4"] - 3 * curvature * (curvature - 2) / (16 * floor**2),
            shown=f"c₄|ridge = {sympy.factor(derived['c4'])}",
        ),
        report_identity(
            name="C14 ridge c₆",
            check_id="gap_series.ridge_c6",
            claim=(
                "on the ridge, c₆ = −κ(7κ + 9)(13κ − 3)/(48R₀³), the form "
                "sigma_max_edge.c6 hard-codes at R₀ = 1"
            ),
            source=RIDGE_SEXTIC_SOURCE,
            residual=derived["c6"]
            + curvature * (7 * curvature + 9) * (13 * curvature - 3) / (48 * floor**3),
            shown=f"c₆|ridge = {sympy.factor(derived['c6'])}",
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
        check_sextic_basis,
        check_counting_rule_against_the_known_orders,
        check_sextic_cumulant_reach,
        check_sextic_arms_agree,
        check_odd_orders_vanish,
        check_exponential_family,
        check_ridge_specialisation,
    )
    reports = [report for stage in stages for report in stage()]
    reports += check_scale_covariance(
        gap_from_definition(ORDER), "fixed y, σ²", "fixed_y_sigma2"
    )
    reports += check_scale_covariance(
        averaged_gap(ORDER), "averaged, σ²", "averaged_sigma2"
    )
    reports += check_scale_covariance(
        averaged_gap(ORDER), "averaged, σ⁴", "averaged_sigma4", 4
    )
    return reports


def _print_setup() -> None:
    """Print the coefficients the checks are about, for the bare run."""
    fixed = gap_from_definition(ORDER)
    averaged = averaged_gap(ORDER)
    found, remainder = basis_coefficients()
    print(
        f"R̄ symbolic, l₁..l₆ free, ORDER = {ORDER} and {SEXTIC_ORDER}. "
        "No family chosen, no numbers.\n"
    )
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
        description="The gap's expansion coefficients, symbolically: c₂, c₄ and c₆."
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
