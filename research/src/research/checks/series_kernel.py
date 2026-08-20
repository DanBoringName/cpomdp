"""The symbolic log-ratio and its expansion in prior spread, defined once.

The inference gap under state-dependent noise is a cumulant difference. With `q` the
agent's plug-in Gaussian and `p(x|y)` the exact posterior::

    KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]

where `W` is the log-ratio of the true likelihood to the plug-in one::

    W(x, y)  =  log N(y; x, R(x))  −  log N(y; x, R̄),        R̄ = R(μ)

This module owns `W`, its expansion in `σ`, the Gaussian moment operator and the
cumulants built from them. Callers own the questions: ``log_ratio_series`` pins the
structural properties, ``gap_series`` extracts coefficients. Extracted from
``log_ratio_series`` when the second caller arrived, on the same principle that put the
quadrature in ``gap_kernel``.

Notation, matching ``research/c4_hand_derivation.md``:

===========  ==========================================================
`s`          prior variance, `σ²`
`ν`          innovation, `y − μ`
`h`          displacement from the **prior** mean, `x − μ`
`δ`          `l(x) − l(μ)` where `l = log R`, carried as `l₁..l₄`
`z`          standard normal draw under `q`
===========  ==========================================================

**Truncation, not ``sympy.series``.** Every primitive here is built as an explicit
polynomial in `σ` (a geometric series for the gain, a binomial one for the posterior
width, the exponential series for `e^{−δ}`), and :func:`truncate` drops what is above
the working order by coefficient extraction, which runs in milliseconds.

The reason is cost growth, not a wall. Measured on the assembled `W`,
``series(together(W), sigma, 0, n)`` costs about 2 s at `σ⁴`, 8.5 s at `σ⁵` and 83.5 s
at `σ⁶`, near enough an order of magnitude per order. That is affordable at the orders
this module works to and not at the ones it is built to reach, and it is a poor
foundation for a pipeline that expands, multiplies and truncates repeatedly.
``sympy.series`` is therefore used only in the checks, as an independent arm against the
truncation path.

**Truncate inside every product.** Intermediate swell is the whole cost of the pipeline,
and every factor here is of non-negative order in `σ`, so dropping high powers early
cannot discard a term that would have landed below the working order.

Run the checks::

    uv run --no-sync python -m research.checks.series_kernel --check

Symbolic throughout. No floats, no numerics, and no functional form chosen for `R`: the
log-derivatives `l₁..l₄` stay free symbols and `R̄` stays symbolic and is never set to
one. A check that passes only at `R̄ = 1` is a check that has lost a variable.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import sympy

from warrantlib import (
    CheckReport,
    Outcome,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = [
    "Source",
    "cumulants",
    "displacement_series",
    "exact_predictive_expectation",
    "exp_series",
    "gain_series",
    "gaussian_expectation",
    "gaussian_moment",
    "increment_series",
    "innovation_series",
    "kalman_gain",
    "log_noise_increment",
    "log_ratio",
    "log_ratio_in_sigma",
    "posterior_sd",
    "posterior_sd_series",
    "predictive_expectation",
    "report_condition",
    "report_identity",
    "run_checks",
    "truncate",
]

#: Prior spread. Positive, so `sqrt` simplifies without a modulus.
SIGMA = sympy.Symbol("sigma", positive=True)

#: `R̄ = R(μ)`, the plug-in the agent freezes. Positive and never given a value.
RBAR = sympy.Symbol("Rbar", positive=True)

#: The innovation `ν = y − μ`.
NU = sympy.Symbol("nu", real=True)

#: Displacement from the prior mean, `h = x − μ`.
H = sympy.Symbol("h", real=True)

#: A standard normal draw, so `x = m_q + √v_q · z` under `q`.
Z = sympy.Symbol("z", real=True)

#: The two standard normals the predictive average needs, kept **distinct from** `Z`.
#: `ν = σz₁ + √R̄·e^{δ/2}·z₂` is exact at every order, and the posterior and predictive
#: expectations are integrals over different distributions. One shared symbol would
#: conflate them silently. Reserved here, used by the σ⁴ work.
Z1, Z2 = sympy.symbols("z1 z2", real=True)

#: `δ = l(x) − l(μ)`, carried as an opaque symbol where no expansion is wanted.
DELTA = sympy.Symbol("delta", real=True)

#: The prior mean.
MU = sympy.Symbol("mu", real=True)

#: Taylor coefficients of `l = log R` at the prior mean: `l₁ = l'(μ)` and so on. Free
#: symbols, so no family is chosen and no check can pass by accident of one.
L1, L2, L3, L4 = sympy.symbols("l1 l2 l3 l4", real=True)

#: Highest log-derivative carried.
DERIVATIVE_ORDER = 4

#: Prior variance.
PRIOR_VARIANCE = SIGMA**2


@dataclass(frozen=True)
class Source:
    """A correspondence and its refs, carried together.

    Every reduction here names where its setup was hand derived, and separately which
    ref registered that derivation. Held as two constants they would be edited one at a
    time, and a stale ref beside a fresh correspondence reads as a checked ordering
    without being one.

    Attributes:
        correspondence: where the symbolic setup was analytically checked, for
            `SymbolicReduction.correspondence`.
        provenance: which ref registered it, and which ref measured against it.
    """

    correspondence: str
    provenance: Provenance


#: The commit that first carried the hand derivation. Every source below pointing into
#: `c4_hand_derivation.md` was registered here.
_DERIVATION_REF = "99e3c34"

#: Where the hand derivation's construction of `W` is recorded, for the correspondence
#: field of every reduction resting on it.
#:
#: The derivation landed 2026-08-17, after `log_ratio_series` was measuring against it
#: (`23f0c47`, 2026-08-15). `measured_at` names that earliest reliance rather than the
#: latest commit to touch the file, so the ordering reads as what it was. ADR-037
#: discloses the same thing for the result this backs.
CONSTRUCTION_SOURCE = Source(
    correspondence=(
        "research/c4_hand_derivation.md, Steps 1-2 "
        "(the log-ratio and the reciprocal identity)"
    ),
    provenance=Provenance(
        registered_at=_DERIVATION_REF,
        measured_at="23f0c47",
        registered="Steps 1-2, the log-ratio and the reciprocal identity",
    ),
)

#: Where the σ-expansion of `h`, `δ` and `W` is recorded. Same ordering problem as
#: `CONSTRUCTION_SOURCE`, and this one backs 21 of the suite's checks.
EXPANSION_SOURCE = Source(
    correspondence=(
        "research/c4_hand_derivation.md, Step 3 (expansion in prior spread)"
    ),
    provenance=Provenance(
        registered_at=_DERIVATION_REF,
        measured_at="23f0c47",
        registered="Step 3, the expansion in prior spread",
    ),
)

#: The Gaussian moment integral, evaluated in this module. Its registration is the
#: check itself, so the two refs are one and the render says history orders nothing.
MOMENT_SOURCE = Source(
    correspondence=(
        "the Gaussian integral ∫ z^n φ(z) dz, evaluated symbolically in this check, "
        "which is what defines the moment the operator claims"
    ),
    provenance=Provenance(
        registered_at="53a668f",
        measured_at="53a668f",
        registered="the moment table, stated and checked in one commit",
    ),
)

#: Truncation as a projection, defined in this module's docstring. Self-backing, so the
#: refs are one.
TRUNCATION_SOURCE = Source(
    correspondence=(
        "truncation is the projection onto span{σ^k : k ≤ order}, stated in this "
        "module's docstring"
    ),
    provenance=Provenance(
        registered_at="53a668f",
        measured_at="53a668f",
        registered="the truncation operator, defined in this module's docstring",
    ),
)

#: The cumulant-moment recursion, a standard identity. Self-backing, so the refs
#: are one.
RECURSION_SOURCE = Source(
    correspondence=(
        "the cumulant-moment recursion κ_n = μ_n − Σ C(n−1, m−1)·κ_m·μ_{n−m}, standard"
    ),
    provenance=Provenance(
        registered_at="53a668f",
        measured_at="53a668f",
        registered="the cumulant-moment recursion, a standard identity",
    ),
)

#: The scope every reduction in this module inherits. `R` smooth and positive at `μ` is
#: what lets `l = log R` be Taylored at all; the expansion is formal, so no convergence
#: is claimed; and the three modelling choices are the ones `gap_kernel` implements in
#: quadrature.
STANDING_ASSUMPTIONS = (
    "R smooth and positive at μ, so l = log R has a Taylor expansion there",
    "the expansion is formal in σ, with no convergence claim",
    "reverse KL, R frozen at the prior mean R(μ), the average taken under the exact "
    "predictive",
)


def kalman_gain() -> sympy.Expr:
    """`K = s/(s + R̄)`, the gain the agent uses with its noise frozen.

    Returns:
        The gain, exact and unexpanded.
    """
    return PRIOR_VARIANCE / (PRIOR_VARIANCE + RBAR)


def posterior_sd() -> sympy.Expr:
    """`√v_q` where `v_q = (1 − K)s` is the plug-in posterior variance.

    Returns:
        The standard deviation, exact and unexpanded.
    """
    return sympy.sqrt((1 - kalman_gain()) * PRIOR_VARIANCE)


def truncate(expression: sympy.Expr, order: int) -> sympy.Expr:
    """Drop powers of `σ` above `order`.

    A projection onto the span of `σ⁰ … σ^order`, so applying it twice at the same order
    changes nothing and applying it at a lower order afterwards is the same as applying
    the lower one first.

    Args:
        expression: a polynomial in `σ`, with coefficients free of `σ`.
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The truncated expression, expanded.
    """
    polynomial = sympy.Poly(sympy.expand(expression), SIGMA)
    kept = sum(
        (
            coefficient * SIGMA ** int(monomial[0])
            for monomial, coefficient in polynomial.terms()
            if int(monomial[0]) <= order
        ),
        sympy.Integer(0),
    )
    return sympy.expand(kept)


# The expansions below are keyed on the working order and nothing else, and sympy
# expressions are immutable, so the caches return the same object rather than an equal
# one. Every check that reads a coefficient rebuilds the whole series first, and the
# rebuild cost grows steeply with the order.
@cache
def gain_series(order: int) -> sympy.Expr:
    """`K` as a polynomial in `σ`, by geometric expansion rather than by `series`.

    `K = u/(1 + u)` with `u = σ²/R̄`, so `K = Σ_{k≥1} (−1)^{k+1} u^k`. Even powers only.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The gain's expansion.
    """
    return sum(
        (
            (-1) ** (k + 1) * SIGMA ** (2 * k) / RBAR**k
            for k in range(1, order // 2 + 1)
        ),
        sympy.Integer(0),
    )


@cache
def posterior_sd_series(order: int) -> sympy.Expr:
    """`√v_q` as a polynomial in `σ`, by binomial expansion.

    `√v_q = σ(1 + σ²/R̄)^{−1/2}`, so the coefficients are `binom(−1/2, k)`. Odd powers
    only, starting at `σ`.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The width's expansion.
    """
    return sum(
        (
            sympy.binomial(sympy.Rational(-1, 2), k) * SIGMA ** (2 * k + 1) / RBAR**k
            for k in range((order - 1) // 2 + 1)
        ),
        sympy.Integer(0),
    )


@cache
def displacement_series(order: int) -> sympy.Expr:
    """`h = Kν + √v_q · z` as a polynomial in `σ`.

    The `σ²` coefficient is the Kalman shift, free of `z`. It is what separates
    displacement from the *prior* mean from displacement from the posterior mean, and a
    derivation that measures `h` from the wrong centre loses exactly this term.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The displacement's expansion.
    """
    gain = gain_series(order)
    width = posterior_sd_series(order)
    return truncate(gain * NU + width * Z, order)


@cache
def log_noise_increment() -> sympy.Expr:
    """`δ = l(μ + h) − l(μ)` as a Taylor polynomial in `h`, to fourth order.

    Built by differentiating an *undefined* function and then naming its derivatives,
    rather than by writing the coefficients down. That way the `1/k!` weights are
    checked rather than restated.

    Returns:
        The increment as a polynomial in `h`, with no constant term.
    """
    # `Function("l")` builds an undefined function, and applying it is how sympy names
    # `l(μ)` so its derivatives can be taken. `ty` reads the returned class as
    # non-callable, which is what the two suppressions are for. Restructuring to avoid
    # the call would mean writing the 1/k! weights down by hand, which is the one thing
    # this function exists not to do.
    noise_log = sympy.Function("l")
    applied = noise_log(MU)  # ty: ignore[call-non-callable]
    taylor = sum(
        applied.diff(MU, order) * H**order / sympy.factorial(order)
        for order in range(1, DERIVATIVE_ORDER + 1)
    )
    named = {
        sympy.Derivative(applied, (MU, order)): coefficient
        for order, coefficient in enumerate((L1, L2, L3, L4), start=1)
    }
    return sympy.expand(taylor.subs(named))


@cache
def increment_series(order: int) -> sympy.Expr:
    """`δ` with `h` expanded, as a polynomial in `σ`.

    The powers of `h` are accumulated one at a time and truncated at each step, so the
    fourth-order term never carries the full product before it is cut.

    `h` is `O(σ)`, so the `k`-th term `l_k·h^k/k!` is `O(σ^k)` and an expansion to
    `σ^order` needs log-derivatives up to `l_order`. Only `l₁..l₄` are carried, so an
    order above `DERIVATIVE_ORDER` would silently drop the terms it cannot express and
    return a polynomial that looks complete. That is the failure this rejects.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The increment's expansion.

    Raises:
        ValueError: if `order` exceeds `DERIVATIVE_ORDER`.
    """
    if order > DERIVATIVE_ORDER:
        raise ValueError(
            f"increment_series({order}) needs log-derivatives up to l{order}, and this "
            f"module carries l1..l{DERIVATIVE_ORDER}. The missing terms would be "
            "dropped rather than reported, so the result would read as a complete "
            f"expansion to σ^{order}. Raise DERIVATIVE_ORDER and extend the symbols "
            "before asking for this order."
        )
    displacement = displacement_series(order)
    power = displacement
    total = L1 * power
    for degree, coefficient in enumerate((L2, L3, L4), start=2):
        power = truncate(power * displacement, order)
        total += coefficient * power / sympy.factorial(degree)
    return truncate(total, order)


def exp_series(exponent: sympy.Expr, order: int) -> sympy.Expr:
    """`e^x` as a polynomial in `σ`, by the exponential series.

    The exponent carries its own sign, so `e^{−δ}` is ``exp_series(-increment, order)``.

    `x` is `O(σ)` with no constant term, so the `j`-th term is `O(σ^j)` and the sum
    closes at `j = order`. Each term is built from the last and truncated before the
    next multiplication.

    Args:
        exponent: `x`, already expanded in `σ` and carrying no `σ⁰` term.
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The exponential's expansion.
    """
    term: sympy.Expr = sympy.Integer(1)
    total: sympy.Expr = sympy.Integer(1)
    for step in range(1, order + 1):
        term = truncate(term * exponent, order) / step
        total += term
    return truncate(total, order)


def log_ratio(increment: sympy.Expr, displacement: sympy.Expr) -> sympy.Expr:
    """`W`, the log-ratio of the true likelihood to the plug-in one.

    The whole `R`-dependence of the gap enters through this one expression, via the
    reciprocal identity `1/R(x) − 1/R̄ = (e^{−δ} − 1)/R̄`. Written with `δ` and `h` as
    arguments so the same definition serves the opaque-`δ` checks and the expanded ones.

    Args:
        increment: `δ = l(x) − l(μ)`, opaque or expanded.
        displacement: `h = x − μ`, opaque or expanded.

    Returns:
        `W = −δ/2 + (ν − h)²/(2R̄)·(1 − e^{−δ})`.
    """
    return -increment / 2 + (NU - displacement) ** 2 / (2 * RBAR) * (
        1 - sympy.exp(-increment)
    )


@cache
def log_ratio_in_sigma(order: int) -> sympy.Expr:
    """`W` with `h`, `δ` and `e^{−δ}` all expanded, as a polynomial in `σ`.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        `W` as a polynomial in `σ`.
    """
    displacement = displacement_series(order)
    increment = increment_series(order)
    residual = truncate((NU - displacement) ** 2, order)
    relief = truncate(1 - exp_series(-increment, order), order)
    assembled = -increment / 2 + truncate(residual * relief, order) / (2 * RBAR)
    return truncate(assembled, order)


def gaussian_moment(order: int) -> sympy.Expr:
    """`E[z^n]` for a standard normal: `(n−1)!!` when `n` is even, zero when odd.

    Args:
        order: the moment's order.

    Returns:
        The moment, as an exact integer.
    """
    if order % 2:
        return sympy.Integer(0)
    return sympy.factorial2(order - 1)


def gaussian_expectation(
    expression: sympy.Expr, symbol: sympy.Symbol = Z
) -> sympy.Expr:
    """`E[·]` over one standard normal, by replacing each power with its moment.

    Args:
        expression: a polynomial in `symbol`, with coefficients free of it.
        symbol: which standard normal to average over. Defaults to `z`, the draw under
            `q`.

    Returns:
        The expectation.
    """
    polynomial = sympy.Poly(sympy.expand(expression), symbol)
    return sympy.simplify(
        sum(
            coefficient * gaussian_moment(int(monomial[0]))
            for monomial, coefficient in polynomial.terms()
        )
    )


def predictive_expectation(expression: sympy.Expr, order: int) -> sympy.Expr:
    """`E_{p*}[·]` over the innovation, at leading order in `σ`.

    At leading order the exact innovation collapses to `N(0, R̄)`, so the average is one
    Gaussian integral in `ν/√R̄`. Correct at `σ²` and wrong at `σ⁴`, where the neglected
    terms enter. Use :func:`exact_predictive_expectation` at or above `σ⁴`.

    Args:
        expression: a polynomial in `ν`, with coefficients free of it.
        order: the highest power of `σ` to keep in the result.

    Returns:
        The expectation, truncated.
    """
    standardised = expression.subs(NU, sympy.sqrt(RBAR) * Z2)
    return truncate(gaussian_expectation(standardised, Z2), order)


@cache
def innovation_series(order: int) -> sympy.Expr:
    """`ν` under the true predictive, expanded in `σ`.

    Not a model of `p*`, but the generative process written out. The latent is
    `x = μ + σz₁` and the sensor is `y = x + √R(x)·ε`, so with `R(x) = R̄·e^δ` and
    `ε = z₂`::

        ν  =  y − μ  =  σz₁ + √R̄·e^{δ(σz₁)/2}·z₂

    exact at every order. `δ` is evaluated at the **prior** displacement `σz₁`, the true
    latent, and never at the posterior one: the observation is generated before any
    inference happens.

    This is what replaces treating `p*` as a Gaussian with a corrected variance.
    ``predictive_truncation`` measures what that treatment costs: `p*` is a scale
    mixture with exponential tails, so no Gaussian stands in for it at any variance.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The innovation as a polynomial in `σ`, carrying `z₁` and `z₂`.
    """
    latent = SIGMA * Z1
    increment = truncate(log_noise_increment().subs(H, latent), order)
    scale = exp_series(increment / 2, order)
    return truncate(latent + sympy.sqrt(RBAR) * truncate(scale * Z2, order), order)


def exact_predictive_expectation(expression: sympy.Expr, order: int) -> sympy.Expr:
    """`E_{p*}[·]` over the innovation, by nesting the two Gaussian draws.

    The innovation carries both draws, so the average is two nested integrals rather
    than
    one: `z₂` over the sensor noise and `z₁` over the prior. They are independent, so
    neither integral needs the other's result and the order between them does not
    matter.

    Args:
        expression: a polynomial in `ν`, with coefficients free of it.
        order: the highest power of `σ` to keep in the result.

    Returns:
        The expectation, truncated.
    """
    substituted = truncate(expression.subs(NU, innovation_series(order)), order)
    over_sensor = truncate(gaussian_expectation(substituted, Z2), order)
    return truncate(gaussian_expectation(over_sensor, Z1), order)


def cumulants(
    expression: sympy.Expr, upto: int, order: int, symbol: sympy.Symbol = Z
) -> dict[int, sympy.Expr]:
    """`κ₁ … κ_upto` of `expression`, truncated at `σ^order`.

    Built from the raw moments by the standard recursion
    `κ_n = μ_n − Σ_{m=1}^{n−1} C(n−1, m−1)·κ_m·μ_{n−m}`, so no cumulant formula is
    written down and a request for a higher one costs nothing but the moments.

    Args:
        expression: the random variable, a polynomial in `symbol`.
        upto: the highest cumulant to return.
        order: the highest power of `σ` to keep, inclusive.
        symbol: which standard normal the expectation is over.

    Returns:
        The cumulants, keyed by index.

    Raises:
        ValueError: if `upto` is below one.
    """
    if upto < 1:
        raise ValueError(f"cumulants asked for κ up to {upto}, which is not a cumulant")
    moments: dict[int, sympy.Expr] = {0: sympy.Integer(1)}
    power: sympy.Expr = sympy.Integer(1)
    for index in range(1, upto + 1):
        power = truncate(power * expression, order)
        moments[index] = truncate(gaussian_expectation(power, symbol), order)
    found: dict[int, sympy.Expr] = {}
    for index in range(1, upto + 1):
        lower = sum(
            (
                sympy.binomial(index - 1, step - 1)
                * found[step]
                * moments[index - step]
                for step in range(1, index)
            ),
            sympy.Integer(0),
        )
        found[index] = truncate(moments[index] - lower, order)
    return found


def _reduction(claim: str, source: Source) -> SymbolicReduction:
    """A reduction carrying this module's standing scope.

    Args:
        claim: the analytic statement the identity stands for.
        source: where the setup was hand derived, and its refs.

    Returns:
        The evidence a `PROVED` report here carries.
    """
    return SymbolicReduction(
        claim=claim,
        correspondence=source.correspondence,
        assumptions=STANDING_ASSUMPTIONS,
    )


def _proved(name: str, claim: str, source: Source, shown: str) -> CheckReport:
    """A holding identity, reported at the warrant symbolic computation earns.

    Args:
        name: the check's label.
        claim: what the identity is, in words.
        source: where it was hand derived, and its refs.
        shown: the symbolic result, printed whether or not it held.

    Returns:
        The report.
    """
    return CheckReport(
        name=name,
        warrant=Warrant.PROVED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail=f"PASS — {claim}. got: {shown}",
        evidence=(_reduction(claim, source),),
        provenance=(source.provenance,),
    )


def _refuted(
    name: str, claim: str, shown: str, residual: str | None = None
) -> CheckReport:
    """A failed identity. The refutation is the result, and it is corroborative.

    `CORROBORATED` rather than `PROVED`, and the asymmetry is the point. A residual the
    CAS reduces to zero decides the identity. A residual it fails to reduce decides
    nothing, since simplification is incomplete and a true zero it could not find looks
    the same from here. So a passing check is theorem-grade and a firing one is evidence
    that something is wrong rather than proof of what.

    Args:
        name: the check's label.
        claim: what the identity was claimed to be.
        shown: what the symbolic computation actually returned.
        residual: the difference that failed to vanish, where there was one. Most
            identities here print one arm and leave the other in the claim, so without
            this the message says two things disagree and not by how much.

    Returns:
        The report.
    """
    detail = f"FAIL — claimed {claim}. got: {shown}"
    if residual is not None:
        detail += f". residual: {residual}"
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED,
        tier=Tier.EXACT,
        detail=detail,
    )


def report_identity(
    name: str,
    claim: str,
    source: Source,
    residual: sympy.Expr,
    shown: sympy.Expr | str,
) -> CheckReport:
    """Report whether a residual vanishes, printing the result either way.

    A failing identity that prints nothing is a failing identity nobody can diagnose,
    so `shown` is rendered before the verdict is read, and the residual joins it when it
    fails to vanish.

    Args:
        name: the check's label.
        claim: what the identity is, in words.
        source: where it was hand derived, and its refs.
        residual: the difference that must be identically zero.
        shown: what to print, whether or not the residual vanished.

    Returns:
        A `PROVED` report when the residual vanishes, a refutation when it does not.
    """
    reduced = sympy.simplify(residual)
    if reduced == 0:
        return _proved(name, claim, source, str(shown))
    return _refuted(name, claim, str(shown), str(reduced))


def report_condition(
    name: str,
    claim: str,
    source: Source,
    holds: bool,
    shown: sympy.Expr | str,
) -> CheckReport:
    """Report a property that is not the vanishing of a residual.

    Non-zero, free of a variable, of a given degree: statements a difference cannot
    express, so they are decided by the caller and labelled here.

    Args:
        name: the check's label.
        claim: what the property is, in words.
        source: where it was hand derived, and its refs.
        holds: whether the property obtained.
        shown: what to print, whether or not it held.

    Returns:
        A `PROVED` report when the property holds, a refutation when it does not.
    """
    if holds:
        return _proved(name, claim, source, str(shown))
    return _refuted(name, claim, str(shown))


def check_moment_table() -> list[CheckReport]:
    """K1: the moment operator against the Gaussian integral, `n = 0..10`.

    Checked against `∫ z^n φ(z) dz` rather than against the double-factorial identity
    the operator is written as. Comparing the function to its own body would pass by
    construction and refute nothing. `n = 0` is included on purpose: it rests on
    `(−1)!! = 1`, which is a convention a library may change under us.

    Returns:
        One report per order.
    """
    density = sympy.exp(-(Z**2) / 2) / sympy.sqrt(2 * sympy.pi)
    reports = []
    for order in range(11):
        integrated = sympy.integrate(Z**order * density, (Z, -sympy.oo, sympy.oo))
        claimed = gaussian_moment(order)
        reports.append(
            report_identity(
                name=f"K1 moment z^{order}",
                claim=f"E[z^{order}] = {claimed}",
                source=MOMENT_SOURCE,
                residual=integrated - claimed,
                shown=f"∫ z^{order} φ(z) dz = {integrated}",
            )
        )
    return reports


def check_truncation() -> list[CheckReport]:
    """K2: `truncate` is a projection, and it drops exactly what it should.

    Three properties, each of which a mis-ordered truncation breaks: idempotence at one
    order, agreement between truncating twice and truncating once at the lower order,
    and dropping the powers above the cut while leaving those below untouched.

    Returns:
        One report per property.
    """
    source = TRUNCATION_SOURCE
    probe = sum(
        (SIGMA**power * L1**power for power in range(6)),
        sympy.Integer(0),
    )
    return [
        report_identity(
            name="K2 truncate is idempotent",
            claim="truncating twice at the same order changes nothing",
            source=source,
            residual=truncate(truncate(probe, 3), 3) - truncate(probe, 3),
            shown=f"truncate(probe, 3) = {truncate(probe, 3)}",
        ),
        report_identity(
            name="K2 truncate composes downward",
            claim="truncating at 4 then at 2 equals truncating at 2",
            source=source,
            residual=truncate(truncate(probe, 4), 2) - truncate(probe, 2),
            shown=f"truncate at 4 then at 2 = {truncate(truncate(probe, 4), 2)}",
        ),
        report_identity(
            name="K2 truncate keeps what is below the cut",
            claim=(
                "the kept coefficients are the original ones, "
                "and nothing above the cut stays"
            ),
            source=source,
            residual=(
                sum(
                    (
                        truncate(probe, 3).coeff(SIGMA, power)
                        - probe.coeff(SIGMA, power)
                        for power in range(4)
                    ),
                    sympy.Integer(0),
                )
                + sum(
                    (truncate(probe, 3).coeff(SIGMA, power) for power in range(4, 6)),
                    sympy.Integer(0),
                )
            ),
            shown=f"truncate(probe, 3) = {truncate(probe, 3)}",
        ),
    ]


def check_primitive_series() -> list[CheckReport]:
    """K3: the hand-built expansions against `sympy.series` on the exact forms.

    The pipeline cannot afford `series` on the assembled expression, so the gain and the
    posterior width are written as explicit geometric and binomial sums. That makes them
    the one place an arithmetic slip would be silent. `series` on a two-term rational
    function is cheap, so it serves as the independent arm here and nowhere else.

    Returns:
        One report per primitive.
    """
    source = EXPANSION_SOURCE
    gain = sympy.expand(kalman_gain().series(SIGMA, 0, 8).removeO())
    width = sympy.expand(posterior_sd().series(SIGMA, 0, 7).removeO())
    return [
        report_identity(
            name="K3 gain expansion",
            claim="the geometric expansion of K matches series(K) to σ⁶",
            source=source,
            residual=gain_series(6) - truncate(gain, 6),
            shown=f"K = {gain_series(6)}",
        ),
        report_identity(
            name="K3 width expansion",
            claim="the binomial expansion of √v_q matches series(√v_q) to σ⁵",
            source=source,
            residual=posterior_sd_series(5) - truncate(width, 5),
            shown=f"√v_q = {posterior_sd_series(5)}",
        ),
        report_identity(
            name="K3 exponential expansion",
            claim="e^{−δ} built by the exponential series matches series(exp) to σ³",
            source=source,
            residual=(
                exp_series(-L1 * SIGMA * Z, 3)
                - truncate(
                    sympy.expand(
                        sympy.exp(-L1 * SIGMA * Z).series(SIGMA, 0, 4).removeO()
                    ),
                    3,
                )
            ),
            shown=f"e^(−l₁σz) = {exp_series(-L1 * SIGMA * Z, 3)}",
        ),
    ]


def check_assembled_against_series() -> list[CheckReport]:
    """K4: the truncation path reproduces `series` on the assembled `W`.

    The one check that licenses the swap. Everything else here tests a primitive, and a
    pipeline can be built from correct primitives and still assemble them wrongly: a
    truncation applied one factor too early drops a term that would have landed below
    the working order.

    The `series` arm is built here rather than imported, so it stays independent of the
    module being checked. It runs to `σ⁴`, which is where `DERIVATIVE_ORDER` stops the
    truncation path rather than where `series` stops being affordable: only `l₁..l₄` are
    carried, so `σ⁵` has no expressible left-hand side to compare against. Raising the
    constant is what moves this check, not a faster CAS.

    **Order conventions differ, and the difference is real.** ``sympy.series(expr, x, 0,
    n)`` keeps powers *below* `n`. :func:`truncate` keeps powers *up to and including*
    `order`. So the arms are compared at `n` against `order = n − 1`. Reading one as the
    other silently drops the top term, which is how this was first misread.

    Returns:
        One report per order checked.
    """
    reports = []
    for exclusive in range(2, DERIVATIVE_ORDER + 2):
        displacement = sympy.expand(
            (kalman_gain() * NU + posterior_sd() * Z)
            .series(SIGMA, 0, exclusive)
            .removeO()
        )
        increment = log_noise_increment().subs(H, displacement)
        assembled = sympy.expand(
            log_ratio(increment, displacement).series(SIGMA, 0, exclusive).removeO()
        )
        inclusive = exclusive - 1
        built = log_ratio_in_sigma(inclusive)
        reports.append(
            report_identity(
                name=f"K4 assembled W to σ^{inclusive}",
                claim=(
                    f"the truncation path equals series(W) through σ^{inclusive}, "
                    "term for term"
                ),
                source=EXPANSION_SOURCE,
                residual=built - assembled,
                shown=(
                    f"[σ^{inclusive}] W = {sympy.expand(built.coeff(SIGMA, inclusive))}"
                ),
            )
        )
    return reports


def check_cumulants() -> list[CheckReport]:
    """K5: the cumulant recursion against the closed forms for `κ₁` and `κ₂`.

    `κ₁` is the mean and `κ₂` is the variance. The recursion is what a later order will
    rest on, so it is checked where the answer is already known rather than where it is
    not.

    Returns:
        One report per cumulant.
    """
    source = RECURSION_SOURCE
    probe = L1 * SIGMA * Z + L2 * SIGMA**2 * Z**2
    found = cumulants(probe, 2, 4)
    mean = gaussian_expectation(probe)
    variance = truncate(
        gaussian_expectation(truncate(probe**2, 4)) - truncate(mean**2, 4), 4
    )
    return [
        report_identity(
            name="K5 first cumulant is the mean",
            claim="κ₁ = E[W]",
            source=source,
            residual=found[1] - mean,
            shown=f"κ₁ = {found[1]}",
        ),
        report_identity(
            name="K5 second cumulant is the variance",
            claim="κ₂ = E[W²] − E[W]²",
            source=source,
            residual=found[2] - variance,
            shown=f"κ₂ = {found[2]}",
        ),
    ]


def run_checks() -> list[CheckReport]:
    """Run every kernel check, in the order the construction needs them.

    Returns:
        Every check's report.
    """
    return [
        *check_moment_table(),
        *check_truncation(),
        *check_primitive_series(),
        *check_assembled_against_series(),
        *check_cumulants(),
    ]


def _print_setup() -> None:
    """Print the symbolic objects the checks are about, for the bare run."""
    print("R̄ symbolic, l₁..l₄ free. No family chosen, no numbers.\n")
    print(f"K        = {kalman_gain()}   -> {gain_series(4)}")
    print(f"√v_q     = {posterior_sd()}   -> {posterior_sd_series(3)}")
    print(f"δ(h)     = {log_noise_increment()}")
    print(f"h(σ)     = {displacement_series(3)}")
    print(f"W        = {log_ratio(DELTA, H)}")
    print(f"W(σ)     = {log_ratio_in_sigma(2)}")
    print(
        "\nNo gap coefficient is computed here. This module owns the construction; "
        "\nthe callers own the questions."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every identity holds, one otherwise.
    """
    parser = argparse.ArgumentParser(
        description="The symbolic log-ratio and its expansion in prior spread."
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
    return 1 if any(r.outcome is Outcome.FIRED for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
