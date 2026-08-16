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

Notation, matching the hand derivation:

===========  ==========================================================
`s`          prior variance, `σ²`
`ν`          innovation, `y − μ`
`h`          displacement from the **prior** mean, `x − μ`
`δ`          `l(x) − l(μ)` where `l = log R`, carried as `l₁..l₄`
`z`          standard normal draw under `q`
===========  ==========================================================

**Truncation, not ``sympy.series``.** A nested ``series(together(W), sigma, 0, 5)`` on
the assembled expression does not terminate in fifteen minutes. Every primitive here is
therefore built as an explicit polynomial in `σ` (a geometric series for the gain, a
binomial one for the posterior width, the exponential series for `e^{−δ}`), and
:func:`truncate` drops what is above the working order by coefficient extraction, which
runs in milliseconds. ``sympy.series`` appears only in the checks, as an independent arm
on the primitives, where the expressions are small rational functions and it is cheap.

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

import sympy

from cpomdp.warrant import (
    CheckReport,
    Outcome,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = [
    "cumulants",
    "displacement_series",
    "gain_series",
    "gaussian_expectation",
    "gaussian_moment",
    "kalman_gain",
    "log_noise_increment",
    "log_ratio",
    "log_ratio_in_sigma",
    "posterior_sd",
    "posterior_sd_series",
    "predictive_expectation",
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

#: Where the hand derivation's construction of `W` is recorded, for the correspondence
#: field of every reduction resting on it.
CONSTRUCTION_SOURCE = (
    "hand derivation, Steps 1-2 (log-ratio and the reciprocal identity)"
)

#: Where the σ-expansion of `h`, `δ` and `W` is recorded.
EXPANSION_SOURCE = "hand derivation, Step 3 (expansion in prior spread)"

#: The scope every reduction in this module inherits. `R` smooth and positive at `μ` is
#: what lets `l = log R` be Taylored at all; the expansion is formal, so no convergence
#: is claimed; and the three modelling choices are the ones `gap_kernel` implements in
#: quadrature.
STANDING_ASSUMPTIONS = (
    "R smooth and positive at μ, so l = log R has a Taylor expansion there",
    "the expansion is formal in σ, with no convergence claim",
    "reverse KL, R frozen at the prior mean R(μ), the average taken under q",
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


def increment_series(order: int) -> sympy.Expr:
    """`δ` with `h` expanded, as a polynomial in `σ`.

    The powers of `h` are accumulated one at a time and truncated at each step, so the
    fourth-order term never carries the full product before it is cut.

    Args:
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The increment's expansion.
    """
    displacement = displacement_series(order)
    power = displacement
    total = L1 * power
    for degree, coefficient in enumerate((L2, L3, L4), start=2):
        power = truncate(power * displacement, order)
        total += coefficient * power / sympy.factorial(degree)
    return truncate(total, order)


def exp_neg_series(increment: sympy.Expr, order: int) -> sympy.Expr:
    """`e^{−δ}` as a polynomial in `σ`, by the exponential series.

    `δ` is `O(σ)`, so the `j`-th term is `O(σ^j)` and the sum closes at `j = order`.
    Each term is built from the last and truncated before the next multiplication.

    Args:
        increment: `δ`, already expanded in `σ`.
        order: the highest power of `σ` to keep, inclusive.

    Returns:
        The exponential's expansion.
    """
    term = sympy.Integer(1)
    total = sympy.Integer(1)
    for step in range(1, order + 1):
        term = truncate(term * (-increment), order) / step
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
    relief = truncate(1 - exp_neg_series(increment, order), order)
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

    At leading order the exact predictive `ν = σz₁ + √R̄·e^{δ/2}·z₂` collapses to
    `N(0, R̄)`, so the average is one Gaussian integral in `ν/√R̄`. Correct at `σ²` and
    wrong at `σ⁴`, where the neglected terms enter, which is why the exact nesting is
    left to the work that needs it.

    Args:
        expression: a polynomial in `ν`, with coefficients free of it.
        order: the highest power of `σ` to keep in the result.

    Returns:
        The expectation, truncated.
    """
    standardised = expression.subs(NU, sympy.sqrt(RBAR) * Z2)
    return truncate(gaussian_expectation(standardised, Z2), order)


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


def _reduction(claim: str, correspondence: str) -> SymbolicReduction:
    """A reduction carrying this module's standing scope.

    Args:
        claim: the analytic statement the identity stands for.
        correspondence: where the setup was hand derived against the problem.

    Returns:
        The evidence a `PROVED` report here carries.
    """
    return SymbolicReduction(
        claim=claim,
        correspondence=correspondence,
        assumptions=STANDING_ASSUMPTIONS,
    )


def _proved(name: str, claim: str, correspondence: str, shown: str) -> CheckReport:
    """A holding identity, reported at the warrant symbolic computation earns.

    Args:
        name: the check's label.
        claim: what the identity is, in words.
        correspondence: where it was hand derived.
        shown: the symbolic result, printed whether or not it held.

    Returns:
        The report.
    """
    return CheckReport(
        name=name,
        warrant=Warrant.PROVED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.A,
        detail=f"PASS — {claim}. got: {shown}",
        evidence=(_reduction(claim, correspondence),),
    )


def _refuted(name: str, claim: str, shown: str) -> CheckReport:
    """A failed identity. The refutation is the result, and it carries no warrant.

    Args:
        name: the check's label.
        claim: what the identity was claimed to be.
        shown: what the symbolic computation actually returned.

    Returns:
        The report.
    """
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED,
        tier=Tier.A,
        detail=f"FAIL — claimed {claim}. got: {shown}",
    )


def report_identity(
    name: str,
    claim: str,
    correspondence: str,
    residual: sympy.Expr,
    shown: sympy.Expr | str,
) -> CheckReport:
    """Report whether a residual vanishes, printing the result either way.

    A failing identity that prints nothing is a failing identity nobody can diagnose,
    so `shown` is rendered before the verdict is read.

    Args:
        name: the check's label.
        claim: what the identity is, in words.
        correspondence: where it was hand derived against the analytic problem.
        residual: the difference that must be identically zero.
        shown: what to print, whether or not the residual vanished.

    Returns:
        A `PROVED` report when the residual vanishes, a refutation when it does not.
    """
    if sympy.simplify(residual) == 0:
        return _proved(name, claim, correspondence, str(shown))
    return _refuted(name, claim, str(shown))


def report_condition(
    name: str,
    claim: str,
    correspondence: str,
    holds: bool,
    shown: sympy.Expr | str,
) -> CheckReport:
    """Report a property that is not the vanishing of a residual.

    Non-zero, free of a variable, of a given degree: statements a difference cannot
    express, so they are decided by the caller and labelled here.

    Args:
        name: the check's label.
        claim: what the property is, in words.
        correspondence: where it was hand derived against the analytic problem.
        holds: whether the property obtained.
        shown: what to print, whether or not it held.

    Returns:
        A `PROVED` report when the property holds, a refutation when it does not.
    """
    if holds:
        return _proved(name, claim, correspondence, str(shown))
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
                correspondence=(
                    "the Gaussian integral ∫ z^n φ(z) dz, evaluated symbolically in "
                    "this check, which is what defines the moment the operator claims"
                ),
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
    correspondence = (
        "truncation is the projection onto span{σ^k : k ≤ order}, stated in this "
        "module's docstring"
    )
    probe = sum(
        (SIGMA**power * L1**power for power in range(6)),
        sympy.Integer(0),
    )
    return [
        report_identity(
            name="K2 truncate is idempotent",
            claim="truncating twice at the same order changes nothing",
            correspondence=correspondence,
            residual=truncate(truncate(probe, 3), 3) - truncate(probe, 3),
            shown=f"truncate(probe, 3) = {truncate(probe, 3)}",
        ),
        report_identity(
            name="K2 truncate composes downward",
            claim="truncating at 4 then at 2 equals truncating at 2",
            correspondence=correspondence,
            residual=truncate(truncate(probe, 4), 2) - truncate(probe, 2),
            shown=f"truncate at 4 then at 2 = {truncate(truncate(probe, 4), 2)}",
        ),
        report_identity(
            name="K2 truncate keeps what is below the cut",
            claim=(
                "the kept coefficients are the original ones, "
                "and nothing above the cut stays"
            ),
            correspondence=correspondence,
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
    correspondence = EXPANSION_SOURCE
    gain = sympy.expand(kalman_gain().series(SIGMA, 0, 8).removeO())
    width = sympy.expand(posterior_sd().series(SIGMA, 0, 7).removeO())
    return [
        report_identity(
            name="K3 gain expansion",
            claim="the geometric expansion of K matches series(K) to σ⁶",
            correspondence=correspondence,
            residual=gain_series(6) - truncate(gain, 6),
            shown=f"K = {gain_series(6)}",
        ),
        report_identity(
            name="K3 width expansion",
            claim="the binomial expansion of √v_q matches series(√v_q) to σ⁵",
            correspondence=correspondence,
            residual=posterior_sd_series(5) - truncate(width, 5),
            shown=f"√v_q = {posterior_sd_series(5)}",
        ),
        report_identity(
            name="K3 exponential expansion",
            claim="e^{−δ} built by the exponential series matches series(exp) to σ³",
            correspondence=correspondence,
            residual=(
                exp_neg_series(L1 * SIGMA * Z, 3)
                - truncate(
                    sympy.expand(
                        sympy.exp(-L1 * SIGMA * Z).series(SIGMA, 0, 4).removeO()
                    ),
                    3,
                )
            ),
            shown=f"e^(−l₁σz) = {exp_neg_series(L1 * SIGMA * Z, 3)}",
        ),
    ]


def check_assembled_against_series() -> list[CheckReport]:
    """K4: the truncation path reproduces `series` on the assembled `W`.

    The one check that licenses the swap. Everything else here tests a primitive, and a
    pipeline can be built from correct primitives and still assemble them wrongly: a
    truncation applied one factor too early drops a term that would have landed below
    the working order.

    The `series` arm is built here rather than imported, so it stays independent of the
    module being checked. It is capped at `σ³`. Past that the call is the one measured
    not to terminate, which is the reason the truncation path exists.

    **Order conventions differ, and the difference is real.** ``sympy.series(expr, x, 0,
    n)`` keeps powers *below* `n`. :func:`truncate` keeps powers *up to and including*
    `order`. So the arms are compared at `n` against `order = n − 1`. Reading one as the
    other silently drops the top term, which is how this was first misread.

    Returns:
        One report per order checked.
    """
    reports = []
    for exclusive in (2, 3, 4):
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
                correspondence=EXPANSION_SOURCE,
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
    correspondence = (
        "the cumulant-moment recursion κ_n = μ_n − Σ C(n−1, m−1)·κ_m·μ_{n−m}, standard"
    )
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
            correspondence=correspondence,
            residual=found[1] - mean,
            shown=f"κ₁ = {found[1]}",
        ),
        report_identity(
            name="K5 second cumulant is the variance",
            claim="κ₂ = E[W²] − E[W]²",
            correspondence=correspondence,
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
