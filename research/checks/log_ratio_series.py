"""Symbolic pins for the log-ratio series, before any coefficient is derived.

The inference gap is a cumulant difference. With `q` the agent's plug-in Gaussian and
`p(x|y)` the exact posterior::

    KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]

where `W` is the log-ratio of the true likelihood to the plug-in one::

    W(x, y)  =  log N(y; x, R(x))  −  log N(y; x, R̄),        R̄ = R(μ)

Everything the expansion of the gap in prior spread rests on is a property of `W`: that
it vanishes when the noise is flat, that it carries no zeroth-order term in spread, and
that its leading term averages away under `q`. This module pins those properties
symbolically, so the hand derivation of the quartic coefficient has fixed ground to
stand on rather than conventions recalled from memory.

**Nothing here computes a gap coefficient.** Not `c₂`, not `c₄`, not `c₆`, and nothing
at second order in spread or beyond. The derivation is a hand exercise and a test that
printed its answer would spoil the checkpoint it exists to protect. The line is drawn at
the first-order term in `σ` and its expectation, which is where the structure lives and
where the arithmetic does not.

Run it::

    uv run --no-sync python -m research.checks.log_ratio_series --check
    uv run --no-sync python -m research.checks.log_ratio_series

Symbolic throughout. No floats, no numerics, no functional form chosen for `R`: the
log-derivatives `l₁..l₄` stay free symbols and `R̄` stays symbolic and is never set to
one. A check that passes only at `R̄ = 1` is a check that has lost a variable.

Conventions this fixes, in the notation the hand derivation uses:

===========  ==========================================================
`s`          prior variance, `σ²`
`ν`          innovation, `y − μ`
`h`          displacement from the **prior** mean, `x − μ`
`δ`          `l(x) − l(μ)` where `l = log R`, carried as `l₁..l₄`
`z`          standard normal draw under `q`
===========  ==========================================================

`h` is measured from the *prior* mean, not the posterior mean. The two differ at order
`σ²` by the Kalman shift, and T4 asserts that shift is non-zero precisely so the
distinction cannot quietly collapse.

Where this sits against the numeric checks. ``gap_kernel`` implements the same three
conventions in quadrature — reverse KL, `R` frozen at `R(μ)`, averaged under the true
predictive — and this module is where they are stated rather than coded. T6 and T8
together say the gap carries nothing at `σ⁰` or `σ¹`, so it starts at `σ²`.
``gap_expansion``'s G4a measures the residual after `c₂σ²` scaling as `σ⁴`. The symbolic
side supplies the reason, the numeric side the confirmation, and neither computes a
coefficient.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import sympy

__all__ = [
    "displacement_series",
    "kalman_gain",
    "log_noise_increment",
    "log_ratio",
    "posterior_sd",
    "run_checks",
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


def displacement_series(order: int) -> sympy.Expr:
    """`h = Kν + √v_q · z` expanded in `σ` to the given order.

    Args:
        order: the exclusive order in `σ` to expand to.

    Returns:
        The displacement as a polynomial in `σ`, with the order term dropped.
    """
    exact = kalman_gain() * NU + posterior_sd() * Z
    return sympy.expand(exact.series(SIGMA, 0, order).removeO())


def log_noise_increment() -> sympy.Expr:
    """`δ = l(μ + h) − l(μ)` as a Taylor polynomial in `h`, to fourth order.

    Built by differentiating an *undefined* function and then naming its derivatives,
    rather than by writing the coefficients down. That way T5 checks the `1/k!` weights
    instead of restating them.

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


def log_ratio(increment: sympy.Expr, displacement: sympy.Expr) -> sympy.Expr:
    """`W`, the log-ratio of the true likelihood to the plug-in one.

    The whole `R`-dependence of the gap enters through this one expression, via the
    reciprocal identity T1 checks. Written with `δ` and `h` as arguments so the same
    definition serves the opaque-`δ` checks and the expanded ones.

    Args:
        increment: `δ = l(x) − l(μ)`, opaque or expanded.
        displacement: `h = x − μ`, opaque or expanded.

    Returns:
        `W = −δ/2 + (ν − h)²/(2R̄)·(1 − e^{−δ})`.
    """
    return -increment / 2 + (NU - displacement) ** 2 / (2 * RBAR) * (
        1 - sympy.exp(-increment)
    )


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


def gaussian_expectation(expression: sympy.Expr) -> sympy.Expr:
    """`E_q[·]` over `z`, by replacing each power of `z` with its moment.

    Args:
        expression: a polynomial in `z`, with coefficients free of `z`.

    Returns:
        The expectation.
    """
    polynomial = sympy.Poly(sympy.expand(expression), Z)
    return sympy.simplify(
        sum(
            coefficient * gaussian_moment(int(monomial[0]))
            for monomial, coefficient in polynomial.terms()
        )
    )


@dataclass(frozen=True)
class Pin:
    """One symbolic assertion: what it claims, and what it got.

    Args:
        name: the check's label.
        claim: what the identity is, in words.
        shown: the symbolic result, printed whether or not it holds.
        holds: whether the assertion passed.
    """

    name: str
    claim: str
    shown: sympy.Expr | str
    holds: bool

    def __str__(self) -> str:
        """The pin as two lines: the verdict, then what it actually got."""
        verdict = "PASS" if self.holds else "FAIL"
        return f"{self.name}: {verdict} — {self.claim}\n    got: {self.shown}"


def _vanishes(expression: sympy.Expr) -> bool:
    """Whether an expression simplifies to exactly zero.

    Args:
        expression: the expression to test.

    Returns:
        Whether it is identically zero.
    """
    return sympy.simplify(expression) == 0


def check_reciprocal_identity() -> list[Pin]:
    """T1: `1/R(x) − 1/R̄ == (e^{−δ} − 1)/R̄`, exactly and with no expansion.

    The identity that lets the whole `R`-dependence be carried by `δ` alone. It is
    exact, so nothing downstream inherits a truncation from it.

    Returns:
        The pin, as a one-item list.
    """
    noise = RBAR * sympy.exp(DELTA)  # R(x) = R̄·e^δ, by the definition of δ
    difference = 1 / noise - 1 / RBAR
    claimed = (sympy.exp(-DELTA) - 1) / RBAR
    residual = sympy.simplify(difference - claimed)
    return [
        Pin(
            name="T1 reciprocal identity",
            claim="1/R(x) − 1/R̄ = (e^−δ − 1)/R̄, exact",
            shown=f"difference − claim = {residual}",
            holds=residual == 0,
        )
    ]


def check_flat_noise_vanishing() -> list[Pin]:
    """T2: `W ≡ 0` at `δ = 0`, for every `ν` and `h`.

    Flat noise is the case where the Kalman filter is exact, so the log-ratio must
    vanish identically rather than to some order. Checked by substitution, not by
    series: an expansion would only show it vanishes to the order expanded.

    Returns:
        The pin, as a one-item list.
    """
    flat = log_ratio(sympy.Integer(0), H)
    simplified = sympy.simplify(flat)
    return [
        Pin(
            name="T2 flat noise",
            claim="W = 0 identically at δ = 0, for all ν and h",
            shown=f"W|(δ=0) = {simplified}",
            holds=simplified == 0,
        )
    ]


def check_gain_and_sd_series() -> list[Pin]:
    """T3: the gain and the posterior width, expanded in `σ`.

    `K = σ²/R̄ − σ⁴/R̄² + O(σ⁶)` and `√v_q = σ − σ³/(2R̄) + O(σ⁵)`.

    Returns:
        The pin, as a one-item list.
    """
    gain = sympy.expand(kalman_gain().series(SIGMA, 0, 6).removeO())
    width = sympy.expand(posterior_sd().series(SIGMA, 0, 5).removeO())
    gain_claim = SIGMA**2 / RBAR - SIGMA**4 / RBAR**2
    width_claim = SIGMA - SIGMA**3 / (2 * RBAR)
    holds = _vanishes(gain - gain_claim) and _vanishes(width - width_claim)
    return [
        Pin(
            name="T3 gain and width",
            claim="K = σ²/R̄ − σ⁴/R̄² + O(σ⁶),  √v_q = σ − σ³/(2R̄) + O(σ⁵)",
            shown=f"K = {gain};  √v_q = {width}",
            holds=holds,
        )
    ]


def check_displacement_series() -> list[Pin]:
    """T4: `h = σz + (ν/R̄)σ² − (z/2R̄)σ³ + O(σ⁴)`, coefficient by coefficient.

    The `σ²` coefficient gets its own pin. It is the Kalman shift, it is free of `z`,
    and it is what separates displacement from the prior mean from displacement from the
    posterior mean. A derivation that measures `h` from the wrong centre loses exactly
    this term, so asserting it is non-zero is asserting the two are not interchangeable.

    Returns:
        One pin per coefficient, plus the shift pin.
    """
    expansion = displacement_series(4)
    claims = {
        1: Z,
        2: NU / RBAR,
        3: -Z / (2 * RBAR),
    }
    pins = [
        Pin(
            name=f"T4 displacement σ^{power}",
            claim=f"coefficient of σ^{power} is {claim}",
            shown=f"{expansion.coeff(SIGMA, power)}",
            holds=_vanishes(expansion.coeff(SIGMA, power) - claim),
        )
        for power, claim in claims.items()
    ]
    shift = expansion.coeff(SIGMA, 2)
    pins.append(
        Pin(
            name="T4 Kalman shift",
            claim="the σ² coefficient is non-zero and free of z",
            shown=f"{shift}, with d/dz = {sympy.diff(shift, Z)}",
            holds=shift != 0 and sympy.diff(shift, Z) == 0,
        )
    )
    return pins


def check_increment_coefficients() -> list[Pin]:
    """T5: `δ` has no constant term, and its `h^k` coefficients are `l_k/k!`.

    A convention pin rather than a discovery. It exists so that a later change to how
    the log-derivatives are carried breaks a test here instead of surfacing as a factor
    of `2` or `24` inside the hand derivation.

    Returns:
        The constant-term pin, then one per order.
    """
    increment = log_noise_increment()
    constant = increment.subs(H, 0)
    pins = [
        Pin(
            name="T5 δ constant term",
            claim="δ vanishes at h = 0",
            shown=f"δ|(h=0) = {constant}",
            holds=constant == 0,
        )
    ]
    for order, coefficient in enumerate((L1, L2, L3, L4), start=1):
        claim = coefficient / sympy.factorial(order)
        pins.append(
            Pin(
                name=f"T5 δ h^{order}",
                claim=f"coefficient of h^{order} is l{order}/{order}!",
                shown=f"{increment.coeff(H, order)} against {claim}",
                holds=_vanishes(increment.coeff(H, order) - claim),
            )
        )
    return pins


def _log_ratio_in_sigma(order: int) -> sympy.Expr:
    """`W` with `h` and `δ` both expanded, as a series in `σ`.

    Args:
        order: the exclusive order in `σ` to expand to.

    Returns:
        `W` as a polynomial in `σ`.
    """
    displacement = displacement_series(order + 2)
    increment = log_noise_increment().subs(H, displacement)
    expanded = log_ratio(increment, displacement)
    return sympy.expand(expanded.series(SIGMA, 0, order).removeO())


def check_no_zeroth_order() -> list[Pin]:
    """T6: `W` has no `σ⁰` term.

    At zero spread the belief is a point mass at `μ`, so `δ` vanishes and T2 applies.
    The gap therefore starts at first order in `σ` at the earliest, which is what makes
    an expansion in spread meaningful at all.

    Returns:
        The pin, as a one-item list.
    """
    zeroth = _log_ratio_in_sigma(2).coeff(SIGMA, 0)
    simplified = sympy.simplify(zeroth)
    return [
        Pin(
            name="T6 no σ⁰ term",
            claim="W carries no zeroth-order term in σ",
            shown=f"[σ⁰] W = {simplified}",
            holds=simplified == 0,
        )
    ]


def check_first_order_term() -> list[Pin]:
    """T7: `[σ¹] W == (l₁z/2)·(ν²/R̄ − 1)`, linear in `z` with no constant part.

    Only `l₁` appears. The higher log-derivatives cannot reach first order because `δ`
    enters through `h`, and `h` is `O(σ)`. That is the structural reason the first-order
    term is the one that averages away, which T8 then shows.

    Returns:
        The value pin, the degree pin and the constant-term pin.
    """
    first = sympy.expand(_log_ratio_in_sigma(2).coeff(SIGMA, 1))
    claim = (L1 * Z / 2) * (NU**2 / RBAR - 1)
    degree = sympy.Poly(first, Z).degree()
    return [
        Pin(
            name="T7 first-order term",
            claim="[σ¹] W = (l₁z/2)(ν²/R̄ − 1)",
            shown=f"{sympy.factor(first)}",
            holds=_vanishes(first - claim),
        ),
        Pin(
            name="T7 degree in z",
            claim="[σ¹] W is degree 1 in z",
            shown=f"degree = {degree}",
            holds=degree == 1,
        ),
        Pin(
            name="T7 constant in z",
            claim="[σ¹] W has no z-free part",
            shown=f"[σ¹] W|(z=0) = {sympy.simplify(first.subs(Z, 0))}",
            holds=_vanishes(first.subs(Z, 0)),
        ),
    ]


def check_moment_table() -> list[Pin]:
    """T8 support: `E[z^{2n}] = (2n−1)!!` for `n = 0..5`, and odd moments vanish.

    Checked against the Gaussian integral rather than against the double-factorial
    identity, so the moment operator has an arm independent of the formula it uses.

    Returns:
        One pin per order.
    """
    density = sympy.exp(-(Z**2) / 2) / sympy.sqrt(2 * sympy.pi)
    pins = []
    for order in range(11):
        integrated = sympy.integrate(Z**order * density, (Z, -sympy.oo, sympy.oo))
        claim = gaussian_moment(order)
        pins.append(
            Pin(
                name=f"T8 moment z^{order}",
                claim=f"E[z^{order}] = {claim}",
                shown=f"∫ z^{order} φ(z) dz = {integrated}",
                holds=sympy.simplify(integrated - claim) == 0,
            )
        )
    return pins


def check_first_order_expectation() -> list[Pin]:
    """T8: `E_q[[σ¹] W] == 0`.

    The first-order term is odd in `z`, so it averages away and the gap's leading
    behaviour is pushed to second order. This is the last statement before the
    arithmetic starts, and it is where this module stops.

    Returns:
        The pin, as a one-item list.
    """
    first = _log_ratio_in_sigma(2).coeff(SIGMA, 1)
    expectation = gaussian_expectation(first)
    return [
        Pin(
            name="T8 first-order expectation",
            claim="E_q[[σ¹] W] = 0",
            shown=f"E_q[[σ¹] W] = {expectation}",
            holds=expectation == 0,
        )
    ]


def run_checks() -> list[Pin]:
    """Run every pin, in the order the derivation needs them.

    Returns:
        Every pin's result.
    """
    stages: Sequence[Callable[[], list[Pin]]] = (
        check_reciprocal_identity,
        check_flat_noise_vanishing,
        check_gain_and_sd_series,
        check_displacement_series,
        check_increment_coefficients,
        check_no_zeroth_order,
        check_first_order_term,
        check_moment_table,
        check_first_order_expectation,
    )
    return [pin for stage in stages for pin in stage()]


def _print_setup() -> None:
    """Print the symbolic objects the pins are about, for the bare run."""
    print("R̄ symbolic, l₁..l₄ free. No family chosen, no numbers.\n")
    print(f"K        = {kalman_gain()}")
    print(f"√v_q     = {posterior_sd()}")
    print(f"δ(h)     = {log_noise_increment()}")
    print(f"h(σ)     = {displacement_series(4)}")
    print(f"W        = {log_ratio(DELTA, H)}")
    print(f"[σ¹] W   = {sympy.factor(_log_ratio_in_sigma(2).coeff(SIGMA, 1))}")
    print(
        "\nNo gap coefficient is computed here. The quartic is a hand exercise, and "
        "\nprinting c₂ would spoil the checkpoint this module protects."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pins and print them.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Zero when every pin holds, one otherwise.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--check" not in arguments:
        _print_setup()
        return 0

    pins = run_checks()
    for pin in pins:
        print(pin)
    failed = [pin for pin in pins if not pin.holds]
    print(f"\n{len(pins)} pins, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
