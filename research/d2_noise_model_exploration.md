# What `δ_ref` does to the fitted exponent: an exploration

**Status: EXPLORATORY. Nothing here is registered and no check cites it.** It reports
what a reading of `δ_ref` implies, not a measurement of any filter, and the filter it
would apply to does not exist yet.

## The code

`research/src/research/explorations/noise_model.py`, run with
`python -m research.explorations.noise_model`. It prints every number below, checks the
covariance integral against an actual least-squares fit that shares no code with it, and
asserts both.

## The question

`research/gate_d4_registration.md` models D2's statistical term as
`σ_p ≈ ε√12/(D√N)`, falling with `N`, the number of `σ` samples fitted. That is the
standard error of a slope under **independent random** errors on each point.

The quantity being propagated is `δ_ref`, which the same document's opening line calls a
*certified discretisation bound*, and which PR-8 exists to produce by interval arithmetic
or a proved quadrature error bound. A bound on a deterministic error is not a random
draw. The reference filter evaluated at neighbouring `σ` is wrong in the same direction
for the same reason.

If the errors are correlated rather than independent, averaging more of them does not
average them away, and `1/√N` is the wrong functional form rather than the wrong
constant.

## What the reading implies

An additive error on the gap enters the log fit as a relative error `ε(v)`, with
`v = ln(σ/σ_min)`. To first order the OLS slope moves by `Cov(v, ε)/Var(v)`. **There is
no `N` in that expression.** The field is a function rather than a sample, so refining
the grid converges the integral instead of shrinking the answer.

**The comparison depends on the estimator, so that is settled first.** A deterministic
shift under an unweighted fit is not the same number as under one weighted by `1/ε²`,
and the registration names both: its bias is derived "with `v = ln σ` uniform over
`L = D·ln10`", which is unweighted, and its noise is "propagat[ed] through OLS with
heteroscedastic weights". Those describe different fits.

Its four published `σ_p` values decide it. Unweighted OLS reproduces all four at one
`N ≈ 60`. Weighted needs `N = 6.5` for three of them and has no solution for the fourth.

| | published | unweighted, `N=60` | weighted, `N=60` |
| --- | --- | --- | --- |
| `k=5, D=0.4343` | 0.0890 | 0.0880 | 0.0333 |
| `k=10, D=0.4343` | 0.0450 | 0.0440 | 0.0166 |
| `k=30, D=0.4343` | 0.0150 | 0.0147 | 0.0055 |
| `k=10, D=0.520` | 0.0359 | 0.0367 | 0.0113 |

**The numbers are unweighted**, whatever the prose says. So `N` is not undeclared after
all: it is recoverable at about 60. And the deterministic readings below are compared
against the estimator the figures actually use.

At the registered operating point, `k_min = 10` and `D* = 0.520`, against `β = 0.05`,
signs kept:

| reading of `δ_ref` | shift in the exponent | share of `β` |
| --- | --- | --- |
| the registration's `σ_p` | 0.0359 | 72% |
| an error tracking the gap, flat in ratio | 0.0000 | 0% |
| a constant absolute error of one bound | **−0.0666** | 133% |
| the worst field the bound admits | **+0.1083** | 217% |

The signs differ and are not decoration. A constant positive offset **flattens** the
exponent, to 1.933. The worst admissible field **steepens** it, to 2.108. Whether that
matters depends on whether D2's registered `2 ± 0.5` is read as two-sided, which it
appears to be.

**Two of the three deterministic readings blow the whole budget on their own**, under
the estimator the registration's own numbers use, before any truncation bias is added.

Under a `1/ε²`-weighted fit they would not: the offset falls to 0.0450 and the worst case
to 0.0474, both inside `β`. That is the reading the prose at line 501 describes and the
figures do not support. If the estimator were changed to match the prose, the conclusion
here reverses, and the registered `σ_p` values would all need recomputing.

## A defect in the formula, separately

The document states the statistical term as `ε√12/(D√N)`. The slope's standard error is
`σ_y√12/(L√N)` with `L` the window width in nats, and `D` is in decades, so the stated
form is larger than the true one by `ln 10`:

```text
as written, N = 345    0.035866
corrected,  N = 345    0.015576
exact OLS,  N = 345    0.015531
exact OLS,  N = 63     0.035876      the registered 0.0359
```

The registered figures are consistent with either the stated formula at `N = 345` or a
corrected one at `N ≈ 60`, and the two cannot be told apart from the numbers. `N ≈ 60`
is the more plausible fit design.

## The check

The covariance integral is checked against running the fit. A pure power law, a constant
absolute error of one bound added, ordinary least squares on the logs, and the slope read
off. It shares no code with the integral, uses the exact logarithm rather than the
first-order `ln(1 + ε) ≈ ε`, and samples rather than integrates:

| field | `N = 500` | `N = 50,000` | integral |
| --- | --- | --- | --- |
| tracking the gap | +0.0000 | +0.0000 | 0 |
| constant offset | −0.0666 | −0.0666 | 0.0695 |
| worst case | +0.1082 | +0.1083 | 0.1050 |

The flatness across two orders of magnitude in `N` is the point: a hundredfold increase
in samples moves the fitted exponent by nothing.

The gaps to the integral are the linearisation, `ε` reaching 0.1 at the bottom edge, and
they go **opposite ways**. It overstates the offset by 4% and understates the worst field
by 3%, so the worst case is 217% of `β` rather than the 210% the integral alone gives.

## What this does and does not establish

**It does not show the registration is wrong.** The benign reading is real: an error
proportional to the gap has a flat relative error, zero covariance with `v`, and moves
the exponent not at all however large it is. If the reference filter behaves that way,
`σ_p` is a non-issue and the registered figures stand.

**It does show `N` is the wrong thing to declare.** Under the random model `N` sets
`σ_p`. Under every deterministic reading `N` is irrelevant and the *shape* of the
filter's error sets it instead. Declaring `N` now would fix a parameter that two of the
three readings do not contain, and it is in any case recoverable at about 60 from the
published figures rather than genuinely free.

**And it leaves a question about how the two terms combine.** The registered constraint
is `√(bias² + σ_p²) ≤ β`, which adds them as independent contributions. A deterministic
error's effect on the exponent is a bias, not a variance. Two biases add, with whatever
cancellation their signs happen to give, rather than in quadrature. If the deterministic
reading is the operative one, the constraint's form needs revisiting alongside its terms.

**And the shape is not knowable yet.** It is a property of the reference filter, which
PR-7 builds. Until then the honest range on the exponent's error is 0 to 0.105, against a
total budget of 0.05.

## What would settle it

The reference filter, with its error measured across the window rather than bounded at a
point. `Cov(v, ε)/Var(v)` evaluated on the measured field is then the statistical term,
and it needs no `N`. If that is how it goes, `σ_p`'s current form is superseded rather
than recalibrated, and `D*` and `f*` follow from whatever replaces it.
