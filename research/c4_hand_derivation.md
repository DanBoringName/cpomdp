# The inference gap under state-dependent noise: the hand derivation

The analytic derivation the symbolic suites check themselves against. A
`SymbolicReduction` emitted by `research.checks.series_kernel`,
`research.checks.log_ratio_series` or `research.checks.gap_series` names a step of this
document in its `correspondence` field when it rests on the construction (Steps 1-2),
the σ-expansion (Step 3), or the cumulant statement of the gap (Step 4). The rest name
something else. A standard identity they lean on, such as the Gaussian moment integral.
A registered section or a dated entry in `research/gate_d4_registration.md`. Or a
definition stated in the emitting module's own docstring, where the reduction rests on
how that module sets a term up rather than on the analytic derivation. That field exists
because a CAS establishes that one expression equals another and has nothing to say
about whether those are the expressions the analytic claim is about. This is where that
second question is answered.

Steps 1 to 4 and `c₂` are here. `c₄` and the `tanh` worked example are appended by the
quartic work and are not in this file yet. Everything in this file is the typed up version of my notes.

## The object

A scalar latent `x` with a Gaussian prior, a scalar observation `y`, and observation
noise that depends on the state:

```text
prior       x ~ N(μ, s),              s = σ²
likelihood  y | x ~ N(x, R(x))
```

The agent does not use `R(x)`. It freezes the noise at the prior mean, `R̄ = R(μ)`, and
runs an ordinary Kalman update with that plug-in value. Call the resulting Gaussian `q`.
The exact posterior `p(x|y)` is not Gaussian, and the inference gap is what separates
them:

```text
KL(q ‖ p(·|y))  =  log E_q[e^W]  −  E_q[W]
```

where `W` is the log-ratio of the true likelihood to the plug-in one:

```text
W(x, y)  =  log N(y; x, R(x))  −  log N(y; x, R̄)
```

Reverse KL, `R` frozen at the prior mean, and the average taken under the exact
predictive. Those three conventions are pinned rather than rediscovered per caller, and
`research.checks.gap_kernel` implements the same three in quadrature.

Notation used throughout, matching the modules:

| symbol | meaning |
| --- | --- |
| `s` | prior variance, `σ²` |
| `ν` | innovation, `y − μ` |
| `h` | displacement from the **prior** mean, `x − μ` |
| `l` | `log R`, with `l₁ = l'(μ)`, `l₂ = l''(μ)`, … |
| `δ` | `l(x) − l(μ)` |
| `z` | standard normal draw under `q` |

`l₁ … l₄` stay free symbols and `R̄` is never given a value. A check that passes only at
`R̄ = 1` has lost a variable.

## Step 1: the log-ratio

Write both Gaussians out and subtract. The `(y − x)²` terms share a numerator, so only
the log-determinant and the reciprocal survive:

```text
W  =  −½·log(R(x)/R̄)  −  ½·(y − x)²·(1/R(x) − 1/R̄)
```

Define `δ = l(x) − l(μ)` with `l = log R`. Then `R(x) = R̄·e^δ` by construction, and the
first term is `−δ/2` exactly.

## Step 2: the reciprocal identity

The second term still carries `R(x)` in a denominator. Substituting `R(x) = R̄·e^δ`:

```text
1/R(x) − 1/R̄  =  (e^{−δ} − 1)/R̄
```

Exact, with no expansion. This is what lets the entire `R`-dependence of the gap be
carried by `δ` alone, so nothing downstream inherits a truncation from this step.
Substituting it, and writing `y − x = ν − h`:

```text
W  =  −δ/2  +  (ν − h)²/(2R̄) · (1 − e^{−δ})
```

That is `log_ratio(δ, h)` in `series_kernel`, and it is the only place `R` enters.

Two consequences worth stating, because both are pinned as checks. `W ≡ 0` when `δ = 0`,
identically in `ν` and `h`. Flat noise is the case where the Kalman filter is exact
(T2). And the identity above is asserted exactly rather than to some order (T1).

## Step 3: expansion in prior spread

Everything is now expanded in `σ`, with `s = σ²`.

**The gain and the posterior width.** The Kalman gain and the width of `q` are rational
in `σ²`:

```text
K      =  σ²/(R̄ + σ²)   =  σ²/R̄ − σ⁴/R̄² + O(σ⁶)
√v_q   =  σ·√(1 − K)    =  σ − σ³/(2R̄) + O(σ⁵)
```

The first is a geometric series, the second a binomial one. Both are written as explicit
polynomials rather than obtained from `sympy.series`, and each is checked against
`series` on its own small rational function (K3).

**The displacement.** Under `q`, `x = m_q + √v_q·z`, so displacement from the *prior*
mean is `h = Kν + √v_q·z`:

```text
h  =  σz  +  (ν/R̄)·σ²  −  (z/2R̄)·σ³  +  O(σ⁴)
```

The `σ²` coefficient is the Kalman shift. It is free of `z`, and it is what separates
displacement from the prior mean from displacement from the posterior mean. A derivation
that measures `h` from the wrong centre loses exactly this term. It is harmless at `σ²`
and wrong at `σ⁴`, which is why it is pinned by a check of its own (T4) rather than left
implicit.

**The noise increment.** `δ` is a Taylor polynomial in `h`:

```text
δ  =  l₁h  +  l₂h²/2  +  l₃h³/6  +  l₄h⁴/24
```

Since `h = O(σ)`, the term `l_k·h^k/k!` is `O(σ^k)`. An expansion to `σ^n` therefore
needs log-derivatives up to `l_n`. Only `l₁ … l₄` are carried, so `σ⁴` is the ceiling
and `increment_series` refuses an order above it rather than silently dropping the terms
it cannot express.

**The assembly.** Substituting `h` and `δ` into `W` and truncating gives, at the two
orders `c₂` needs:

```text
[σ¹] W  =  l₁z(ν² − R̄) / (2R̄)

[σ²] W  =  ( −R̄²l₂z²  +  R̄ν(−l₁²νz² − 4l₁z² − 2l₁ + l₂νz²)  +  2l₁ν³ ) / (4R̄²)
```

`W` carries no `σ⁰` term (T6), and the first-order term averages to zero under `q`
(T8), because it is odd in `z`. Those two together are why the gap starts at `σ²`.

**On truncation rather than `sympy.series`.** The assembled `W` is built by explicit
polynomial truncation, with the cut applied inside every product. Every factor is of
non-negative order in `σ`, so dropping high powers early cannot discard a term that would
have landed below the working order. `sympy.series` on the assembled expression is
affordable to `σ⁴` (about two seconds, against 8.5 s at `σ⁵` and 83.5 s at `σ⁶`), and
K4 uses it as an independent arm at every order the truncation path can express. The cap is `DERIVATIVE_ORDER`, not the cost of the CAS.

## Step 4: the gap as half the variance

The gap is a difference of cumulant generating functions. With `Λ(t) = log E_q[e^{tW}]`:

```text
KL(q ‖ p(·|y))  =  Λ(1) − Λ'(0)  =  Σ_{n≥2} κ_n/n!
```

so at leading order the gap is half the variance of `W` under `q`:

```text
KL  =  ½·Var_q(W)  +  O(κ₃)
```

The two routes to this are computed separately and checked against each other (C1): one
expands `log E_q[e^W] − E_q[W]` directly through the generating function, the other takes
`κ₂/2` from the cumulant recursion. They agree through `σ²` and part company at `κ₃`,
which is why the statement is scoped to `σ²` rather than asserted generally. Neither arm
calls the other, but both read the same `W` and take expectations through the same moment
operator, so the agreement separates the two routes and not the kernel beneath them.

At fixed observation:

```text
½·Var_q(W)  =  l₁²σ²(R̄ − ν²)² / (8R̄²)  +  O(σ⁴)
```

Non-negative, being a square over a positive. Still a function of `ν`: the average over
the innovation has not happened yet.

## `c₂`

Average over the innovation. At leading order the exact predictive
`ν = σz₁ + √R̄·e^{δ/2}·z₂` collapses to `N(0, R̄)`, so the average is a single Gaussian
integral in `ν/√R̄`. Using `E[ν²] = R̄` and `E[ν⁴] = 3R̄²`:

```text
E_{p*}[ (R̄ − ν²)² ]  =  R̄² − 2R̄·E[ν²] + E[ν⁴]  =  2R̄²
```

so

```text
E_{p*}[KL]  =  l₁²σ²/4  +  O(σ⁴)
```

and therefore

```text
c₂  =  l₁²/4  =  ( R'(μ) / 2R(μ) )²
```

using `l₁ = l'(μ) = R'(μ)/R(μ)`.

**The independent arm.** `research/gate_d4_registration.md`, RESULT 2026-08-07, derived
`c₂ = (R'(μ)/2R(μ))²` in closed form before this series existed. The two are the same
statement reached by different routes, which is agreement of two independently computed
closed forms rather than a tolerance being met.

**Scope of the leading-order predictive.** Collapsing the predictive to `N(0, R̄)` is
correct at `σ²` and wrong at `σ⁴`, where the neglected terms enter. The exact nesting
belongs to the work that needs it. Doing it early would put an unchecked object under a
checked result.

**Two consequences, both pinned.** For an exponential noise profile `R = A·e^{bx}` every
higher log-derivative vanishes and `c₂ = b²/4` (C7). And `c₂` is invariant under
`R → aR` for positive `a`, since `log(aR) = log a + log R` leaves every log-derivative
alone (C6).

## What this document does not establish

`c₄`. Nothing here fits, extracts or guesses one, and neither does any module citing this
file. A coefficient produced before its derivation lands becomes the thing the derivation
is then checked against, and the ledger ends up carrying a fit under a Prover 1 label.
The quartic needs three things this file does not have: the exact predictive rather than
its leading-order collapse, `κ₃` and `κ₄` rather than `κ₂` alone, and the Kalman shift in
`h` carried correctly through a term where it stops being harmless.
