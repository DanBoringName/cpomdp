# The sextic coefficient: the hand derivation

What the symbolic suite is checked against when it runs to `σ⁶`. It extends
`research/c4_hand_derivation.md`, which establishes the object, the log-ratio, the
reciprocal identity, the expansion in prior spread, the gap as a cumulant series, and
`c₂`. None of that is repeated here. Read it first.

That document closes by saying what it does not establish, and `c₄` is on the list. `c₄`
was derived on the branch recorded in `research/gate_d4_registration.md` section 7,
which also discloses that the derivation ran before anything registered it. This
document is written before the suite computes `c₆`, so that disclosure is not inherited.

The same caveat applies here as there. A CAS establishes that one expression equals
another. It has nothing to say about whether those are the expressions the analytic
claim is about. That is what this file answers.

## Why the sextic is wanted

`σ_max = √(f·c₂/|c₄|)` is the registered upper edge of the fit window, the spread at
which the quartic term reaches a fraction `f` of the quadratic. Along the ridge
`μ = μ*(κ)` of the declared family `R(x) = R₀ + κx²`,

```text
R̄ = 2R₀      c₂ = κ/(4R₀)      c₄ = 3κ(κ − 2)/(16R₀²)
```

so `c₄` is **zero at κ = 2**, where the edge diverges and the quartic bounds nothing.
`c₂²/|c₄| = κ/(3|κ − 2|)` is 0.33 at `κ = 1` and 66 at `κ = 1.99`. The sextic is what
bounds the window in that neighbourhood, and the registration already names redefining
`σ_max` against `c₆` as one of the two ways to close its open question there.

## Which cumulants reach `σ⁶`

The gap is `Σ_{n≥2} κ_n/n!` with `W = σW₁ + σ²W₂ + …`, so `κ_n(W)` starts at `σⁿ` and
the sum at `σ⁶` runs over `n = 2 … 6`.

`W₁` is linear in the standard normal `z` and so is Gaussian under `q`. Every cumulant
of it above the second vanishes. That is what confines `σ⁴` to `κ₂` and `κ₃`: `κ₄`'s
leading term is `σ⁴κ₄(W₁)`, and it is zero.

The same argument does **not** clear `κ₅` and `κ₆` at `σ⁶`. A joint cumulant of five
factors reaches `σ⁶` through the degree assignment `(1,1,1,1,2)`, which is
`κ(W₁,W₁,W₁,W₁,W₂)`. Four Gaussian arguments do not make a joint cumulant vanish when
the fifth is not one of them. `κ₆` reaches `σ⁶` only through `(1,1,1,1,1,1)`, which is
`κ₆(W₁)` and is zero.

So five cumulants are live at `σ⁶` and one is not:

| cumulant | lowest order | contributes at `σ⁶` |
| --- | --- | --- |
| `κ₂` | `σ²` | yes |
| `κ₃` | `σ³` | yes |
| `κ₄` | `σ⁴`, and that term is zero | yes, from `σ⁵` up |
| `κ₅` | `σ⁵`, and that term is zero | yes, from `σ⁶` |
| `κ₆` | `σ⁶`, and that term is zero | no |

The suite computes the gap through the generating function rather than term by term, so
this table is not an input to it. It is the accounting a reader checks the result
against, and the entry to watch is `κ₆`: a `c₆` carrying a genuine sixth-cumulant
contribution would contradict it.

## The dimensional basis

Give the state `x` a length `L`. Then `σ ~ L`, and `l_n = dⁿ log R/dxⁿ ~ L^{-n}`. With
`C = 1` the observation carries the state's units, so `R ~ L²` and `R̄^{-1} ~ L^{-2}`.

A KL divergence is dimensionless, so `c_{2n}σ^{2n}` is, and `c_{2n} ~ L^{-2n}`.

Every term the expansion can produce is a product of log-derivatives and inverse powers
of `R̄`. Writing one as `(∏ l_{n_i}) · R̄^{-k}`, the dimension is `L^{-Σn_i - 2k}`, so

```text
Σ n_i  =  2n − 2k
```

For each `k` the admissible `l`-monomials are the partitions of `2n − 2k` into parts of
size at least one, one monomial per partition. That is `p(2n − 2k)` of them.

**One monomial is excluded, and only one.** At `k = n` the constraint reads `Σn_i = 0`,
a bare `R̄^{-n}` carrying no log-derivative at all. A constant `R` has every `l_n = 0`,
and there the plug-in noise is the true noise, `q` is the exact posterior and the gap is
identically zero. A term surviving with no `l` would make it non-zero. So `k` runs to
`n − 1`.

```text
basis size of c_{2n}  =  Σ_{k=0}^{n-1} p(2n − 2k)
```

**Parity is a check rather than a constraint here.** Under `x → −x` each `l_n` picks up
`(−1)ⁿ` and the gap is unchanged, so every surviving monomial needs `Σn_i` even. The
dimensional constraint already forces `Σn_i = 2n − 2k`, which is even for every `k`. The
two agree, which is the check passing rather than a second filter reducing the count.

### Checked against the two coefficients already known

```text
c₂:  p(2)                    =  2
c₄:  p(4) + p(2)             =  5 + 2  =  7
c₆:  p(6) + p(4) + p(2)      =  11 + 5 + 2  =  18
```

The registration fixed `c₄`'s basis at **seven terms** on 2026-08-07, from a dimensional
argument with a parity check, and `research.checks.gap_series` resolves `c₄` onto those
seven with no remainder. The rule reproduces that count without being told it, and
reproduces `c₂`'s two. Neither was used to derive the rule.

`c₂ = l₁²/4` occupies one of its two admissible terms, the `l''` coefficient being zero.
A basis is what the coefficient may contain, not what it does.

### The eighteen terms

```text
k = 0    l^(6)      l' l^(5)    l'' l''''      l'² l''''    l'''²
         l' l'' l'''            l'³ l'''       l''³         l'² l''²
         l'⁴ l''    l'⁶

k = 1    l'''' / R̄  l' l''' / R̄  l''² / R̄      l'² l'' / R̄   l'⁴ / R̄

k = 2    l'' / R̄²   l'² / R̄²
```

A resolution onto these with a non-zero remainder refutes either the basis or the
expansion, and the remainder is the quantity to report rather than to absorb.

## What is new in the expansion at `σ⁶`

Three things enter that `σ⁴` did not need, and each is a place a derivation can go wrong
quietly.

**Log-derivatives to `l₆`.** The increment `δ = l(μ + h) − l(μ)` is a Taylor series in
`h`, and `h` is `O(σ)`, so the `k`-th term is `O(σᵏ)`. Reaching `σ⁶` needs `l₅` and `l₆`
carried. A module holding only `l₁ … l₄` and asked for `σ⁶` would drop the terms it
cannot express and return a polynomial that reads as complete. The suite raises there
rather than truncating silently, and that guard is the reason this is stated rather than
assumed.

**Further terms of the exact predictive.** The innovation is
`ν = σz₁ + √R̄·e^{δ/2}·z₂`, and `c₄` already required the exact form rather than its
collapse to `N(0, R̄)`: collapsing leaves `c₂` untouched and gives a `c₄` 5.7 times the
exact one. At `σ⁶` the same nesting is carried two orders further, so the exponential's
own expansion contributes where at `σ⁴` it did not.

**The Kalman shift at higher order.** The `σ²` coefficient of the displacement is the
shift separating displacement from the prior mean from displacement from the posterior
mean. Deleting it fires five checks at `σ⁴`. It enters `σ⁶` in products where it did not
appear alone, so the checks that catch its absence at `σ⁴` are not sufficient to catch a
wrong shift at `σ⁶`.

## What this document does not establish

The value of `c₆`. Nothing here fits, extracts or guesses one, and nothing here should be
read as predicting which of the eighteen coefficients are non-zero.

It also does not establish that the two coefficients reported consistent with zero in
`c₄`'s basis have sextic analogues, nor that `c₆` is direction-free. `c₂` is
direction-free and `c₄` is not, `κ₃` separating reverse from forward KL above `σ²`, so
the direction is pinned as reverse at every order and a claim about forward KL needs its
own derivation.

Nor does it settle `σ_max`. Redefining the upper edge against `c₆` is one of two options
the registration holds open, and the choice is registered there rather than here.
