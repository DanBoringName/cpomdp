# GATE-D4 pre-registration

*Opened 2026-08-07. GATE-D4 compares a certified discretisation bound `δ_ref` against R6's
inference-gap signal by a pre-agreed factor. `fep_falsification_battery.md`'s D4 entry
states the timing requirement: the factor is written down **before the bound is computed**.
This document fixes what can be fixed while neither quantity exists.*

**Why it opens this early.** D4's predicate is that the bound be small relative to R6's
signal, so evaluating the gate needs the signal, and the signal is produced by the exact
reference filter whose error the bound certifies. By the time the gate is evaluable, both
the signal and the bound exist. A factor fixed then can be sized to what the bound turned
out to be. The only window in which neither quantity exists is the one before the reference
filter is built, which is now.

**Why it opens partially.** Section 1 is dated today and the rest is not, deliberately. The
family is the largest free parameter in the construction: `c₂`, `c₄` and the curvature
ceiling are all functions of it, so choosing the family chooses the bar. Holding the
declaration back until the coefficients were in hand would mean choosing the family with its
consequences in view. Versioning catches a family *changed* after results. Nothing catches
one *chosen toward* them except declaring it first.

Sections marked **OUTSTANDING** carry their method and their trigger, and each is dated when
it lands.

---

## 1. The model family (DECLARED 2026-08-07)

**`d4-family-v1`**

```text
scalar chain, 1-D latent:   A = B = C = 1,  Q = 1
sensor noise:               R(x) = R₀ + κ·x²          (R₀ = 1)
swept axis 1 (spread):      prior variance σ²
swept axis 2 (curvature):   κ,  where R''(x) = 2κ
```

**Selection reason, prior to and independent of D4.** This is the family Paper 1's `R(x)`
result already uses. `tests/test_theorem.py` declares it as "the scalar chain the worked
example uses: A = B = C = Q = 1, R(x) = 1 + x²", and has done since 2026-06-19, before
GATE-D4 was scoped or named. The only generalisation here is promoting the fixed coefficient
on `x²` to a swept parameter `κ`, which D2 requires because it sweeps the curvature of `R`.
`κ = 1` recovers the worked example exactly.

It also sits inside the reference filter's declared scope, which `warrant_ledger.md`
section 7 limits to roughly one or two latent dimensions.

**What a later change costs.** The version tag is what makes a changed family appear in a
diff rather than in prose (standing prohibition 9). A replacement family requires its own
declaration and its own dated reason, written *before* any measurement on it.

## 2. The Taylor coefficients `c₂` and `c₄` (OUTSTANDING)

The inference gap as a function of belief spread, expanded about zero:

```text
gap(σ) = c₂·σ² + c₄·σ⁴ + O(σ⁶)
upper window edge (spread axis) = √(c₂/|c₄|)
```

`|c₄|` rather than `c₄`. The edge is the spread at which the quartic term matches the
quadratic, and that crossing exists either sign: a negative `c₄` cancels the quadratic there
rather than overtaking it, and it is the same breakdown scale.

**Method.** Nest `jax.jacfwd` four times on the closed-form fixed-`R` path, where the Kalman
filter *is* the exact Bayesian filter, giving both coefficients to machine precision with no
step size. Fourth-order finite differencing would amplify round-off as `h⁻⁴` and yield a
step-size-dependent number with no clean error statement, which is a poor thing to hand a
gate about certified numerical error.

Two traps named in advance. `c₂ = f''(0)/2` and `c₄ = f⁗(0)/24`. Using raw derivatives puts
a factor of 12 in the ratio and moves the upper edge by `√12 ≈ 3.5×`. And if the gap is even
in belief spread by symmetry, the odd derivatives vanish identically under autodiff, which
is a free correctness check to run before trusting the even ones.

**Diagnosis rule, registered in advance (ledger section 8, third standing rule).** A
finite-difference cross-check runs as an independent method with a tolerance stated before
it runs. Disagreement *inside* that tolerance is reported as the expected discrepancy.
Disagreement *outside* it is attributed **first to an implementation error in one of the
two**, not to precision. Registering "autodiff wins" would be a rule for walking past a bug.

**Trigger.** If the coefficients are not solved by end of day two, register `T` below with
`c₂/|c₄|` bracketed numerically rather than solved. Still a pre-registration, still dated
before the reference filter exists, weaker only in the tightness of the bracket.

## 3. The curvature ceiling (SETTLED 2026-08-07): vacuous for this family

The belief-smoothed rung needs `E[R(x)]` under the Gaussian prediction to exist, and H1 puts
no ceiling on how fast `R` may grow, so a family can lose a rung at high curvature. That is
a second, independent way D1 and D2 can fail to be tests (`warrant_ledger.md` section 5).

For `d4-family-v1` it does not arise. With `x ~ N(μ, σ²)`,

```text
E[R(x)] = R₀ + κ·(μ² + σ²)
```

finite for every finite `κ` and `σ²`. The rung cannot drop off the ladder on this family at
any curvature, so the second stop branch is closed by construction rather than by
observation. A family with `R` growing faster than `exp(x²/2σ²)` would reopen it, which is
one more thing a replacement family would have to re-establish.

## 4. `D`, `k_min` and the threshold `T` (OUTSTANDING, needs section 2)

The gate is stated as an **absolute threshold**, not as a factor:

```text
window edges (spread axis):  lower √(k·δ_ref / curvature),  upper √(c₂/|c₄|)
at exactly D decades:        k·δ_ref = (c₂/|c₄|)·curvature·10^(−2D)  ≡  T

GATE-D4 passes iff:          gap > T
D1/D2 are tests iff:         δ_ref ≤ T / k_min
reported diagnostic:         k = T / δ_ref
```

`T` contains no `δ_ref`. It is a constant of the declared family and `D`, computable before
the reference filter exists, and nothing about it moves when the filter lands. That is what
makes the gate immune to the measurement rather than merely arguably immune.

**`D` is not neutral, and this document will not claim otherwise.** Since `T ∝ 10^(−2D)`, a
larger `D` lowers the threshold monotonically, so more decades makes GATE-D4 easier *and*
D2's fit better. Both pressures push the same way and nothing bounds `D` from above. `D` is
declared on D2 fitting grounds, and it must be frozen and independently defensible for that
reason rather than because it is neutral. This registration prints `T` at `D−1`, `D` and
`D+1`, since one decade moves `T` by 100× and a reader should see that rather than infer it.

**`k_min`** is the floor below which C6/R6 stops being a test, since at `k = 1` the claim is
a bare inequality between two computed numbers. It is not derivable, but it is declarable on
grounds independent of D4.

**`D` and `k_min` are frozen at declaration.** Stated explicitly, because the move available
on seeing a disappointing `δ_ref` is to lower `D`.

## 5. D1's resolution threshold (OUTSTANDING): an expression, not a value

Registered as a **propagated expression**, not a constant. A formula fixed before the
measurement has no knobs left to turn.

The four rungs are all scored against one reference filter, so `δ_ref` is common-mode and
largely cancels in their differences. The error on a difference is therefore far smaller
than `δᵢ + δⱼ`, and the sum-of-bars rule is safe but understates the test
(`warrant_ledger.md` section 4). What gets registered is the residual differential
sensitivity, derived and written down before any rung is evaluated.

## 6. D2's exponent interval (OUTSTANDING): derived, not chosen

"A pre-registered interval around 2" leaves the width unstated, which is the same
choice-versus-rule problem sections 4 and 5 exist to remove. The width follows from the fit
design already being declared: `D` decades, a declared point count and a declared residual
model give a standard error on the fitted exponent, and the interval is a stated multiple of
it. `D` then does double duty.

Also registered here: the two failure modes the battery already calls *boring* and excludes
from the agreement criterion in advance.

## Stop conditions (DECLARED 2026-08-07)

Three branches, disjoint, split by which half of the instrument failed.

| branch | evaluable | report |
| --- | --- | --- |
| `c₂ ≤ 0` | analytically | quadratic dominance refuted. A finding about the geometry, and it belongs in the paper |
| the curvature ceiling binds inside the swept range | analytically | the belief-smoothed rung drops off the ladder, and D1 reports on three rungs or `VOID`. **Closed for `d4-family-v1` by section 3** |
| `δ_ref > T / k_min` | after the reference filter | the filter is not accurate enough for D1 and D2 to be tests at `D` decades. `VOID` on unmeasurability, **not** a gate `FAIL` |

The first two are family-side. The third is filter-side and is the likelier class.

There is no family-only way to empty the window: as `δ_ref → 0` the lower edge goes to zero
and the window has unboundedly many decades, so with `c₂ > 0` and finite curvature the
family alone cannot empty it. Emptiness is always the third row.

**The response to an empty window is not "pick another family"** unless the replacement and
its reason were declared first. Otherwise the stop condition becomes the contamination route
it exists to close.

## What was known when

| date | fixed | not yet known |
| --- | --- | --- |
| 2026-08-07 | the family and its version, the curvature ceiling being vacuous, the three stop branches, and the gate's form as `gap > T` | `c₂`, `c₄`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |

Every number GATE-D4 turns on was unknown on the date the family was declared. That is the
claim this document exists to make checkable, and this file's git history is the evidence.
