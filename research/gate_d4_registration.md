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

### AMENDMENT 2026-08-07: the prior mean is part of the family, and `R'(μ) ≠ 0` is required

`d4-family-v1` as declared above left the prior mean `μ` unstated. It cannot be left
unstated, because it decides the exponent D2 sets out to fit.

Measured on the quadrature gap of section 2, at `κ = 1`, as `σ → 0`:

| prior mean | local exponent `d log gap / d log σ` | consequence |
| --- | --- | --- |
| `μ = 0` | → 4 | `c₂ = 0`. D2 fits 4, not 2 |
| `μ = 1` | → 2 | `c₂ ≠ 0`. D2 fits 2, as predicted |

The reason is structural. `R(x) = R₀ + κx²` has `R'(0) = 0`, so at `μ = 0` the belief sits
exactly on `R`'s minimum, `R` varies only quadratically across it, and the gap drops two
orders. Section 2's law makes this exact: `c₂ = (R'(μ)/2R(μ))²`, which vanishes precisely
at a stationary point of `R`.

**`d4-family-v1` is amended to fix `μ = 1`**, recovering `R(x) = 1 + x²` evaluated away from
its minimum, with `R'(1) = 2κ ≠ 0`. `tests/test_theorem.py`'s `scalar_chain` defaults to
`mean=0.0`, so Paper 1's worked example sits at the degenerate point. The
generalisation to `μ = 1` is part of this declaration and not inherited from it.

**Discovery order, stated plainly.** `μ` was fixed *after* observing the exponent collapse
at `μ = 0`, not before. Two things bound what that could have bought. The ground is
structural and checkable without any measurement, since `R'(μ) ≠ 0` is a property of the
declared `R` and nothing else. And `δ_ref`, R6's signal and the threshold `T` did not exist
on this date and still do not, so no choice here could be aimed at the gate's outcome. What
it does affect is whether D2 is a test at all, which is the thing a reader should weigh.

## 2. The Taylor coefficients `c₂` and `c₄` (OUTSTANDING)

The inference gap as a function of belief spread, expanded about zero:

```text
gap(σ) = c₂·σ² + c₄·σ⁴ + O(σ⁶)
upper window edge (spread axis) = √(c₂/|c₄|)
```

`|c₄|` rather than `c₄`. The edge is the spread at which the quartic term matches the
quadratic, and that crossing exists either sign: a negative `c₄` cancels the quadratic there
rather than overtaking it, and it is the same breakdown scale.

### AMENDMENT 2026-08-07: the registered method was wrong

As first written, this section said to nest `jax.jacfwd` on "the closed-form fixed-`R` path,
where the Kalman filter *is* the exact Bayesian filter". That path returns zero. Under fixed
`R` with `p = p*` and exact inference, both divergences are **identically zero by
construction**, which is R1's content. The gap exists only under `R(x)`, where the exact
posterior is non-Gaussian. The original method computed the wrong quantity, and it is
recorded here rather than overwritten.

**The method that works.** Build the gap by quadrature: the exact posterior
`p(x|y) ∝ N(x; μ, σ²)·N(y; x, R(x))` on a grid, the agent's Gaussian `q` from a Kalman
update with the plug-in `R̂ = R(μ)`, then `E_{y∼p*}[KL(q ‖ p)]` with the reverse KL PR-7
specifies. It converges to twelve digits at a few thousand grid points.

One precondition the grid imposes: `R` must stay positive across the whole quadrature span,
not merely near `μ`. A family that violates it silently produces `log` of a negative number,
and the span is wide (many `σ`) precisely where `σ` is large.

Two traps still stand. `c₂ = f''(0)/2` and `c₄ = f⁗(0)/24`. Using raw derivatives puts a
factor of 12 in the ratio and moves the upper edge by `√12 ≈ 3.5×`. And a free correctness
check remains available wherever the gap is even in belief spread.

**Diagnosis rule, registered in advance (ledger section 8, third standing rule).** A second
independent method cross-checks with a tolerance stated before it runs. Disagreement
*inside* that tolerance is reported as the expected discrepancy. Disagreement *outside* it
is attributed **first to an implementation error in one of the two**, not to precision.
Registering "one method wins" would be a rule for walking past a bug.

### RESULT 2026-08-07: `c₂` in closed form

```text
c₂ = ( R'(μ) / (2·R(μ)) )²
```

Half the log-derivative of `R` at the prior mean, squared. Verified against the quadrature
gap on five structurally different `R` families (quadratic, exponential, linear, cubic, and
the quadratic at nine `(μ, κ)` combinations), agreeing to five significant figures in every
case. It is a general law of the construction, not a fit to one family.

Three consequences.

**`c₂ ≥ 0` always**, being a square. So the stop branch "`c₂ ≤ 0`" collapses to `c₂ = 0`,
which holds exactly when `R'(μ) = 0`. That is a structural condition checkable from the
declared `R` without any measurement, and it is what the section 1 amendment acts on.

**For `d4-family-v1`, `c₂ = κ²μ² / (R₀ + κμ²)²`.** At `μ = 1, κ = 1` that is exactly `0.25`,
and the quadrature returns `0.250009`.

**The battery's stated scaling is wrong on the curvature exponent.** D2 predicts
`gap ∝ (curvature of R) × (belief spread)²`. The spread exponent of 2 is confirmed. The
curvature dependence is not: `c₂` is quadratic in `κ` at small `κ`, not linear, and the
quantity that actually governs the gap is the *relative gradient* `R'/R` rather than the
curvature `R''`. This is a correction to a registered prediction, found analytically before
any measurement, and it bears on how D2's second axis should be swept.

### AMENDMENT 2026-08-07: the upper edge, and why `√(c₂/|c₄|)` is unusable

`c₄ < 0` is the measured case, not a defensive possibility, and it breaks the edge as first
written. With `gap = c₂σ² + c₄σ⁴` and `c₄` negative, `σ = √(c₂/|c₄|)` is where the quartic
*cancels* the quadratic, so the expansion predicts a gap of zero there. The gap turns over
earlier still, at `σ = √(c₂/2|c₄|)`, and decreases beyond it. The old edge therefore sits
`√2 ≈ 1.414×` past the point where the gap stops being monotone in spread, which is fatal
for fitting a power-law exponent.

**Redefined.** The edge is the spread at which the quartic correction reaches a declared
fraction `f` of the quadratic term:

```text
σ_max = √( f · c₂ / |c₄| )
```

`f` is declarable on the same footing as `D`, being a statement about how well exponent-2
must hold across the fit range, and it carries no route to D4's outcome for the same reason.

**`f` must be settled before `D` is frozen.** It moves the edge directly: `f = 0.1` scales it
by `√0.1 = 0.316`, costing exactly half a decade, which comes straight out of `D`. Freezing
`D` first would freeze it against an edge that then moves.

The threshold picks up `f` with it:

```text
T = f · (c₂/|c₄|) · curvature · 10^(−2D)
```

### AMENDMENT 2026-08-07: `μ` by rule rather than by value

The first amendment fixed `μ = 1`. That was a value, and `R'(μ) ≠ 0` rules out `μ = 0`
without picking `1`. Since `c₂` depends on `μ` quantitatively, `μ` moves `T`, which is the
same residual freedom the threshold construction exists to remove.

**The rule: `μ` sits at the argmax of `|R'/2R|`.** For this family,

```text
μ* = √(R₀/κ)        c₂(μ*) = κ / (4·R₀)
```

Both verified exactly against a numerical maximisation at `κ ∈ {0.25, 0.5, 1, 2, 4}`. The
rule is family-general, derivable now, and severity-maximising in `c₂`, since a larger `c₂`
raises `T`. At `κ = 1, R₀ = 1` it returns `μ* = 1`, so the earlier value coincides with the
rule at that point rather than being contradicted by it.

**A consequence for the sweep.** `μ*` depends on `κ`, and `κ` is a swept axis, so `μ` must
co-vary with `κ` along it rather than staying pinned. Holding `μ` fixed would put a single
`κ` at the optimum and the rest off it.

**And it changes the curvature story again.** At the optimum `c₂ = κ/(4R₀)` is *linear* in
`κ`. The battery's `gap ∝ curvature × spread²` is therefore recovered along the optimal
ridge, even though the mechanism it names remains wrong: what governs the gap is the
relative gradient `R'/R`, and linearity in `κ` is a property of the ridge rather than of the
curvature.

**Declared target.** `μ` maximises `c₂`, not `c₂/|c₄|`. `c₄` does not exist yet, so `c₂` is
the only target that can be declared before the reference filter, and declaring the other
later would be choosing the operating point with its consequences in view. `T`'s sensitivity
to `μ` is printed beside its sensitivity to `D`.

### `c₄` (OUTSTANDING)

Not yet in closed form, and harder than `c₂` for a reason worth recording: **`c₄` is not a
function of `R'/R` and `R''/R` alone.** Two exponential cases with identical values of both
ratios, differing only in `R(μ)`, return different `c₄`. So the basis needs `R(μ)` itself,
and a first fit on `{a⁴, a²b, b²}` with `a = R'/R`, `b = R''/R` leaves residuals of 24%.

The numerics are also not yet converged: two quadrature settings gave `c₄ = −0.2615` and
`−0.1941` at `μ = 1, κ = 1`, a 35% spread. `c₂` was stable to five digits under the same
change, so the instability is `c₄`'s own and has to be resolved before any number is
registered from it.

**The basis is fixed by dimensions, so the refit is a linear solve rather than a search.**
With `y = x + v` and `var(v) = R(x)`, `R` carries units of `x²`, so `1/R` has the units of
`ℓ''` and `ℓ'²`, where `ℓ = log R`. Every term in `c₄` must have units `x⁻⁴` *and* vanish
when `R` is constant, which is R1 again. That gives seven terms:

```text
ℓ'⁴,  ℓ'²ℓ'',  ℓ''²,  ℓ'ℓ''',  ℓ'''',  ℓ'²/R(μ),  ℓ''/R(μ)
```

A bare `1/R²` is excluded because it survives constant `R`. This also explains the
`R(μ)` dependence found empirically rather than leaving it a curiosity, and it recasts `c₂`
as `ℓ'²/4`.

The exponential family isolates two coefficients directly: for `R = A·e^{bx}` every log-
derivative above the first vanishes, so `c₄ = α·b⁴ + ζ·b²/R(μ)`. Same `b` with different
`A` moves only `R(μ)` and gives `ζ`. A second `b` at fixed `A` then gives `α`.

**Extraction, corrected.** The gap converges to twelve digits while `c₄` moved 35%, so the
loss is in extraction rather than quadrature. Two fixes, both available now that `c₂` is
closed-form. Subtract the dominant term exactly and fit `gap(σ) − c₂σ²`. And fit at *large*
spread rather than small: as `σ → 0` the ratio `c₄σ⁴/c₂σ²` vanishes and the quartic sits
below the quadrature floor, which is exactly the setting-dependence observed. Choose `σ`
just under `σ_max`, where the quartic is a few percent of the total. The quartic is only
visible where it is nearly significant.

**Trigger.** If `c₄` is not solved by end of day two, register `T` with `c₂/|c₄|` bracketed
numerically rather than solved. Still a pre-registration, still dated before the reference
filter exists, weaker only in the tightness of the bracket.

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
at exactly D decades:        k·δ_ref = f·(c₂/|c₄|)·curvature·10^(−2D)  ≡  T

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

**`D`, `f` and `k_min` are frozen at declaration**, and `f` is settled first, since it
moves the edge that `D` is counted across.

Stated explicitly, because the move available on seeing a disappointing `δ_ref` is to lower
`D`, or to loosen `f` so that `D` survives.

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
| `c₂ = 0`, equivalently `R'(μ) = 0` | analytically, from the declared `R` alone | the sweep sits on a stationary point of `R` and the spread exponent is 4, not 2. Structural, so it is caught at declaration rather than measured. **Closed for `d4-family-v1` by the section 1 amendment** |
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
| 2026-08-07 (opened) | the family and its version, the curvature ceiling being vacuous, the three stop branches, and the gate's form as `gap > T` | `c₂`, `c₄`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (amended) | the quadrature method replacing one that computed zero, and `c₂ = (R'(μ)/2R(μ))²` | `c₄`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (amended again) | `σ_max = √(f·c₂/\|c₄\|)` replacing an edge that sat past the gap's turnover, `μ` by the argmax rule `μ* = √(R₀/κ)` rather than by value, and `c₄`'s seven-term dimensional basis | `c₄`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |

Every number GATE-D4 turns on was unknown on the date the family was declared, and `T`,
R6's signal and `δ_ref` are unknown still. That is the claim this document exists to make
checkable, and this file's git history is the evidence.

Amendments are appended and dated rather than folded into the text they correct, so a
reader can see what was believed when. Two stand so far: the prior mean was missing from
the family, and the registered method for `c₂` and `c₄` computed a quantity that is zero by
construction.
