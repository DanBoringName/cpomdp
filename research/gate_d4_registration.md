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
derived (not swept):        μ = μ*(κ) = √(R₀/κ)      -- see the amendments
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

### AMENDMENT 2026-08-07: `μ` is *derived*, which E2's dichotomy has no slot for

Once `μ = μ*(κ)`, `μ` stops being an independent swept axis: the spec named two, and one is
now a function of the other. The family block above is corrected to say so.

E2 requires every parameter "declared **fixed or free in the model specification**, never at
analysis time". `μ` is neither. It is **derived**, and that is a third category with its own
contamination profile: it cannot be tuned directly, which is stronger than *free*, but it
inherits whatever freedom sits in the rule that derives it, which is weaker than *fixed*.
The freedom here is the choice to maximise `c₂`, declared above and dated.

### AMENDMENT 2026-08-07: `R'(μ) ≠ 0` is a precondition on any fixture without a mean-moving action

Paper 1's worked example is safe at `mean = 0.0` because its *actions* displace the
predicted mean, so `R(μ⁻)` takes 1 and 5 and `R'(μ⁻) ≠ 0` where it matters. That
generalises: the degeneracy arises exactly where nothing displaces the mean.

**So any fixture without a mean-moving action asserts `R'(μ) ≠ 0` rather than assuming it.**
Checkable from the declared `R` alone, at no cost, and it joins the span-positivity
precondition already recorded from the linear family. Both were found the same way in one
sitting, which suggests the class is worth checking for rather than waiting to trip over.

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

A bare `1/R²` is excluded because it survives constant `R`. A parity check confirms the
enumeration is complete rather than merely sufficient: for symmetric `R`, counting `ℓ'` and
`ℓ'''` as odd, all seven terms are even under `x → −x`, nothing of odd parity can appear,
and nothing of even parity at that dimension is missing. This also explains the
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
window edges (spread axis):  lower √(k·δ_ref / c₂),  upper √(f·c₂/|c₄|)
at exactly D decades:        k·δ_ref = f·c₂²/|c₄|·10^(−2D)  ≡  T      (see the audit)

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

### AMENDMENT 2026-08-07: `T` rests on three declared parameters, and the span is 10⁴

The opening framing had `T` fixed by the family's analytic properties and `D`. That is no
longer true. `T = f·(c₂/|c₄|)·curvature·10^(−2D)` and `c₂` depends on the `μ`-rule, so three
things are declared: **`D`, `f`, and the `μ`-rule**. Each is individually defensible and all
three precede `δ_ref`, so the construction holds. Presenting one while carrying three would
not.

Joint sensitivity of the known factors `f·c₂·10^(−2D)`, with `|c₄|` still pending, over
defensible ranges `D ∈ [1.5, 3.0]`, `f ∈ [0.02, 0.2]`, and the `μ`-rule against a pinned
`μ = 1`:

| knob | range | factor on `T` |
| --- | --- | --- |
| `D` | 1.5 to 3.0 decades | **1000×** |
| `f` | 0.02 to 0.2 | 10× |
| `μ`-rule vs pinned `μ = 1` | at `κ ∈ {0.25, 1, 4}` | 1.56× at most, and exactly 1× at `κ = 1` |
| **joint** | | **≈ 1.0 × 10⁴** |

The answer to "is the construction robust" is no, and it is worth having before the
reference filter rather than after. Essentially the whole span is `D`: it moves `T` by 100×
per decade while `f` and the `μ`-rule together contribute about 16×. **`D`'s justification
carries the severity almost by itself**, and the fitting argument behind it has to be made
to that standard. `f` and the `μ`-rule are second-order by comparison.

### AMENDMENT 2026-08-07: D2 would pass for the wrong reason, and gets a second leg

Along the ridge `μ = μ*(κ)`, `c₂ = κ/(4R₀)` is linear in `κ`, so the measured behaviour is
`gap ∝ κ·σ²`. That is the battery's registered D2 prediction *verbatim*. D2 would report
`NOT TRIGGERED` on a mechanism already refuted analytically above, which is a severity
failure rather than a pass: a falsifier that cannot fire on a true refutation is not testing
anything.

**Two corrections, both registered now.**

D2's prediction is restated to cover the **σ-exponent only**, with the κ-dependence
explicitly out of scope on the ridge, since linearity there is a property of the ridge
rather than of the mechanism.

And a **second leg is added: a κ-sweep at pinned `μ`**, off the ridge. There
`c₂ = κ²μ²/(R₀+κμ²)²`, which is quadratic in `κ` at small `κ`, so the two mechanisms
separate on an axis already being swept. Measured local exponent `d log c₂ / d log κ` at
`μ = 1`:

| `κ` | 0.05 | 0.10 | 0.20 | 0.40 | 0.80 |
| --- | --- | --- | --- | --- | --- |
| exponent | 1.87 | 1.75 | 1.56 | 1.28 | 0.94 |

Approaching 2 as `κ → 0` off-ridge, against exactly 1 on it. The leg therefore needs small
`κ`, roughly `κ ≤ 0.1`, to resolve cleanly. This converts an analytic refutation into a
measured one and costs one extra sweep on an existing axis.

### AMENDMENT 2026-08-07: corrections to the sensitivity, and what actually sets `D`

**The `μ`-rule factor was understated, and understated exactly where the new leg lives.**
The ratio is `c₂(μ*)/c₂(μ=1) = (R₀+κ)²/(4R₀κ)`, which by AM-GM is `≥ 1` everywhere and
equals 1 only at `κ = R₀`, where `μ* = 1` and the rule coincides with the pin. It diverges
as `κ → 0`.

| `κ` | 0.05 | 0.10 | 0.25 | 1.00 | 4.00 |
| --- | --- | --- | --- | --- | --- |
| factor | 5.51× | 3.03× | 1.56× | 1.00× | 1.56× |

The previous amendment computed 1.56× over `κ ∈ {0.25, 1, 4}`. The D2 leg registered in the
same sitting needs `κ ≲ 0.1`, where the factor is 3.03× to 5.51×. **The joint span is
therefore nearer 5 × 10⁴, not 1 × 10⁴.** The conclusion is unchanged, `D` still carrying
essentially all of it. Worth recording is the mechanism: two amendments written in one
sitting, and one silently invalidated the other's inputs by moving the `κ` range it was
computed over.

**`T` is a single number and `c₂` is not.** Both `c₂` and `c₄` vary along the `κ` sweep, so
`σ_max` varies and the decades between the edges vary with it. `T` is not well-defined until
the `κ` is named. **Registered rule: `T` is evaluated at the `κ` minimising the window
width**, the binding cell, rather than at a reference `κ` or the sweep's midpoint. That
makes `T` the most demanding admissible value and puts the choice beyond argument. It is
derivable once `c₄` lands, so the rule is declared now and evaluated after.

### `D` is a bias argument, not a statistics argument

This is why `D` read as freely choosable. For an OLS log-log fit over `D` decades with `N`
points and relative error `ε`, the standard error on the exponent is about `ε√12/(D√N)`. The
quadrature gap converges to twelve digits, so `ε ~ 10⁻¹⁰` and the random error is nowhere
near binding. **A statistical argument gives no lower bound on `D` at all.**

What binds is systematic. The gap is not a power law: with `a = |c₄|/c₂`,

```text
gap = c₂σ²(1 − aσ²)
local log-log slope = 2 − 2aσ²/(1 − aσ²)
at the top of the window (aσ_max² = f):  2 − 2f/(1−f)
```

At `f = 0.1` the local exponent at the top is 1.778, not 2. The fitted exponent is biased
low, and `D` has to buy that bias down rather than buy down noise.

**The exact bias, from the OLS geometry.** With `v = ln σ` uniform over `L = D·ln10`, the
first-order-in-`f` result is `b = −3f/(D·ln10)²`. That envelope has the right shape and the
wrong size:

| | `D=1` | `D=2` | `D=3` |
| --- | --- | --- | --- |
| `f = 0.02` | 1.71× | 1.27× | 1.16× |
| `f = 0.10` | 1.67× | 1.24× | 1.14× |

overstating the true bias by 14% to 71%, worst at small `D`. **The exact OLS integral is
what gets registered**, not the closed form, which stands only as a scaling guide.

**`D`, `f` and D2's interval are one declaration, not three.** Declare the bias budget `β`,
how far the fitted exponent may sit from 2 through truncation alone. Then `D` follows from
`f` through the bias relation, and `f` follows from maximising `T` subject to it. Since
`T ∝ f·10^(−2D)`, the smallest `D` the budget allows is the severity-maximising choice, which
is the same shape as every other rule here.

Carrying that through gives a closed form, verified numerically at `β ∈ {0.01, 0.02, 0.05}`:

```text
f* = β/3        D* = 1/ln10 = 0.4343 decades        independent of β
```

A window of exactly `e ≈ 2.718` in spread. **This shrinks the 10⁴ span**, which assumed the
three knobs move independently when they cannot.

**An open decision, to be taken before any of the three is declared.** The optimum above
*dilutes* the truncation bias by widening the window. Since `c₄` will be known, the bias can
instead be **subtracted analytically**, which removes it rather than diluting it and
decouples `D` from `f` entirely. The two constructions give different `D`, so choosing
between them after seeing either would be choosing with the consequences in view. There is
also a tension the optimisation does not see: 0.43 decades is statistically ample given
`ε ~ 10⁻¹⁰`, and is a short range to present as a power-law fit. That is a judgement about
what a reader accepts rather than a fact the algebra supplies, and it belongs in the
declaration rather than in a footnote.

**D2's interval inherits this.** Whatever bias survives, the registered interval must
accommodate it, or the leg fires on truncation rather than on the world. Same coupling,
arriving at the registration rather than at the derivation.

### AMENDMENT 2026-08-07: `T` carries `c₂` squared, and "curvature" was the refuted mechanism's name

The lower edge is where the gap clears the error: `gap = c₂σ² = k·δ_ref`, so
`σ_min = √(k·δ_ref/c₂)`. `warrant_ledger.md` section 5 writes it as `√(k·δ_ref/curvature)`,
which is the same thing *only if* "curvature" means `c₂`. Working `T` through both edges:

```text
σ_min² = k·δ_ref/c₂        σ_max² = f·c₂/|c₄|        ratio = 10^(2D)

T  =  k·δ_ref  =  f · c₂² / |c₄| · 10^(−2D)
```

**`c₂` squared**, because it appears in both edges: once setting where the signal clears the
error, once setting where the expansion breaks. Verified numerically, `c₂·σ_min² = T` exactly.

The earlier form in section 4 carried `curvature` as a separate factor beside `c₂/|c₄|`,
inherited from the ledger. That is wrong either way. If "curvature" means `c₂` the formula
was right and misnamed after the very mechanism refuted above. If it means something
`R''`-flavoured, the lower edge inherited the falsified mechanism and is simply wrong. The
ledger's wording is audited before `T` is declared.

**This is the third instance of the tracked pattern** of mechanism wording outliving its
falsified rationale, after the epistemic-pull phrasing and the "optimal reach" row. The
first two were presentational. This one is load-bearing.

**Sensitivity, again revised.** The `μ`-rule now enters `T` squared: `5.51² = 30.4×` at
`κ = 0.05`, `9.2×` at `κ = 0.1`. Joint span `1000 × 10 × 30 ≈ 3 × 10⁵`. The line that `D`
carries the severity almost by itself needs re-checking rather than re-quoting: `D` still
dominates, and `30×` is no longer a rounding error.

### AMENDMENT 2026-08-07: the fitting argument used the wrong `ε`

`ε ~ 10⁻¹⁰` is the *quadrature's* precision. D2 measures against the reference filter, and
the lower edge is *defined* as where the gap is exactly `k` times `δ_ref`. So the relative
error on the gap at the bottom of the window is `1/k` by construction, and 10% at `k = 10`.
It improves as `σ²` across the window, reaching `1/(k·10^(2D))` at the top.

Propagating that through OLS with heteroscedastic weights, at the bias-only optimum
`D = 0.4343`:

| `k` | 5 | 10 | 30 |
| --- | --- | --- | --- |
| `σ_p` | 0.089 | 0.045 | 0.015 |

`σ_p = 0.045` at `k = 10` exceeds every `β` tested (0.01, 0.02, 0.05). **The statistical term
does not merely bind, it dominates the bias it was dismissed in favour of.** `D* = 0.4343`
answered a bias-only optimisation the measurement will not be operating under.

Re-optimising `T` subject to `√(bias² + σ_p²) ≤ β`:

| `β` | `k = 10` | `k = 30` |
| --- | --- | --- |
| 0.02 | `D* = 0.97`, `f* = 0.031` | `D* = 0.47`, `f* = 0.019` |
| 0.05 | `D* = 0.51`, `f* = 0.045` | `D* = 0.26`, `f* = 0.035` |

Both terms fall with `D`, the bias as `1/D²` and the noise as `1/D`, so the corrected optimum
is generally larger, `T` is smaller, and the construction is less severe and more honest. The
presentational worry about a 0.43-decade window largely dissolves at the values that matter,
which is why that argument is not spent here.

**A consequence to accept.** `σ_p` depends on `k`, hence on `δ_ref`, hence on the reference
filter. So `D` is registered as an **expression in `k`, evaluated conservatively at `k_min`**,
since smaller `k` means more noise and demands more decades. That is the
register-expressions-not-constants discipline arriving where a number was hoped for.

### AMENDMENT 2026-08-07: dilute-versus-subtract becomes a rule

The decision was flagged rather than taken. It converts to a criterion.

Subtraction uses `c₄` to correct the fit, so the residual bias after subtracting is `c₄`'s own
relative error times the bias. At `c₄`'s current 35% that buys a factor of about 3, not
removal. Dilution buys `1/D²` with no dependence on `c₄`'s accuracy at all.

**Registered rule: subtract if and only if `c₄`'s relative error is below `X`, dilute
otherwise.** `X` is declared now, before the refit reveals which side the result lands on.
Same move as `f*` and `μ*`: a choice that would otherwise be made with its consequences in
view becomes a rule fixed before they are visible.

### AMENDMENT 2026-08-07: the closed form does not survive the exact bias, and `D` is set by noise

The convergence check proposed for the re-optimisation was that switching the noise off
should recover the bias-only closed form `D* = 1/ln10 = 0.4343`. **It fails.**

| `k` | 10 | 30 | 100 | 1000 | 10⁶ |
| --- | --- | --- | --- | --- | --- |
| `D*` at `β = 0.02` | 0.960 | 0.436 | 0.204 | 0.150 | 0.150 |

As `k → ∞` the optimum runs to the search boundary, not to 0.4343. The agreement at `k = 30`
is a coincidence of where the noise constraint happens to bind.

**The cause.** `D* = 1/ln10` was derived from the *first-order-in-`f`* bias `3f/(D·ln10)²`,
already recorded above as overstating the true bias by 14% to 71%. At small `D` it is far
worse: at `D = 0.15` the exact integral admits `f = 0.0138` where the analytic permits
`0.0008`. Taking the largest feasible `f` at each `D` and forming `T ∝ f·10^(−2D)`:

| `D` | 0.15 | 0.434 | 1.0 | 2.0 | 3.0 |
| --- | --- | --- | --- | --- | --- |
| `T`, exact bias | 6.9e−3 | 3.3e−3 | 6.0e−4 | 1.7e−5 | 3.4e−7 |
| `T`, first-order bias | 4.0e−4 | 9.0e−4 | 3.5e−4 | 1.4e−5 | 3.2e−7 |

The first-order column has an interior peak at 0.4343. The exact column is **monotone
decreasing**, so the bias-only problem has no interior optimum at all: `T` is maximised by
shrinking the window to nothing.

**Two things follow, and the second inverts an earlier conclusion.**

`f* = β/3` and `D* = 1/ln10` are **withdrawn as results**. They stand only as a first-order
illustration, and the exact OLS integral governs.

And the bias-only problem is **ill-posed**, since a zero-width window fits no exponent.
What makes it well-posed is the noise term, which diverges as `D → 0`. So *noise*, not bias,
is what sets `D` from below. The earlier amendment concluded the opposite, and did so
resting on the same first-order approximation this one retires.

### AMENDMENT 2026-08-07: the numbers, committed before the refit

Anything the `c₄` refit could inform is fixed here, as figures rather than intents. A rule
whose deciding quantity is already on screen is not a rule.

| symbol | value | derivation |
| --- | --- | --- |
| `k_min` | **10** | the gap clears the certified error by one order of magnitude. At `k = 1` the claim is a bare inequality between two computed numbers, which is the defect the flip-margin bar was introduced to fix. One order is the smallest round separation that makes "separated from zero" mean anything |
| `β` | **0.05**, and it is a **total** budget | D2's interval must separate an exponent of 2 from the nearest alternative. The degenerate `μ = 0` case gives 4 and a cubic gap would give 3, so the half-width is 0.5. `β` is the instrument's share of it, declared at one tenth. Total, not bias-only, matching the `√(bias² + σ_p²) ≤ β` constraint the re-optimisation actually used |
| D2's interval | **2 ± 0.5** | the separation above. The instrument consumes at most 10% of it, so the leg fires on the world rather than on truncation or noise |
| `X` | **0.1** | subtraction must buy at least an order of magnitude on the bias, since its residual is `X ×` the bias. Below that, dilution's `1/D²` is competitive and carries no dependence on `c₄`'s accuracy |

At `k_min = 10` and `β = 0.05` the constraint is feasible: `D* = 0.520`, `f* = 0.0488`,
`σ_p = 0.0359`. `D` is registered as an expression in `k` evaluated at `k_min`, so this is
the conservative corner rather than a typical one.

### AMENDMENT 2026-08-07: the range the `c₄` refit fits over

"Just under `σ_max`" is circular: `σ_max = √(f·c₂/|c₄|)` needs `c₄`. Registered instead is a
**derivation range chosen purely on conditioning grounds, declared not to be the registered
window**:

```text
σ ∈ [0.06, 0.30]        quadrature converged to ≥ 10 digits throughout,
                        spanning a factor of 5 in σ and 25 in σ²
```

`f` does not enter it. What the quartic's share of the total turns out to be across that
range is reported afterwards as an outcome, never used to choose the range, so `f` cannot
leak into how `c₄` is measured while `c₄` sets the window `f` parameterises.

### DISCLOSURE 2026-08-07, before the refit: the declared convergence is not met at the bottom

The derivation range was registered with "quadrature converged to ≥ 10 digits throughout".
Measured, by refining the `x`-grid from 16001 to 32001 points at span 9σ:

| `σ` | 0.06 | 0.12 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- |
| relative change | 1.8e−9 | 4.6e−10 | 1.0e−10 | 6.8e−11 |

So roughly **8.7 digits at the bottom** of the range against the 10 declared, reaching 10.2
at the top. The `y`-grid is the reverse, 8.6e−15 at `σ = 0.06` and 4.0e−10 at `σ = 0.30`, so
the binding axis swaps across the range. Recorded before the refit runs rather than after,
because a precondition disclosed once the result is visible is not a precondition.

**What it propagates to.** At `σ = 0.06` the quartic residual after subtracting `c₂σ²` is
`2.58e−6`, which is 0.288% of the gap. A relative gap error of 1.8e−9 is `1.62e−12`
absolute, so the error carried onto `c₄` is

```text
1.62e−12 / 2.58e−6  =  6.3e−7  relative
```

Seven orders below the 0.1 the dilute-versus-subtract rule turns on, and five below the 1%
that would matter for `T`. The shortfall is real and inconsequential, and both halves of
that are stated rather than the convenient one.

The declaration is **not** amended to match the measurement. It stands at 10 digits with the
shortfall recorded against it, since rewriting a precondition to fit what was achieved is
the move the document exists to prevent.

### RESULT 2026-08-07: `c₂ > 0` is an analytic positivity statement at leading order

`c₂ = (ℓ'(μ)/2)² ≥ 0`, with equality only at a stationary point of `R`. C6 already claims
positivity "by theorem" and needs the reference filter to *certify the separation* rather
than to observe a sign, so this does not move C6 off the filter. It is more than the
existing claim in one respect: it supplies the leading coefficient analytically, where
before there was a sign and a numerical magnitude.

The caveat is load-bearing and keeps it from reaching further. `c₄ < 0` means the truncated
expansion turns over, so what is established is positivity in a **neighbourhood of zero
spread**, not globally, and C6 measures at finite spread where exactly that validity is in
question. Recorded because a Tier-A-flavoured statement sitting inside an afternoon's
algebra is better found now than in draft.

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

Two branches, disjoint, split by which half of the instrument failed.

**A third branch was retired rather than left standing.** `c₂ ≤ 0` was registered when
`c₂`'s sign was open. `c₂ = (ℓ'/2)²` is a perfect square, so it is non-negative identically
and vanishes only at `ℓ'(μ) = 0`, which the `R'(μ) ≠ 0` precondition now catches upstream at
declaration. The branch did not fail. It collapsed into the precondition, and a row that can
never fire reads as a check when it is not one.

| branch | evaluable | report |
| --- | --- | --- |
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
| 2026-08-07 (fourth) | `μ` reclassified as *derived*, `R'(μ) ≠ 0` as a fixture precondition, `T`'s three declared parameters, and D2's second leg off the ridge | `c₄`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (fifth) | the `μ`-factor correction, `T` evaluated at the binding `κ`, and `D` as a bias argument | `c₄`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |
| 2026-08-07 (sixth) | `T = f·c₂²/\|c₄\|·10^(−2D)` with the "curvature" naming audited, the noise term set by `k` rather than by quadrature, `D` as an expression in `k`, and the dilute-versus-subtract rule | `c₄`, `X`, `β`, `f`, `D`, `k_min`, `T`, R6's signal, `δ_ref` |

Every number GATE-D4 turns on was unknown on the date the family was declared, and `T`,
R6's signal and `δ_ref` are unknown still. That is the claim this document exists to make
checkable, and this file's git history is the evidence.

Amendments are appended and dated rather than folded into the text they correct, so a
reader can see what was believed when. Two stand so far: the prior mean was missing from
the family, and the registered method for `c₂` and `c₄` computed a quantity that is zero by
construction.
