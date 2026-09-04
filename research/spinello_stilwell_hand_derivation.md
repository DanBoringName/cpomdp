# The Spinello–Stilwell rung: the hand derivation

`research/spinello_stilwell_hand_derivation.pdf` came first. It is a scan of the
notebook the derivation was worked in, dated 31/08/2026 to 04/09/2026. It is the
original. This is a typed up version of that scan by AI-Claude Opus: the same steps, in
the same order, under the notes' own numbering, so the two read side by side.

The steps transcribe the scan and stop where it stops. The dated sections at the foot
are not in the scan and were added after the code ran. Each says what in the scan it
corrects. Where a step and the scan disagree, the scan is what happened and the step
has the transcription error.

`research/spinello_stilwell_rung.md` records what the paper leaves open. This file
records what was derived from it: where equations (35c), (35d) and (35e) come from, which
block of (35d) fails to be a square, and what deleting that block costs. ADR-056 carries
the part that is decided.

Nothing in the steps is a measured result. Two claims are marked unrun below and stay
that way until a module produces them.

## The paper

Spinello and Stilwell, *Nonlinear estimation with state-dependent Gaussian observation
noise*, IEEE Transactions on Automatic Control 55(6), 1358–1366, June 2010,
[doi:10.1109/TAC.2010.2042006](https://doi.org/10.1109/TAC.2010.2042006). Equation
numbers throughout are the paper's. The preprint carries a 2009 IEEE copyright notice
forbidding redistribution to a server, so it is cited here and not committed.

The lineage the rung inherits runs through Bell and Cathey 1993, the paper's reference
[34]. That is what makes one Gauss–Newton step from the prediction the EKF update, and
the iteration to convergence the iterated EKF.

## Convention

The paper writes gradients as **row vectors**. So `h' = ∇h` is a row, and `h'ᵀh'` is
column times row, an `n × n` matrix. Every product below reads that way.

The full paper-to-cpomdp notation swap is in `research/spinello_stilwell_rung.md` and is
not repeated. Three entries in it matter here:

- The paper's `Σ(x)`, scalar `σ(x)`, is cpomdp's `R(x)`.
- The paper's `R` is a Gauss–Newton curvature matrix, the `JᵀJ`. It has nothing to do
  with noise. The letter has swapped meanings between the two literatures.
- The paper's `ℓ` is the whole negative-log objective. My `ℓ` in the `c₂`/`c₄` work is
  `log R`. Different objects, connected through the paper's `½ ln σ` term, which is `½ℓ`
  in my notation. The object the gap expansion of `research/c4_hand_derivation.md` is
  built from is the object that misbehaves here.

## What the rung has to produce

The ladder reports

```text
gap = E_{y~p*}[ KL( q(x) ‖ p(x|y) ) ]
```

`p(x|y)` is the exact posterior, which the S2 lattice computes. `q(x)` is whatever the
rung hands back. For the KL to be the thing the ladder measures, `q` must be a proper
distribution over `x`. For this rung it is Gaussian.

So the rung's job per timestep: take a Gaussian prior `N(μ, P)` and one observation `y`,
return a Gaussian `q = N(x̂, P₊)`. Two numbers.

The mean has to be a MAP estimate. The best single point to centre a Gaussian on is the
posterior mode, and finding the mode means minimising the negative log posterior. That
turns an inference problem into an optimisation problem, which is where Gauss–Newton
enters.

The covariance is a Fisher information. A Gaussian centred on the mode with covariance
equal to inverse curvature at the mode is the Laplace approximation. The paper uses the
expected curvature rather than the observed one.

## Step 1: the objective

The posterior is prior times likelihood:

```text
p(x|y) ∝ N(x; μ, P) · N(y; h(x), σ(x))
```

Take minus the log and drop constants that do not depend on `x`:

```text
ℓ(x) = ½ ξᵀP⁻¹ξ  +  ½ ζ²/σ(x)  +  ½ ln σ(x)          ξ = x − μ,  ζ = y − h(x)
```

Term 1 is distance from the prior mean weighted by prior precision. Term 2 is
measurement mismatch weighted by noise precision. Term 3 is the Gaussian normaliser.

Term 3 exists because a Gaussian density carries `1/√σ` in front. Where `σ` is constant,
`ln σ` is constant and drops out of the minimisation. With `R(x)` the normaliser is a
function of the thing being minimised over, so it stays. That is the whole origin of the
trouble below.

This is the right objective for cpomdp because it is exactly minus the log of the
posterior the lattice computes: same prior, same likelihood, same `R(x)`. The rung
minimises the function the ground truth integrates. No approximation has been made yet.

## Step 2: the residual vector

Gauss–Newton needs `ℓ = ½ rᵀr`, so write each penalty as half a square. Equation (20):

```text
r(x) = ( P^{−½}ξ ,  σ^{−½}ζ ,  (ln σ)^{½} )ᵀ         rows r₁, r₂, r₃
```

`½(r₁² + r₂² + r₃²)` gives the `ℓ(x)` of step 1 back exactly.

`r₁` and `r₂` are honest real residuals. `r₃` is real only for `σ ≥ 1`. Below that
`ln σ < 0` and `r₃` is imaginary. The paper's own footnote to (20) concedes this and
notes that the update equations stay real. Both true, and neither is the problem this
derivation runs into.

## Step 3: the Gauss–Newton step

```text
x̂⁽ᶥ⁺¹⁾ = x̂⁽ᶥ⁾ − [∇ᵀr ∇r]⁻¹ ∇ᵀr r                                (22)
```

Three reasons this is the correct method here, rather than merely an available one.

**It is the iterated EKF.** Bell and Cathey 1993: one step of the above from `x̂⁽⁰⁾ = μ`
is the EKF update, algebraically. Iterating to convergence is the iterated EKF. So the
rung is the standard Gaussian filter for this likelihood, and nothing beyond it.

**It carries a shape guarantee.** For real `J`, `wᵀ(JᵀJ)w = ‖Jw‖² ≥ 0`, so the step
matrix is positive-semi-definite and every step is a descent direction. Full Newton
keeps a `Σ rᵢ ∇²rᵢ` term that has no such guarantee and needs second derivatives of
`σ(x)`.

**The prior term is free damping.** The step matrix splits as `P⁻¹ + R`, where `P⁻¹`
comes from `r₁` and is positive-definite on its own. So `P⁻¹ + R ≻ 0` wherever `R ⪰ 0`.
Levenberg–Marquardt buys that floor with a tuned `λI`. The Bayesian prior gives it away.

For cpomdp specifically, the observation map is linear, so `∇²h = 0` and the
second-derivative term Gauss–Newton discards carries only `∇²σ` contributions. The
approximation is strictly smaller here than the one the paper made for arctangent
bearings.

## Step 4: the three matrices, derived from `r`

Scalar case, row vectors throughout.

```text
∂r₁/∂x = P^{−½}
∂r₂/∂x = −σ^{−½}( h' + (ζ/2σ)σ' )
∂r₃/∂x = ½(ln σ)^{−½} σ'/σ
```

The `r₂` line is a product rule on `r₂ = σ(x)^{−½}·ζ(x)` with `ζ' = −h'`, organised
around `σ^{−½}`.

**Gradient.**

```text
Jᵀr = g = P⁻¹ξ + r₂ ∂r₂ + r₃ ∂r₃
        = P⁻¹ξ − (ζ/σ)h' − (ζ²/2σ²)σ' + (1/2σ)σ'
        = P⁻¹ξ + [ −(ζ/σ)h' + (1/2σ)(1 − ζ²/σ)σ' ]
```

The bracket is `s`, equation (35c), reproduced exactly.

The `r₃` contribution is `(ln σ)^{½}·½(ln σ)^{−½}·σ'/σ = σ'/2σ`. **The logs cancel.** It
is real for any `σ > 0`, and the sign of `ln σ` is irrelevant to it. Step 8 turns on this
fact.

**Curvature.**

```text
JᵀJ = P⁻¹ + (∂r₂)ᵀ(∂r₂) + (∂r₃)ᵀ(∂r₃)
    = P⁻¹ + (1/σ)( h' + (ζ/2σ)σ' )ᵀ( h' + (ζ/2σ)σ' )  +  σ'ᵀσ'/(4σ² ln σ)
```

The part after `P⁻¹` is `R`, equation (35d). Nothing cancels in the `r₃` piece this time.

## The Fisher information

Unnumbered in the notes, sitting between steps 4 and 5.

`Ū = E[sᵀs]`, with `s` the measurement part of the gradient from step 4 and the
expectation over the innovation `ζ`, treated as random with `ζ ~ N(0, σ)`.

`ζ` is random here because `Ū`'s job differs from `R`'s. `R` shapes the step for this
measurement, so it uses the actual `ζ`. `Ū` sets the reported posterior precision, which
answers how much a measurement from this sensor, at this state, teaches on average.
Averaging over what the sensor could emit means averaging over `ζ`. At the state in
question the model says `y = h + noise`, so `ζ = y − h` is the noise itself, mean zero
and variance `σ`.

Four facts about `ζ ~ N(0, σ)` do the work: `E[ζ] = 0`, `E[ζ²] = σ`, `E[ζ³] = 0`, and the
Gaussian fourth moment `E[ζ⁴] = 3σ²`.

Split (35c) into two rows:

```text
s = a + c        a = −(ζ/σ)h'        c = (1/2σ)(1 − ζ²/σ)σ'
```

`a` is the ordinary Kalman innovation term. `c` is the new term the noise model brings.

**Sanity check first.** A score function has mean zero. `E[a] = −(E[ζ]/σ)h' = 0`, and
`E[c] = (1/2σ)(1 − σ/σ)σ' = 0`.

**Square it.** `sᵀs = aᵀa + aᵀc + cᵀa + cᵀc`.

`aᵀa = (ζ²/σ²)h'ᵀh'`, so `E[aᵀa] = (1/σ)h'ᵀh'`. That is the first term of (35e).

`aᵀc` has scalar part `−(1/2σ²)(ζ − ζ³/σ)`. Every power of `ζ` in it is odd, so the
expectation is zero, and `cᵀa` is the same with the matrices the other way round. The
cross terms are alive in the printed `R` because that uses the actual innovation. On
average they contribute nothing, because the observation channel and the noise channel
are uncorrelated sources of information.

`cᵀc = (1/4σ²)(1 − ζ²/σ)²σ'ᵀσ'`, and

```text
E[(1 − ζ²/σ)²] = E[1] − 2E[ζ²]/σ + E[ζ⁴]/σ² = 1 − 2 + 3 = 2
```

so `E[cᵀc] = (1/2σ²)σ'ᵀσ'`. Collecting:

```text
Ū = (1/σ)h'ᵀh' + (1/2σ²)σ'ᵀσ'
```

which is (35e) verbatim. **No `ln σ` appears anywhere in `Ū`**, so the deletion of step 6
cannot reach the posterior covariance.

`Ū ⪰ 0` unconditionally: two outer products with coefficients `1/σ > 0` and
`1/2σ² > 0`. So `P₊⁻¹ = P⁻¹ + Ū ≻ 0` always.

Comparing (35e) against (35d) term for term, the averaging did three things. It killed
the cross terms, which are odd in `ζ`. It replaced the `ζ²`-dependent coefficient on
`σ'ᵀσ'` with its mean. And it produced no counterpart at all to the `1/ln σ` term. `R` is
per measurement and may be a gain. `Ū` is the average and is not.

Setting `σ' = 0` leaves `Ū = (1/σ)h'ᵀh'`, the standard Kalman information for a linear
observation with fixed noise. That is route 4 of the rung file, and it is the rung's only
external check.

## Step 5: `R` is exactly two Jacobian blocks

Expand the `r₂` row against the printed (35d):

```text
(1/σ)( h' + (ζ/2σ)σ' )ᵀ( h' + (ζ/2σ)σ' )
    = (1/σ)h'ᵀh' + (ζ/2σ²)( h'ᵀσ' + σ'ᵀh' ) + (ζ²/4σ³)σ'ᵀσ'
```

Those are the first three terms of (35d), term for term. So

```text
R = (∂r₂)ᵀ(∂r₂) + (∂r₃)ᵀ(∂r₃)
```

with the fourth printed term, `(1/4σ² ln σ)σ'ᵀσ'`, being the whole of the `r₃` block.

## Step 6: the anatomy of `R`, and the block that has to go

The step matrix in (35a) is not `R` alone. It is `M = P⁻¹ + R`, with `P⁻¹` arriving as
`(∂r₁)ᵀ(∂r₁) = (P^{−½})ᵀ(P^{−½})`. `P` is a covariance, so `P⁻¹` is symmetric
positive-definite: every direction `w` gives `wᵀP⁻¹w > 0`. Add any
positive-semi-definite matrix and

```text
wᵀ(P⁻¹ + R)w = wᵀP⁻¹w + wᵀRw > 0
```

strictly, for every `w`. Then `M ≻ 0`, so `M` is invertible and `−M⁻¹g` points downhill.
The prior acts as a floor, the one Levenberg–Marquardt has to build by hand. The argument
needs only `wᵀRw ≥ 0`, so interrogate the two blocks separately.

**The `r₂` block.** With `b = h' + (ζ/2σ)σ'`, the sandwich `wᵀ(∂r₂)ᵀ(∂r₂)w` is `t·t = t²`
for the scalar `t = −σ^{−½}·b·w`. All three ingredients of `t` are real: `σ^{−½}` because
`σ > 0` is a variance, `b` because `h'`, `ζ` and `σ'` are real quantities of the model,
`w` because it was chosen. So `t² ≥ 0` for every `w`, every `ζ`, every state and every
`σ > 0`. This block is positive-semi-definite everywhere the mode is defined.

**The `r₃` block.** The same sandwich gives the scalar `u = ½(ln σ)^{−½}(σ'·w)/σ`, and

```text
u² = (σ'w)² / (4σ² ln σ)
```

`(σ'w)²` is a real square and `4σ²` is positive. `ln σ` alone decides the sign:

| regime | `ln σ` | the block |
| --- | --- | --- |
| `σ > 1` | positive | `≥ 0` |
| `σ = 1` | zero | pole, division by zero |
| `σ < 1` | negative | `≤ 0`, strictly negative wherever `σ'w ≠ 0` |

So `R` is one piece that is a real square and positive-semi-definite unconditionally,
plus one piece whose sign tracks `ln σ` and which is negative exactly where `σ < 1`. Q3
and Q4 of the rung file are this table read as failure modes.

**The deletion.** Drop the `r₃` block from `R` (35d). Keep `s` (35c), `Ū` (35e) and the
objective (18) verbatim.

```text
R_mod = (∂r₂)ᵀ(∂r₂) = (1/σ) bᵀb
```

Four reasons.

1. **It buys positive-semi-definiteness outright.** `R_mod` is a real square, so
   `R_mod ⪰ 0` everywhere. With `P⁻¹ ≻ 0` as the floor, `M ≻ 0` and every step descends.
   No box on the state and no assumption about where the iterates go.
2. **Gauss–Newton's guarantee is a theorem about real Jacobians.** `r₃ = (ln σ)^{½}` is
   imaginary for `σ < 1`, which is the paper's own footnote. The guarantee was never
   available there to begin with.
3. **It is surgical.** Step 8 sets that out line by line.
4. **The published runs never cashed the term in.** Step 9.

The cost is a name. This is no longer literal Gauss–Newton on (20). It is a
modified-metric Newton iteration that keeps the Gauss–Newton fixed point, and cpomdp has
to say so at first use.

## Step 7: invariance under a change of observation units

Multiply the measurement by a constant `λ > 0`. Radians to degrees is `λ = 180/π`.
Nothing physical happens. Only which numbers get written down changes.

The model says `y ≈ h(x)`, so a rescaled measurement needs a rescaled prediction:
`h → λh`. The innovation `ζ = y − h → λy − λh = λζ`, both terms scaling together.
Variance is an expectation of a square, so `σ = E[(y − h)²] → λ²σ`. Differentiating in
`x` does not touch `λ`, so the derivatives inherit: `h' → λh'` and `σ' → λ²σ'`. The
state-side objects `x`, `μ`, `P`, `ξ` and `w` are untouched, because `λ` never reaches
them.

```text
h → λh      ζ → λζ      σ → λ²σ      h' → λh'      σ' → λ²σ'
```

Count powers of `λ`. A term is invariant when they cancel to `λ⁰`.

| object | term | powers |
| --- | --- | --- |
| `s` (35c) | `(ζ/σ)h'` | `λ·λ/λ² = λ⁰` |
| `s` | `ζ²/σ` | `λ²/λ² = λ⁰` |
| `s` | `σ'/σ` | `λ²/λ² = λ⁰` |
| `R_mod` | `(1/σ)h'ᵀh'` | `λ·λ/λ² = λ⁰` |
| `R_mod` | `(ζ/2σ²)(h'ᵀσ' + σ'ᵀh')` | `λ·λ·λ²/λ⁴ = λ⁰` |
| `R_mod` | `(ζ²/4σ³)σ'ᵀσ'` | `λ²·λ²·λ²/λ⁶ = λ⁰` |
| `Ū` (35e) | `(1/σ)h'ᵀh'` | `λ⁰` |
| `Ū` | `(1/2σ²)σ'ᵀσ'` | `λ⁰` |
| **full `R`** | `σ'ᵀσ'/(4σ² ln σ)` | numerator `λ⁴` over `σ² = λ⁴` cancels, then **`ln σ → ln σ + 2 ln λ`** |

Seven of eight. The eighth is the deleted block, and it fails because a logarithm turns a
scale into an additive shift that nothing else can absorb.
`research/src/research/spinello_stilwell/invariance.py` is where this count is asserted
symbolically rather than read off the page.

**Why the ladder is entitled to demand invariance.** The gap is a number in nats. The KL
compares two distributions over the state, and rescaling `y → λy` only relabels the
conditioning value: for the same physical measurement `q` and `p(x|·)` are unchanged, so
the KL is unchanged. The outer expectation visits the same physical observations with the
same probabilities. So the gap is unit-free by construction, before any filter enters,
and the lattice computes exactly that number. A rung is an approximation of it. If a
rung's output moves with `λ`, it is not a rough approximation of the gap. It is an
approximation of something else.

With the `1/ln σ` term in place, the same physical problem in degrees and in radians
reports two different gaps for one facet. Step 6 faults the route, since no descent step
is guaranteed. This faults the report. Two independent reasons, one term convicted.

## Step 8: why the deletion is surgical

Point by point.

- **The fixed point of `x ← x − M⁻¹g` is where `g = 0`.** `M` steers the route. `g` picks
  the destination.
- **`ln σ` cancels out of the gradient.** From step 4, `r₃ ∂r₃ = σ'/2σ`, real everywhere,
  and it is already printed in (35c). `s` stays verbatim.
- **The deleted object is the curvature block alone.**
  `(∂r₃)ᵀ(∂r₃) = σ'ᵀσ'/(4σ² ln σ)`. Nothing cancels there, which is why it is the block
  that goes.
- **`Ū` is built from `s` alone.** No `ln σ` ever appears in it. (35e) stays verbatim and
  `P₊` is untouched.
- **The objective (18) is untouched**, so the mode is unmoved.

Net: same `x̂`, same `P₊`, same `g`, same gap in nats. Only the path of iterates changes,
and it now provably descends.

> **Unrun.** That the fixed point is genuinely unmoved is an argument here, not a
> measurement. Nothing has compared modified against unmodified iterates in code. Routes
> 3 and 5 of the rung file are where that gets settled.

## Step 9: the original paper never cashed the term in

Section IV's constants give `σ < 1` throughout figures 2 to 4 for any `N ≥ 3`, where `N`
is the number of hydrophones. `κ = 24/(α²N³)` shrinks as `N⁻³`, so larger arrays only
push `σ` further down. If that holds, the `1/ln σ` block was **negative in every
published run**, and the filter won its comparison against the EKF anyway.

> **Unrun.** This is arithmetic on the paper's stated constants and no module has done
> it. It gets pinned in Python before it is repeated anywhere. Until then it is a route
> and not a result.

## Step 10: what cpomdp owes

**A name.** Spinello–Stilwell iterated filter with a documented modification, declared at
first use in Paper 2.

**Three tests, with what each is blind to stated beside it.**

> **Superseded.** Test 1's row is wrong about where the failure appears. The RESULT
> and CLARIFICATION of 2026-09-04 at the foot of this document carry the measurement
> and the reason. Read those first.

| # | Test | Blind to |
| --- | --- | --- |
| 1 | Gap invariant under `y → λy`. Fails unmodified by the predicted `2 ln λ` shift, passes modified | a wrong fixed point that is nonetheless unit-free |
| 2 | `σ' = 0` gives exact agreement with the Kalman oracle | every `σ'` term |
| 3 | `σ ≫ 1` gives near agreement with the printed (35d) | the `σ < 1` regime, which is the whole problem |

No one of the three covers what the other two miss, which is the reason for declaring all
three rather than picking the strongest.

**The argument for citing and repairing rather than deriving from scratch.** A published
derivation exists. It has a known EKF lineage through Bell and Cathey. One change is
declared, with a cost Python can monitor. Deriving a fresh filter would trade all three
for the appearance of independence.

## RESULT 2026-09-04: the three tests, run

`research/src/research/spinello_stilwell/repair.py`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.repair
```

and asserted from `tests/test_spinello_stilwell_repair.py`. The scheme itself moved to
`scheme.py` so the printed form and the modification are one implementation with a flag,
rather than two copies of (35) that can drift apart.

**Test 1 is not the test step 10 predicted.** The table above says the unmodified filter
fails by the `2 ln λ` shift. It does not. Run to convergence the printed scheme is
already unit-free: the estimate spans 0.0 across `λ ∈ {1, ½, 3, 7}`, and so does the
modification. Route 1's numeric half had measured that for the printed scheme in August
and the argument at step 7 did not absorb it.

The dependence is real and it bites at a **finite budget**. At a budget of one the
printed estimate spans `2.955e-05` over the same four unit choices, while the
modification spans `1.11e-16`, which is machine zero. A budget of one is the single-step
filter (36), and ADR-056 declares that as a rung of its own. So the defect belongs to a
rung the ladder reports, not to a path nobody sees. That makes the test sharper than the
one the notes asked for, and it also means the warrant at step 7 has to be stated as
being about the reported rung rather than about the fixed point.

**Test 2 is exact.** At `∇σ = 0` both variants reproduce the Kalman posterior with a
departure of `0.0` in the mean and `0.0` in the variance, at every `λ` tried. Bit for
bit, not to a tolerance. The rung's only external check is discharged in the one regime
where an oracle exists.

One thing the run turned up that the notes did not anticipate: with `∇σ = 0` **and**
`σ = 1`, the printed fourth term of (35d) is `0/0` and the scheme has no value. The
modification reduces to Kalman there as it does everywhere else. So the pole reaches
even the fixed-noise reduction, and in the printed form the rung's only external check
is not evaluable on it.

**Test 3 behaves as argued.** The printed (35d) and the modification converge as the
noise grows, the relative difference falling from `5.45e-01` at `σ = 2` to `7.24e-08` at
`σ = 10⁶`. At `σ = 0.9` the printed curvature is `-9.74`, negative, while the
modification is `+1.98`. That sign flip is the blindness the table claims for this test,
pinned by a check rather than left as a caveat.

**Still unrun.** The gap itself, which needs the reference lattice and a rung with a
declared budget. Routes 3, 5, 6 and 7 of `research/spinello_stilwell_rung.md`. Nothing
here reports a warrant, and the modification remains a derived proposal rather than a
decided one.

## CLARIFICATION 2026-09-04: why test 1 could not fail by `2 ln λ`

This section is not in the scan as claude found it later when tasked to verify my tests and hand derivation. I agree with its findings. The scan ends at step 10 predicting that the
unmodified gap fails invariance by a `2 ln λ` shift, and the RESULT above measured that
prediction wrong. What follows is the reason the predicted failure was never available,
derived at the keyboard after `repair.py` ran. The scan keeps its prediction, since it
records what was believed before the code existed, and the Q1 amendment in
`research/spinello_stilwell_rung.md` is the correction of record.

**The unit change is an additive constant in the objective.** Rescale the observation.
The mismatch term cancels and the normaliser splits:

```text
ℓ_λ(x) = ½ξᵀP⁻¹ξ + ½(λζ)²/(λ²σ) + ½ ln(λ²σ) = ℓ(x) + ln λ
```

That constant is the entire effect. A constant in `x` moves neither the mode nor the
normalised posterior, and a KL between distributions over the state cannot see it.
Step 7 had every substitution needed for that one line and never wrote it.

**Exactly one object fails to treat the shift as a constant.** Trace
`ln σ → ln σ + 2 ln λ` through the three places `r₃` reaches:

| where `r₃` lands | value | under `σ → λ²σ` |
| --- | --- | --- |
| objective, `r₃²` | `ln σ` | gains a constant the argmin ignores |
| gradient, `r₃ ∂r₃` | `σ'/2σ` | the logs already cancelled, nothing to shift |
| curvature, `(∂r₃)ᵀ(∂r₃)` | `σ'ᵀσ'/(4σ² ln σ)` | the shift lands in the denominator and sticks |

The square root is why. `(ln σ)^½` is not affine in `ln σ`, so squaring its Jacobian
turns an additive shift into a change of value. That gives the deleted block a second
independent characterisation beside step 6's: the one piece of the scheme that is not a
real square, and the one piece non-affine in the quantity the unit choice shifts.

**The curvature steers and does not choose.** `M = P⁻¹ + R` shapes the step. The
stopping condition is `g = 0`, and `g` is invariant, so its root is the same in every
unit system. `Ū` contains no `ln σ` and is evaluated at that root. Run to convergence,
the printed scheme therefore reports a unit-free `(x̂, P₊)` with the term still in it.
Only the path moves. The route 1 numbers of 2026-08-24 already had the iteration counts
at 6, 9, 8, 8 across four unit choices while the answer sat still, and step 10's test
was written without absorbing them.

**A finite budget reports an iterate, and iterates are steered.** Stop at a budget of
one and the reported mean is a single `M`-preconditioned step, so the unit dependence
reaches the report. A budget of one is (36). ADR-056 declares it as a rung, so the
number that moves is one the ladder publishes. The spreads in the RESULT above are the
conviction, and the modification's machine-zero spread is the acquittal beside it.
Step 7's sentence that one facet reports two gaps survives with a qualifier: true at
every finite budget, false at convergence, and the gap itself stays unmeasured until
the rung exists.

**A gap that did shift by `2 ln λ` would mean the ladder is broken.** An additive
constant survives into a reported divergence only where the number was computed from
unnormalised log-densities, which is the entropy-subtraction family of defects the
standing prohibition exists for. A normalised KL cannot inherit the normaliser's
constant. The ladder's own construction is what made the predicted failure
unobservable, and the prediction failing is that construction working.

Two limits. Each variant's converged answer being unit-free does not discharge step 8's
claim that printed and modified share a fixed point, since that comparison is across
variants at one unit choice and stays with routes 3 and 5. And the unit-dependent
iteration count is a cost result in its own right: the unmodified iterated rung's
per-decision compute depends on the observation's units, which RFC-001's energy
accounting has to care about even where the report is clean.

## ADOPTED 2026-09-04: the modification ships

ADR-057 takes step 6's deletion for rungs (36) and (35), on step 8's argument and the
RESULT above. The rung is named as modified at first use in Paper 2, as step 10 asks.
No `λ` is declared: the deleted block was the only term step 7's table moved, so the
shipped scheme has no pole and a unit convention would change nothing it reports.
Step 8's "provably descends" is taken as a descent direction at every step, which is
what a positive definite step matrix gives, and no more.
