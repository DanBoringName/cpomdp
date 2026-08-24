# The Spinello–Stilwell rung: what the paper leaves open

Rung two of PR-7's evaluation ladder is the iterated extended Kalman filter for
state-dependent observation noise, Spinello and Stilwell, *IEEE Transactions on
Automatic Control*, 2009. This file records what building it requires deciding that the
paper does not decide, and what has to be measured before those decisions are made.
ADR-056 carries the decisions that are settled and points here for the rest.

Nothing below is a result. These are routes and tests, written before the rung exists,
so that a choice made later can be read against what was known when.

## Notation: a complete three-way swap

The paper's symbols collide with cpomdp's, and not one of the three means the same
thing in both. Transcribing an equation without renaming is the obvious way to produce
a filter that runs and is wrong.

| paper | means | cpomdp |
| --- | --- | --- |
| `Σ(x)` | observation-noise covariance, state-dependent | `observation_noise`, `R(x)` |
| `P[k\|k−1]`, `P[k\|k]` | belief covariance, predicted and posterior | `Belief.cov`, `Σ` |
| `R⁽ᶥ⁾` | the Gauss–Newton approximate Hessian | no equivalent. Name it `gn_hessian` |
| `U`, `Ū` | Fisher information contribution | no equivalent |
| `h` | observation mean function | `observation_matrix @ x` in the linear case |
| `ζ` | innovation `z − h(x)` | the residual |

The paper's `R` must never be spelled `R` in cpomdp source. It is the Hessian of a cost,
not a noise covariance, and the tree already uses `R` for the thing the paper calls `Σ`.

The paper assumes `p ≤ n` throughout, no more observation channels than states. cpomdp
does not enforce this anywhere. The rung should assert it rather than inherit it
silently.

## The scheme, scalar case (§III-D-2, `p = 1`)

```
x̂⁽ᶥ⁺¹⁾ = x̂⁽ᶥ⁾ − [P⁻¹ + R⁽ᶥ⁾]⁻¹ ( P⁻¹(x̂⁽ᶥ⁾ − x̂⁻) + s⁽ᶥ⁾ )        (35a)

s⁽ᶥ⁾ = −(ζ/σ)∇h + (1/2σ)(1 − ζ²/σ)∇σ                            (35c)
R⁽ᶥ⁾ = (1/σ)∇h∇h + (ζ/2σ²)(∇h∇σ + ∇σ∇h)
       + (1/4σ²)(ζ²/σ + 1/ln σ)∇σ∇σ                              (35d)
Ū    = (1/σ)∇h∇h + (1/2σ²)∇σ∇σ                                   (35e)
P⁻¹[k|k] = P⁻¹[k|k−1] + Ū                                        (35b)
```

Two matrices, two jobs. `R⁽ᶥ⁾` shapes the step. `Ū`, the Fisher information, sets the
posterior covariance, evaluated at the converged estimate per (33e). Conflating them
gives a filter with a plausible mean and a wrong covariance, which the ladder cannot
detect by inspection because the ladder is a comparison of posteriors.

## Q1. `1/ln σ` is the only unit-dependent object in the estimator

Rescale the observation, `o → λo`. Then `z → λz`, `h → λh`, `σ → λ²σ`, `ζ → λζ`,
`∇h → λ∇h`, `∇σ → λ²∇σ`. Every term above is invariant under that substitution except
one:

```
1/ln σ   →   1/(ln σ + 2 ln λ)
```

Verified symbolically: seven of the eight terms in `s`, `R⁽ᶥ⁾` and `Ū` return to
themselves, and only the log-determinant term moves.

The quantity the ladder reports is `E_p*[ KL(q ‖ p(x|y)) ]`, a divergence between
distributions over the *state*. It does not depend on the units the observation is
measured in. So this term makes the estimator depend on a choice the reported number
cannot see, which is the same defect the standing prohibition on entropy subtraction
guards against and belongs in the same family of arguments.

That is the warrant for dropping the term, if it is dropped: not that it changes the
path rather than the fixed point, but that it is the sole place an arbitrary unit
choice enters an estimator whose output is unit-free.

**Route.** An exploration that rescales one worked case over several decades of `λ` and
reports which quantities move. **Test.** The converged estimate and the reported gap are
invariant under rescaling once the term is removed, and are not before.

## Q2. Ask whether the pole is a units artefact before writing any guard

`R⁽ᶥ⁾` is singular at `σ = 1`. Under rescaling the pole sits at `σ = λ⁻²`, so the unit
choice alone decides where it falls: `λ = 2` moves it to `σ = 1/4`, `λ = ½` to `σ = 4`.

Since the fixed point of (35a) is where `P⁻¹(x̂ − x̂⁻) + s = 0`, and `s` is invariant,
rescaling moves the pole without moving the answer. Choosing observation units that put
the whole reachable `σ` on one side of 1 therefore removes the hazard by construction
and costs nothing.

That is a declared unit choice, not a tuned parameter. It must still be declared, and
before any number exists, because an undeclared choice that happens to avoid a pole is
exactly what standing rule 2 exists to catch.

A guard and its instrumentation become necessary only where no such choice exists, on
a family whose reachable `σ` straddles every achievable pole.

**Route.** Determine, for each declared `R` family and its swept range, whether a single
`λ` puts the whole reachable set on `σ > 1`. On the registered ridge
`R(x) = R₀ + κx²` at `μ* = √(R₀/κ)`, `σ(μ*) = 2R₀`, so `R₀ = 0.5` puts the pole on the
operating point at `λ = 1`. **Test.** The declared `λ` keeps `σ > 1` across the swept
range, asserted rather than assumed.

## Q3. The failure is two-sided, and one side is silent

**From above.** As `σ → 1⁺`, `1/ln σ → +∞`, so `R⁽ᶥ⁾ → +∞` and the step
`[P⁻¹ + R⁽ᶥ⁾]⁻¹(…) → 0`. The iteration freezes. Under a fixed budget it returns
`x̂[k|k−1]`, the prediction, wearing the appearance of a converged posterior.

**From below.** For `σ < 1`, `1/ln σ < 0`. `P⁻¹ + R⁽ᶥ⁾` passes through singular and
becomes indefinite, giving unbounded steps and steps that climb the objective.

On the ridge with `R₀ < 1` a single run crosses from one regime to the other.

The stall is the severity-one case: it produces a number, the number is wrong, and
nothing about it looks wrong. The blow-up at least announces itself.

## Q4. Below `σ = 1`, Gauss–Newton is outside its domain

`R⁽ᶥ⁾` is meant to be `∇ᵀr∇r` for the residual vector `r` of (20), and any such matrix
is positive-semi-definite. The negative term is precisely the third component of `r`,
`(ln det Σ)^½`, being imaginary and then squared. The paper's own footnote concedes the
component is imaginary when `det Σ < 1`.

The objective (18) is real and correct throughout. It simply is not a sum of squares
where `σ < 1`, so the method used to minimise it does not apply there. That is the
reason for any guard, and it belongs in the record as a statement about the method's
domain rather than as a numerical footnote.

## Q5. The ladder changes two things at once between rung one and rung two

Rung one, plug-in at the predicted mean, gives

```
P⁺⁻¹ = P⁻⁻¹ + (1/σ)∇ᵀh∇h
```

The paper's single-step filter (36e) gives that **plus** `(1/2σ²)∇ᵀσ∇σ`.

So moving from rung one to the iterated rung changes the inclusion of the
derivative-of-covariance terms *and* the iteration to convergence, together. An
improvement between them cannot be attributed to either.

(36) is (35) at a budget of one, so adding it as its own rung is nearly free. It turns a
four-rung ordering into a five-rung one in which two adjacent differences isolate
distinct mechanisms.

**This has to be decided before any R7 number exists.** A rung added after an ordering
is seen is what standing rule 7 refuses.

## Q6. What rung one is blind to

The term rung one discards, `(1/2σ²)∇ᵀσ∇σ`, is non-zero exactly when the noise varies
with the state. That is the same derivative-of-covariance information that makes the
channel epistemically active at all.

So rung one does not merely approximate the `R(x)` posterior badly. It approximates it
in a way that cannot see the mechanism under test.

Worth a sentence in Paper 2 Part 2, stated carefully. The filter's posterior information
and expected free energy's epistemic term are related quantities and not the same
object, and a sentence that runs them together would claim more than this observation
supports.

## Q7. Budget exhaustion corrupts the covariance too

`Ū` is evaluated at `x̂⁽ᶥ⁾` per (33e). A truncated iteration therefore returns a wrong
`P[k|k]` as well as a wrong mean, so both halves of the divergence move.

That argues for a non-convergent step being routed to `VOID` rather than reported as a
number. A gap computed from a posterior whose mean and covariance are both wrong by an
unmeasured amount is not a measurement of anything.

## Q8. Rung two has no published numerical validation

Every figure in §IV uses (36), the single-step filter. The iterated scheme (35) is
derived and never simulated.

So the rung arrives with no external check of any kind, and whatever oracle the test
suite provides is the only thing standing behind it. That raises what the fixed-`R`
reduction has to carry: at `∇σ = 0` the scheme must reproduce the ordinary Kalman
update exactly, and that is the one place an independent answer exists.

## The routes, collected

| # | Route | What it decides |
| --- | --- | --- |
| 1 | Rescaling exploration over several decades of `λ` | whether `1/ln σ` is the only term that moves |
| 2 | Reachable-`σ` survey per declared family | whether one `λ` clears the pole for all of them |
| 3 | Two-sided failure probe near `σ = 1` | that the stall returns the prediction, silently |
| 4 | `∇σ = 0` reduction against the Kalman oracle | the only external check the rung has |
| 5 | Budget-exhaustion probe | that both mean and covariance move, so `VOID` is right |
| 6 | Rung-one-versus-(36) separation | whether the two mechanisms are separately visible |

Routes 1, 2 and 4 are prerequisites for the rung. Routes 3, 5 and 6 decide how it
reports and how many rungs the ladder declares.

### ROUTE 2026-08-24: routes 1 and 2's symbolic half are run, and Q2's premise is measured

`research/src/research/spinello_stilwell/invariance.py`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.invariance
```

Three things, asserted inside the module rather than printed:

- **Route 1, symbolic half.** Seven of the eight terms in `s`, `R⁽ᶥ⁾` and `Ū` return to
  themselves under `o → λo`. Only `(1/4σ²)(1/ln σ)∇σ∇σ` moves, becoming
  `1/(ln σ + 2 ln λ)`. That is the whole of Q1's claim.
- **Route 2, symbolic half.** The pole therefore sits at `σ = λ⁻²`. On the ridge,
  `σ` at the operating point is `2R₀` whatever `κ` is, so `R₀ = ½` puts the pole exactly
  there at `λ = 1`. Both derived rather than substituted.
- **Q2's premise, which the original pass asserted without checking.** The converged
  estimate and the posterior variance are unmoved by the rescaling, to 1e-12 across
  `λ ∈ {1, ½, 3, 7}`, while the iteration count is not: 6, 9, 8, 8. If rescaling moved
  the answer, choosing units to clear the pole would be choosing an answer, and it would
  be a tuned parameter rather than a declared convention. It does not.

The numeric part uses a reference implementation of (35) written for that check alone.
It is not the rung. It carries no guard at the pole and takes its iteration budget as an
argument rather than declaring one, both deliberately, so that nothing in it can stand
in for a decision the rung still owes.

What is **not** run: the empirical half of route 1 (a rescaling sweep of the reported
gap, which needs the rung), and routes 3 to 6 entirely. The two-sided failure of Q3, the
budget-exhaustion behaviour of Q7 and the rung-one-versus-(36) separation of Q5 remain
routes, and no number here bears on any of them.


### RESOLVED 2026-08-24: route 2 is run, and only one declared family resists a unit choice

`research/src/research/spinello_stilwell/reachable_noise.py`, run with

```text
uv run --no-sync python -m research.spinello_stilwell.reachable_noise
```

The infimum of `R` over the whole state line, per declared family:

| family | `inf R` | clears the pole at |
| --- | --- | --- |
| `1 + x²` | 1 | any `λ > 1` |
| `1.5 + 0.5 tanh(x)` | 1 | any `λ > 1` |
| `1.5 + 0.5 sin(x)` | 1, **attained** | any `λ > 1` |
| `2` (fixed) | 2 | already clear at `λ = 1` |
| ridge, `R₀ = ½` at `μ*` | 0.5 | `λ > 1.5492` |
| `exp(x)` | 0 | **no `λ`** |

Four of the six never dip below `R = 1`, so the pole is outside their range for any
`λ > 1`. That was not obvious in advance and it is a property of how they were declared
rather than of any sweep width.

**`1.5 + 0.5 sin(x)` attains `R = 1` exactly**, wherever `sin(x) = −1`. At `λ = 1` the
pole is *on* the family, not near it. That single family is what makes the unit choice
compulsory rather than prudent, and it was already in the registered set before any of
this was noticed.

**One `λ` covers the entire survey.** Over the reachable window, taken at nine prior
standard deviations either side of the mean, `λ = 2.5630` puts every family at every
declared spread a factor 1.2 clear of the pole. A single declared convention suffices,
which is what keeps this a unit choice rather than a per-cell tuning.

**`exp(x)` is the exception, and it is a real one.** Its infimum over the line is zero,
so no `λ` clears every state. It is cleared only over a declared box, and the required
`λ` grows without bound as that box widens: at spread 0.30 the reachable minimum is
0.183 and needs `λ > 2.56`, and a wider box needs more. So for that family the guard
question of Q2 stays open, and it is bound to route 3: a run whose iterate escapes the
declared window can still reach the pole, and Q3 says the approach from above is silent.

What this does **not** settle. The `λ` above is a measurement, not a declaration. What
gets registered is the owner's call and belongs in an ADR, and until then no rung may
read a unit choice off this table. Routes 3, 4, 5 and 6 are untouched, and route 4, the
`∇σ = 0` reduction against the Kalman oracle, remains the rung's only external check.
