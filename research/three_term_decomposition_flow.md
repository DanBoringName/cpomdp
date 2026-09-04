# How a cell gets its two numbers

I'm writing this as the evaluator gets built, one step per commit, so the logic flow
is readable without opening the diffs. The maths contract is
`research/fep_falsification_battery.md` section C. The code is `cpomdp.scoring`.

## The cross comes first

`build_cross` builds every cell of the two declared axes and refuses to hand back fewer
than `|model axis| × |inference axis|`. The certificate is a
`ProductCompletenessCertificate` naming both axes and both versions, so a run over a
different cross cannot be mistaken for this one. This is the Route 2 purchase from the
build plan, and it is why every separation reported later is a comparison between two
cells of one decided set rather than two cells someone happened to run.

## The shape comes next

`Decomposition` holds `misspecification` and `inference_gap`. Nothing else. The floor
`H(p*)` belongs to the process, not to any cell, and prohibition 1 bans reaching a
term by subtracting it. The type has no slot to put an entropy in, and a test pins the
field list, so growing it is a visible decision.

## The divergence is a sum of non-negative parts

Both terms are KL divergences between Gaussians, so one primitive serves both.
`gaussian_kl` does not use the textbook form. That form subtracts `n` from a trace and
one log-determinant from another, and near equality it returns rounding noise of either
sign. A term reading `1e-16` under it says nothing about whether the term is zero
(ledger section 4, "absence of catastrophic cancellation").

The primitive whitens by `cov_b`'s Cholesky factor and reads the eigenvalues `λ` of
`cov_b⁻¹ cov_a` off a singular value decomposition. Each contributes `λ − 1 − ln λ`,
which is non-negative on its own, and the mean term is a sum of squares. Near `λ = 1`
even `λ − 1 − ln λ` is a cancellation, so below `1e-4` it is evaluated by its series
instead. Nothing is subtracted from anything of like size. Identical inputs return `0.0` exactly, and two
Gaussians `1e-13` apart return `~1e-26` with a sign that cannot flip. That is what lets
a later cell say "below `1e-12`" and mean it.

The tests hold it against the scalar closed form, the textbook form where that form is
accurate, and an affine change of coordinates, which is the invariance the ledger
relies on to treat nats as a scale-free unit.

## The misspecification term reads two predictives

`observation_predictive` pushes a belief through the dynamics and then through the
sensor and returns the Gaussian `p(y | u)` for the next reading. `misspecification_step`
is the divergence from the true model's predictive to the cell model's, each predicting
from its own **exact** belief.

That last word is the design decision. If the cell's predictive came from the agent's
own belief, a cell that filtered badly under the correct model would show a nonzero
misspecification, and C3 would fail by construction rather than by finding. So the
term is a function of `(p*, p, y_{1:t−1}, u)` and nothing about the filter. Two
separately built models with equal numbers score `0.0` exactly at every step, since
both sides run the same arithmetic on the same values and identical Gaussians diverge
by exactly zero.

Both functions refuse state-dependent noise. Under `R(x)` the predictive is a scale
mixture with no covariance that describes it, and scoring that is the reference
filter's job at PR-9, not a formula's.

## The inference gap is an average with a closed form

`inference_gap_step` takes two functions of the reading: the cell's filter step at its
current belief, and the exact step under the cell's own model at the exact filter's
belief. Every Gaussian filter step is affine in `y`, with a covariance that never sees
the reading, so the average over `y ∼ p*(y | u)` collapses to

```
E_y[KL] = KL at y = m*  +  ½ · tr(Σ_p⁻¹ · D · S* · Dᵀ)
```

with `D` the difference of the two gains and `S*` the true predictive covariance. The
trace is a Frobenius norm of a whitened `D`, so the whole thing is squares and
`λ − 1 − ln λ` terms, like the divergence it is built on.

The gains are not asked for. A backend exposes `infer_states` and nothing about how it
got there, so the step is probed: `m + 1` calls read the map off unit offsets, and one
more call in the opposite direction checks the map predicts it. A step that is not
affine, or whose covariance moves with the reading, is refused. The closed form then
cannot be applied to a rule it does not describe, which matters more than the probe's
cost (it is an instrument, not the agent's per-cycle path).

Three oracles hold it. The scalar closed form `test_reference_gap` already uses. The
grid engine in `cpomdp.reference`, on the fixed-`R` case that ADR-053 says is the one
both engines can reach, agreeing to `1e-6` with an integrator that knows nothing about
gains. And a two-dimensional case with the gains written out by formula and the
average taken by the textbook trace. The exact rule scores `0.0` exactly, since it and
the exact step are one computation.
