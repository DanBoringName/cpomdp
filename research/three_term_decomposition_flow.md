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
