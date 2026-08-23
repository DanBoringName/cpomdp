# The fit window under subtraction: an exploration

**Status: EXPLORATORY. Nothing here is registered and no check cites it.** The numbers
below rest on a stand-in for a quantity the registration derives differently, stated in
"What this cannot support" at the end. They are here to record what was looked at and
what it suggests, not to be quoted.

## The code

`research/src/research/explorations/c6_window.py`, run with
`python -m research.explorations.c6_window`. It prints every number below and asserts the
two validations, so a reader can disagree with the method rather than with the prose. It
reports no warrant and is in no manifest: an exploration has none to report.

## The question

`research/gate_d4_registration.md` registers `D* = 0.520` and `f* = 0.0488` at
`k_min = 10` and `β = 0.05`. Both came from an optimisation whose bias term is the
*quartic* truncation, diluted by widening the window: the entry describes the bias as
falling like `1/D²`, which is the dilution scaling.

The dilute-versus-subtract rule has since fired for **subtract**, on `c₄`'s relative
error being far inside the registered `X = 0.1`. Subtraction removes the quartic from the
fit, so the residual the fit still carries is the sextic. `c₆` is now in closed form.

So: does the optimisation that produced `D*` and `f*` still hold when the term it
optimised against is no longer the one left over?

## What was checked, and against what

**The bias integral reproduces a registered table.** For a relative correction
`−f·e^{m·u}` with `u = ln σ − ln σ_max` uniform over `[−L, 0]`, `L = D·ln10`, the exact
OLS bias in the fitted exponent is `Cov(u, ln(1 + ε))/Var(u)`. Its ratio to the
registration's first-order envelope `−3f/L²` at `f = 0.02`:

| | `D=1` | `D=2` | `D=3` |
| --- | --- | --- | --- |
| computed here | 1.712 | 1.270 | 1.163 |
| registration | 1.71 | 1.27 | 1.16 |

**The sextic arm satisfies a scaling identity.** The first-order envelope generalises to
`−(6/m)·f/L²`, giving `−3f/L²` at `m = 2` and `−1.5f/L²` at `m = 4`. Rescaling `u` by the
exponent turns the exact integral into `bias(m, L) = m·bias(1, m·L)`, so the `m = 4` bias
at `D` is **twice** the `m = 2` bias at `2D`, and the two arms are related exactly rather
than approximately. The computed ratio is 2.000000 at both widths tried.

An earlier draft of this file claimed the two were *equal* rather than a factor of two
apart. They are not, and the assertion in
`research.explorations.c6_window` is what caught it: the ratio to the first-order
envelope is what matches between the arms, and that was misread as the biases matching.
Nothing else in this file rested on the wrong version.

## What it suggests

**`T` changes shape, not only value.** Defining `f` as the fractional size of the leading
unmodelled term at the top edge:

```text
quartic edge   σ_max² = f·c₂/|c₄|      T = f · c₂²/|c₄| · 10^(−2D)
sextic edge    σ_max⁴ = f·c₂/|c₆|      T = c₂^(3/2) · √(f/|c₆|) · 10^(−2D)
```

Linear in `f` becomes square-root in `f`, which is what moves the optimiser.

**`D*` looks robust.** Re-optimising against the sextic residual puts the optimum near
`D ≈ 0.466` rather than `0.520`, a factor of 0.895. That factor held between 0.887 and
0.903 under a ±15% perturbation of the noise calibration. Holding `D` at the registered
0.520 costs about 6% on `T`, against a quantity that moves 100× per decade.

**`f*` cannot be carried across at all, and the reason is not numerical.** `f` is defined
against whichever term sets the edge. Under subtraction that term is the sextic. The
registered 0.0488 and the sextic optimum near 0.034 are different quantities wearing the
same letter, so `f` needs deriving under the new edge rather than transferring.

## What this cannot support

**The noise term is a stand-in.** `σ_p` here is `A/D`, with `A` calibrated so the quartic
optimisation lands on the registered `D* = 0.520`. It reproduces the rest to about ten
percent: `f* = 0.0445` against 0.0488, `σ_p = 0.0388` against 0.0359. The registration
propagates the error through OLS with heteroscedastic weights, and that propagation has
not been recovered here.

Ten percent is enough to say whether a quantity moves. It is not enough to declare one.
A new `f*` taken from this would be a fitted constant under a registered label, which is
the failure ADR-037 exists to disclose rather than repeat.

**And it surfaced a gap.** The registration's own statistical formula reads
`ε√12/(D√N)`, so `σ_p` depends on `N`, the number of `σ` samples the D2 fit uses. `N`
appears nowhere else in that document and is not declared. It is a free choice that moves
`σ_p`, hence the constraint, hence `D*` and `f*`. Whatever `σ_p` propagation is recovered
will need `N` fixed before it produces a number, and fixing it now is cheaper than fixing
it once the answer is visible.

## What would make any of this registrable

> **Note, 2026-08-23.** The first sentence below is superseded.
> `research/d2_noise_model_exploration.md`, added alongside this file, finds that
> the registration's four published `σ_p` values are all reproduced by *unweighted*
> OLS at one `N ≈ 60`, while the weighted fit has no solution for the fourth. The
> heteroscedastic propagation is therefore not what produced them, so it is not the
> thing to recover. The rest of this section still stands.

Recover the heteroscedastic propagation the registration used, with `N` declared.
Reproduce `D* = 0.520`, `f* = 0.0488` and `σ_p = 0.0359` from it rather than to ten
percent. Only then re-run the sextic optimisation, and register the result as an
amendment.
