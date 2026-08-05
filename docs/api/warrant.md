# Warrant

Warrant is a property of the check, not of the number. A grid sample over a continuous action range and an exhaustive enumeration over a declared finite set can both report that no policy flips. Only the second decided it.

| Prover | What it does | Label |
| --- | --- | --- |
| 1 | pen-and-paper theorem, within stated hypotheses | `PROVED` |
| 2 | symbolic computation: closed-form identities, algebraic non-existence | `PROVED` |
| 3a | sampling a continuum | `CORROBORATED` |
| 3b | exhaustive enumeration over a finite domain | `PROVED`, with a completeness certificate |
| 3c | validated numerics over a compact domain | `CERTIFIED` |

An action sweep over a continuous range is a finite grid over an infinite domain, so 3a. A policy enumeration over a declared finite set is 3b. `EFESelector` reports `CORROBORATED` and `EnumeratedEfeSearch` reports `PROVED` for that reason.

`CERTIFIED` sits between the two. Validated numerics prove a universal over a compact domain, and the proof carries the bound it was computed with. Borrowing `PROVED` overclaims. Borrowing `CORROBORATED` throws the bound away.

::: cpomdp.Warrant

## What a check emits

Outcome is orthogonal to warrant. A check reports both, so a green run that is entirely corroborative reads as one rather than as a column of `PASS`.

`Outcome` has three values because a registered falsifier has three answers. It held, it fired, or it decided neither way. That third case covers a tie and a falsifier void by construction, which cannot fire and so is evidence for nothing. Forcing either into `PASS` counts a check that decided nothing among the ones that did.

`Tier` says what the check was measured against, and cuts across the other two rather than ranking them. A Tier A closed-form reference can be sampled, and an exhaustive enumeration can produce a Tier C number.

A `PROVED` report needs evidence, enforced at construction. `CompletenessCertificate` is the only kind today, since exhaustive enumeration is the only decisive prover the suite runs. A theorem citation joins it when a Prover 1 check needs one. The weaker levels need none, because a bound and a sample carry their story in `detail`. Report `PROVED` with nothing behind it and the constructor raises.

::: cpomdp.Outcome

::: cpomdp.Tier

::: cpomdp.CheckReport
