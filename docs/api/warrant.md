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

A registered falsifier does not pass. It fires or it does not, and `PASS` is absent from the vocabulary rather than disambiguated by a column beside it.

`Outcome` has five values because five things can happen to a falsifier, and they are not interchangeable. It ran and did not fire, so the claim survives it. It fired, and the refutation is the result. It ran and the ordering came out genuinely undetermined, because the two quantities' intervals overlap. It was void by construction and could not have fired here, so it is evidence for nothing and is not a survivor. Or it was measured elsewhere and did not run here at all. Collapsing the last three loses the survivor accounting, and burns the word a real tie needs.

`Tier` says what the check was measured against, and cuts across the other two rather than ranking them. A Tier A closed-form reference can be sampled, and an exhaustive enumeration can produce a Tier C number.

A check that never ran carries no warrant. `CORROBORATED` means sampling-grade evidence was obtained, so attributing it to a falsifier that sampled nothing claims evidence that does not exist. The warrant is `None` there and prints as `—`, enforced at construction.

A `PROVED` report needs evidence, enforced at construction. `CompletenessCertificate` is the only kind today, since exhaustive enumeration is the only decisive prover the suite runs. A theorem citation joins it when a Prover 1 check needs one. The weaker levels need none, because a bound and a sample carry their story in `detail`. Report `PROVED` with nothing behind it and the constructor raises.

::: cpomdp.Outcome

::: cpomdp.Tier

::: cpomdp.CheckReport

## Reading a run

Registering four falsifiers and testing two is a different claim from testing four, and one number cannot carry both. The header separates them and names how many fired. The rows underneath say what warrant the tested ones carried, so a run that survived everything without deciding anything reads as exactly that.

```text
4 registered, 2 tested here, none fired
   PROVED        NOT TRIGGERED   2
   —             NOT APPLICABLE  1
   —             NOT RUN HERE    1
```

::: cpomdp.check_summary
