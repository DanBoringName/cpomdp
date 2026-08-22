# warrantlib

A column of `PASS` cannot say whether anything was decided. A grid sample over a
continuous range and an exhaustive enumeration over a declared finite set can both come
back clean. Only the second settled the question. warrantlib is a small vocabulary for
keeping that difference in a check suite's output instead of losing it there.

```bash
pip install warrantlib
```

Python 3.11 and up. The standard library is the only dependency.

## Warrant

`Warrant` says how well a claim is warranted, by the prover class behind it.

| Prover | What it does | Label |
| --- | --- | --- |
| 1 | pen-and-paper theorem, within stated hypotheses | `PROVED` |
| 2 | symbolic computation: closed-form identities, algebraic non-existence | `PROVED` |
| 3 · enumeration | exhaustive enumeration over a finite domain | `PROVED`, with a completeness certificate |
| 3 · validated | validated numerics over a compact domain | `CERTIFIED` |
| 3 · sample | sampling a continuum | `CORROBORATED` |

`CERTIFIED` sits between the other two. Validated numerics prove a universal over a
compact domain, and the proof carries the bound it was computed with. Borrowing `PROVED`
overclaims. Borrowing `CORROBORATED` throws the bound away.

An action sweep over a continuous range is a finite grid over an infinite domain, so it
samples. A policy enumeration over a declared finite set enumerates. The warrant follows
from which of those the check did, not from how clean the answer looked.

## Outcome

`Outcome` says what a registered falsifier did. A falsifier does not pass. It fires or it
does not, and `PASS` is absent from the vocabulary rather than disambiguated by a column
beside it.

| Value | What happened |
| --- | --- |
| `NOT_TRIGGERED` | it ran, the condition did not obtain, the claim survives it |
| `FIRED` | the condition obtained. The claim is refuted, and that is the result |
| `NOT_RESOLVED` | it ran and the ordering is genuinely undetermined, because the two quantities' intervals overlap |
| `NOT_APPLICABLE` | void by construction, so it is evidence for nothing and is not a survivor |
| `NOT_RUN_HERE` | measured elsewhere, or not yet. The detail says where |

Collapsing the last three loses the survivor accounting, and burns the word a real tie
needs. The last two never ran, so they carry no warrant, and `CheckReport` enforces that.

## The rest

`Tier` says what the check was measured against. `EXACT` against a closed form,
`BOUNDED` against a stated bar, `COMPUTED` where there is no bar to state. It cuts across
the other two rather than ranking them.

`CheckReport` is what a check emits. Frozen, because editing a report after the check ran
is editing the finding. It refuses `PROVED` with nothing behind it.

Evidence comes in two kinds, one per decisive prover. `CompletenessCertificate` backs an
exhaustive enumeration, recording the domain it covered against the count it visited.
`SymbolicReduction` backs a theorem or a symbolic identity, and names where the symbolic
setup was checked by hand against the analytic problem it stands for. A CAS establishes
that one expression equals another. Whether those are the right expressions is a human
obligation, and this is where it is discharged rather than assumed.

`check_summary` prints a run as counts per `(warrant, outcome)`.

## Provenance

Evidence says a claim was decided. It does not say when the bar was set, and a bar chosen
after the number is visible decides nothing. `Provenance` is the pointer a reviewer
follows: the ref where the prediction or the derivation was registered, the ref whose
tree produced the number, and one line saying what they will find at the first.

A ref is a git commit SHA, an http(s) URL or a DOI. A path, a branch, a tag and `HEAD`
are refused, because each resolves to a different tree every time it is read. A `PROVED`
report requires one, on the same terms as it requires evidence.

Where the two refs name one commit, the render says the ordering is not established by
history. That is not a failure. It is what happens whenever a check and the derivation
behind it land together, and the marker keeps it from reading as something a reviewer
could verify.

The type compares refs. It cannot order them, so a registration written after the fact
still renders without a marker. That is a `git merge-base --is-ancestor` away, which is
why the refs are refs.

## Two names per check

A report carries prose and a key, and they do different jobs.

`name` reads in a summary line, so it is reworded whenever the wording improves.
`check_id` is what a manifest declares before a run and what joins one run's report to
the next: dot-separated segments of letters, digits and underscores, refused at
construction if anything else appears in it.

Deriving the key from the prose ties the two together, and the first reworded name then
reads as one check dropped and one added, which is the reading a comparison exists to
rule out.

## Use

```python
from warrantlib import (
    CheckReport, Outcome, Provenance, SymbolicReduction, Tier, Warrant, check_summary,
)

report = CheckReport(
    name="second gap coefficient",
    check_id="gap_series.c2_closed_form",
    warrant=Warrant.PROVED,
    outcome=Outcome.NOT_TRIGGERED,
    tier=Tier.EXACT,
    detail="the CAS reduces the integral to the quoted constant",
    evidence=(
        SymbolicReduction(
            claim="the second gap coefficient equals the quoted constant",
            correspondence="hand derivation, section 3",
            assumptions=("the expansion is formal, not convergent",),
        ),
    ),
    provenance=(
        Provenance(
            registered_at="a76cf1b",
            measured_at="9baaa22",
            registered="the coefficient's closed form, registered 2026-08-07",
        ),
    ),
)

print(check_summary([report]))
```

```text
1 registered, 1 tested here, none fired
   PROVED        NOT TRIGGERED   1
```

Registering four falsifiers and testing two is a different claim from testing four, and
one number cannot carry both. The header separates them.

## Writing a run down

A run that prints and exits leaves nothing to compare against. `report_to_dict` writes
one report as a JSON-ready mapping, and `report_from_dict` reads one back.

```python
from warrantlib import SCHEMA_VERSION, report_from_dict, report_to_dict

record = report_to_dict(report)
assert report_from_dict(record) == report
```

Key order is fixed rather than taken from the input, so two runs of one suite produce
the same bytes and a diff shows what changed rather than what moved. Enums travel as
their values, which is why those values are words.

Reading goes through the constructor rather than around it, so every precondition still
applies. A record naming `PROVED` with its evidence stripped does not construct. A
record from a schema version this one does not know, an evidence kind with no class
behind it, or an enum value no member carries is refused rather than resolved to
whatever fits.

Nothing here touches a filesystem. The caller decides where the bytes go. A JSON Schema
for the record ships beside the code at `warrantlib/report.schema.json`, for a consumer
reading a ledger without Python.

## Under pytest

`pip install "warrantlib[pytest]"` adds a plugin, loaded through pytest's `pytest11`
entry point. A test hands its findings over and the run reports them in the vocabulary
the check used.

```python
def test_the_coefficient_is_closed_form(record_check):
    record_check(measure_c2())
```

```text
test_the_coefficient_is_closed_form NOT TRIGGERED  [100%]

============== warrant summary ==============
1 registered, 1 tested here, none fired
   PROVED        NOT TRIGGERED   1
```

A fired check fails its test, and so does an unresolved one. The two that never ran here
skip, and the progress letters keep them apart: `v` for void by construction, `e` for
measured elsewhere.

A manifest takes it further. It declares every check a suite is registered to report, so
a check that stops reporting fails by name instead of shrinking a count that could not
say which one left. Generate one with `python -m warrantlib.manifest <path>` and ask
whether it is current with `--check`.

## Where it comes from

warrantlib was factored out of [cpomdp](https://github.com/inferogenesis/cpomdp), where
it labels a research programme's falsification battery. It is developed in that
repository and released separately. The API reference is at
[cpomdp.inferogenesis.com/api/warrant](https://cpomdp.inferogenesis.com/api/warrant/).
