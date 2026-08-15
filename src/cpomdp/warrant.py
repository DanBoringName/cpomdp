"""How well a claim is warranted, kept separate from whether it held.

Warrant is a property of the check, not of the number it produced. A grid sample and an
exhaustive enumeration can both come back clean. Only one of them decided anything, and
printing both as ``PASS`` loses that from the record.

Every check in the suite labels itself from this enum. It started as ``SearchWarrant``
in ``cpomdp.enumeration``, covering searches alone. That name survives as an alias.

``CheckReport`` is what a check emits. It pairs the warrant with an ``Outcome`` and a
``Tier``, which vary independently: a run can be green throughout and have decided
nothing with the summary should saying so.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only. `enumeration` imports this module, so a runtime import here would
    # close the cycle. Nothing below reads the evidence's fields. The alias this feeds
    # is declared under `SymbolicReduction`, which is its other arm.
    from cpomdp.enumeration import CompletenessCertificate

__all__ = [
    "CheckReport",
    "Outcome",
    "SymbolicReduction",
    "Tier",
    "Warrant",
    "check_summary",
]


class Warrant(Enum):
    """How well a claim is warranted, by the prover class that produced it.

    ``PROVED`` — the claim is decided. A pen-and-paper theorem within its stated
    hypotheses (Prover 1), a symbolic identity (Prover 2), or a finite domain enumerated
    in full, where ¬∃ ≡ ∀¬ (Prover 3b). Under 3b it is earned only with a completeness
    certificate. Without one the enumeration is a sample wearing a decision's label.

    ``CERTIFIED`` — validated numerics prove a universal over a compact domain by
    construction (Prover 3c). Stronger than a sample, weaker than a decision, and it
    carries the bound it was computed with. Collapsing it into ``PROVED`` overclaims.
    Collapsing it into ``CORROBORATED`` throws the bound away.

    ``CORROBORATED`` — a sample of a continuum (Prover 3a). It exhibits existence and
    refutes a universal by counterexample. It never decides one, at any seed count.

    Orthogonal to outcome. A check reports both, and the three levels print in distinct
    vocabulary so a corroborative green run is visibly that.
    """

    PROVED = "PROVED"
    CERTIFIED = "CERTIFIED"
    CORROBORATED = "CORROBORATED"


class Outcome(Enum):
    """What a registered falsifier did, independent of how well it was warranted.

    A falsifier does not pass. It fires or it does not, and the words are chosen so that
    a run cannot be read as a column of ``PASS`` with the interesting distinctions
    flattened out of it.

    ``NOT_TRIGGERED`` — it ran, the condition did not obtain, the claim survives it.
    ``FIRED`` — the condition obtained. The claim is refuted, and that is the result.
    ``NOT_RESOLVED`` — it ran and the ordering is genuinely undetermined, because the
    two quantities' intervals overlap. Narrow on purpose: this is a measured tie, not a
    stand-in for a check that did not run.
    ``NOT_APPLICABLE`` — void by construction. It could not have fired here, so it is
    evidence for nothing and does not count among the survivors.
    ``NOT_RUN_HERE`` — measured elsewhere, or not yet. The detail says where.

    The last two never ran, so they carry no warrant. ``CheckReport`` enforces that.
    """

    NOT_TRIGGERED = "NOT TRIGGERED"
    FIRED = "FIRED"
    NOT_RESOLVED = "NOT RESOLVED"
    NOT_APPLICABLE = "NOT APPLICABLE"
    NOT_RUN_HERE = "NOT RUN HERE"


#: The outcomes of a falsifier that actually ran here. The other two did not, so they
#: are not counted among the tested and cannot carry a warrant.
_TESTED_HERE = (Outcome.NOT_TRIGGERED, Outcome.FIRED, Outcome.NOT_RESOLVED)


class Tier(Enum):
    """What the check was measured against.

    ``A`` — a closed-form reference at machine precision.
    ``B`` — a stated bar, or a certified bracket.
    ``C`` — computed, with no statable bar. The word for a Tier C number is *computed*,
    never *certified*.

    Cuts across warrant and outcome rather than ranking them. A Tier A reference can be
    sampled (3a) and a Tier C number can come out of an exhaustive enumeration (3b).
    """

    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class SymbolicReduction:
    """What backs a Prover 2 claim: the correspondence a CAS cannot supply.

    A CAS checks that one expression equals another. It does not check that those
    expressions are the ones the analytic claim is about. That step is a human
    obligation, named as such in the warrant ledger, and this is where it is recorded
    instead of assumed. A reduction is evidence for
    [`CheckReport`][cpomdp.CheckReport] on the same terms as a completeness
    certificate, and the two are interchangeable there.

    Args:
        claim: the analytic statement, in words, that the symbolic identity stands for.
        correspondence: where the symbolic setup was analytically checked against the
            problem it stands for. A hand derivation by file and line, or a dated
            registration result.
        assumptions: what the reduction assumed, one condition per entry. The scope
            travels with the evidence, so the contingency is visible without reading
            the algebra.

    Raises:
        ValueError: if the assumptions are not a tuple, or if either of the other two
            fields is blank.
    """

    claim: str
    correspondence: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a reduction that records no obligation to have discharged."""
        if not isinstance(self.assumptions, tuple):
            raise ValueError(
                "symbolic reduction passed assumptions as "
                f"{type(self.assumptions).__name__}. Assumptions are a tuple, so a "
                "bare string records one condition per character and the identity's "
                "scope reads as gibberish. Wrap a single one: (assumption,)."
            )
        for name, value in (
            ("claim", self.claim),
            ("correspondence", self.correspondence),
        ):
            if not value.strip():
                raise ValueError(
                    f"symbolic reduction has a blank {name}. Prover 2 is theorem-grade "
                    "only where the symbolic setup was hand derived against the "
                    "analytic problem, so a reduction naming neither the statement nor "
                    "where it was checked backs nothing. Fill both, or report "
                    "CORROBORATED and say why in the check's detail."
                )

    def __str__(self) -> str:
        """The reduction as one line: the claim, where it was checked, its scope."""
        scope = (
            f"assuming {'; '.join(self.assumptions)}"
            if self.assumptions
            else "no assumptions recorded"
        )
        return f"symbolic: {self.claim} (per {self.correspondence}, {scope})"


if TYPE_CHECKING:
    # What backs a ``PROVED`` claim, one member per decisive prover the suite runs. A
    # completeness certificate decides by exhausting a finite domain (3b). A symbolic
    # reduction decides by identity (Provers 1 and 2) and enumerates nothing, so a
    # certificate is the wrong evidence for it rather than a missing one.
    Evidence = CompletenessCertificate | SymbolicReduction


@dataclass(frozen=True)
class CheckReport:
    """One check's result: what it found, how well, and against what.

    A record rather than a return value. Frozen, because editing a report after the
    check ran is editing the finding.

    Args:
        name: which check this is, as it appears in the summary.
        warrant: the prover class behind the claim, or ``None`` where the check
            produced no evidence to classify.
        outcome: what the falsifier did.
        tier: what the check was measured against.
        detail: why it reports what it reports, in one line. Required, so a report
            cannot be a bare outcome with extra fields.
        evidence: what backs the claim, as a tuple. Required non-empty when the warrant
            is ``PROVED`` and unused otherwise. A tuple rather than one item because a
            claim quantified over several enumerations rests on all their certificates,
            and carrying one of them understates what was checked.

    Raises:
        ValueError: if the warrant is ``PROVED`` and no evidence was given, if a check
            that never ran here carries a warrant anyway, or if the evidence is not a
            tuple.
    """

    name: str
    warrant: Warrant | None
    outcome: Outcome
    tier: Tier
    detail: str
    evidence: tuple["Evidence", ...] = ()

    def __post_init__(self) -> None:
        """Reject a claim with nothing behind it, at either of two strengths."""
        if not isinstance(self.evidence, tuple):
            raise ValueError(
                f"check {self.name!r} passed evidence as "
                f"{type(self.evidence).__name__}. Evidence is a tuple, so a claim "
                "resting on several enumerations can carry all of their certificates. "
                "Wrap a single one: (certificate,)."
            )
        if self.warrant is Warrant.PROVED and not self.evidence:
            raise ValueError(
                f"check {self.name!r} reports PROVED with no evidence. A decided "
                "universal needs something statable behind it: a completeness "
                "certificate for an exhaustive enumeration (Prover 3b), a "
                "SymbolicReduction for a theorem or a symbolic identity (Provers 1 "
                "and 2). Report CERTIFIED for a bound over a compact domain, "
                "CORROBORATED for a sample."
            )
        if self.outcome not in _TESTED_HERE and self.warrant is not None:
            raise ValueError(
                f"check {self.name!r} is {self.outcome.value} and carries the warrant "
                f"{self.warrant.value}. It produced no evidence here, so there is no "
                "prover class to report. Leave the warrant None."
            )

    def __str__(self) -> str:
        """The report as one summary line, in the warrant's own vocabulary."""
        warrant = self.warrant.value if self.warrant else "—"
        return (
            f"{self.name}: {self.outcome.value} "
            f"({warrant}, tier {self.tier.value}). {self.detail}"
        )


def check_summary(reports: Sequence[CheckReport]) -> str:
    """Counts per ``(warrant, outcome)`` across a run, as a block of lines.

    The header carries the accounting a reader needs first: how many falsifiers were
    registered, how many this run actually tested, and how many fired. Registering four
    and testing two is a different claim from testing four, and one number cannot say
    both. The rows underneath say what warrant the tested ones carried, so a run that
    survived everything without deciding anything prints as exactly that.

    Pairs with no checks are left out, so the block is as long as the run was varied.
    Ordering follows the enum declarations rather than the input, so two runs of the
    same suite produce the same text. Checks with no warrant sort last, under ``—``.

    Args:
        reports: the run's reports, in any order.

    Returns:
        A newline-separated block: the accounting line, then one row per occupied pair.
    """
    counts = Counter((report.warrant, report.outcome) for report in reports)
    tested = sum(1 for report in reports if report.outcome in _TESTED_HERE)
    fired = sum(1 for report in reports if report.outcome is Outcome.FIRED)
    lines = [
        f"{len(reports)} registered, {tested} tested here, "
        f"{f'{fired} fired' if fired else 'none fired'}"
    ]
    lines += [
        f"   {(warrant.value if warrant else '—'):<13} "
        f"{outcome.value:<15} {counts[warrant, outcome]}"
        for warrant in (*Warrant, None)
        for outcome in Outcome
        if counts[warrant, outcome]
    ]
    return "\n".join(lines)
