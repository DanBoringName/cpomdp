"""How well a claim is warranted, kept separate from whether it held.

Warrant is a property of the check, not of the number it produced. A grid sample and an
exhaustive enumeration can both come back clean. Only one of them decided anything, and
printing both as ``PASS`` loses that from the record.

Every check in the suite labels itself from this enum. It started as ``SearchWarrant``
in ``cpomdp.enumeration``, covering searches alone. That name survives as an alias.

``CheckReport`` is what a check emits. It pairs the warrant with an ``Outcome`` and a
``Tier``, which vary independently: a run can be green throughout and have decided
nothing, and the summary says so.
"""

import unicodedata
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
    in full, where ¬∃ ≡ ∀¬ (Prover 3 · enumeration). Under that last one it is earned
    only with a completeness certificate. Without one the enumeration is a sample
    wearing a decision's label.

    ``CERTIFIED`` — validated numerics prove a universal over a compact domain by
    construction (Prover 3 · validated). Stronger than a sample, weaker than a decision,
    and it carries the bound it was computed with. Collapsing it into ``PROVED``
    overclaims. Collapsing it into ``CORROBORATED`` throws the bound away.

    ``CORROBORATED`` — a sample of a continuum (Prover 3 · sample). It exhibits
    existence and refutes a universal by counterexample. It never decides one, at any
    sample count.

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

    ``EXACT`` — a closed-form reference at machine precision.
    ``BOUNDED`` — a stated bar, or a certified bracket.
    ``COMPUTED`` — no statable bar. The word for such a number is *computed*, never
    *certified*.

    Cuts across warrant and outcome rather than ranking them. An ``EXACT`` reference can
    be sampled (Prover 3 · sample) and a ``COMPUTED`` number can come out of an
    exhaustive enumeration (Prover 3 · enumeration).
    """

    EXACT = "exact"
    BOUNDED = "bounded"
    COMPUTED = "computed"


#: Unicode general categories that put nothing on the page: the controls, the format
#: characters (soft hyphen, the zero-width space and the joiners, the byte-order mark),
#: and the three kinds of separator. ``str.strip()`` removes the separators and the
#: whitespace controls and leaves the format characters behind, so a field of them alone
#: survives a presence check while reading as empty.
_UNPRINTED = frozenset({"Cc", "Cf", "Zl", "Zp", "Zs"})


def _reject_unreadable(name: str, value: object, blank_reason: str) -> None:
    """Raise unless a reduction's ``value`` is one-line text that puts ink on the page.

    Args:
        name: the field, as it appears in the message.
        value: what was passed for it.
        blank_reason: why this particular field may not be blank, appended to the
            message when it is.

    Raises:
        ValueError: if ``value`` is not a string, is blank, or holds a line break.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"symbolic reduction passed {name} as {type(value).__name__}. The field is "
            "text a reader reads back, so anything else records the obligation in a "
            "form nothing renders."
        )
    if all(unicodedata.category(character) in _UNPRINTED for character in value):
        raise ValueError(
            f"symbolic reduction has a blank {name}. Whitespace and the zero-width "
            "characters count as blank here, since they satisfy a presence check and "
            f"put nothing on the page. {blank_reason}"
        )
    if value.splitlines() != [value]:
        raise ValueError(
            f"symbolic reduction has a line break in {name}. A reduction renders as "
            "one line beside the check it backs, so a second line arrives in the "
            "middle of a summary row."
        )


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
        ValueError: if the assumptions are not a tuple, or if the claim, the
            correspondence or any assumption is not one-line text with a visible
            character in it.
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
            _reject_unreadable(
                name,
                value,
                "Prover 2 is theorem-grade only where the symbolic setup was hand "
                "derived against the analytic problem, so a reduction naming neither "
                "the statement nor where it was checked backs nothing. Fill both, or "
                "report CORROBORATED and say why in the check's detail.",
            )
        for position, assumption in enumerate(self.assumptions, start=1):
            _reject_unreadable(
                f"assumption {position}",
                assumption,
                "An entry nobody filled in is scope a reader cannot check, and it "
                "renders as a gap in the list rather than as a caveat. Say what the "
                "condition is, or drop the entry.",
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
    # completeness certificate decides by exhausting a finite domain. A symbolic
    # reduction decides by identity (Provers 1 and 2) and enumerates nothing, so a
    # certificate is the wrong evidence for it rather than a missing one.
    Evidence = CompletenessCertificate | SymbolicReduction


def _evidence_types() -> tuple[type, ...]:
    """The evidence classes, as a tuple ``isinstance`` accepts.

    Imported on call rather than at module scope: ``enumeration`` imports this module,
    so the certificate is only reachable once that import has finished.
    """
    from cpomdp.enumeration import CompletenessCertificate

    return (CompletenessCertificate, SymbolicReduction)


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
        evidence: what backs the claim, as a tuple of ``CompletenessCertificate`` and
            ``SymbolicReduction``. Required non-empty when the warrant is ``PROVED``
            and unused otherwise. A tuple rather than one item because a claim
            quantified over several enumerations rests on all their certificates, and
            carrying one of them understates what was checked.

    Raises:
        ValueError: if the warrant is ``PROVED`` and no evidence was given, if a check
            that never ran here carries a warrant anyway, if the evidence is not a
            tuple, or if an item in it is neither of the two evidence kinds.
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
        if self.evidence:
            permitted = _evidence_types()
            for item in self.evidence:
                if not isinstance(item, permitted):
                    raise ValueError(
                        f"check {self.name!r} carries a "
                        f"{type(item).__name__} as evidence. There are two kinds, one "
                        "per decisive prover: a CompletenessCertificate for an "
                        "exhaustive enumeration (Prover 3 · enumeration), a "
                        "SymbolicReduction for a theorem or a symbolic identity "
                        "(Provers 1 and 2). Anything "
                        "else satisfies the PROVED precondition by being present and "
                        "backs nothing."
                    )
        if self.warrant is Warrant.PROVED and len(self.evidence) == 0:
            raise ValueError(
                f"check {self.name!r} reports PROVED with no evidence. A decided "
                "universal needs something statable behind it: a completeness "
                "certificate for an exhaustive enumeration (Prover 3 · enumeration), a "
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
