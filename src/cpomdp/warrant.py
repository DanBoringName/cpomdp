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

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only. `enumeration` imports this module, so a runtime import here would
    # close the cycle. Nothing below reads the certificate's fields.
    from cpomdp.enumeration import CompletenessCertificate

__all__ = ["CheckReport", "Outcome", "Tier", "Warrant"]


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
    """What a check found, independent of how well it was warranted.

    ``PASS`` — the condition held.
    ``FAIL`` — it did not, and the refutation is the result.
    ``NOT_RESOLVED`` — the check ran and decided neither way. A tie, or a falsifier void
    by construction that could not have fired here.

    Three values, not two. Forcing a tie into ``PASS`` is how a check that decided
    nothing gets counted among the ones that did, and a falsifier that cannot fire is
    not evidence for the claim it was pointed at.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RESOLVED = "NOT_RESOLVED"


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
class CheckReport:
    """One check's result: what it found, how well, and against what.

    A record rather than a return value. Frozen, because editing a report after the
    check ran is editing the finding.

    Args:
        name: which check this is, as it appears in the summary.
        warrant: the prover class behind the claim.
        outcome: whether the condition held, failed, or went undecided.
        tier: what the check was measured against.
        detail: why it reports what it reports, in one line. Required, so a report
            cannot be a bare outcome with extra fields.
        certificate: the completeness certificate, where the check has one.
    """

    name: str
    warrant: Warrant
    outcome: Outcome
    tier: Tier
    detail: str
    certificate: "CompletenessCertificate | None" = None

    def __str__(self) -> str:
        """The report as one summary line, in the warrant's own vocabulary."""
        return (
            f"{self.name}: {self.outcome.value} "
            f"({self.warrant.value}, tier {self.tier.value}). {self.detail}"
        )
