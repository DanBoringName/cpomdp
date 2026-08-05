"""How well a claim is warranted, kept separate from whether it held.

Warrant is a property of the check, not of the number it produced. A grid sample and an
exhaustive enumeration can both come back clean. Only one of them decided anything, and
printing both as ``PASS`` loses that from the record.

Every check in the suite labels itself from this enum. It started as ``SearchWarrant``
in ``cpomdp.enumeration``, covering searches alone. That name survives as an alias.
"""

from enum import Enum

__all__ = ["Warrant"]


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
