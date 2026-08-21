"""How well a claim is warranted, kept separate from whether it held.

Warrant is a property of the check, not of the number it produced. A grid sample and an
exhaustive enumeration can both come back clean. Only one of them decided anything, and
printing both as ``PASS`` loses that from the record.

Every check in a suite labels itself from this enum. It started as ``SearchWarrant``
inside cpomdp's enumerated search, covering searches alone. That name survives there as
an alias.

``CheckReport`` is what a check emits. It pairs the warrant with an ``Outcome`` and a
``Tier``, which vary independently: a run can be green throughout and have decided
nothing, and the summary says so.

``CompletenessCertificate`` and ``SymbolicReduction`` are the two evidence kinds a
``PROVED`` report carries, one per decisive prover. ``Provenance`` is the other thing it
carries: which ref registered the claim, and which one measured it.

The standard library is the only dependency.
"""

from warrantlib._vocabulary import (
    CheckReport,
    CompletenessCertificate,
    Evidence,
    Outcome,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = [
    "CheckReport",
    "CompletenessCertificate",
    "Evidence",
    "Outcome",
    "Provenance",
    "SymbolicReduction",
    "Tier",
    "Warrant",
    "check_summary",
]
