"""The warrant vocabulary, re-exported from ``warrantlib``.

``warrantlib`` is a standalone package with no dependencies beyond the standard library,
so a check suite can label its findings without installing cpomdp or JAX. cpomdp depends
on it and exposes the same names under this path and at the top level. Every definition
lives in ``warrantlib``. Nothing here adds to them.
"""

from warrantlib import (
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
