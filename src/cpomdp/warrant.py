"""The warrant vocabulary, re-exported from ``warrantlib``.

``warrantlib`` is a standalone package with no dependencies beyond the standard library,
so a check suite can label its findings without installing cpomdp or JAX. cpomdp depends
on it and exposes the same names under this path and at the top level. Every definition
lives in ``warrantlib``. Nothing here adds to them.

``Evidence`` is a union, so widening it widens what this path admits whether or not the
new members are named here. A caller annotating against it could be handed a type it had
no import for. The completeness leaves are carried for that reason, rather than because
any old import path names them.
"""

from warrantlib import (
    AxisDeclaration,
    CheckReport,
    CompletenessCertificate,
    CompletenessEvidence,
    Evidence,
    Outcome,
    ProductCompletenessCertificate,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    check_summary,
)

__all__ = [
    "AxisDeclaration",
    "CheckReport",
    "CompletenessCertificate",
    "CompletenessEvidence",
    "Evidence",
    "Outcome",
    "ProductCompletenessCertificate",
    "Provenance",
    "SymbolicReduction",
    "Tier",
    "Warrant",
    "check_summary",
]
