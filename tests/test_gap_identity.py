"""What the closed-form gap suite is allowed to claim, and what it must keep refusing.

A `PROVED` row asserting something false is worse than no row. These guard the shape of
the claim rather than its arithmetic: the module's own checks establish the algebra, and
these establish that the warrants, the tiers and the disclaimed hypotheses stay put.
"""

from functools import cache

import pytest

from research.checks import gap_identity
from warrantlib import Outcome, SymbolicReduction, Tier, Warrant

#: The six identities symbolic computation decides, and the one comparison it does not.
_SYMBOLIC = {
    "gap_identity.integrand_is_quadratic",
    "gap_identity.predictive_is_a_density",
    "gap_identity.predictive_second_moment",
    "gap_identity.closed_form_assembles",
    "gap_identity.vanishes_at_the_truth",
    "gap_identity.reverse_is_not_forward",
}
_SAMPLED = "gap_identity.engines_agree"


@cache
def _reports():
    """The suite, run once for the module. Two quadrature engines make it slow."""
    return tuple(gap_identity.run_checks())


@pytest.mark.slow
def test_nothing_fires():
    fired = [r.check_id for r in _reports() if r.outcome is Outcome.FIRED]
    assert not fired, fired


@pytest.mark.slow
def test_the_symbolic_identities_are_proved_at_exact():
    for report in _reports():
        if report.check_id not in _SYMBOLIC:
            continue
        assert report.warrant is Warrant.PROVED, report.check_id
        assert report.tier is Tier.EXACT, report.check_id


@pytest.mark.slow
def test_every_proved_row_carries_its_reduction_and_provenance():
    # `CheckReport` refuses PROVED without provenance, so this is about the evidence
    # kind: a symbolic claim has to name the correspondence a CAS cannot supply.
    for report in _reports():
        if report.warrant is not Warrant.PROVED:
            continue
        assert report.provenance, report.check_id
        assert any(isinstance(item, SymbolicReduction) for item in report.evidence), (
            report.check_id
        )


@pytest.mark.slow
def test_the_cross_engine_comparison_does_not_claim_to_be_proved():
    # It samples a continuum of spreads. Adding spreads does not make it a universal,
    # and ADR-052 rests on it staying honest about that.
    (comparison,) = [r for r in _reports() if r.check_id == _SAMPLED]
    assert comparison.warrant is Warrant.CORROBORATED
    assert comparison.tier is Tier.BOUNDED
    assert not comparison.evidence


@pytest.mark.slow
def test_the_disclaimed_hypotheses_travel_with_the_evidence():
    # The three things the identity does not establish. If one of these stops being
    # said, a reader of the report loses the scope and the row overclaims without any
    # number changing.
    reductions = [
        item
        for report in _reports()
        for item in report.evidence
        if isinstance(item, SymbolicReduction)
    ]
    assert reductions
    for reduction in reductions:
        joined = " ".join(reduction.assumptions).lower()
        assert "non-negativity is not asserted" in joined
        assert "scalar state" in joined
        assert "r constant in the state" in joined


def test_the_setup_prints_without_asserting_anything(capsys):
    # The no-argument path is the one a reader runs first. It must not be where the
    # scope disclaimer went missing.
    assert gap_identity.main([]) == 0
    printed = capsys.readouterr().out
    assert "Not proved here" in printed
    assert "Gibbs" in printed
