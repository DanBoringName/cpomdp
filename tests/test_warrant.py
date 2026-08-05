"""The shared `Warrant` vocabulary and the check report that carries it.

`SearchWarrant` had two levels: `PROVED` (3b, exhaustive enumeration) and `CORROBORATED`
(3a, a grid sample). `Warrant` adds `CERTIFIED` (3c, validated numerics over a compact
domain) and moves to `cpomdp.warrant`, where checks can reach it too. `SearchWarrant`
stays as an alias, so existing call sites keep their members and their return type.

`CheckReport` is what a check emits: a warrant, an `Outcome`, a `Tier`, and a reason.
Warrant and outcome are orthogonal, so a green run that is entirely corroborative reads
as one instead of as a column of `PASS`.

Imports `cpomdp.warrant`, so until it lands this module is collection-red — the
`ModuleNotFoundError` naming it is the build cue.
"""

import dataclasses

import pytest

from cpomdp.enumeration import (
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
    OpenLoopSelector,
    RecedingHorizonSelector,
    SearchWarrant,
)
from cpomdp.selection import EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel
from cpomdp.warrant import CheckReport, Outcome, Tier, Warrant


def _model():
    """A plain fixed-sensor model, p = 1."""
    return LinearGaussianModel(
        dynamics=[[1.0, 0.1], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        sensor_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=[[0.0], [1.0]],
    )


def _finite_set(actions):
    return FiniteActionSet([[a] for a in actions], version="test-v1")


def _certificate():
    """A complete certificate: the evidence a 3b `PROVED` claim carries."""
    return CompletenessCertificate(expected=4, visited=4, warrant=Warrant.PROVED)


class TestWarrantLevels:
    def test_the_three_prover_classes(self):
        assert Warrant.PROVED.value == "PROVED"
        assert Warrant.CERTIFIED.value == "CERTIFIED"
        assert Warrant.CORROBORATED.value == "CORROBORATED"

    def test_no_fourth_level(self):
        # A fourth member is a fourth prover class: a decision, not an edit.
        assert [w.name for w in Warrant] == ["PROVED", "CERTIFIED", "CORROBORATED"]

    def test_round_trips_through_its_value(self):
        # Checks report by value, so a value read back must land on the same member.
        for level in Warrant:
            assert Warrant(level.value) is level


class TestOutcome:
    def test_the_three_outcomes(self):
        assert Outcome.PASS.value == "PASS"
        assert Outcome.FAIL.value == "FAIL"
        assert Outcome.NOT_RESOLVED.value == "NOT_RESOLVED"

    def test_no_fourth_outcome(self):
        assert [o.name for o in Outcome] == ["PASS", "FAIL", "NOT_RESOLVED"]

    def test_round_trips_through_its_value(self):
        for outcome in Outcome:
            assert Outcome(outcome.value) is outcome


class TestTier:
    def test_the_three_tiers(self):
        # A: closed-form reference at machine precision. B: a stated bar or certified
        # bracket. C: computed, with no statable bar.
        assert Tier.A.value == "A"
        assert Tier.B.value == "B"
        assert Tier.C.value == "C"

    def test_no_fourth_tier(self):
        assert [t.name for t in Tier] == ["A", "B", "C"]


class TestCheckReport:
    def _report(
        self,
        *,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.PASS,
        tier=Tier.C,
        evidence=None,
    ):
        # Defaults to the report that needs no evidence: a sample, computed, green.
        return CheckReport(
            name="flip-decided",
            warrant=warrant,
            outcome=outcome,
            tier=tier,
            detail="exhaustive argmin over crossover-v1^7 is cue-ward",
            evidence=evidence,
        )

    def test_carries_all_four_labels(self):
        report = self._report()
        assert report.name == "flip-decided"
        assert report.warrant is Warrant.CORROBORATED
        assert report.outcome is Outcome.PASS
        assert report.tier is Tier.C

    def test_detail_is_required(self):
        # A check that cannot say why it reports what it reports is a bare PASS with
        # extra fields. Omitting the reason must not construct.
        with pytest.raises(TypeError):
            CheckReport(  # ty: ignore[missing-argument]
                name="flip-decided",
                warrant=Warrant.CORROBORATED,
                outcome=Outcome.PASS,
                tier=Tier.C,
            )

    def test_is_frozen(self):
        # A report is a record of what a check found. Editing one after the fact is
        # editing the finding.
        with pytest.raises(dataclasses.FrozenInstanceError):
            self._report().outcome = Outcome.FAIL

    def test_evidence_is_absent_by_default(self):
        assert self._report().evidence is None

    def test_carries_the_evidence_it_was_given(self):
        cert = _certificate()
        report = self._report(warrant=Warrant.PROVED, tier=Tier.A, evidence=cert)
        assert report.evidence is cert

    def test_warrant_and_outcome_are_independent(self):
        # The pairing this vocabulary exists for: a green run that decided nothing.
        report = self._report()
        assert report.outcome is Outcome.PASS
        assert report.warrant is Warrant.CORROBORATED

    def test_renders_one_line_naming_all_of_it(self):
        line = str(self._report())
        assert line.count("\n") == 0
        for part in ("flip-decided", "PASS", "CORROBORATED", "C", "cue-ward"):
            assert part in line


class TestProvedNeedsEvidence:
    """`PROVED` with nothing behind it does not construct."""

    def _report(self, *, warrant, evidence=None, outcome=Outcome.PASS):
        return CheckReport(
            name="flip-decided",
            warrant=warrant,
            outcome=outcome,
            tier=Tier.A,
            detail="exhaustive argmin over crossover-v1^7 is cue-ward",
            evidence=evidence,
        )

    def test_proved_without_evidence_does_not_construct(self):
        with pytest.raises(ValueError, match="PROVED"):
            self._report(warrant=Warrant.PROVED)

    def test_proved_with_a_certificate_constructs(self):
        cert = _certificate()
        assert self._report(warrant=Warrant.PROVED, evidence=cert).evidence is cert

    def test_a_failing_proved_check_still_needs_evidence(self):
        # The outcome does not exempt it. A refutation carrying PROVED claims the
        # refutation was decided, which needs the same backing as a decided pass.
        with pytest.raises(ValueError, match="PROVED"):
            self._report(warrant=Warrant.PROVED, outcome=Outcome.FAIL)

    @pytest.mark.parametrize("warrant", [Warrant.CERTIFIED, Warrant.CORROBORATED])
    def test_the_weaker_levels_need_none(self, warrant):
        # A bound and a sample carry their own story in `detail`. Only a claim to have
        # decided a universal needs something enumerable behind it.
        assert self._report(warrant=warrant).evidence is None


class TestPublicSurface:
    def test_the_vocabulary_is_importable_from_the_package_root(self):
        import cpomdp

        assert (cpomdp.Warrant, cpomdp.Outcome, cpomdp.Tier, cpomdp.CheckReport) == (
            Warrant,
            Outcome,
            Tier,
            CheckReport,
        )

    def test_the_vocabulary_is_in_the_package_all(self):
        import cpomdp

        for name in ("Warrant", "Outcome", "Tier", "CheckReport"):
            assert name in cpomdp.__all__


class TestSearchWarrantAlias:
    def test_alias_is_the_same_enum(self):
        # Not merely equal. A separate enum breaks every `is` check in the suite.
        assert SearchWarrant is Warrant

    def test_existing_members_resolve_to_the_same_objects(self):
        assert SearchWarrant.PROVED is Warrant.PROVED
        assert SearchWarrant.CORROBORATED is Warrant.CORROBORATED


class TestExistingLabelsUnchanged:
    def test_enumerated_search_is_still_proved(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert search.warrant is Warrant.PROVED

    def test_grid_selector_is_still_corroborated(self):
        selector = EFESelector(_model(), n_candidates=5, action_bounds=(-1.0, 1.0))
        assert selector.warrant is Warrant.CORROBORATED

    @pytest.mark.parametrize("driver", [RecedingHorizonSelector, OpenLoopSelector])
    def test_enumerated_drivers_are_still_proved(self, driver):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert driver(search).warrant is Warrant.PROVED

    def test_certificate_renders_the_same_string(self):
        # This string reaches the write-up, so not a character of it may move.
        cert = CompletenessCertificate(expected=4, visited=4, warrant=Warrant.PROVED)
        assert str(cert) == "PROVED (finite set, |A|^H = 4, visited 4)"

    def test_search_still_scores(self):
        # The label is a property, so a broken search would still report PROVED.
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        result = search.evaluate(
            Belief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
            Preference(goal=[0.0], precision=[[1.0]]),
        )
        assert result.g.shape == (4,)
        assert search.warrant is Warrant.PROVED
