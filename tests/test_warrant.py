"""The shared `Warrant` vocabulary and the check report that carries it.

`SearchWarrant` had two levels: `PROVED` (3b, exhaustive enumeration) and `CORROBORATED`
(3a, a grid sample). `Warrant` adds `CERTIFIED` (3c, validated numerics over a compact
domain) and moves to `cpomdp.warrant`, where checks can reach it too. `SearchWarrant`
stays as an alias, so existing call sites keep their members and their return type.

`CheckReport` is what a registered falsifier emits: a warrant, an `Outcome`, a `Tier`,
and a reason. A falsifier does not pass, so `PASS` is not in the vocabulary at all. The
prover column disambiguates warrant, never the outcome. A check that never ran carries
no warrant, because attributing one claims evidence it did not produce.

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
from cpomdp.warrant import CheckReport, Outcome, Tier, Warrant, check_summary


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
    def test_a_falsifier_never_passes(self):
        # Standing rule 6 is satisfied by not printing PASS, not by printing PASS
        # beside a column that disambiguates it. The word must not be reachable.
        values = {outcome.value for outcome in Outcome}
        assert "PASS" not in values
        assert not any(name == "PASS" for name in Outcome.__members__)

    def test_the_five_outcomes(self):
        assert Outcome.NOT_TRIGGERED.value == "NOT TRIGGERED"
        assert Outcome.FIRED.value == "FIRED"
        assert Outcome.NOT_RESOLVED.value == "NOT RESOLVED"
        assert Outcome.NOT_APPLICABLE.value == "NOT APPLICABLE"
        assert Outcome.NOT_RUN_HERE.value == "NOT RUN HERE"

    def test_no_sixth_outcome(self):
        assert [o.name for o in Outcome] == [
            "NOT_TRIGGERED",
            "FIRED",
            "NOT_RESOLVED",
            "NOT_APPLICABLE",
            "NOT_RUN_HERE",
        ]

    def test_the_three_that_did_not_run_stay_distinct(self):
        # Void by construction, measured elsewhere, and a genuine tie are three
        # different things. Collapsing them loses the survivor accounting.
        assert (
            len({Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE, Outcome.NOT_RESOLVED})
            == 3
        )

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
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.C,
        evidence=(),
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
        assert report.outcome is Outcome.NOT_TRIGGERED
        assert report.tier is Tier.C

    def test_detail_is_required(self):
        # A check that cannot say why it reports what it reports is a bare outcome
        # with extra fields. Omitting the reason must not construct.
        with pytest.raises(TypeError):
            CheckReport(  # ty: ignore[missing-argument]
                name="flip-decided",
                warrant=Warrant.CORROBORATED,
                outcome=Outcome.NOT_TRIGGERED,
                tier=Tier.C,
            )

    def test_is_frozen(self):
        # A report is a record of what a check found. Editing one after the fact is
        # editing the finding.
        with pytest.raises(dataclasses.FrozenInstanceError):
            self._report().outcome = Outcome.FIRED

    def test_evidence_is_absent_by_default(self):
        assert self._report().evidence == ()

    def test_carries_the_evidence_it_was_given(self):
        cert = _certificate()
        report = self._report(warrant=Warrant.PROVED, tier=Tier.A, evidence=(cert,))
        assert report.evidence == (cert,)

    def test_warrant_and_outcome_are_independent(self):
        # The pairing this vocabulary exists for: a run that survived every falsifier
        # and decided nothing.
        report = self._report()
        assert report.outcome is Outcome.NOT_TRIGGERED
        assert report.warrant is Warrant.CORROBORATED

    def test_renders_one_line_naming_all_of_it(self):
        line = str(self._report())
        assert line.count("\n") == 0
        for part in ("flip-decided", "NOT TRIGGERED", "CORROBORATED", "C", "cue-ward"):
            assert part in line

    def test_a_check_with_no_warrant_renders_a_dash(self):
        line = str(self._report(warrant=None, outcome=Outcome.NOT_APPLICABLE))
        assert "—" in line


class TestChecksThatNeverRanCarryNoWarrant:
    """A check that produced no evidence has no prover verdict to report."""

    def _report(self, *, warrant, outcome):
        return CheckReport(
            name="void-check",
            warrant=warrant,
            outcome=outcome,
            tier=Tier.C,
            detail="why",
        )

    @pytest.mark.parametrize("outcome", [Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE])
    def test_a_check_that_never_ran_may_not_claim_a_prover(self, outcome):
        # CORROBORATED means sampling-grade evidence was obtained. A falsifier void by
        # construction sampled nothing, and one measured elsewhere sampled nothing
        # here, so either would be attributing a warrant to a check that produced none.
        with pytest.raises(ValueError, match="no evidence here"):
            self._report(warrant=Warrant.CORROBORATED, outcome=outcome)

    @pytest.mark.parametrize("outcome", [Outcome.NOT_APPLICABLE, Outcome.NOT_RUN_HERE])
    def test_none_is_the_representable_state(self, outcome):
        assert self._report(warrant=None, outcome=outcome).warrant is None

    @pytest.mark.parametrize(
        "outcome",
        [Outcome.NOT_TRIGGERED, Outcome.FIRED, Outcome.NOT_RESOLVED],
    )
    def test_a_check_that_ran_may_carry_one(self, outcome):
        # A genuine tie ran and produced evidence. It stays warrantable.
        report = self._report(warrant=Warrant.CORROBORATED, outcome=outcome)
        assert report.warrant is Warrant.CORROBORATED


class TestProvedNeedsEvidence:
    """`PROVED` with nothing behind it does not construct."""

    def _report(self, *, warrant, evidence=(), outcome=Outcome.NOT_TRIGGERED):
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
        report = self._report(warrant=Warrant.PROVED, evidence=(cert,))
        assert report.evidence == (cert,)

    def test_a_failing_proved_check_still_needs_evidence(self):
        # The outcome does not exempt it. A refutation carrying PROVED claims the
        # refutation was decided, which needs the same backing as a decided pass.
        with pytest.raises(ValueError, match="PROVED"):
            self._report(warrant=Warrant.PROVED, outcome=Outcome.FIRED)

    def test_evidence_must_be_a_tuple(self):
        # A claim over several enumerations rests on all their certificates. A bare one
        # would iterate as a sequence of fields somewhere downstream, or read as a
        # complete backing for a claim it only half covers.
        with pytest.raises(ValueError, match="tuple"):
            self._report(warrant=Warrant.PROVED, evidence=_certificate())

    def test_carries_a_certificate_per_enumeration(self):
        # A check quantified over two horizons carries both.
        pair = (_certificate(), _certificate())
        assert self._report(warrant=Warrant.PROVED, evidence=pair).evidence == pair

    @pytest.mark.parametrize("warrant", [Warrant.CERTIFIED, Warrant.CORROBORATED])
    def test_the_weaker_levels_need_none(self, warrant):
        # A bound and a sample carry their own story in `detail`. Only a claim to have
        # decided a universal needs something enumerable behind it.
        assert self._report(warrant=warrant).evidence == ()


class TestCheckSummary:
    """Counts per (warrant × outcome), so a green run says what it decided."""

    def _report(self, name, warrant, outcome, evidence=()):
        return CheckReport(
            name=name,
            warrant=warrant,
            outcome=outcome,
            tier=Tier.C,
            detail="why",
            evidence=evidence,
        )

    def test_counts_each_pair(self):
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("c", Warrant.CERTIFIED, Outcome.FIRED),
            ]
        )
        assert "CORROBORATED" in summary
        assert "CERTIFIED" in summary
        assert "2" in summary

    def test_a_corroborative_surviving_run_reads_as_one(self):
        # The reason this function exists. Three falsifiers, none fired, none decisive.
        summary = check_summary(
            [
                self._report(str(i), Warrant.CORROBORATED, Outcome.NOT_TRIGGERED)
                for i in range(3)
            ]
        )
        assert "CORROBORATED" in summary
        assert "PROVED" not in summary

    def test_pairs_with_no_checks_are_omitted(self):
        summary = check_summary(
            [self._report("a", Warrant.CERTIFIED, Outcome.NOT_TRIGGERED)]
        )
        assert "FIRED" not in summary
        assert "NOT RESOLVED" not in summary

    def test_a_firing_falsifier_is_visible(self):
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.FIRED),
            ]
        )
        assert "FIRED" in summary
        assert "1 fired" in summary

    def test_the_header_separates_registered_from_tested(self):
        # The accounting ADR-029 required and a single count cannot carry: four
        # registered, two of which this run actually tested.
        summary = check_summary(
            [
                self._report(
                    "a", Warrant.PROVED, Outcome.NOT_TRIGGERED, (_certificate(),)
                ),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_RESOLVED),
                self._report("c", None, Outcome.NOT_APPLICABLE),
                self._report("d", None, Outcome.NOT_RUN_HERE),
            ]
        )
        assert "4 registered, 2 tested here, none fired" in summary.splitlines()[0]

    def test_checks_with_no_warrant_sort_under_a_dash(self):
        summary = check_summary(
            [
                self._report("a", None, Outcome.NOT_APPLICABLE),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
            ]
        )
        rows = summary.splitlines()[1:]
        assert rows[0].split()[0] == "CORROBORATED"
        assert rows[1].split()[0] == "—"

    def test_a_void_falsifier_is_not_counted_as_tested(self):
        # ADR-029's gloss: evidence for nothing, and not a survivor.
        summary = check_summary([self._report("a", None, Outcome.NOT_APPLICABLE)])
        assert "1 registered, 0 tested here" in summary

    def test_unresolved_is_not_folded_into_survival(self):
        # The ADR-029 rule that survives the vocabulary change: a check that decided
        # neither way is not counted among the ones that did.
        summary = check_summary(
            [
                self._report("a", Warrant.CORROBORATED, Outcome.NOT_TRIGGERED),
                self._report("b", Warrant.CORROBORATED, Outcome.NOT_RESOLVED),
            ]
        )
        assert "NOT RESOLVED" in summary
        assert "NOT TRIGGERED" in summary

    def test_counts_are_stable_under_input_order(self):
        reports = [
            self._report("a", Warrant.CERTIFIED, Outcome.NOT_TRIGGERED),
            self._report("b", Warrant.CORROBORATED, Outcome.FIRED),
        ]
        assert check_summary(reports) == check_summary(list(reversed(reports)))

    def test_no_checks_is_not_a_blank_line(self):
        # An empty suite is a finding, not a silence.
        assert check_summary([]).strip() != ""


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
