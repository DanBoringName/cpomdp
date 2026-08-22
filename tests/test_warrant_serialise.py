"""A report survives the round trip to a machine record and back.

The vocabulary is a set of frozen records with preconditions, and nothing until now
wrote one anywhere but a terminal. A ledger comparing two runs needs the record on
disk, and needs the form it takes to be fixed: a field renamed between versions turns
every row into one dropped and one added, which is the reading the comparison exists
to rule out.

So the wire form is pinned here rather than left to `dataclasses.asdict`. The golden
record below is a literal, never regenerated, and it is what a reader of a later
version compares against.

The shipped JSON Schema is checked against real records here. A schema nothing
validates against drifts from the writer silently, and it is the one artefact a
consumer outside Python has to go on.
"""

import json
from pathlib import Path

import jsonschema
import pytest

import warrantlib
from warrantlib import (
    SCHEMA_VERSION,
    CheckReport,
    CompletenessCertificate,
    Outcome,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
    report_from_dict,
    report_to_dict,
)

#: A `PROVED` report carrying one of each evidence kind and one provenance, which is
#: the widest record the vocabulary can construct.
GOLDEN_RECORD = {
    "schema_version": "1.0",
    "check_id": "gap_series.c2_closed_form",
    "name": "C4 c₂ closed form",
    "warrant": "PROVED",
    "outcome": "NOT TRIGGERED",
    "tier": "exact",
    "detail": "the CAS reduces the residual to zero",
    "evidence": [
        {
            "kind": "completeness_certificate",
            "expected": 81,
            "visited": 81,
            "warrant": "PROVED",
            "action_set_size": 3,
            "horizon": 4,
            "action_set_version": "crossover-v1",
        },
        {
            "kind": "symbolic_reduction",
            "claim": "c₂ equals the quoted constant",
            "correspondence": "hand derivation, section 3",
            "assumptions": ["the expansion is formal, not convergent"],
        },
    ],
    "provenance": [
        {
            "registered_at": "a76cf1b",
            "measured_at": "9baaa22",
            "registered": "the coefficient's closed form",
        }
    ],
}


def _golden_report():
    return CheckReport(
        name="C4 c₂ closed form",
        check_id="gap_series.c2_closed_form",
        warrant=Warrant.PROVED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail="the CAS reduces the residual to zero",
        evidence=(
            CompletenessCertificate(
                expected=81,
                visited=81,
                warrant=Warrant.PROVED,
                action_set_size=3,
                horizon=4,
                action_set_version="crossover-v1",
            ),
            SymbolicReduction(
                claim="c₂ equals the quoted constant",
                correspondence="hand derivation, section 3",
                assumptions=("the expansion is formal, not convergent",),
            ),
        ),
        provenance=(
            Provenance(
                registered_at="a76cf1b",
                measured_at="9baaa22",
                registered="the coefficient's closed form",
            ),
        ),
    )


def _bare_report(
    *,
    warrant: Warrant | None = Warrant.CORROBORATED,
    outcome: Outcome = Outcome.NOT_TRIGGERED,
    tier: Tier = Tier.COMPUTED,
    evidence: tuple[CompletenessCertificate | SymbolicReduction, ...] = (),
    provenance: tuple[Provenance, ...] = (),
) -> CheckReport:
    return CheckReport(
        name="flip-decided",
        check_id="crossover.flip_not_clean_at_h_star",
        warrant=warrant,
        outcome=outcome,
        tier=tier,
        detail="sampled, no bar",
        evidence=evidence,
        provenance=provenance,
    )


class TestTheWireForm:
    def test_the_widest_record_matches_the_golden_literal(self):
        # Regenerating this literal from the code would make it agree by construction
        # and pin nothing. It is written by hand and stays written by hand.
        assert report_to_dict(_golden_report()) == GOLDEN_RECORD

    def test_the_key_order_is_the_golden_order(self):
        # Comparing mappings ignores order, so the fixed order the writer documents
        # would go unpinned and reordering the return literal would stay green. The
        # promise is that two runs of one suite produce the same bytes, so bytes are
        # what this compares.
        assert json.dumps(report_to_dict(_golden_report())) == json.dumps(GOLDEN_RECORD)

    def test_the_record_is_json(self):
        # A tuple or an enum survives equality against a literal and then fails at the
        # point the record is written, which is after the run that produced it.
        assert json.loads(json.dumps(report_to_dict(_golden_report()))) == GOLDEN_RECORD

    def test_the_golden_literal_still_reads_back(self):
        assert report_from_dict(GOLDEN_RECORD) == _golden_report()

    def test_the_version_travels_with_the_record(self):
        assert report_to_dict(_bare_report())["schema_version"] == SCHEMA_VERSION

    def test_enums_serialise_as_their_values(self):
        record = report_to_dict(_bare_report())
        assert record["warrant"] == "CORROBORATED"
        assert record["outcome"] == "NOT TRIGGERED"
        assert record["tier"] == "computed"


class TestRoundTrip:
    @pytest.mark.parametrize("outcome", list(Outcome))
    def test_every_outcome_round_trips(self, outcome):
        warrant = (
            Warrant.CORROBORATED
            if outcome in {Outcome.NOT_TRIGGERED, Outcome.FIRED, Outcome.NOT_RESOLVED}
            else None
        )
        report = _bare_report(outcome=outcome, warrant=warrant)
        assert report_from_dict(report_to_dict(report)) == report

    @pytest.mark.parametrize("tier", list(Tier))
    def test_every_tier_round_trips(self, tier):
        report = _bare_report(tier=tier)
        assert report_from_dict(report_to_dict(report)) == report

    def test_a_check_with_no_warrant_round_trips(self):
        report = _bare_report(warrant=None, outcome=Outcome.NOT_APPLICABLE)
        assert report_to_dict(report)["warrant"] is None
        assert report_from_dict(report_to_dict(report)) == report

    def test_the_widest_record_round_trips(self):
        report = _golden_report()
        assert report_from_dict(report_to_dict(report)) == report

    def test_an_unscoped_reduction_round_trips(self):
        report = _bare_report(
            warrant=Warrant.PROVED,
            evidence=(
                SymbolicReduction(claim="a claim", correspondence="a derivation"),
            ),
            provenance=(
                Provenance(
                    registered_at="a76cf1b",
                    measured_at="9baaa22",
                    registered="what a reviewer finds",
                ),
            ),
        )
        assert report_from_dict(report_to_dict(report)) == report


class TestReadingRefuses:
    """A record it cannot read is refused, rather than read as something else."""

    def test_an_unknown_schema_version_is_refused(self):
        # Refusing beats guessing: a differ that mis-reads an older record reports
        # changes nobody made, which is worse than reporting that it cannot compare.
        record = {**GOLDEN_RECORD, "schema_version": "2.0"}
        with pytest.raises(ValueError, match="schema_version"):
            report_from_dict(record)

    def test_a_record_with_no_version_is_refused(self):
        record = {
            key: value
            for key, value in GOLDEN_RECORD.items()
            if key != "schema_version"
        }
        with pytest.raises(ValueError, match="schema_version"):
            report_from_dict(record)

    def test_an_unknown_evidence_kind_is_refused(self):
        record = {**GOLDEN_RECORD, "evidence": [{"kind": "a_write_up", "path": "x.md"}]}
        with pytest.raises(ValueError, match="kind"):
            report_from_dict(record)

    def test_evidence_with_no_kind_is_refused(self):
        record = {**GOLDEN_RECORD, "evidence": [{"expected": 81, "visited": 81}]}
        with pytest.raises(ValueError, match="kind"):
            report_from_dict(record)

    def test_an_unknown_outcome_is_refused(self):
        record = {**GOLDEN_RECORD, "outcome": "PASS"}
        with pytest.raises(ValueError, match="outcome"):
            report_from_dict(record)

    def test_an_unknown_warrant_is_refused(self):
        record = {**GOLDEN_RECORD, "warrant": "PROBABLY"}
        with pytest.raises(ValueError, match="warrant"):
            report_from_dict(record)

    def test_an_unknown_tier_is_refused(self):
        record = {**GOLDEN_RECORD, "tier": "A"}
        with pytest.raises(ValueError, match="tier"):
            report_from_dict(record)

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "check_id",
            "name",
            "warrant",
            "outcome",
            "tier",
            "detail",
            "evidence",
            "provenance",
        ],
    )
    def test_every_field_the_writer_emits_is_required_on_the_way_in(self, field):
        # `.get` on any of these reads a truncated record as a real one. A record with
        # no `warrant` key would come back `None`, and a differ would report the
        # `PROVED` it used to carry as a status change nobody made.
        record = {key: value for key, value in GOLDEN_RECORD.items() if key != field}
        with pytest.raises(ValueError, match=field):
            report_from_dict(record)

    @pytest.mark.parametrize("field", ["evidence", "provenance"])
    def test_a_list_field_that_is_not_a_list_is_refused(self, field):
        with pytest.raises(ValueError, match=field):
            report_from_dict({**GOLDEN_RECORD, field: "one item"})

    def test_a_bare_string_of_assumptions_is_refused(self):
        # `tuple("formal")` is six one-character assumptions, and each one passes the
        # blank check. `SymbolicReduction` refuses a bare string for exactly this
        # reason, and it reads a tuple by the time it looks.
        record = {
            **GOLDEN_RECORD,
            "evidence": [
                {
                    "kind": "symbolic_reduction",
                    "claim": "a claim",
                    "correspondence": "a derivation",
                    "assumptions": "the expansion is formal",
                }
            ],
        }
        with pytest.raises(ValueError, match="assumptions"):
            report_from_dict(record)

    def test_a_reduction_with_no_assumptions_field_is_refused(self):
        record = {
            **GOLDEN_RECORD,
            "evidence": [
                {
                    "kind": "symbolic_reduction",
                    "claim": "a claim",
                    "correspondence": "a derivation",
                }
            ],
        }
        with pytest.raises(ValueError, match="assumptions"):
            report_from_dict(record)

    def test_a_certificate_names_which_warrant_is_wrong(self):
        record = {
            **GOLDEN_RECORD,
            "evidence": [
                {
                    "kind": "completeness_certificate",
                    "expected": 81,
                    "visited": 81,
                    "warrant": "PROBABLY",
                    "action_set_size": 3,
                    "horizon": 4,
                    "action_set_version": "crossover-v1",
                }
            ],
        }
        with pytest.raises(ValueError, match="evidence warrant"):
            report_from_dict(record)

    def test_a_missing_field_is_refused(self):
        record = {key: value for key, value in GOLDEN_RECORD.items() if key != "detail"}
        with pytest.raises(ValueError, match="detail"):
            report_from_dict(record)

    def test_the_report_preconditions_still_apply_on_the_way_in(self):
        # A record naming PROVED with its evidence stripped must not construct, or the
        # wire form is a way round the precondition the type exists to enforce.
        record = {**GOLDEN_RECORD, "evidence": []}
        with pytest.raises(ValueError, match="evidence"):
            report_from_dict(record)


REPORT_SCHEMA = json.loads(
    (Path(warrantlib.__file__).parent / "report.schema.json").read_text()
)


class TestTheShippedSchema:
    """The schema ships in the wheel, so it is checked against what the writer emits."""

    def test_it_is_a_valid_schema(self):
        jsonschema.Draft202012Validator.check_schema(REPORT_SCHEMA)

    def test_the_golden_record_validates(self):
        jsonschema.validate(GOLDEN_RECORD, REPORT_SCHEMA)

    @pytest.mark.parametrize("outcome", list(Outcome))
    def test_every_outcome_the_writer_emits_validates(self, outcome):
        tested_here = outcome in (
            Outcome.NOT_TRIGGERED,
            Outcome.FIRED,
            Outcome.NOT_RESOLVED,
        )
        report = _bare_report(
            outcome=outcome, warrant=Warrant.CORROBORATED if tested_here else None
        )
        jsonschema.validate(report_to_dict(report), REPORT_SCHEMA)

    @pytest.mark.parametrize("tier", list(Tier))
    def test_every_tier_the_writer_emits_validates(self, tier):
        jsonschema.validate(report_to_dict(_bare_report(tier=tier)), REPORT_SCHEMA)

    def test_the_schema_pins_the_version_it_describes(self):
        assert REPORT_SCHEMA["properties"]["schema_version"]["const"] == SCHEMA_VERSION

    def test_a_prose_name_as_a_key_is_refused(self):
        record = {**GOLDEN_RECORD, "check_id": "K5 first cumulant is the mean"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REPORT_SCHEMA)

    def test_a_proved_record_with_no_evidence_is_refused(self):
        # The constructor refuses this too. Stated in both places on purpose: a
        # consumer reading the ledger without the vocabulary has only the schema.
        record = {**GOLDEN_RECORD, "evidence": []}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REPORT_SCHEMA)

    def test_a_void_record_carrying_a_warrant_is_refused(self):
        record = {**GOLDEN_RECORD, "outcome": "NOT APPLICABLE"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REPORT_SCHEMA)

    def test_an_unknown_evidence_kind_is_refused(self):
        record = {**GOLDEN_RECORD, "evidence": [{"kind": "a_write_up"}]}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(record, REPORT_SCHEMA)
