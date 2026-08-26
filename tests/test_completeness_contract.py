"""The contract every completeness leaf holds, whatever domain it declares.

Two predicates decide a `PROVED` completeness claim. **Domain**: the count the claim was
obliged to visit is the declared set's own cardinality. **Coverage**: it visited all of
them. The base owns both, so a leaf supplies only what its domain means and cannot
weaken the gate by forgetting to call it.

These run over every leaf the serialiser registers, so a leaf added later is held to the
same contract without anyone remembering to add it here. `_LEAVES` below supplies the
fields each leaf needs; a leaf registered without an entry fails loudly rather than
silently going unchecked.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import warrantlib
from warrantlib import (
    AxisDeclaration,
    CheckReport,
    CompletenessCertificate,
    CompletenessEvidence,
    Outcome,
    ProductCompletenessCertificate,
    Provenance,
    Tier,
    Warrant,
    report_from_dict,
    report_to_dict,
)
from warrantlib._serialise import _EVIDENCE_CODECS

FIXTURES = Path(__file__).parent / "fixtures" / "warrantlib"

#: The schema warrantlib ships, which is the consumer contract a record is judged by.
REPORT_SCHEMA = json.loads(
    (Path(warrantlib.__file__).parent / "report.schema.json").read_text()
)


def _report_carrying(evidence):
    """The narrowest `PROVED` report that can carry one piece of evidence."""
    return CheckReport(
        name="a decided universal",
        check_id="contract.decided",
        warrant=Warrant.PROVED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail="the declared domain was enumerated in full",
        evidence=(evidence,),
        provenance=(
            Provenance(
                registered_at="a76cf1b",
                measured_at="9baaa22",
                registered="the bar this was measured against",
            ),
        ),
    )


@dataclass(frozen=True)
class _Leaf:
    """One completeness leaf, and enough of it to build a valid instance."""

    cls: type[CompletenessEvidence]
    extra: dict[str, Any]
    cardinality: int
    versions: tuple[str, ...]


_LEAVES = {
    CompletenessCertificate: _Leaf(
        cls=CompletenessCertificate,
        extra={
            "action_set_size": 3,
            "horizon": 4,
            "action_set_version": "acts-v1",
        },
        cardinality=81,
        versions=("acts-v1",),
    ),
    ProductCompletenessCertificate: _Leaf(
        cls=ProductCompletenessCertificate,
        extra={
            "axes": (
                AxisDeclaration(name="model", size=3, version="models-v1"),
                AxisDeclaration(name="inference", size=4, version="rules-v1"),
            )
        },
        cardinality=12,
        versions=("models-v1", "rules-v1"),
    ),
}


def _registered_leaves() -> tuple[type[CompletenessEvidence], ...]:
    """Every completeness leaf the serialiser knows how to write."""
    return tuple(
        codec.cls
        for codec in _EVIDENCE_CODECS.values()
        if issubclass(codec.cls, CompletenessEvidence)
    )


def _built(leaf: _Leaf, *, warrant=Warrant.PROVED, expected=None, visited=None):
    """One instance of `leaf`, defaulting to a complete, correctly declared one."""
    cardinality = leaf.cardinality if expected is None else expected
    return leaf.cls(
        expected=cardinality,
        visited=cardinality if visited is None else visited,
        warrant=warrant,
        **leaf.extra,
    )


LEAF_CASES = [pytest.param(cls, id=cls.__name__) for cls in _registered_leaves()]


@pytest.mark.parametrize("cls", LEAF_CASES)
class TestEveryLeafHoldsTheContract:
    def test_a_valid_certificate_constructs_and_is_complete(self, cls):
        certificate = _built(_LEAVES[cls])
        assert certificate.domain_declared
        assert certificate.complete

    def test_proved_over_a_count_the_domain_does_not_give_raises(self, cls):
        leaf = _LEAVES[cls]
        with pytest.raises(ValueError, match="declared"):
            _built(leaf, expected=leaf.cardinality + 1)

    def test_proved_with_a_shortfall_raises(self, cls):
        leaf = _LEAVES[cls]
        with pytest.raises(ValueError, match=r"complete|sampled"):
            _built(leaf, visited=leaf.cardinality - 1)

    def test_corroborated_constructs_with_both_predicates_false(self, cls):
        # A partial enumeration over a count the domain does not give is a sample, and
        # a sample is exactly what CORROBORATED is for. The gate is the PROVED gate.
        leaf = _LEAVES[cls]
        certificate = _built(
            leaf,
            warrant=Warrant.CORROBORATED,
            expected=leaf.cardinality + 1,
            visited=1,
        )
        assert not certificate.domain_declared
        assert not certificate.complete

    def test_the_leaf_is_a_frozen_dataclass(self, cls):
        assert is_dataclass(cls)
        certificate = _built(_LEAVES[cls])
        with pytest.raises(FrozenInstanceError):
            certificate.visited = 0

    def test_the_string_names_every_declared_version(self, cls):
        leaf = _LEAVES[cls]
        rendered = str(_built(leaf))
        for version in leaf.versions:
            assert version in rendered, rendered

    def test_the_record_round_trips_to_an_equal_object(self, cls):
        certificate = _built(_LEAVES[cls])
        tag = next(tag for tag, codec in _EVIDENCE_CODECS.items() if codec.cls is cls)
        record = _EVIDENCE_CODECS[tag].to_record(certificate)
        assert record["kind"] == tag
        assert _EVIDENCE_CODECS[tag].from_record(record) == certificate


@pytest.mark.parametrize("cls", LEAF_CASES)
class TestEveryLeafCrossesTheWire:
    """Through the public functions and against the shipped schema, not the codecs.

    Exercising `to_record` and `from_record` directly leaves `_evidence_to_dict`, its
    exact-class dispatch, and the schema branch for the kind all unrun. A wrong `const`,
    a missing `required` entry or a dispatch that never matches would ship green.
    """

    def test_the_record_validates_against_the_shipped_schema(self, cls):
        record = report_to_dict(_report_carrying(_built(_LEAVES[cls])))
        jsonschema.validate(record, REPORT_SCHEMA)

    def test_the_report_round_trips_through_the_public_functions(self, cls):
        report = _report_carrying(_built(_LEAVES[cls]))
        assert report_from_dict(report_to_dict(report)) == report

    def test_the_record_carries_the_kind_the_registry_names(self, cls):
        record = report_to_dict(_report_carrying(_built(_LEAVES[cls])))
        tag = next(tag for tag, codec in _EVIDENCE_CODECS.items() if codec.cls is cls)
        assert record["evidence"][0]["kind"] == tag


@pytest.mark.parametrize("cls", LEAF_CASES)
def test_the_render_shows_the_declared_count_not_a_recomputed_one(cls):
    # Only a certificate whose declaration disagrees with its domain can see this, and
    # a PROVED one cannot disagree. The render is what a reader is shown, so a leaf that
    # quietly substitutes its own arithmetic reports a count nobody declared.
    leaf = _LEAVES[cls]
    disagreeing = _built(
        leaf,
        warrant=Warrant.CORROBORATED,
        expected=leaf.cardinality + 1,
        visited=1,
    )
    assert str(leaf.cardinality + 1) in str(disagreeing), str(disagreeing)


def test_the_base_refuses_a_leaf_that_validates_itself():
    # The gate is held once. A leaf with its own `__post_init__` shadows the base's and
    # the PROVED preconditions stop running, which no test of that leaf would notice.
    with pytest.raises(TypeError, match="__post_init__"):

        @dataclass(frozen=True)
        class _SelfValidating(CompletenessEvidence):
            def __post_init__(self) -> None:
                pass

            @property
            def domain_declared(self) -> bool:
                return True

            @property
            def set_description(self) -> str:
                return "a set"


def test_every_leaf_in_the_vocabulary_is_serialisable():
    # A leaf that exists and cannot be written is a claim the ledger silently loses.
    #
    # warrantlib's own leaves only. A third party may define one this serialiser never
    # sees, which is why `__init_subclass__` couples to no registry. A class rejected at
    # creation also stays in `__subclasses__` until it is collected, since the type
    # exists before `__init_subclass__` runs.
    defined_here = {
        leaf
        for leaf in CompletenessEvidence.__subclasses__()
        if leaf.__module__.startswith("warrantlib")
    }
    assert defined_here == set(_registered_leaves())


def test_this_module_covers_every_registered_leaf():
    assert set(_LEAVES) == set(_registered_leaves())


@pytest.mark.parametrize(
    "fixture", ["tree_certificate_report.json", "symbolic_reduction_report.json"]
)
def test_a_record_written_before_the_split_still_reads(fixture):
    # Generated from a844c9d, before any of this existed, and never regenerated. A
    # regenerated golden pins nothing.
    written = json.loads((FIXTURES / fixture).read_text())
    report = report_from_dict(written)
    assert report_to_dict(report) == written
