"""A report as a machine record, and back.

A ledger comparing two runs answers whether a registered check changed status between
them. That only works if the form the record takes is fixed. A renamed field turns
every row into one dropped and one added, which is the reading a comparison exists to
rule out, so the wire form is written out here rather than left to ``asdict``.

Reading refuses what it cannot read. A record from a later schema, an evidence kind
this version has no class for, an enum value that has since been renamed: each is an
error rather than a best guess, because a differ that mis-reads an old record reports
changes nobody made.

The standard library is the only dependency, and nothing here touches a filesystem.
The caller decides where the bytes go.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from warrantlib._vocabulary import (
    AxisDeclaration,
    CheckReport,
    CompletenessCertificate,
    Evidence,
    Outcome,
    ProductCompletenessCertificate,
    Provenance,
    SymbolicReduction,
    Tier,
    Warrant,
)

_EnumT = TypeVar("_EnumT", bound=Enum)

SCHEMA_VERSION = "1.0"
"""The version of the serialised form, carried on every record.

Bumped whenever a field is added, removed or renamed, or an enum value changes
spelling. Enum *members* are additive: a new one is readable by an older version only
in the sense that the older version refuses it by name, which is the intended
behaviour. A record whose version this module does not know is refused.

An evidence *kind* is additive under one version and does not move it. The ``kind``
field self-describes, and a reader meeting an unknown one refuses it by name, which is
the designed failure rather than a gap. The version moves when the record envelope
changes. ``_from_dict`` compares it exactly, so an undocumented bump orphans every
record already in a ledger.
"""

#: The discriminator each evidence kind serialises under. `Evidence` is a union, and a
#: record without a tag is a set of dataclasses with disjoint fields that a reader has
#: to guess between. Guessing is what this module refuses to do.
_CERTIFICATE_KIND = "completeness_certificate"
_PRODUCT_CERTIFICATE_KIND = "product_completeness_certificate"
_REDUCTION_KIND = "symbolic_reduction"


@dataclass(frozen=True)
class _Codec:
    """How one evidence class crosses the wire, both ways.

    Args:
        cls: the class the tag names.
        to_record: the class to a record, tag included.
        from_record: a record of that tag back to the class.
    """

    cls: type
    to_record: Callable[[Any], dict[str, Any]]
    from_record: Callable[[Mapping[str, Any]], Any]


def _certificate_to_record(item: CompletenessCertificate) -> dict[str, Any]:
    """A tree certificate as a record."""
    return {
        "kind": _CERTIFICATE_KIND,
        "expected": item.expected,
        "visited": item.visited,
        "warrant": item.warrant.value,
        "action_set_size": item.action_set_size,
        "horizon": item.horizon,
        "action_set_version": item.action_set_version,
    }


def _certificate_from_record(data: Mapping[str, Any]) -> CompletenessCertificate:
    """A tree certificate from its record."""
    return CompletenessCertificate(
        expected=_require(data, "expected"),
        visited=_require(data, "visited"),
        warrant=_member(Warrant, _require(data, "warrant"), "evidence warrant"),
        action_set_size=_require(data, "action_set_size"),
        horizon=_require(data, "horizon"),
        action_set_version=_require(data, "action_set_version"),
    )


def _product_to_record(item: ProductCompletenessCertificate) -> dict[str, Any]:
    """A product certificate as a record, every axis carried."""
    return {
        "kind": _PRODUCT_CERTIFICATE_KIND,
        "expected": item.expected,
        "visited": item.visited,
        "warrant": item.warrant.value,
        "axes": [
            {"name": axis.name, "size": axis.size, "version": axis.version}
            for axis in item.axes
        ],
    }


def _product_from_record(
    data: Mapping[str, Any],
) -> ProductCompletenessCertificate:
    """A product certificate from its record."""
    return ProductCompletenessCertificate(
        expected=_require(data, "expected"),
        visited=_require(data, "visited"),
        warrant=_member(Warrant, _require(data, "warrant"), "evidence warrant"),
        axes=tuple(
            AxisDeclaration(
                name=_require(axis, "name"),
                size=_require(axis, "size"),
                version=_require(axis, "version"),
            )
            for axis in _sequence(data, "axes")
        ),
    )


def _reduction_to_record(item: SymbolicReduction) -> dict[str, Any]:
    """A symbolic reduction as a record."""
    return {
        "kind": _REDUCTION_KIND,
        "claim": item.claim,
        "correspondence": item.correspondence,
        "assumptions": list(item.assumptions),
    }


def _reduction_from_record(data: Mapping[str, Any]) -> SymbolicReduction:
    """A symbolic reduction from its record."""
    return SymbolicReduction(
        claim=_require(data, "claim"),
        correspondence=_require(data, "correspondence"),
        assumptions=_sequence(data, "assumptions"),
    )


#: Every evidence kind this module can write and read. A future kind is one entry, and
#: the refusal below names what it knows from these keys rather than from a hand-written
#: list that can fall behind them.
_EVIDENCE_CODECS: dict[str, _Codec] = {
    _CERTIFICATE_KIND: _Codec(
        CompletenessCertificate, _certificate_to_record, _certificate_from_record
    ),
    _PRODUCT_CERTIFICATE_KIND: _Codec(
        ProductCompletenessCertificate, _product_to_record, _product_from_record
    ),
    _REDUCTION_KIND: _Codec(
        SymbolicReduction, _reduction_to_record, _reduction_from_record
    ),
}


def _require(data: Mapping[str, Any], field: str) -> Any:
    """Read a field, or say which one the record is missing.

    Args:
        data: the record.
        field: the field to read.

    Returns:
        Its value.

    Raises:
        ValueError: if the record has no such field.
    """
    if field not in data:
        raise ValueError(
            f"record has no {field}. Every field the vocabulary declares is written, "
            f"so one missing is a record from a form this version cannot read rather "
            f"than a value left out."
        )
    return data[field]


def _sequence(data: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    """Read a field the writer emits as a list, without coercing what is not one.

    `tuple` accepts any iterable, and a bare string is one. Coercing turns a sentence
    into one entry per character, and the constructor's own guard against that reads a
    tuple by the time it looks, so the record would construct.

    Args:
        data: the record.
        field: the field to read.

    Returns:
        Its entries.

    Raises:
        ValueError: if the record has no such field, or the field is not a list.
    """
    value = _require(data, field)
    if not isinstance(value, list):
        raise ValueError(
            f"record has {field}={value!r}, which is not a list. The writer emits one, "
            f"and converting whatever arrives instead is how a bare string becomes one "
            f"entry per character with every entry passing its own checks."
        )
    return tuple(value)


def _member(enum: type[_EnumT], value: Any, field: str) -> _EnumT:
    """Resolve an enum member from its serialised value.

    Args:
        enum: the enum to resolve against.
        value: the serialised value.
        field: the field it came from, as the message names it.

    Returns:
        The member.

    Raises:
        ValueError: if no member carries that value.
    """
    for member in enum:
        if member.value == value:
            return member
    known = ", ".join(repr(member.value) for member in enum)
    raise ValueError(
        f"record has {field}={value!r}, which {enum.__name__} does not carry. Known "
        f"values are {known}. A value this version has no member for is refused rather "
        f"than mapped to a neighbour, since the neighbour would read as a real result."
    )


def _evidence_to_dict(item: Evidence) -> dict[str, Any]:
    """One evidence item as a record, tagged with its kind.

    Args:
        item: the evidence.

    Returns:
        The record.

    Raises:
        TypeError: if nothing in the registry writes that class. A leaf the vocabulary
            defines and this module cannot write is a claim a ledger silently loses.
    """
    for codec in _EVIDENCE_CODECS.values():
        if type(item) is codec.cls:
            return codec.to_record(item)
    raise TypeError(
        f"no evidence kind writes {type(item).__name__}. Exact class rather than "
        "isinstance: a subclass written under its parent's tag loses its own fields "
        "and reads back as the parent, which is a different claim."
    )


def _evidence_from_dict(data: Mapping[str, Any]) -> Evidence:
    """One evidence record as the class its kind names.

    Args:
        data: the record.

    Returns:
        The evidence.

    Raises:
        ValueError: if the record carries no kind, or a kind with no class.
    """
    kind = _require(data, "kind")
    codec = _EVIDENCE_CODECS.get(kind)
    if codec is None:
        known = ", ".join(repr(tag) for tag in _EVIDENCE_CODECS)
        raise ValueError(
            f"evidence record has kind={kind!r}, which is none of {known}. There is "
            f"one kind per decisive prover, and another would have to be a class here "
            f"before a record could name it."
        )
    return codec.from_record(data)


def _provenance_to_dict(item: Provenance) -> dict[str, Any]:
    """One provenance as a record.

    Args:
        item: the provenance.

    Returns:
        The record.
    """
    return {
        "registered_at": item.registered_at,
        "measured_at": item.measured_at,
        "registered": item.registered,
    }


def report_to_dict(report: CheckReport) -> dict[str, Any]:
    """A report as a JSON-ready record, carrying the schema version.

    Every value is a string, an integer, a list, a mapping or ``None``, so the result
    passes through ``json.dumps`` unchanged. Key order is fixed rather than taken from
    the input, so two runs of one suite produce the same bytes and a diff shows what
    changed rather than what moved.

    Args:
        report: the report to write.

    Returns:
        The record.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "check_id": report.check_id,
        "name": report.name,
        "warrant": report.warrant.value if report.warrant else None,
        "outcome": report.outcome.value,
        "tier": report.tier.value,
        "detail": report.detail,
        "evidence": [_evidence_to_dict(item) for item in report.evidence],
        "provenance": [_provenance_to_dict(item) for item in report.provenance],
    }


def report_from_dict(data: Mapping[str, Any]) -> CheckReport:
    """A record as a report, through the constructor rather than around it.

    Every precondition the type enforces applies on the way in. A record naming
    ``PROVED`` with its evidence stripped does not construct, or the wire form would be
    a route around the guard the type exists to be.

    Args:
        data: the record, as `report_to_dict` wrote it.

    Returns:
        The report.

    Raises:
        ValueError: if the record's schema version is not this one, if a field is
            missing, if an enum value has no member, or if the report's own
            preconditions refuse what the record describes.
    """
    version = _require(data, "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"record has schema_version={version!r}, and this is {SCHEMA_VERSION!r}. "
            f"Reading it would mean guessing which fields moved, and a differ that "
            f"guesses reports changes nobody made."
        )
    warrant = _require(data, "warrant")
    return CheckReport(
        name=_require(data, "name"),
        check_id=_require(data, "check_id"),
        warrant=_member(Warrant, warrant, "warrant") if warrant is not None else None,
        outcome=_member(Outcome, _require(data, "outcome"), "outcome"),
        tier=_member(Tier, _require(data, "tier"), "tier"),
        detail=_require(data, "detail"),
        evidence=tuple(
            _evidence_from_dict(item) for item in _sequence(data, "evidence")
        ),
        provenance=tuple(
            Provenance(
                registered_at=_require(item, "registered_at"),
                measured_at=_require(item, "measured_at"),
                registered=_require(item, "registered"),
            )
            for item in _sequence(data, "provenance")
        ),
    )
