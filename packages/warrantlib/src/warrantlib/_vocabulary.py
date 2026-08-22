"""The warrant vocabulary: the records, their preconditions, and the run summary.

Every type the package exports is defined here. ``warrantlib/__init__.py`` re-exports
them and owns the public name list, so a caller never imports this module directly.

The split exists because the wire format is a separate responsibility from the records
themselves. ``_serialise`` imports these types. Nothing here imports it back.

The standard library is the only dependency.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Warrant(Enum):
    """How well a claim is warranted, by the prover class that produced it.

    ``PROVED`` — the claim is decided. A pen-and-paper theorem within its stated
    hypotheses (Prover 1), a symbolic identity (Prover 2), or a finite domain enumerated
    in full, where ¬∃ ≡ ∀¬ (Prover 3 · enumeration). Under that last one it is earned
    only with a completeness certificate. Without one the enumeration is a sample
    wearing a decision's label.

    ``CERTIFIED`` — validated numerics prove a universal over a compact domain by
    construction (Prover 3 · validated). Stronger than a sample, weaker than a decision,
    and it carries the bound it was computed with. Collapsing it into ``PROVED``
    overclaims. Collapsing it into ``CORROBORATED`` throws the bound away.

    ``CORROBORATED`` — a sample of a continuum (Prover 3 · sample). It exhibits
    existence and refutes a universal by counterexample. It never decides one, at any
    sample count.

    Orthogonal to outcome. A check reports both, and the three levels print in distinct
    vocabulary so a corroborative green run is visibly that.
    """

    PROVED = "PROVED"
    CERTIFIED = "CERTIFIED"
    CORROBORATED = "CORROBORATED"


class Outcome(Enum):
    """What a registered falsifier did, independent of how well it was warranted.

    A falsifier does not pass. It fires or it does not, and the words are chosen so that
    a run cannot be read as a column of ``PASS`` with the interesting distinctions
    flattened out of it.

    ``NOT_TRIGGERED`` — it ran, the condition did not obtain, the claim survives it.
    ``FIRED`` — the condition obtained. The claim is refuted, and that is the result.
    ``NOT_RESOLVED`` — it ran and the ordering is genuinely undetermined, because the
    two quantities' intervals overlap. Narrow on purpose: this is a measured tie, not a
    stand-in for a check that did not run.
    ``NOT_APPLICABLE`` — void by construction. It could not have fired here, so it is
    evidence for nothing and does not count among the survivors.
    ``NOT_RUN_HERE`` — measured elsewhere, or not yet. The detail says where.

    The last two never ran, so they carry no warrant. ``CheckReport`` enforces that.
    """

    NOT_TRIGGERED = "NOT TRIGGERED"
    FIRED = "FIRED"
    NOT_RESOLVED = "NOT RESOLVED"
    NOT_APPLICABLE = "NOT APPLICABLE"
    NOT_RUN_HERE = "NOT RUN HERE"


#: The outcomes of a falsifier that actually ran here. The other two did not, so they
#: are not counted among the tested and cannot carry a warrant.
_TESTED_HERE = (Outcome.NOT_TRIGGERED, Outcome.FIRED, Outcome.NOT_RESOLVED)


class Tier(Enum):
    """What the check was measured against.

    ``EXACT`` — a closed-form reference at machine precision.
    ``BOUNDED`` — a stated bar, or a certified bracket.
    ``COMPUTED`` — no statable bar. The word for such a number is *computed*, never
    *certified*.

    Cuts across warrant and outcome rather than ranking them. An ``EXACT`` reference can
    be sampled (Prover 3 · sample) and a ``COMPUTED`` number can come out of an
    exhaustive enumeration (Prover 3 · enumeration).
    """

    EXACT = "exact"
    BOUNDED = "bounded"
    COMPUTED = "computed"


#: Unicode general categories that put nothing on the page: the controls, the format
#: characters (soft hyphen, the zero-width space and the joiners, the byte-order mark),
#: and the three kinds of separator. ``str.strip()`` removes the separators and the
#: whitespace controls and leaves the format characters behind, so a field of them alone
#: survives a presence check while reading as empty.
_UNPRINTED = frozenset({"Cc", "Cf", "Zl", "Zp", "Zs"})


def _reject_unreadable(
    subject: str, name: str, value: object, blank_reason: str
) -> None:
    """Raise unless a record's ``value`` is one-line text that puts ink on the page.

    Args:
        subject: the record kind, as the message opens.
        name: the field, as it appears in the message.
        value: what was passed for it.
        blank_reason: why this particular field may not be blank, appended to the
            message when it is.

    Raises:
        ValueError: if ``value`` is not a string, is blank, or holds a line break.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{subject} passed {name} as {type(value).__name__}. The field is text a "
            "reader reads back, so anything else records it in a form nothing renders."
        )
    if all(unicodedata.category(character) in _UNPRINTED for character in value):
        raise ValueError(
            f"{subject} has a blank {name}. Whitespace and the zero-width "
            "characters count as blank here, since they satisfy a presence check and "
            f"put nothing on the page. {blank_reason}"
        )
    if value.splitlines() != [value]:
        raise ValueError(
            f"{subject} has a line break in {name}. A record renders as one line "
            "beside the check it backs, so a second line arrives in the middle of a "
            "summary row."
        )


@dataclass(frozen=True)
class SymbolicReduction:
    """What backs a Prover 2 claim: the correspondence a CAS cannot supply.

    A CAS checks that one expression equals another. It does not check that those
    expressions are the ones the analytic claim is about. That step is a human
    obligation, named as such in the warrant ledger, and this is where it is recorded
    instead of assumed. A reduction is evidence for
    [`CheckReport`][warrantlib.CheckReport] on the same terms as a completeness
    certificate, and the two are interchangeable there.

    Args:
        claim: the analytic statement, in words, that the symbolic identity stands for.
        correspondence: where the symbolic setup was analytically checked against the
            problem it stands for. A hand derivation by file and line, or a dated
            registration result.
        assumptions: what the reduction assumed, one condition per entry. The scope
            travels with the evidence, so the contingency is visible without reading
            the algebra.

    Raises:
        ValueError: if the assumptions are not a tuple, or if the claim, the
            correspondence or any assumption is not one-line text with a visible
            character in it.
    """

    claim: str
    correspondence: str
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a reduction that records no obligation to have discharged."""
        if not isinstance(self.assumptions, tuple):
            raise ValueError(
                "symbolic reduction passed assumptions as "
                f"{type(self.assumptions).__name__}. Assumptions are a tuple, so a "
                "bare string records one condition per character and the identity's "
                "scope reads as gibberish. Wrap a single one: (assumption,)."
            )
        for name, value in (
            ("claim", self.claim),
            ("correspondence", self.correspondence),
        ):
            _reject_unreadable(
                "symbolic reduction",
                name,
                value,
                "Prover 2 is theorem-grade only where the symbolic setup was hand "
                "derived against the analytic problem, so a reduction naming neither "
                "the statement nor where it was checked backs nothing. Fill both, or "
                "report CORROBORATED and say why in the check's detail.",
            )
        for position, assumption in enumerate(self.assumptions, start=1):
            _reject_unreadable(
                "symbolic reduction",
                f"assumption {position}",
                assumption,
                "An entry nobody filled in is scope a reader cannot check, and it "
                "renders as a gap in the list rather than as a caveat. Say what the "
                "condition is, or drop the entry.",
            )

    def __str__(self) -> str:
        """The reduction as one line: the claim, where it was checked, its scope."""
        scope = (
            f"assuming {'; '.join(self.assumptions)}"
            if self.assumptions
            else "no assumptions recorded"
        )
        return f"symbolic: {self.claim} (per {self.correspondence}, {scope})"


#: A check's id: dot-separated segments of ASCII letters, digits and underscores.
#: Spelled out rather than `\w`, which matches unicode by default and would admit the
#: maths glyphs the prose names use (`c₂`), so two ids could differ by a character a
#: reader cannot tell apart.
_CHECK_ID = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")

#: A git commit, abbreviated or full. Seven digits is git's own floor for an unambiguous
#: abbreviation. Case-insensitive, because a pasted hash sometimes is. Accepts a few
#: English words as a side effect, `deadbeef` and `defaced` among them. Refusing those
#: costs more than it buys.
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{7,40}")

#: An http(s) URL with a non-empty authority.
_URL = re.compile(r"https?://[^\s/]+(?:/\S*)?")

#: A DOI, bare or prefixed. The registrant is ``10.`` and four to nine digits.
_DOI = re.compile(r"(?:doi:)?10\.\d{4,9}/\S+", re.IGNORECASE)

#: The three ref shapes, in the order the message lists them. Matched with
#: ``fullmatch``: under ``match`` a branch name passes as a URL's prefix, and a hash
#: with prose after it passes as a commit.
_REF_SHAPES = (_COMMIT_SHA, _URL, _DOI)


def _reject_unresolvable(name: str, value: str) -> None:
    """Raise unless a provenance's ``value`` is a ref that resolves to one fixed thing.

    Args:
        name: the field, as it appears in the message.
        value: the ref passed for it, already known to be one-line text.

    Raises:
        ValueError: if the ref is none of the three shapes.
    """
    if any(shape.fullmatch(value) for shape in _REF_SHAPES):
        return
    raise ValueError(
        f"provenance has {name}={value!r}, which resolves to nothing fixed. A ref is a "
        "git commit SHA (7 to 40 hex digits), an http(s) URL, or a DOI. A path, a "
        "branch, a tag or a phrase names something that moves, so a reviewer checking "
        "the ordering later reaches a different tree than the one this was written "
        "against, or none at all."
    )


@dataclass(frozen=True)
class Provenance:
    """Which ref registered a claim, and which one measured it.

    A number checked against a bar is worth reading only if the bar was fixed before the
    number existed. A report says what was decided and how well it was decided, and
    nothing in it says when the bar was set. This is the ref a reviewer opens to find
    out.

    Where the two refs name one commit, the render says so. Registering and measuring
    together is not refused. The ordering then rests on the account the surrounding
    prose gives, and the marker is what stops a reader taking it for something the
    history shows.

    History orders two refs. A string cannot. Equality is checkable here and ordering is
    not, so a ``registered_at`` that in fact came after ``measured_at`` renders exactly
    like one that came before. Establishing the direction is a reviewer, or a test,
    running ``git merge-base --is-ancestor``.

    Args:
        registered_at: the ref where the prediction, the bar or the derivation was
            registered. A git commit SHA, an http(s) URL, or a DOI. A URL is taken to be
            a permalink. One that tracks a branch moves, which is the defect that rules
            a bare path out.
        measured_at: the ref whose tree produced the number, in the same three shapes.
        registered: what a reviewer will find at ``registered_at``, in one line. A ref
            on its own sends them to a diff and leaves them to work out which part of it
            was the registration.

    Raises:
        ValueError: if either ref is not one-line text in one of the three shapes, or if
            ``registered`` is not one-line text with a visible character in it.
    """

    registered_at: str
    measured_at: str
    registered: str

    def __post_init__(self) -> None:
        """Reject a ref that resolves to nothing, and a statement nobody wrote."""
        for name, value in (
            ("registered_at", self.registered_at),
            ("measured_at", self.measured_at),
        ):
            _reject_unreadable(
                "provenance",
                name,
                value,
                "The ordering is the whole content of a provenance, so one end of it "
                "missing records no ordering at all. Give the ref, or report "
                "CORROBORATED and say in the check's detail why there is none.",
            )
            _reject_unresolvable(name, value)
        _reject_unreadable(
            "provenance",
            "registered",
            self.registered,
            "A bare ref sends a reviewer to a diff and leaves them to work out which "
            "part of it was the registration. Say what they will find there.",
        )

    @property
    def same_ref(self) -> bool:
        """Whether the two refs name one commit, so history orders nothing.

        An abbreviation counts. Without that, lengthening one of the two hashes walks
        away from the marker while still naming the same commit.
        """
        first, second = self.registered_at.casefold(), self.measured_at.casefold()
        if first == second:
            return True
        short, long = sorted((first, second), key=len)
        abbreviated = all(_COMMIT_SHA.fullmatch(ref) for ref in (short, long))
        return abbreviated and long.startswith(short)

    def __str__(self) -> str:
        """The provenance as one line: what was registered, where, against what."""
        if self.same_ref:
            return (
                f"provenance: {self.registered} (registered and measured at "
                f"{self.registered_at}, so the ordering is not established by history)"
            )
        return (
            f"provenance: {self.registered} "
            f"(registered at {self.registered_at}, measured at {self.measured_at})"
        )


@dataclass(frozen=True)
class CompletenessCertificate:
    """Evidence an enumeration was exhaustive: ``expected`` vs ``visited`` (ADR-030).

    Two independent facts, and a ``PROVED`` warrant needs both. **Domain**:
    ``expected == action_set_size ** horizon``, so the set quantified over is the
    declared one. **Coverage**: ``visited == expected``, so it was enumerated in full.
    They come apart wherever ``visited`` is a loop-carried counter rather than an
    array's length, which is where a padding bug lives, and coverage alone is what
    carries the Prover 3 · enumeration licence.

    The certificate names its set. ``expected`` on its own conflates the base with the
    exponent — 81 is ``9**2`` and ``3**4`` — so a bare count is not self-describing and
    two certificates over different sets cannot be told apart. Carrying the size, the
    horizon and the version fixes that at the type rather than in the surrounding prose
    (standing prohibition 9).

    A partial enumeration sampled its set, so its warrant is ``CORROBORATED``. Pairing
    ``PROVED`` with a shortfall does not construct.

    Args:
        expected: the policy count the search was obliged to visit — ``|A|^H``,
            supplied rather than derived, so the domain check compares two routes.
        visited: how many it actually visited.
        warrant: the prover class the enumeration earns.
        action_set_size: the declared action count — ``|A|``.
        horizon: the sequence length — ``H``.
        action_set_version: the declared set's version tag.

    Raises:
        ValueError: if the warrant is ``PROVED`` and either precondition fails.
    """

    expected: int
    visited: int
    warrant: Warrant
    action_set_size: int
    horizon: int
    action_set_version: str

    def __post_init__(self) -> None:
        """Reject a ``PROVED`` certificate failing domain or coverage."""
        if self.warrant is not Warrant.PROVED:
            return
        if not self.domain_declared:
            raise ValueError(
                f"a PROVED certificate must quantify over the declared set, got "
                f"expected={self.expected} against |A|^H = "
                f"{self.action_set_size}^{self.horizon} = "
                f"{self.action_set_size**self.horizon} for set "
                f"{self.action_set_version!r}. The count and the set have come apart."
            )
        if not self.complete:
            raise ValueError(
                f"a PROVED certificate must be complete, got expected="
                f"{self.expected} against visited={self.visited}. A partial "
                "enumeration sampled its set, so its warrant is CORROBORATED."
            )

    @property
    def domain_declared(self) -> bool:
        """Whether ``expected`` is the declared set's own ``|A|^H``."""
        return self.expected == self.action_set_size**self.horizon

    @property
    def complete(self) -> bool:
        """Whether every expected policy was visited."""
        return self.expected == self.visited

    def __str__(self) -> str:
        """The certificate as a one-line warrant string in its own vocabulary."""
        return (
            f"{self.warrant.value} (set {self.action_set_version}, "
            f"|A|^H = {self.action_set_size}^{self.horizon} = {self.expected}, "
            f"visited {self.visited})"
        )


Evidence = CompletenessCertificate | SymbolicReduction
"""What backs a ``PROVED`` claim, one member per decisive prover the suite runs.

A completeness certificate decides by exhausting a finite domain. A symbolic reduction
decides by identity (Provers 1 and 2) and enumerates nothing, so a certificate is the
wrong evidence for it rather than a missing one.
"""

#: The evidence classes as a tuple ``isinstance`` accepts. Both are defined here, so the
#: guard reads them at module scope.
_EVIDENCE_TYPES = (CompletenessCertificate, SymbolicReduction)


@dataclass(frozen=True)
class CheckReport:
    """One check's result: what it found, how well, and against what.

    A record rather than a return value. Frozen, because editing a report after the
    check ran is editing the finding.

    Args:
        name: which check this is, as it appears in the summary.
        check_id: the same check as a key, in dot-separated segments of letters,
            digits and underscores. The name is prose and is reworded; this is what a
            manifest declares and what two runs are joined on, so it is not.
        warrant: the prover class behind the claim, or ``None`` where the check
            produced no evidence to classify.
        outcome: what the falsifier did.
        tier: what the check was measured against.
        detail: why it reports what it reports, in one line. Required, so a report
            cannot be a bare outcome with extra fields.
        evidence: what backs the claim, as a tuple of ``CompletenessCertificate`` and
            ``SymbolicReduction``. Required non-empty when the warrant is ``PROVED``
            and unused otherwise. A tuple rather than one item because a claim
            quantified over several enumerations rests on all their certificates, and
            carrying one of them understates what was checked.
        provenance: which ref registered the claim and which one measured it, as a
            tuple. Required non-empty when the warrant is ``PROVED``, unused otherwise.
            A tuple on the same argument the evidence is one: a claim resting on two
            registrations rests on both, and carrying one of them understates what a
            reviewer has to check.

    Raises:
        ValueError: if the warrant is ``PROVED`` and no evidence or no provenance was
            given, if a check that never ran here carries a warrant anyway, if the
            evidence or the provenance is not a tuple, or if an item in either is not
            one of the kinds it accepts.
    """

    name: str
    check_id: str
    warrant: Warrant | None
    outcome: Outcome
    tier: Tier
    detail: str
    evidence: tuple[Evidence, ...] = ()
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        """Reject a claim with nothing behind it, at either of two strengths."""
        subject = f"check {self.name!r}"
        _reject_unreadable(
            subject,
            "name",
            self.name,
            "A row nobody can attribute is a result with no claim beside it, and it "
            "reads in the summary as a blank where the check should be.",
        )
        _reject_unreadable(
            subject,
            "detail",
            self.detail,
            "A report with no reason is the bare outcome this field exists to stop, "
            "and the outcome alone says what happened without saying what was found.",
        )
        _reject_unreadable(
            subject,
            "check_id",
            self.check_id,
            "The id is what a manifest declares and what joins one run's report to "
            "the next. A report without one is reconcilable with nothing.",
        )
        if not _CHECK_ID.fullmatch(self.check_id):
            raise ValueError(
                f"{subject} has check_id={self.check_id!r}, which is not a key. An id "
                "is dot-separated segments of letters, digits and underscores, as in "
                "gap_series.c2_closed_form or R10.crossover_flip. The prose name is "
                "the other field. A key carrying prose moves whenever the prose is "
                "reworded, and a ledger comparing two runs then reads one check as "
                "one dropped and one added."
            )
        if not isinstance(self.evidence, tuple):
            raise ValueError(
                f"check {self.name!r} passed evidence as "
                f"{type(self.evidence).__name__}. Evidence is a tuple, so a claim "
                "resting on several enumerations can carry all of their certificates. "
                "Wrap a single one: (certificate,)."
            )
        if self.evidence:
            for item in self.evidence:
                if not isinstance(item, _EVIDENCE_TYPES):
                    raise ValueError(
                        f"check {self.name!r} carries a "
                        f"{type(item).__name__} as evidence. There are two kinds, one "
                        "per decisive prover: a CompletenessCertificate for an "
                        "exhaustive enumeration (Prover 3 · enumeration), a "
                        "SymbolicReduction for a theorem or a symbolic identity "
                        "(Provers 1 and 2). Anything "
                        "else satisfies the PROVED precondition by being present and "
                        "backs nothing."
                    )
        if not isinstance(self.provenance, tuple):
            raise ValueError(
                f"check {self.name!r} passed provenance as "
                f"{type(self.provenance).__name__}. Provenance is a tuple, so a claim "
                "resting on two registrations can carry both. Wrap a single one: "
                "(provenance,)."
            )
        for item in self.provenance:
            if not isinstance(item, Provenance):
                raise ValueError(
                    f"check {self.name!r} carries a {type(item).__name__} as "
                    "provenance. A bare ref, or a sentence saying the bar came first, "
                    "satisfies the precondition by being present and leaves a reviewer "
                    "with one end of an ordering and no way to check it. Pass a "
                    "Provenance, which validates both refs and says what was "
                    "registered at the first of them."
                )
        if self.warrant is Warrant.PROVED and len(self.evidence) == 0:
            raise ValueError(
                f"check {self.name!r} reports PROVED with no evidence. A decided "
                "universal needs something statable behind it: a completeness "
                "certificate for an exhaustive enumeration (Prover 3 · enumeration), a "
                "SymbolicReduction for a theorem or a symbolic identity (Provers 1 "
                "and 2). Report CERTIFIED for a bound over a compact domain, "
                "CORROBORATED for a sample."
            )
        if self.warrant is Warrant.PROVED and len(self.provenance) == 0:
            raise ValueError(
                f"check {self.name!r} reports PROVED with no provenance. A decided "
                "universal says which ref registered the claim and which one measured "
                "it, so a reader can check that the bar was fixed before the number "
                "existed rather than take the ordering on trust. Registering and "
                "measuring at one ref is allowed and renders as such. Report CERTIFIED "
                "or CORROBORATED where there was no registration at all."
            )
        if self.outcome not in _TESTED_HERE and self.warrant is not None:
            raise ValueError(
                f"check {self.name!r} is {self.outcome.value} and carries the warrant "
                f"{self.warrant.value}. It produced no evidence here, so there is no "
                "prover class to report. Leave the warrant None."
            )

    def __str__(self) -> str:
        """The report as one summary line, in the warrant's own vocabulary."""
        warrant = self.warrant.value if self.warrant else "—"
        line = (
            f"{self.name}: {self.outcome.value} "
            f"({warrant}, tier {self.tier.value}). {self.detail}"
        )
        if not self.provenance:
            return line
        return f"{line} {' '.join(str(item) for item in self.provenance)}"


def check_summary(reports: Sequence[CheckReport]) -> str:
    """Counts per ``(warrant, outcome)`` across a run, as a block of lines.

    The header carries the accounting a reader needs first: how many falsifiers were
    registered, how many this run actually tested, and how many fired. Registering four
    and testing two is a different claim from testing four, and one number cannot say
    both. The rows underneath say what warrant the tested ones carried, so a run that
    survived everything without deciding anything prints as exactly that.

    Pairs with no checks are left out, so the block is as long as the run was varied.
    Ordering follows the enum declarations rather than the input, so two runs of the
    same suite produce the same text. Checks with no warrant sort last, under ``—``.

    Args:
        reports: the run's reports, in any order.

    Returns:
        A newline-separated block: the accounting line, then one row per occupied pair.
    """
    counts = Counter((report.warrant, report.outcome) for report in reports)
    tested = sum(1 for report in reports if report.outcome in _TESTED_HERE)
    fired = sum(1 for report in reports if report.outcome is Outcome.FIRED)
    lines = [
        f"{len(reports)} registered, {tested} tested here, "
        f"{f'{fired} fired' if fired else 'none fired'}"
    ]
    lines += [
        f"   {(warrant.value if warrant else '—'):<13} "
        f"{outcome.value:<15} {counts[warrant, outcome]}"
        for warrant in (*Warrant, None)
        for outcome in Outcome
        if counts[warrant, outcome]
    ]
    return "\n".join(lines)
