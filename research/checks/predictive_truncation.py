"""Is the `y` quadrature grid wide enough for the density it actually integrates?

The GATE-D4 inference gap is built on a `y` grid sized at ``μ ± 9·√(σ² + R(μ))``. The
half-width comes from the plug-in `R(μ)`. The density integrated over it is `p*(y)`,
built from the true `R(x)`. Those are different objects and the pairing was never
checked.

They differ in kind, not only in scale. `p*` is a scale mixture of Gaussians, so for
unbounded `R` its tails are exponential, and a "`k` standard deviations" rule sizes a
Gaussian tail against a non-Gaussian one. No `k` is safe on principle. Whether a given
`k` is safe depends on where the grid edge falls relative to the crossover, which is
what C2 measures.

This module measures the truncation. It does not fix it. The production grid rule stays
where it is, and the red cells are the output::

    uv run --no-sync python -m research.checks.predictive_truncation --check
    uv run --no-sync python -m research.checks.predictive_truncation

The gap itself lives in ``gap_kernel``, which pins the three conventions the lost
scripts only recorded in prose: reverse KL, `R` frozen at the prior mean, averaged under
the true predictive. This module owns the question, not the integral.

Two departures from the spec this was written against, both deliberate.

**No PASS/FAIL/VOID outcome.** ``cpomdp.warrant.Outcome`` deliberately has no ``PASS``:
a falsifier fires or it does not. Adding those members would undo a documented design.
The mapping used instead, and printed in every summary:

===================  ======================  ====  ====
condition            ``Outcome``             word  exit
===================  ======================  ====  ====
below the floor      ``NOT_TRIGGERED``       PASS  0
above the floor      ``FIRED``               FAIL  1
no measurement made  ``NOT_APPLICABLE``      VOID  2
===================  ======================  ====  ====

**``VoidReason`` is local.** It does not exist in ``cpomdp.warrant``. It lives in
``gap_kernel`` rather than in the package, since a new export there needs a ``docs/api``
page before anything can reference it, and these modules are not on the main suite yet.
Its value goes into ``CheckReport.detail``, so no parallel result type is introduced.

The warrant is ``CORROBORATED`` throughout, not ``CERTIFIED``. Adaptive quadrature over
a sampled grid of `σ` is a sample of a continuum. Certification would need validated
numerics, which is what PR-8's interval-arithmetic bound is for.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from research.checks.gap_kernel import (
    CONVERGENCE_BAR,
    FAMILIES,
    QUADRATURE_FLOOR,
    SIGMAS,
    TAIL_EXTENT,
    NoiseFamily,
    Quadrature,
    VoidReason,
    assert_positive_noise,
    core_and_tail,
    log_predictive,
    plugin_noise_of,
    predictive_sd,
)

from cpomdp.warrant import CheckReport, Outcome, Tier, Warrant, check_summary

__all__ = [
    "TruncationReport",
    "measure_truncation",
    "run_checks",
]

# Every reference value in the spec's section 6 came from a scalar quadrature written
# against an assumed parametrisation, not from the cpomdp pipeline. It bounds the
# phenomenon; it does not validate these numbers. A disagreement is a finding about one
# of the two implementations, so it reports FAIL and gets investigated. It is never
# absorbed by retuning the reference.
_PROVENANCE = "external-scalar-quadrature"

#: Significant figures the external reference is trusted to. Anything tighter would be
#: reading precision into a number that does not carry it.
_REFERENCE_SIG_FIGS = 2

# Declared inputs from research/gate_d4_registration.md, RESULT 2026-08-10. They are
# arguments to C4's materiality question, not measurements this module makes. `c₄` is
# under analytic derivation, so this is the registered figure being used as a scale, not
# a value this suite computes or endorses.
_C4_MAGNITUDE = 0.18980  # |c₄| at the declared operating point κ = 1, μ* = 1
_EXTRACTION_SPREAD = 3.6e-3  # jackknifed extraction spread on c₄, 0.36%
_BASIS_FIT_ERROR = 1.03e-2  # seven-term basis fit error on c₄, 1.03%

#: C4's bar: the truncation must not reach a tenth of the extraction spread already
#: carried on `c₄`. Above that it stops being negligible against what it contaminates.
_MATERIALITY_MARGIN = 0.1

#: The disclosed convergence from the registration's DISCLOSURE block, as a CLI default
#: only. `floor` has no default in the API: a tolerance that lives in the code instead
#: of the call site drifts out of sync with the prose it was disclosed in.
_CLI_DEFAULT_FLOOR = 10.0**-8.7


@dataclass(frozen=True)
class TruncationReport:
    """One `(family, σ, multiplier)` cell: what the grid covered and what it lost.

    Args:
        sigma: σ — the prior standard deviation.
        family: the family's name, as it prints.
        half_width: L — the `y` half-width the rule produced.
        half_width_multiplier: L over the plug-in predictive sd, the rule's `k`.
        true_predictive_sd: `√Var_{p*}[y]`, measured.
        plugin_predictive_sd: `√(σ² + R(μ))`, the sd the rule assumed.
        true_sd_covered: how many *true* sd the half-width actually reaches.
        relative_truncation: `|tail| / (core + tail)`.
        floor: the disclosed tolerance this cell was checked against.
        outcome: fired above the floor, not triggered below it, void where nothing was
            measured.
        void_reason: why there is no measurement, or ``None``.
        provenance: where the comparison reference came from.
    """

    sigma: float
    family: str
    half_width: float
    half_width_multiplier: float
    true_predictive_sd: float
    plugin_predictive_sd: float
    true_sd_covered: float
    relative_truncation: float
    floor: float
    outcome: Outcome
    void_reason: VoidReason | None
    provenance: str


def measure_truncation(
    family: NoiseFamily,
    sigma: float,
    multiplier: float = 9.0,
    *,
    floor: float,
    settings: Quadrature | None = None,
) -> TruncationReport:
    """Measure what the `y` grid throws away at one `(family, σ)` cell.

    Args:
        family: the declared `R` and its prior mean.
        sigma: σ — the prior standard deviation.
        multiplier: the grid rule's `k`, in plug-in predictive standard deviations.
        floor: the disclosed tolerance to check against. Required, and deliberately
            without a default: a tolerance baked into the check drifts out of sync with
            the prose that disclosed it.
        settings: quadrature tolerances. Defaults to the module's.

    Returns:
        The cell's report.
    """
    settings = settings or Quadrature()
    prior_variance = sigma**2
    plugin_noise = plugin_noise_of(family)
    plugin_sd = math.sqrt(prior_variance + plugin_noise)
    half_width = multiplier * plugin_sd
    centre = family.prior_mean
    span = (centre - half_width - TAIL_EXTENT, centre + half_width + TAIL_EXTENT)

    void_reason: VoidReason | None = None
    try:
        assert_positive_noise(family, span)
    except ValueError:
        void_reason = VoidReason.NON_POSITIVE_NOISE

    if void_reason is None:
        core, tail = core_and_tail(family, prior_variance, half_width, settings)
        relative = abs(tail) / (core + tail) if core + tail else 0.0
        _, refined_tail = core_and_tail(
            family, prior_variance, half_width, settings.refined()
        )
        moved = abs(refined_tail - tail) > CONVERGENCE_BAR * abs(tail)
        if relative <= QUADRATURE_FLOOR or moved:
            void_reason = VoidReason.NUMERICAL_FLOOR
        true_sd = predictive_sd(
            family, prior_variance, half_width + TAIL_EXTENT, settings
        )
    else:
        relative, true_sd = math.nan, math.nan

    if void_reason is not None:
        outcome = Outcome.NOT_APPLICABLE
    elif relative < floor:
        outcome = Outcome.NOT_TRIGGERED
    else:
        outcome = Outcome.FIRED

    return TruncationReport(
        sigma=sigma,
        family=family.name,
        half_width=half_width,
        half_width_multiplier=half_width / plugin_sd,
        true_predictive_sd=true_sd,
        plugin_predictive_sd=plugin_sd,
        true_sd_covered=half_width / true_sd if true_sd == true_sd else math.nan,
        relative_truncation=relative,
        floor=floor,
        outcome=outcome,
        void_reason=void_reason,
        provenance=_PROVENANCE,
    )


def _same_to_sig_figs(measured: float, reference: float, figures: int) -> bool:
    """Whether two positive numbers agree to a number of significant figures.

    Args:
        measured: what this module computed.
        reference: the external value.
        figures: how many significant figures to compare on.

    Returns:
        Whether they agree at that precision.
    """
    if measured <= 0.0 or reference <= 0.0:
        return False
    return abs(math.log10(measured / reference)) < 10.0 ** (1 - figures) / math.log(10)


def _check_truncation(report: TruncationReport) -> CheckReport:
    """C1: the truncation against the disclosed floor, at one cell.

    Args:
        report: the measured cell.

    Returns:
        The check's report.
    """
    name = f"C1 truncation [{report.family}, σ={report.sigma:.2f}]"
    if report.void_reason is not None:
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail=f"VOID — {report.void_reason.value}",
        )
    verdict = "FAIL" if report.outcome is Outcome.FIRED else "PASS"
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=report.outcome,
        tier=Tier.B,
        detail=(
            f"{verdict} — truncation {report.relative_truncation:.2e} against a floor "
            f"of {report.floor:.2e}; L = {report.half_width:.3f} covers "
            f"{report.true_sd_covered:.3f} true sd"
        ),
    )


def _check_reference(family: NoiseFamily, report: TruncationReport) -> CheckReport:
    """Section 2's rule: agreement with the external reference, at two sig figs.

    A disagreement is a finding about one of the two implementations and reports FIRED.
    It is never absorbed by moving the reference.

    Args:
        family: the family the cell was measured on, carrying its own reference values.
        report: the measured cell.

    Returns:
        The check's report.
    """
    name = f"ref agreement [{report.family}, σ={report.sigma:.2f}]"
    table = family.reference or {}
    reference = table.get(round(report.sigma, 2))
    if reference is None or report.void_reason is not None:
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail="no external reference for this cell",
        )
    agrees = _same_to_sig_figs(
        report.relative_truncation, reference, _REFERENCE_SIG_FIGS
    )
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED if agrees else Outcome.FIRED,
        tier=Tier.B,
        detail=(
            f"{'PASS' if agrees else 'FAIL'} — measured "
            f"{report.relative_truncation:.2e} against {reference:.1e} "
            f"({_PROVENANCE}), at {_REFERENCE_SIG_FIGS} sig figs"
        ),
    )


def _check_tail_shape(
    family: NoiseFamily, sigma: float, settings: Quadrature
) -> CheckReport:
    """C2: does `log p*` decay linearly or quadratically in the offset?

    The 2% mismatch between the plug-in and true predictive sd is not what breaks the
    grid rule. `p*` is a scale mixture, so for unbounded `R` the tails are exponential
    and a "`k` sd" rule is sizing a Gaussian tail against a non-Gaussian one. This is
    what stops the next family being sized by the same reasoning.

    Args:
        family: the declared `R` and its prior mean.
        sigma: σ — the prior standard deviation.
        settings: the tolerances and the window scale.

    Returns:
        The check's report.
    """
    name = f"C2 tail shape [{family.name}, σ={sigma:.2f}]"
    offsets = np.array([6.0, 10.0, 14.0, 20.0, 28.0, 40.0])
    log_density = np.array(
        [
            log_predictive(family, sigma**2, family.prior_mean + float(nu), settings)
            for nu in offsets
        ]
    )
    if not np.all(np.isfinite(log_density)):
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail=f"VOID — {VoidReason.NUMERICAL_FLOOR.value}, density underflowed",
        )
    top_nu, top_log = offsets[-3:], log_density[-3:]
    linear = np.polyfit(top_nu, top_log, 1)
    quadratic = np.polyfit(top_nu**2, top_log, 1)
    linear_residual = float(np.sum((np.polyval(linear, top_nu) - top_log) ** 2))
    quadratic_residual = float(
        np.sum((np.polyval(quadratic, top_nu**2) - top_log) ** 2)
    )
    linear_wins = linear_residual < quadratic_residual
    crossover = family.crossover(sigma**2) if family.crossover else math.nan
    fired = family.unbounded and not linear_wins
    plugin_noise = plugin_noise_of(family)
    edge = 9.0 * math.sqrt(sigma**2 + plugin_noise)
    shape = "linear" if linear_wins else "Gaussian"
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if fired else Outcome.NOT_TRIGGERED,
        tier=Tier.B,
        detail=(
            f"{'FAIL' if fired else 'PASS'} — {shape} fit wins (residuals "
            f"{linear_residual:.2e} vs {quadratic_residual:.2e}), slope "
            f"{float(linear[0]):.3f}/unit; crossover ν* ≈ {crossover:.2f} against a "
            f"grid edge at {edge:.2f}"
        ),
    )


def _check_family_class(
    family: NoiseFamily, reports: Sequence[TruncationReport]
) -> CheckReport:
    """C3: which families the grid rule exposes.

    Bounded `R` is genuinely Gaussian-tailed and its tail integral underflows double
    precision. That is a PASS, but only with a printed bound: `nan` and `0.0` are not
    measurements and neither is printed as one.

    Args:
        family: the declared `R` and its prior mean.
        reports: that family's cells across the `σ` grid.

    Returns:
        The check's report.
    """
    name = f"C3 family class [{family.name}]"
    if family.unbounded:
        return _exposed_family_report(name, reports)
    return _bounded_family_report(name, reports)


def _bounded_family_report(
    name: str, reports: Sequence[TruncationReport]
) -> CheckReport:
    """C3 for a family whose `R` is bounded, where the tail should not resolve at all.

    The bound printed is this module's own quadrature floor, not double precision. The
    external reference reports underflow; adaptive quadrature returns roundoff instead,
    and quoting a number it cannot resolve as a measurement is the failure mode C5
    exists to catch.

    Args:
        name: the check's name.
        reports: that family's cells across the `σ` grid.

    Returns:
        The check's report.
    """
    unresolved = [
        report
        for report in reports
        if report.void_reason is VoidReason.NUMERICAL_FLOOR
        or report.relative_truncation < 1e-300
    ]
    clean = len(unresolved) == len(reports)
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED if clean else Outcome.FIRED,
        tier=Tier.B,
        detail=(
            f"{'PASS' if clean else 'FAIL'} — bounded R, tail unresolved at "
            f"{len(unresolved)} of {len(reports)} cells, under the quadrature floor "
            f"{QUADRATURE_FLOOR:.0e}. Not resolved further, so no number is quoted"
        ),
    )


def _exposed_family_report(
    name: str, reports: Sequence[TruncationReport]
) -> CheckReport:
    """C3 for a family whose `R` grows without bound, where the tail is exponential.

    Args:
        name: the check's name.
        reports: that family's cells across the `σ` grid.

    Returns:
        The check's report.
    """
    fired = [report for report in reports if report.outcome is Outcome.FIRED]
    worst = max((report.relative_truncation for report in reports), default=math.nan)
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if fired else Outcome.NOT_TRIGGERED,
        tier=Tier.B,
        detail=(
            f"{'FAIL' if fired else 'PASS'} — unbounded R, {len(fired)} of "
            f"{len(reports)} cells above the floor, worst truncation {worst:.2e}"
        ),
    )


def _check_materiality(
    family: NoiseFamily, report: TruncationReport, settings: Quadrature
) -> CheckReport:
    """C4: does the truncation move the quantity being extracted?

    C1 answers whether the disclosure is breached. This answers whether the result is
    contaminated. Those have different answers, and collapsing them into one check loses
    the distinction the amendment needs.

    Args:
        family: the declared `R` and its prior mean.
        report: the measured cell.
        settings: the tolerances and the window scale.

    Returns:
        The check's report.
    """
    name = f"C4 materiality [{report.family}, σ={report.sigma:.2f}]"
    if report.void_reason is not None:
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail=f"VOID — {report.void_reason.value}",
        )
    core, tail = core_and_tail(family, report.sigma**2, report.half_width, settings)
    total_gap = core + tail
    quartic = _C4_MAGNITUDE * report.sigma**4
    ratio = abs(tail) / quartic
    bar = _MATERIALITY_MARGIN * _EXTRACTION_SPREAD
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if ratio >= bar else Outcome.NOT_TRIGGERED,
        tier=Tier.B,
        detail=(
            f"{'FAIL' if ratio >= bar else 'PASS'} — gap {total_gap:.3e}, quartic "
            f"residual {quartic:.3e}, truncation {abs(tail):.2e}, ratio {ratio:.1e} "
            f"against {bar:.1e} (extraction spread {_EXTRACTION_SPREAD:.1e}, basis fit "
            f"{_BASIS_FIT_ERROR:.2e})"
        ),
    )


def _check_convergence(
    family: NoiseFamily, report: TruncationReport, settings: Quadrature
) -> CheckReport:
    """C5: is the tail integral converged, or is this quadrature noise?

    The grid-width study saturates near 3e-15 and stops moving, which is the quadrature
    floor rather than a measurement. A cell at or below it reports VOID.

    Args:
        family: the declared `R` and its prior mean.
        report: the measured cell.
        settings: the tolerances and the window scale.

    Returns:
        The check's report.
    """
    name = f"C5 convergence [{report.family}, σ={report.sigma:.2f}]"
    if report.void_reason is VoidReason.NON_POSITIVE_NOISE:
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail=f"VOID — {report.void_reason.value}",
        )
    core, tail = core_and_tail(
        family, report.sigma**2, report.half_width, settings.refined()
    )
    refined = abs(tail) / (core + tail) if core + tail else 0.0
    if refined <= QUADRATURE_FLOOR or report.relative_truncation <= QUADRATURE_FLOOR:
        return CheckReport(
            name=name,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.B,
            detail=(
                f"VOID — {VoidReason.NUMERICAL_FLOOR.value}: {refined:.2e} at or under "
                f"{QUADRATURE_FLOOR:.0e}"
            ),
        )
    moved = abs(refined - report.relative_truncation) / report.relative_truncation
    return CheckReport(
        name=name,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if moved >= CONVERGENCE_BAR else Outcome.NOT_TRIGGERED,
        tier=Tier.B,
        detail=(
            f"{'FAIL' if moved >= CONVERGENCE_BAR else 'PASS'} — refinement moves the "
            f"truncation by {moved:.1%} against a {CONVERGENCE_BAR:.0%} bar "
            f"({report.relative_truncation:.2e} → {refined:.2e})"
        ),
    )


def run_checks(
    *,
    floor: float,
    families: Sequence[NoiseFamily] = tuple(FAMILIES.values()),
    sigmas: Sequence[float] = SIGMAS,
) -> list[CheckReport]:
    """Run C1 to C5 across the families and the `σ` grid.

    Args:
        floor: the disclosed tolerance to check against. Required.
        families: the declared families to run.
        sigmas: the prior standard deviations to run at.

    Returns:
        Every check's report, in check order.
    """
    settings = Quadrature()
    reports: list[CheckReport] = []
    measured: dict[str, list[TruncationReport]] = {}

    for family in families:
        cells = [
            measure_truncation(family, sigma, floor=floor, settings=settings)
            for sigma in sigmas
        ]
        measured[family.name] = cells
        reports += [_check_truncation(cell) for cell in cells]
        if family.reference:
            reports += [_check_reference(family, cell) for cell in cells]

    for family in families:
        reports.append(_check_tail_shape(family, sigmas[-1], settings))
        reports.append(_check_family_class(family, measured[family.name]))

    for family in families:
        for cell in measured[family.name]:
            if family.crossover is not None:
                reports.append(_check_materiality(family, cell, settings))
            reports.append(_check_convergence(family, cell, settings))

    return reports


def _print_table(reports: Sequence[TruncationReport]) -> None:
    """Print the measured cells as one table.

    Args:
        reports: the cells to print.
    """
    print(f"{'family':<20} {'σ':>6} {'L':>8} {'true sd':>9} {'plug-in':>9} ", end="")
    print(f"{'true sd cov':>12} {'rel trunc':>11}")
    for report in reports:
        print(
            f"{report.family:<20} {report.sigma:>6.2f} {report.half_width:>8.3f} "
            f"{report.true_predictive_sd:>9.5f} {report.plugin_predictive_sd:>9.5f} "
            f"{report.true_sd_covered:>12.3f} {report.relative_truncation:>11.2e}"
        )


def _print_width_ladder(family: NoiseFamily, sigma: float, floor: float) -> None:
    """Print what widening the grid rule would buy, at the worst `σ`.

    This prices the open question the C1 failure creates. It does not answer it, and it
    does not change the rule: which multiplier the production grid uses is a
    registration decision, and taking it here would be taking it with the answer in
    view.

    Args:
        family: the declared `R` and its prior mean.
        sigma: σ — the prior standard deviation to price at.
        floor: the disclosed tolerance, for the verdict column.
    """
    print(f"\ngrid width at σ = {sigma:.2f}, {family.name}")
    print(f"{'multiplier':>10} {'L':>8} {'rel trunc':>11}  verdict")
    for multiplier in (9.0, 11.0, 13.0, 15.0, 18.0):
        cell = measure_truncation(family, sigma, multiplier, floor=floor)
        if cell.void_reason is not None:
            verdict = f"VOID — {cell.void_reason.value}"
        else:
            verdict = "above the floor" if cell.outcome is Outcome.FIRED else "below it"
        print(
            f"{multiplier:>10.0f} {cell.half_width:>8.2f} "
            f"{cell.relative_truncation:>11.2e}  {verdict}"
        )


def _exit_code(reports: Sequence[CheckReport]) -> int:
    """Zero when nothing fired, one on a firing, two when something went unmeasured.

    Args:
        reports: the run's check reports.

    Returns:
        The process exit code.
    """
    if any(report.outcome is Outcome.FIRED for report in reports):
        return 1
    if any(report.outcome is Outcome.NOT_APPLICABLE for report in reports):
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the suite and print it.

    Args:
        argv: command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Measure what the GATE-D4 y grid throws away."
    )
    parser.add_argument("--check", action="store_true", help="run the check suite")
    parser.add_argument(
        "--floor",
        type=float,
        default=_CLI_DEFAULT_FLOOR,
        help="the disclosed tolerance to check against (default: 10**-8.7)",
    )
    parser.add_argument(
        "--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES)
    )
    parser.add_argument("--sigmas", nargs="+", type=float, default=list(SIGMAS))
    arguments = parser.parse_args(argv)

    chosen = [FAMILIES[key] for key in arguments.families]
    if not arguments.check:
        cells = [
            measure_truncation(family, sigma, floor=arguments.floor)
            for family in chosen
            for sigma in arguments.sigmas
        ]
        _print_table(cells)
        _print_width_ladder(chosen[0], max(arguments.sigmas), arguments.floor)
        return 0

    reports = run_checks(
        floor=arguments.floor, families=chosen, sigmas=arguments.sigmas
    )
    for report in reports:
        print(report)
    print(f"\n{check_summary(reports)}")
    print(f"\nreference provenance: {_PROVENANCE}, compared at 2 sig figs")
    return _exit_code(reports)


if __name__ == "__main__":
    sys.exit(main())
