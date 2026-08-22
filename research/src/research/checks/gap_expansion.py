"""Structure of the gap's small-spread expansion, without its coefficients.

The gap expands in prior spread as::

    gap(σ) = c₂σ² + c₄σ⁴ + c₆σ⁶ + O(σ⁸)

`c₂` is derived and published: `(R'(μ)/2R(μ))²`, in the registration's RESULT
2026-08-07. `c₄` has a closed form too, in RESULT 2026-08-16, which AMENDMENT
2026-08-17 pins to the reverse KL direction. `c₆` is unmeasured.

**This module does not fit `c₄` or `c₆`, and that is a constraint rather than an
omission.** A number extracted here before the derivation lands becomes the thing the
derivation gets checked against, and then the warrant ledger carries a `COMPUTED` fit
wearing a Prover 1 label. The derivation has to be primary. So what is measured here
is *structure*, which a derivation must reproduce and which cannot stand in for one:

- the gap itself, certified on three axes
- `c₂` against its closed form, since that one is already derived
- the residual's **exponent** after each known term comes off, never its coefficient
- the gap vanishing identically under fixed `R`

Pass a derived value in, one family at a time, and G4 tries to refute it::

    uv run --no-sync python -m research.checks.gap_expansion --check \
        --families tanh --c4 0.0061107361819873193

`--families` defaults to all five, and a single `--c4` applied to a run of several
mis-tests all but one, so the parser refuses that combination. Supplying a candidate
is the only way a `c₄` figure enters this module. It never produces one.

**The candidate needs fifteen significant figures or more**, and the registration makes
a cell measured with a rounded one VOID rather than passed or fired. The value above is
the closed form evaluated for `tanh` at `μ = 1`, tabled in the registration. Rounding it
to the five figures that document also tables reads `σ^6.302` against a `σ^6` claim and
a ±0.25 bar, where the full-precision candidate reads `σ^6.148` and clears it. `tanh` is
where this bites: its residual is the only one within an order of magnitude of the
quadrature floor.

Why an exponent test is a sharp falsifier. Subtract a candidate `ĉ₄` and the residual is
`(c₄ − ĉ₄)σ⁴ + c₆σ⁶`. If the candidate is exact the quartic cancels and the residual
scales as σ⁶. If it is wrong by any amount, the σ⁴ term survives and dominates as σ → 0,
so the measured slope falls back toward 4. The test reads the exponent alone, so it can
refute a candidate without ever computing the coefficient it would have to be.

Without a candidate, G4 reports ``NOT_RUN_HERE`` and says what it is waiting for::

    uv run --no-sync python -m research.checks.gap_expansion --check
    uv run --no-sync python -m research.checks.gap_expansion

Outcome mapping and the ``CORROBORATED`` warrant follow ``predictive_truncation``.

``log_ratio_series`` is the symbolic counterpart. It proves the gap carries nothing at
`σ⁰` or `σ¹`, which is *why* G4a finds a quartic residual after `c₂σ²` comes off. That
module supplies the reason and this one the measurement. Neither produces a coefficient.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from research.checks.gap_kernel import (
    FAMILIES,
    QUADRATURE_FLOOR,
    TAIL_EXTENT,
    NoiseFamily,
    Quadrature,
    VoidReason,
    assert_positive_noise,
    core_and_tail,
    plugin_noise_of,
    sigma_slug,
)
from warrantlib import CheckReport, Outcome, Tier, Warrant, check_summary

__all__ = [
    "EXPANSION_SIGMAS",
    "GapMeasurement",
    "closed_form_c2",
    "measure_gap",
    "run_checks",
]

#: The `σ` grid the expansion is read on. Small, because every check here is a statement
#: about the σ → 0 limit. This is not the registered derivation range `[0.06, 0.30]`,
#: which exists to condition a *fit*; nothing here fits.
EXPANSION_SIGMAS: tuple[float, ...] = (0.02, 0.025, 0.03, 0.035, 0.04, 0.05)

#: Half-width of the `y` grid these measurements run on, in plug-in predictive standard
#: deviations. Thirteen rather than the production rule's nine because
#: ``predictive_truncation`` C1 fires at nine, and a structural check must not be
#: limited by a truncation it is not measuring. This is the measurement grid for the
#: expansion. It is not a change to the registered production rule, which is a decision
#: for whoever amends the registration.
MEASUREMENT_MULTIPLIER = 13.0

#: The exponent the residual must show once every known term has come off. Structural,
#: from the expansion being even in `σ`, so a derivation cannot argue with it.
QUARTIC_EXPONENT = 4.0
SEXTIC_EXPONENT = 6.0

#: How far a measured log-log slope may sit from its predicted integer. Wide enough that
#: quadrature noise at the small-σ end does not fire it, narrow enough that a candidate
#: `c₄` wrong by a percent does. Passed in at every call site that can vary it.
_CLI_DEFAULT_SLOPE_TOLERANCE = 0.25

#: The relative agreement `c₂` must show against its closed form.
_CLI_DEFAULT_C2_TOLERANCE = 1e-3

#: What "identically zero" means for the fixed-`R` family, in absolute nats.
_CLI_DEFAULT_ZERO_TOLERANCE = 1e-13

#: The certification bar each of G3's three axes is checked against.
_CLI_DEFAULT_CERTIFICATION = 1e-10


def closed_form_c2(family: NoiseFamily) -> float:
    """`c₂ = (R'(μ)/2R(μ))²`, half the log-derivative of `R` at the prior mean, squared.

    Derived and published in the registration's RESULT 2026-08-07, verified there on
    five families. Reproduced from the family's declared log-derivative rather than
    re-derived, so this arm of the `c₂` check is independent of the quadrature.

    Args:
        family: the declared `R` and its prior mean.

    Returns:
        The quadratic coefficient of the expansion.

    Raises:
        ValueError: if the family declares no log-derivative.
    """
    if family.log_noise_derivative is None:
        raise ValueError(
            f"family {family.name!r} declares no log-derivative, so its closed-form c₂ "
            "cannot be formed. Declare `log_noise_derivative` or drop it from the c₂ "
            "check rather than differentiating it here."
        )
    return (family.log_noise_derivative / 2.0) ** 2


@dataclass(frozen=True)
class GapMeasurement:
    """The gap at one `(family, σ)`, with what it is certified to on three axes.

    Args:
        sigma: σ — the prior standard deviation.
        family: the family's name, as it prints.
        gap: `E_{y∼p*}[KL(q ‖ p(·|y))]` in nats.
        relative_truncation: what the `y` grid throws away, relative to the total.
        tolerance_sensitivity: relative move when the quadrature tolerances tighten.
        extent_sensitivity: relative move when the inner `x` window doubles.
        binding_axis: which of the three is largest.
        binding_error: that largest relative error.
        outcome: fired when the binding error is above the certification bar.
        void_reason: why there is no measurement, or ``None``.
    """

    sigma: float
    family: str
    gap: float
    relative_truncation: float
    tolerance_sensitivity: float
    extent_sensitivity: float
    binding_axis: str
    binding_error: float
    outcome: Outcome
    void_reason: VoidReason | None


def measure_gap(
    family: NoiseFamily,
    sigma: float,
    *,
    certification: float,
    zero_floor: float = _CLI_DEFAULT_ZERO_TOLERANCE,
    multiplier: float = MEASUREMENT_MULTIPLIER,
    settings: Quadrature | None = None,
) -> GapMeasurement:
    """The gap at one `(family, σ)`, certified on tolerance, `x` extent and `y` extent.

    Three axes rather than one summary figure. The registration disclosed its
    convergence as a single "8.7 digits", measured by refining the `x` grid at a fixed
    span, which cannot see a truncation at all. Reporting the axes separately is what
    makes the binding one visible.

    Args:
        family: the declared `R` and its prior mean.
        sigma: σ — the prior standard deviation.
        certification: the bar each axis is checked against. Required, and without a
            default, on the same grounds as ``measure_truncation``'s floor.
        zero_floor: below this the gap is treated as vanishing and the relative axes as
            undefined, rather than reporting a ratio against numerical noise.
        multiplier: the `y` half-width, in plug-in predictive standard deviations.
        settings: quadrature tolerances. Defaults to the kernel's.

    Returns:
        The measurement and what binds it.
    """
    settings = settings or Quadrature()
    prior_variance = sigma**2
    half_width = multiplier * math.sqrt(prior_variance + plugin_noise_of(family))
    centre = family.prior_mean
    span = (centre - half_width - TAIL_EXTENT, centre + half_width + TAIL_EXTENT)

    try:
        assert_positive_noise(family, span)
    except ValueError:
        return GapMeasurement(
            sigma=sigma,
            family=family.name,
            gap=math.nan,
            relative_truncation=math.nan,
            tolerance_sensitivity=math.nan,
            extent_sensitivity=math.nan,
            binding_axis="none",
            binding_error=math.nan,
            outcome=Outcome.NOT_APPLICABLE,
            void_reason=VoidReason.NON_POSITIVE_NOISE,
        )

    core, tail = core_and_tail(family, prior_variance, half_width, settings)
    total = core + tail
    if abs(total) < zero_floor:
        # Every axis here is a *relative* error, and a relative error against zero is
        # not a large number, it is an undefined one. The fixed-R family lands here by
        # construction. Its gap is still carried, since G2's claim is about that value.
        return GapMeasurement(
            sigma=sigma,
            family=family.name,
            gap=total,
            relative_truncation=math.nan,
            tolerance_sensitivity=math.nan,
            extent_sensitivity=math.nan,
            binding_axis="none",
            binding_error=math.nan,
            outcome=Outcome.NOT_APPLICABLE,
            void_reason=VoidReason.NUMERICAL_FLOOR,
        )
    # A factor of ten. The kernel already runs at epsrel 1e-12, and
    # 1e-14 sits close enough to machine precision that quad subdivides for hours
    # without buying a digit. The refinement has to be a check on the reported number,
    # not a different and slower computation of it.
    tighter = Quadrature(
        epsabs=settings.epsabs / 10.0,
        epsrel=settings.epsrel / 10.0,
        limit=settings.limit * 2,
    )
    wider = Quadrature(
        epsabs=settings.epsabs,
        epsrel=settings.epsrel,
        x_window_scale=settings.x_window_scale * 2.0,
        limit=settings.limit,
    )
    axes = {
        "y extent": abs(tail) / abs(total) if total else 0.0,
        "tolerance": _relative_move(total, family, prior_variance, half_width, tighter),
        "x extent": _relative_move(total, family, prior_variance, half_width, wider),
    }
    binding_axis = max(axes, key=lambda axis: axes[axis])
    binding_error = axes[binding_axis]
    return GapMeasurement(
        sigma=sigma,
        family=family.name,
        gap=total,
        relative_truncation=axes["y extent"],
        tolerance_sensitivity=axes["tolerance"],
        extent_sensitivity=axes["x extent"],
        binding_axis=binding_axis,
        binding_error=binding_error,
        outcome=(
            Outcome.FIRED if binding_error > certification else Outcome.NOT_TRIGGERED
        ),
        void_reason=None,
    )


def _relative_move(
    baseline: float,
    family: NoiseFamily,
    prior_variance: float,
    half_width: float,
    settings: Quadrature,
) -> float:
    """How far the gap moves under one refinement, relative to itself.

    Args:
        baseline: the gap at the reference settings.
        family: the declared `R` and its prior mean.
        prior_variance: σ² — the prior variance.
        half_width: L — the `y` half-width.
        settings: the refined settings to compare against.

    Returns:
        The relative move.
    """
    core, tail = core_and_tail(family, prior_variance, half_width, settings)
    return abs((core + tail) - baseline) / abs(baseline) if baseline else 0.0


def _log_log_slope(sigmas: np.ndarray, magnitudes: np.ndarray) -> float:
    """The exponent in `|residual| ∝ σ^p`, by least squares on the logs.

    Args:
        sigmas: the spreads measured at.
        magnitudes: the residual magnitudes.

    Returns:
        The fitted exponent.
    """
    slope, _ = np.polyfit(np.log(sigmas), np.log(magnitudes), 1)
    return float(slope)


def _check_preconditions(family: NoiseFamily) -> CheckReport:
    """G0: the fixture preconditions the registration records.

    `R > 0` across the span, and `R'(μ) ≠ 0` so `c₂` does not vanish. The second is
    violated on purpose by the fixed-`R` family, which is the point of it.

    Args:
        family: the declared `R` and its prior mean.

    Returns:
        The check's report.
    """
    name = f"G0 preconditions [{family.name}]"
    check_id = f"gap_expansion.preconditions_{family.key}"
    widest = max(EXPANSION_SIGMAS)
    half_width = MEASUREMENT_MULTIPLIER * math.sqrt(widest**2 + plugin_noise_of(family))
    centre = family.prior_mean
    span = (centre - half_width - TAIL_EXTENT, centre + half_width + TAIL_EXTENT)
    try:
        assert_positive_noise(family, span)
    except ValueError as failure:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=Warrant.CORROBORATED,
            outcome=Outcome.FIRED,
            tier=Tier.BOUNDED,
            detail=f"FAIL — {failure}",
        )
    derivative = family.log_noise_derivative
    if derivative is not None and derivative == 0.0:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=Warrant.CORROBORATED,
            outcome=Outcome.NOT_TRIGGERED,
            tier=Tier.BOUNDED,
            detail=(
                "PASS — R > 0 on the span; R'(μ) = 0 by construction, so c₂ = 0 "
                "and the gap is expected to vanish identically. Checked by G2"
            ),
        )
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.NOT_TRIGGERED,
        tier=Tier.BOUNDED,
        detail=(
            f"PASS — R > 0 on [{span[0]:.2f}, {span[1]:.2f}], "
            f"ℓ'(μ) = {derivative:.6f} ≠ 0"
        ),
    )


def _check_certification(
    family: NoiseFamily, measurement: GapMeasurement, certification: float
) -> CheckReport:
    """G3: is the gap converged on all three axes at this `σ`?

    Args:
        family: the declared `R` the cell was measured on.
        measurement: the measured cell.
        certification: the bar each axis is checked against.

    Returns:
        The check's report.
    """
    name = f"G3 certification [{measurement.family}, σ={measurement.sigma:.3f}]"
    check_id = (
        f"gap_expansion.certification_{family.key}_sigma{sigma_slug(measurement.sigma)}"
    )
    if measurement.void_reason is not None:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=f"VOID — {measurement.void_reason.value}",
        )
    fired = measurement.outcome is Outcome.FIRED
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=measurement.outcome,
        tier=Tier.BOUNDED,
        detail=(
            f"{'FAIL' if fired else 'PASS'} — gap {measurement.gap:.10e}; binding axis "
            f"{measurement.binding_axis} at {measurement.binding_error:.2e} against "
            f"{certification:.1e} (y extent {measurement.relative_truncation:.1e}, "
            f"tolerance {measurement.tolerance_sensitivity:.1e}, x extent "
            f"{measurement.extent_sensitivity:.1e})"
        ),
    )


def _check_c2(
    family: NoiseFamily, measurements: Sequence[GapMeasurement], tolerance: float
) -> CheckReport:
    """G1: the measured `c₂` against `(R'(μ)/2R(μ))²`.

    Read by Richardson extrapolation of `gap/σ²` on the two smallest spreads, which
    cancels the quartic term. `c₂` is the one coefficient this module computes, because
    it is the one already derived: checking it validates the quadrature against a known
    answer rather than producing a new one.

    Args:
        family: the declared `R` and its prior mean.
        measurements: that family's cells, smallest σ first.
        tolerance: the relative agreement required.

    Returns:
        The check's report.
    """
    name = f"G1 c₂ closed form [{family.name}]"
    check_id = f"gap_expansion.c2_closed_form_{family.key}"
    if family.log_noise_derivative is None:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail="no closed form declared for this family",
        )
    expected = closed_form_c2(family)
    if expected == 0.0:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail="c₂ = 0 by construction; the vanishing gap is G2's claim",
        )
    usable = [cell for cell in measurements if cell.void_reason is None]
    if len(usable) < 2:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=f"only {len(usable)} usable cells; Richardson needs two",
        )
    first, second = usable[0], usable[1]
    ratio_first = first.gap / first.sigma**2
    ratio_second = second.gap / second.sigma**2
    # gap/σ² = c₂ + c₄σ² + O(σ⁴). Two points kill the σ² term and leave O(σ⁴).
    measured = (second.sigma**2 * ratio_first - first.sigma**2 * ratio_second) / (
        second.sigma**2 - first.sigma**2
    )
    error = abs(measured / expected - 1.0)
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if error > tolerance else Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail=(
            f"{'FAIL' if error > tolerance else 'PASS'} — quadrature {measured:.8f} "
            f"against closed form {expected:.8f}, {error:.2e} relative, bar "
            f"{tolerance:.0e}"
        ),
    )


def _check_vanishing(
    family: NoiseFamily, measurements: Sequence[GapMeasurement], tolerance: float
) -> CheckReport:
    """G2: under fixed `R` the Kalman filter is exact, so the gap is identically zero.

    R1's content, and a falsifier of this implementation rather than of the world. The
    registration records a method that computed exactly this quantity and was therefore
    measuring nothing, so a suite that cannot detect the degenerate case would repeat
    that mistake silently.

    Args:
        family: the declared `R` and its prior mean.
        measurements: that family's cells.
        tolerance: what counts as zero, in absolute nats.

    Returns:
        The check's report.
    """
    name = f"G2 vanishing gap [{family.name}]"
    check_id = f"gap_expansion.vanishing_gap_{family.key}"
    if family.log_noise_derivative != 0.0:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.EXACT,
            detail="R varies, so the gap is not expected to vanish",
        )
    worst = max((abs(cell.gap) for cell in measurements), default=math.nan)
    fired = not (worst < tolerance)
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if fired else Outcome.NOT_TRIGGERED,
        tier=Tier.EXACT,
        detail=(
            f"{'FAIL' if fired else 'PASS'} — largest |gap| {worst:.2e} across "
            f"{len(measurements)} spreads, against {tolerance:.0e}"
        ),
    )


def _check_residual_exponent(
    family: NoiseFamily,
    measurements: Sequence[GapMeasurement],
    tolerance: float,
    c4_candidate: float | None,
    c6_candidate: float | None,
) -> list[CheckReport]:
    """G4: the residual's exponent after each known term comes off.

    Two legs. The first subtracts the closed-form `c₂σ²` and requires what is left to
    scale as σ⁴, which says the leading correction is quartic without saying what its
    coefficient is. The second runs only when a candidate `c₄` is supplied, and requires
    the residual after subtracting it to scale as σ⁶.

    Only the exponent is reported. The intercept of the same fit would be the
    coefficient, and printing it is what this module exists not to do.

    Args:
        family: the declared `R` and its prior mean.
        measurements: that family's cells.
        tolerance: how far the slope may sit from its predicted integer.
        c4_candidate: a derived `c₄` to try to refute, or ``None``.
        c6_candidate: a derived `c₆`, used only alongside a `c₄`.

    Returns:
        One report per leg.
    """
    name = f"G4 residual exponent [{family.name}]"
    check_id = f"gap_expansion.residual_exponent_{family.key}"
    if family.log_noise_derivative is None or closed_form_c2(family) == 0.0:
        return [
            CheckReport(
                name=name,
                check_id=check_id,
                warrant=None,
                outcome=Outcome.NOT_APPLICABLE,
                tier=Tier.BOUNDED,
                detail="the expansion vanishes; there is no residual to read",
            )
        ]
    usable = [cell for cell in measurements if cell.void_reason is None]
    if len(usable) < 3:
        return [
            CheckReport(
                name=name,
                check_id=check_id,
                warrant=None,
                outcome=Outcome.NOT_APPLICABLE,
                tier=Tier.BOUNDED,
                detail=f"only {len(usable)} usable cells; a slope needs three",
            )
        ]
    sigmas = np.array([cell.sigma for cell in usable])
    gaps = np.array([cell.gap for cell in usable])
    quadratic = closed_form_c2(family) * sigmas**2

    reports = [
        _exponent_report(
            name=f"G4a quartic leading [{family.name}]",
            check_id=f"gap_expansion.quartic_leading_{family.key}",
            sigmas=sigmas,
            residual=gaps - quadratic,
            expected=QUARTIC_EXPONENT,
            tolerance=tolerance,
            subtracted="c₂σ²",
        )
    ]
    if c4_candidate is None:
        reports.append(
            CheckReport(
                name=f"G4b candidate c₄ [{family.name}]",
                check_id=f"gap_expansion.candidate_c4_{family.key}",
                warrant=None,
                outcome=Outcome.NOT_RUN_HERE,
                tier=Tier.BOUNDED,
                detail=(
                    "no candidate declared. Pass --c4 with the closed form evaluated "
                    "for this family and this leg tries to refute it"
                ),
            )
        )
        return reports

    residual = gaps - quadratic - c4_candidate * sigmas**4
    subtracted = "c₂σ² + ĉ₄σ⁴"
    if c6_candidate is not None:
        residual = residual - c6_candidate * sigmas**6
        subtracted += " + ĉ₆σ⁶"
    reports.append(
        _exponent_report(
            name=f"G4b candidate c₄ [{family.name}]",
            check_id=f"gap_expansion.candidate_c4_{family.key}",
            sigmas=sigmas,
            residual=residual,
            expected=SEXTIC_EXPONENT + (2.0 if c6_candidate is not None else 0.0),
            tolerance=tolerance,
            subtracted=subtracted,
        )
    )
    reports.append(
        _stability_report(
            name=f"G4c exponent stability [{family.name}]",
            check_id=f"gap_expansion.exponent_stability_{family.key}",
            sigmas=sigmas,
            residual=residual,
            tolerance=tolerance,
        )
    )
    return reports


def _stability_report(
    *,
    name: str,
    check_id: str,
    sigmas: np.ndarray,
    residual: np.ndarray,
    tolerance: float,
) -> CheckReport:
    """G4c: how far G4b's exponent moves when one `σ` cell is dropped.

    A diagnostic, not a falsifier of the candidate. It asks whether the exponent G4b
    read is a property of the residual or of the grid it was read on, by refitting once
    per omitted cell and reporting the spread. Every refit uses the whole declared grid
    minus one point, so no window is selected and nothing here revises G4b's outcome.

    The rule and its readings were registered in `research/gate_d4_registration.md`
    before this function existed. A spread above the same bar G4b uses means the
    exponent is not stable on this grid for this family, and a fired G4b there is a
    statement about the measurement rather than about the coefficient.

    Args:
        name: the check's name.
        check_id: the check's key, as a manifest and a ledger name it.
        sigmas: the spreads measured at.
        residual: what is left after the subtraction.
        tolerance: the spread bar, shared with G4b so the two are commensurable.

    Returns:
        The diagnostic's report.
    """
    magnitudes = np.abs(residual)
    # The registered readings were declared against EXPANSION_SIGMAS literally. On any
    # other grid the spread is a number this document's rule does not interpret.
    if tuple(np.round(sigmas, 12)) != tuple(np.round(sorted(EXPANSION_SIGMAS), 12)):
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=(
                "VOID — the registered readings are declared against the grid "
                f"{tuple(sorted(EXPANSION_SIGMAS))}, and this ran on "
                f"{tuple(float(s) for s in sigmas)}"
            ),
        )
    if len(sigmas) < 4 or float(np.min(magnitudes)) < QUADRATURE_FLOOR:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=(
                f"VOID — a leave-one-out spread needs four cells above the floor; "
                f"{len(sigmas)} declared, minimum residual "
                f"{float(np.min(magnitudes)):.1e}"
            ),
        )
    slopes = [
        _log_log_slope(np.delete(sigmas, index), np.delete(magnitudes, index))
        for index in range(len(sigmas))
    ]
    spread = float(max(slopes) - min(slopes))
    unstable = spread > tolerance
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if unstable else Outcome.NOT_TRIGGERED,
        tier=Tier.BOUNDED,
        detail=(
            f"{'FAIL' if unstable else 'PASS'} — leaving out one σ cell moves the "
            f"exponent over a spread of {spread:.3f} (bar {tolerance:.2f}), "
            f"range σ^{min(slopes):.3f} to σ^{max(slopes):.3f}"
        ),
    )


def _exponent_report(
    *,
    name: str,
    check_id: str,
    sigmas: np.ndarray,
    residual: np.ndarray,
    expected: float,
    tolerance: float,
    subtracted: str,
) -> CheckReport:
    """One exponent leg, as a report.

    Args:
        name: the check's name.
        check_id: the check's key, as a manifest and a ledger name it.
        sigmas: the spreads measured at.
        residual: what is left after the subtraction.
        expected: the exponent the structure predicts.
        tolerance: how far the measured slope may sit from it.
        subtracted: what came off, for the detail line.

    Returns:
        The check's report.
    """
    magnitudes = np.abs(residual)
    if bool(np.any(magnitudes <= 0.0)) or float(np.min(magnitudes)) < QUADRATURE_FLOOR:
        return CheckReport(
            name=name,
            check_id=check_id,
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=(
                f"VOID — {VoidReason.NUMERICAL_FLOOR.value}: the residual after "
                f"{subtracted} reaches {float(np.min(magnitudes)):.1e}"
            ),
        )
    slope = _log_log_slope(sigmas, magnitudes)
    drift = abs(slope - expected)
    fired = drift > tolerance
    return CheckReport(
        name=name,
        check_id=check_id,
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if fired else Outcome.NOT_TRIGGERED,
        tier=Tier.BOUNDED,
        detail=(
            f"{'FAIL' if fired else 'PASS'} — after {subtracted} the residual "
            f"scales as σ^{slope:.3f}, against a predicted σ^{expected:.0f} "
            f"({drift:.3f} off, bar {tolerance:.2f})"
        ),
    )


def run_checks(
    *,
    certification: float,
    slope_tolerance: float,
    c2_tolerance: float,
    zero_tolerance: float,
    c4_candidate: float | None = None,
    c6_candidate: float | None = None,
    families: Sequence[NoiseFamily] = tuple(FAMILIES.values()),
    sigmas: Sequence[float] = EXPANSION_SIGMAS,
) -> list[CheckReport]:
    """Run G0 to G4 across the families and the expansion `σ` grid.

    Args:
        certification: the bar each of G3's three axes is checked against.
        slope_tolerance: how far a measured exponent may sit from its integer.
        c2_tolerance: the relative agreement `c₂` must show against its closed form.
        zero_tolerance: what counts as zero for the fixed-`R` family.
        c4_candidate: a derived `c₄` for G4b to try to refute, or ``None``.
        c6_candidate: a derived `c₆`, used only alongside a `c₄`.
        families: the declared families to run.
        sigmas: the prior standard deviations to run at, smallest first.

    Returns:
        Every check's report, in check order.
    """
    ordered = sorted(sigmas)
    reports: list[CheckReport] = []
    for family in families:
        cells = [
            measure_gap(
                family, sigma, certification=certification, zero_floor=zero_tolerance
            )
            for sigma in ordered
        ]
        reports.append(_check_preconditions(family))
        reports += [_check_certification(family, cell, certification) for cell in cells]
        reports.append(_check_c2(family, cells, c2_tolerance))
        reports.append(_check_vanishing(family, cells, zero_tolerance))
        reports += _check_residual_exponent(
            family, cells, slope_tolerance, c4_candidate, c6_candidate
        )
    if c4_candidate is not None:
        reports.append(_control_report(reports, slope_tolerance))
    return reports


def _control_report(reports: Sequence[CheckReport], tolerance: float) -> CheckReport:
    """The registered control on G4c: whether any family's exponent was stable.

    The pre-registration reads a `tanh` spread below the bar as a real deviation *only*
    if the diagnostic discriminates at all. Its uninformative branch fires when every
    family that produced a spread read unstable, and until now that branch lived in
    prose with nothing in code to evaluate it.

    A candidate run is one family at a time, since the parser refuses `--c4` across
    several, so this reports on one family per run rather than on the four the
    registration tabulates together. The registration's four-family reading is the
    union of four such runs, not one invocation of this function.

    Args:
        reports: the run's reports so far, G4c's among them.
        tolerance: the spread bar G4c used.

    Returns:
        The control's report, one per run.
    """
    stability = [report for report in reports if report.name.startswith("G4c ")]
    ran = [
        report
        for report in stability
        if report.outcome in (Outcome.NOT_TRIGGERED, Outcome.FIRED)
    ]
    if not ran:
        return CheckReport(
            name="G4c control",
            check_id="gap_expansion.control",
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.BOUNDED,
            detail=f"VOID — no family produced a spread ({len(stability)} attempted)",
        )
    unstable = [report for report in ran if report.outcome is Outcome.FIRED]
    return CheckReport(
        name="G4c control",
        check_id="gap_expansion.control",
        warrant=Warrant.CORROBORATED,
        outcome=Outcome.FIRED if len(unstable) == len(ran) else Outcome.NOT_TRIGGERED,
        tier=Tier.BOUNDED,
        detail=(
            f"FAIL — every family read unstable at bar {tolerance:.2f}, so a spread "
            "below it says nothing about any one of them"
            if len(unstable) == len(ran)
            else f"PASS — {len(ran) - len(unstable)} of {len(ran)} families read "
            f"stable at bar {tolerance:.2f}, so the diagnostic discriminates"
        ),
    )


def _print_table(measurements: Sequence[GapMeasurement]) -> None:
    """Print the measured gaps and what certifies them.

    No residual column and no ratio column. Either would converge to `c₄`, which is what
    this module withholds until the derivation is primary.

    Args:
        measurements: the cells to print.
    """
    print(
        f"{'family':<20} {'σ':>7} {'gap':>18} {'y extent':>10} {'tol':>10} "
        f"{'x extent':>10}  binding"
    )
    for cell in measurements:
        print(
            f"{cell.family:<20} {cell.sigma:>7.3f} {cell.gap:>18.12e} "
            f"{cell.relative_truncation:>10.1e} {cell.tolerance_sensitivity:>10.1e} "
            f"{cell.extent_sensitivity:>10.1e}  {cell.binding_axis}"
        )


def _exit_code(reports: Sequence[CheckReport]) -> int:
    """Zero when nothing fired, one on a firing, two when something went unmeasured.

    ``NOT_RUN_HERE`` does not reach the exit code. A run with no ``--c4`` leaves G4b
    unmeasured by construction, which is a run that asked less, not a run that failed.

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
        description="Structure of the gap's small-spread expansion, without its "
        "coefficients."
    )
    parser.add_argument("--check", action="store_true", help="run the check suite")
    parser.add_argument(
        "--c4",
        type=float,
        default=None,
        help=(
            "a derived c₄ for G4b to try to refute, at fifteen significant figures or "
            "more. A rounded candidate makes the cell VOID"
        ),
    )
    parser.add_argument(
        "--c6", type=float, default=None, help="a derived c₆, used only with --c4"
    )
    parser.add_argument(
        "--certification", type=float, default=_CLI_DEFAULT_CERTIFICATION
    )
    parser.add_argument(
        "--slope-tolerance", type=float, default=_CLI_DEFAULT_SLOPE_TOLERANCE
    )
    parser.add_argument("--c2-tolerance", type=float, default=_CLI_DEFAULT_C2_TOLERANCE)
    parser.add_argument(
        "--zero-tolerance", type=float, default=_CLI_DEFAULT_ZERO_TOLERANCE
    )
    parser.add_argument(
        "--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES)
    )
    parser.add_argument(
        "--sigmas", nargs="+", type=float, default=list(EXPANSION_SIGMAS)
    )
    arguments = parser.parse_args(argv)

    if arguments.c6 is not None and arguments.c4 is None:
        parser.error("--c6 needs --c4: a sextic candidate cannot be tested on its own")

    # A wrong candidate fires G4b near σ^4 and gives G4c a spread of 0.000, which reads
    # as a stable real deviation rather than as the mistyped command it is.
    if arguments.c4 is not None and len(arguments.families) > 1:
        parser.error(
            f"--c4 takes one family, got {len(arguments.families)}: "
            f"{', '.join(arguments.families)}. Each family has its own c₄. "
            "Re-run with --families <one>."
        )

    chosen = [FAMILIES[key] for key in arguments.families]
    if not arguments.check:
        cells = [
            measure_gap(
                family,
                sigma,
                certification=arguments.certification,
                zero_floor=arguments.zero_tolerance,
            )
            for family in chosen
            for sigma in sorted(arguments.sigmas)
        ]
        _print_table(cells)
        return 0

    reports = run_checks(
        certification=arguments.certification,
        slope_tolerance=arguments.slope_tolerance,
        c2_tolerance=arguments.c2_tolerance,
        zero_tolerance=arguments.zero_tolerance,
        c4_candidate=arguments.c4,
        c6_candidate=arguments.c6,
        families=chosen,
        sigmas=arguments.sigmas,
    )
    for report in reports:
        print(report)
    print(f"\n{check_summary(reports)}")
    if arguments.c4 is None:
        print("\nG4b ran with no candidate. No coefficient is fitted here.")
    return _exit_code(reports)


if __name__ == "__main__":
    sys.exit(main())
