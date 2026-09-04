"""The three tests step 10 of the hand derivation names.

`research/spinello_stilwell_hand_derivation.md` derives the modification and names what
each test is blind to. None of them is the rung, and none reports a warrant.

Running the module is checking it: every claim it prints is asserted inside it. These
call the same functions so a failure names the claim rather than the whole run.
"""

from itertools import pairwise

import pytest

from research.spinello_stilwell import repair, scheme


def test_it_runs_and_its_assertions_hold(capsys):
    repair.main()
    printed = capsys.readouterr().out
    assert "the printed single step is unit-dependent" in printed
    assert "reproduce the Kalman posterior in every unit choice" in printed


class TestUnitDependence:
    """Test 1. Blind to a fixed point that is wrong in every unit alike."""

    @pytest.mark.parametrize("log_block", [True, False])
    def test_the_converged_estimate_is_unit_free_either_way(self, log_block):
        # The notes predicted the printed scheme would fail here by 2 ln lambda. It does
        # not: run to convergence both variants land on the same estimate in every unit,
        # which is what route 1's numeric half already measured for the printed one.
        span = repair.estimate_spread(
            log_block=log_block, max_iterations=200, tolerance=1e-14
        )
        assert span < 1e-15, span

    def test_the_printed_single_step_is_not_unit_free(self):
        # Where the dependence actually bites. A budget of one is rung (36), which
        # ADR-056 declares separately, so this is a defect of a rung the ladder reports
        # rather than of a path nobody sees.
        span = repair.estimate_spread(log_block=True, max_iterations=1, tolerance=0.0)
        assert span > 1e-5, span

    def test_the_modified_single_step_is_unit_free(self):
        span = repair.estimate_spread(log_block=False, max_iterations=1, tolerance=0.0)
        assert span < 1e-15, span


class TestFixedNoiseReduction:
    """Test 2, route 4. Blind to every term carrying `grad_noise`."""

    @pytest.mark.parametrize("log_block", [True, False])
    @pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 7.0])
    def test_it_reproduces_the_kalman_posterior(self, log_block, scale):
        # The rung's only external check. Every figure in the paper's section IV uses
        # the single-step filter, so the iterated scheme has no published validation.
        mean_gap, variance_gap = repair.fixed_noise_departure(
            log_block=log_block, scale=scale
        )
        assert mean_gap < 1e-12, mean_gap
        assert variance_gap < 1e-12, variance_gap

    def test_the_printed_scheme_is_undefined_on_the_pole(self):
        # With `grad_noise = 0` the deleted block is zero over zero at `noise = 1`, so
        # the printed form cannot be evaluated in the one regime that has an oracle.
        assert repair.pole_is_reachable_at_fixed_noise()


class TestCurvatureAgreement:
    """Test 3. Blind to `noise < 1`, which is the regime in question."""

    def test_the_two_agree_as_the_noise_grows(self):
        departures = [repair.curvature_departure(n) for n in repair.GROWING_NOISE]
        assert all(later < earlier for earlier, later in pairwise(departures))
        assert departures[-1] < 1e-6, departures[-1]

    def test_the_printed_curvature_goes_negative_below_the_pole(self):
        # What the test cannot see, pinned so the blindness is visible rather than
        # inferred. A negative curvature is Q3's failure from below.
        assert scheme.gauss_newton_curvature(0.9, 2.0, 1.0, 0.3, log_block=True) < 0.0
        assert scheme.gauss_newton_curvature(0.9, 2.0, 1.0, 0.3, log_block=False) > 0.0


class TestTheScheme:
    """The shared implementation the two routes now share."""

    def test_the_real_square_is_the_first_three_printed_terms(self):
        # Step 5 of the derivation: the `r2` block collapses to `(1/noise) b b`, which
        # is why it is a square. Checked here rather than asserted in prose.
        noise, noise_slope, mean_slope, residual = 2.0, 0.7, 1.3, 0.4
        combined = mean_slope + (residual / (2.0 * noise)) * noise_slope
        modified = scheme.gauss_newton_curvature(
            noise, noise_slope, mean_slope, residual, log_block=False
        )
        assert abs(modified - combined**2 / noise) < 1e-14

    def test_the_fisher_information_ignores_the_modification(self):
        # (35e) is built from `s` alone, so no `ln noise` reaches it. This is why the
        # deletion cannot move the posterior covariance.
        assert scheme.fisher_information(0.9, 2.0, 1.0) > 0.0
        assert scheme.fisher_information(1.0, 2.0, 1.0) > 0.0

    def test_it_reports_no_warrant(self):
        for module in (repair, scheme):
            assert not hasattr(module, "run_checks")
            assert not any(name.endswith("_SOURCE") for name in dir(module))
