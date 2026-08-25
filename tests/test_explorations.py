import pytest

import research.explorations.averaged_gap_identity as gap_identity
import research.explorations.c6_window as window
import research.explorations.noise_model as noise
import research.explorations.operating_point as point
import research.explorations.sigma_max_edge as edge


def test_the_window_exploration_runs_and_its_assertions_hold(capsys):
    # The module asserts its two validations rather than only printing them, so running
    # it is the check. Without this the write-up's numbers rot silently.
    window.main()
    printed = capsys.readouterr().out
    assert "registered: D* = 0.5200" in printed
    assert "D moves by a factor of 0.895" in printed


@pytest.mark.parametrize("module", [window, noise, edge, point, gap_identity])
def test_no_exploration_reports_a_warrant(module):
    # The invariant is the package's, stated in explorations/__init__.py, so it is
    # checked over the package rather than over whichever module was written first.
    # A `run_checks` or a `_SOURCE` is the shape that would let one be collected.
    assert not hasattr(module, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(module))


def test_the_noise_model_exploration_runs_and_its_assertions_hold(capsys):
    # It checks the covariance integral against an actual fit, and that the fit does not
    # move with N. Both assertions live in the module; this runs them.
    noise.main()
    printed = capsys.readouterr().out
    assert "OVER BUDGET on its own" in printed
    assert "tracking the gap" in printed


def test_a_deterministic_error_does_not_average_away():
    # The claim the write-up rests on: a thousandfold increase in samples leaves the
    # fitted exponent where it was, where the random model would drop it 30-fold.
    k, decades = point.K_MIN, point.DECADES

    def constant_offset(v):
        return noise.relative_error(v, k)

    coarse = noise.fitted_shift(constant_offset, decades, 50)
    fine = noise.fitted_shift(constant_offset, decades, 50_000)
    assert abs(coarse - fine) < 5e-4, (coarse, fine)
    random_coarse = noise.independent_random(k, decades, 50)
    random_fine = noise.independent_random(k, decades, 50_000)
    assert random_coarse / random_fine > 25


def test_the_registered_sigma_p_is_unweighted():
    # The registration says heteroscedastic weights. Its four published values are
    # reproduced by unweighted OLS at one N near 60, and not by weighted at any.
    for k, decades, published in (
        (5.0, 0.4343, 0.089),
        (10.0, 0.4343, 0.045),
        (30.0, 0.4343, 0.015),
        (10.0, 0.520, 0.0359),
    ):
        plain = noise.unweighted_standard_error(k, decades, 60)
        weighted = noise.weighted_standard_error(k, decades, 60)
        assert abs(plain - published) / published < 0.03, (k, decades, plain)
        assert weighted < published / 2, (k, decades, weighted)


def test_the_random_formula_carries_a_decades_for_nats_slip():
    k, decades, samples = point.K_MIN, point.DECADES, 345
    written = noise.independent_random(k, decades, samples)
    corrected = noise.independent_random_corrected(k, decades, samples)
    exact = noise.unweighted_standard_error(k, decades, samples)
    assert abs(written / corrected - point.LN10) < 1e-9
    assert abs(corrected - exact) / exact < 0.01


def test_the_edge_exploration_runs_and_its_assertions_hold(capsys):
    edge.main()
    printed = capsys.readouterr().out
    assert "floor binds" in printed
    assert "c6 vanishes" in printed


def test_the_declared_floor_binds_under_either_edge():
    # T is evaluated at the kappa minimising the window, so which kappa that is decides
    # the number. Under both edges it is the floor, for every ceiling tried.
    for factor in (edge.quartic_window_factor, edge.sextic_window_factor):
        for ceiling in (0.5, 1.0, 4.0, 10.0):
            kappa, _ = edge.binding_kappa(factor, edge.KAPPA_MIN, ceiling)
            assert abs(kappa - edge.KAPPA_MIN) < 1e-4, (factor.__name__, ceiling, kappa)


def test_the_two_edges_never_fail_together():
    # c4 vanishes at 2 and c6 at 3/13, so one edge is always available.
    assert abs(edge.c4(edge.QUARTIC_ZERO)) < 1e-12
    assert abs(edge.c6(edge.SEXTIC_ZERO)) < 1e-12
    assert abs(edge.c6(edge.QUARTIC_ZERO)) > 1.0
    assert abs(edge.c4(edge.SEXTIC_ZERO)) > 0.01


def test_the_gap_identity_exploration_runs_and_its_assertions_hold(capsys):
    # Three routes to one formula, each asserted inside the module. This is what
    # stands behind treating the Gaussian averaged gap as a closed form at all.
    gap_identity.main()
    printed = capsys.readouterr().out
    assert "45 triples" in printed
    assert "vanishes identically at R_plug = R_true: True" in printed
    assert "sympy decides non-negativity: None" in printed


def test_the_identity_is_general_in_the_observation_matrix():
    # The hypothesis that bounds what may be claimed. A scalar-only identity and one
    # general in C are different universals, and only the second licences a general
    # symbolic check. The non-square case is the one that decides it.
    shapes = {case[2].shape for case in gap_identity.MULTIVARIATE_CASES}
    assert (1, 2) in shapes, "a non-square C must be among the cases"
    assert (2, 2) in shapes


def test_a_correct_rule_has_exactly_zero_gap():
    # Not merely small. The expression cancels, which is why the calibration test in
    # tests/test_reference_gap.py is entitled to assert a hard zero.
    assert gap_identity.averaged_gap(0.7, 0.7, 1.3) == 0.0
