import research.explorations.c6_window as window
import research.explorations.noise_model as noise


def test_the_window_exploration_runs_and_its_assertions_hold(capsys):
    # The module asserts its two validations rather than only printing them, so running
    # it is the check. Without this the write-up's numbers rot silently.
    window.main()
    printed = capsys.readouterr().out
    assert "registered: D* = 0.5200" in printed
    assert "D moves by a factor of 0.895" in printed


def test_the_exploration_reports_no_warrant():
    # It is not a check suite and must not look like one: nothing here may be collected
    # into the manifest or read as carrying a warrant.
    assert not hasattr(window, "run_checks")
    assert not any(name.endswith("_SOURCE") for name in dir(window))


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
    k, decades = 10.0, 0.520
    offset = noise.systematic_offset  # noqa: F841  named for the reader
    field = lambda point: noise._relative_error(point, k)  # noqa: E731
    coarse = noise.fitted_shift(field, decades, 50)
    fine = noise.fitted_shift(field, decades, 50_000)
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
    k, decades, samples = 10.0, 0.520, 345
    written = noise.independent_random(k, decades, samples)
    corrected = noise.independent_random_corrected(k, decades, samples)
    exact = noise.unweighted_standard_error(k, decades, samples)
    assert abs(written / corrected - noise.LN10) < 1e-9
    assert abs(corrected - exact) / exact < 0.01
