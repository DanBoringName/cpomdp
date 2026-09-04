"""The two divergences a cell is scored by, and the shape that keeps them honest."""

from dataclasses import fields

import numpy as np
import pytest

from cpomdp.scoring import _SERIES_BELOW, Decomposition, _excess_over_log, gaussian_kl


def test_the_type_carries_the_two_divergences_and_nothing_else():
    # Standing prohibition 1: never obtain a term by subtracting H(p*). No entropy
    # field, no estimator slot, no total. Adding one is a deliberate edit here first.
    assert [f.name for f in fields(Decomposition)] == [
        "misspecification",
        "inference_gap",
    ]


# --- the divergence both terms are built from ----------------------------------------


def _naive_gaussian_kl(mean_a, cov_a, mean_b, cov_b):
    """The textbook form, subtractions included, as the oracle."""
    n = len(mean_a)
    precision_b = np.linalg.inv(cov_b)
    shift = mean_a - mean_b
    return 0.5 * (
        np.trace(precision_b @ cov_a)
        - n
        + np.linalg.slogdet(cov_b)[1]
        - np.linalg.slogdet(cov_a)[1]
        + shift @ precision_b @ shift
    )


MEAN_A = np.array([0.3, -1.2, 2.0])
COV_A = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.5]])
MEAN_B = np.array([0.0, -1.0, 2.5])
COV_B = np.array([[1.5, -0.1, 0.0], [-0.1, 1.3, 0.4], [0.0, 0.4, 0.9]])


def test_identical_gaussians_diverge_by_exactly_zero():
    assert gaussian_kl(MEAN_A, COV_A, MEAN_A, COV_A) == 0.0


def test_the_scalar_closed_form():
    mean_a, var_a, mean_b, var_b = 0.4, 0.8, -0.1, 1.7
    expected = (
        0.5 * np.log(var_b / var_a)
        + (var_a + (mean_a - mean_b) ** 2) / (2 * var_b)
        - 0.5
    )
    assert gaussian_kl([mean_a], [[var_a]], [mean_b], [[var_b]]) == pytest.approx(
        expected, abs=1e-14
    )


def test_matches_the_textbook_form_where_that_form_is_accurate():
    assert gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B) == pytest.approx(
        _naive_gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B), abs=1e-12
    )


def test_the_divergence_is_directed():
    forward = gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B)
    reverse = gaussian_kl(MEAN_B, COV_B, MEAN_A, COV_A)
    assert abs(forward - reverse) > 1e-3


def test_an_affine_change_of_coordinates_leaves_it_unchanged():
    # The reparameterisation invariance the ledger relies on to call nats a scale-free
    # unit. Both Gaussians move through the same map, so the divergence must not.
    transform = np.array([[3.0, 1.0, 0.0], [0.0, -2.0, 0.5], [1.0, 0.0, 4.0]])
    offset = np.array([10.0, -3.0, 0.25])
    before = gaussian_kl(MEAN_A, COV_A, MEAN_B, COV_B)
    after = gaussian_kl(
        transform @ MEAN_A + offset,
        transform @ COV_A @ transform.T,
        transform @ MEAN_B + offset,
        transform @ COV_B @ transform.T,
    )
    assert after == pytest.approx(before, rel=1e-12)


def test_near_equality_reads_small_and_never_negative():
    # Two Gaussians a relative 1e-13 apart in covariance diverge by ~n·(1e-13)²/4. The
    # textbook subtraction lands within rounding of zero, of either sign, which is
    # exactly the reading a 1e-12 bar cannot interpret.
    nearly = COV_A * (1.0 + 1e-13)
    measured = gaussian_kl(MEAN_A, nearly, MEAN_A, COV_A)
    assert 0.0 <= measured < 1e-24


def test_a_degenerate_covariance_is_refused_by_name():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="cov_b"):
        gaussian_kl([0.0, 0.0], np.eye(2), [0.0, 0.0], singular)


def test_a_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape"):
        gaussian_kl([0.0, 0.0], np.eye(2), [0.0], np.eye(1))


def test_the_series_and_the_direct_form_agree_at_the_switch():
    # Two evaluations of one function. Either side of the crossover they must return
    # the same number, or the divergence would step where the branch changes.
    for excess in (-_SERIES_BELOW, _SERIES_BELOW):
        for side in (excess * (1 - 1e-9), excess * (1 + 1e-9)):
            direct = side - np.log1p(side)
            assert float(_excess_over_log(np.array([side]))[0]) == pytest.approx(
                direct, rel=1e-9
            )
