"""Records what the crossover demo's NumPy oracle returns at H=7, before it changes.

`examples/ffg/crossover.py::_numpy_score` is the independent kernel the headline
`H* = 7` is checked against. Its per-step epistemic term calls `np.linalg.slogdet` and
keeps only the log-magnitude, so a covariance block with an even number of negative
eigenvalues returns a finite, plausible number instead of NaN. The shipped kernel does
not: `_efe_step` and `_state_info_gain` route through `_logdet_pd`, which tests positive
definiteness by Cholesky. The two routines were documented as differing, never audited.

Guarding the oracle is a precondition, not a change of arithmetic, so it must leave
every value here untouched. The anchors below make that checkable rather than asserted.
They are measured off the unguarded path and recorded first, so the guard cannot be
introduced and blessed in one step. An anchor carries no theory: it says the number did
not move, nothing about whether it was right to begin with.

`crossover.check()` gates the same numbers but enumerates ~150k policies and is marked
slow, so it is deselected on pull requests. Scoring two known policies costs a backend
build and fourteen propagation steps, which keeps the headline gated on every run.
"""

import crossover
import numpy as np
import pytest

# Declared bars; full justification in warrant_numbers.md.
# The guard under audit adds a positive-definiteness precondition ahead of an unchanged
# `slogdet`, so on a positive-definite rollout the expected movement is zero and a bar
# at the ULP would hold. This sits about three orders above it (4.3e-9 absolute at
# G ~ 425), matching the atol = 1e-9 the demo already trusts for shipped-against-oracle
# agreement, so cross-platform BLAS variation cannot flake it while a real change in the
# epistemic term cannot hide under it.
ORACLE_RTOL = 1e-11
# The flip margin is a difference of two values near 425, so it inherits ~8.5e-9 of
# absolute slack from the pair above. This is that slack, rounded up.
MARGIN_ATOL = 1e-8

# Measured on the unguarded oracle at H = 7 (float64, x64 enabled).
ANCHOR_WALK = 425.163110098734
ANCHOR_REACH = 425.3151092512748
ANCHOR_MARGIN = -0.15199915254078178


@pytest.fixture(scope="module")
def oracle_scores():
    """`(walk, reach)` from the NumPy oracle at H*, computed once for the module."""
    horizon = crossover.FLIP_H
    return (
        crossover._numpy_score(crossover._walk(horizon)),
        crossover._numpy_score(crossover._reach(horizon)),
    )


class TestOracleAnchors:
    """The oracle's H=7 scores, recorded so a guard cannot move them unnoticed."""

    def test_walk_score(self, oracle_scores):
        walk, _ = oracle_scores
        np.testing.assert_allclose(walk, ANCHOR_WALK, rtol=ORACLE_RTOL)

    def test_reach_score(self, oracle_scores):
        _, reach = oracle_scores
        np.testing.assert_allclose(reach, ANCHOR_REACH, rtol=ORACLE_RTOL)

    def test_flip_margin(self, oracle_scores):
        # ΔG(7) as the ledger registers it: G(walk) − G(reach), relative size 3.6e−4.
        walk, reach = oracle_scores
        np.testing.assert_allclose(walk - reach, ANCHOR_MARGIN, atol=MARGIN_ATOL)

    def test_walk_wins_at_flip_horizon(self, oracle_scores):
        # The direction the headline rests on, asserted with no tolerance in it:
        # whatever the anchors drift to, the oracle must rank the walk first at H*.
        walk, reach = oracle_scores
        assert walk < reach

    def test_horizon_under_audit_is_the_registered_one(self):
        # The anchors are H-specific. If FLIP_H moves, they are stale, not passing.
        assert crossover.FLIP_H == 7
