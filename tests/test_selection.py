"""Preference value type and the LQRSelector pass-through adapter."""

import numpy as np
import pytest

from cpomdp.control import LQRController
from cpomdp.selection import ActionSelector, LQRSelector, Preference
from cpomdp.types import Belief, LinearGaussianModel

# A double-integrator point mass, matching the control-test plant: state =
# [position, velocity], one force on the velocity. Reused to build a controller.
DYNAMICS = [[1.0, 0.1], [0.0, 1.0]]
CONTROL = [[0.0], [0.1]]
GOAL_PRECISION = [[1.0, 0.0], [0.0, 1.0]]
EFFORT_PENALTY = [[0.1]]


def _point_mass_model():
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        sensor_noise=[[1e-2]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=CONTROL,
    )


def _controller():
    return LQRController(
        _point_mass_model(),
        goal_precision=GOAL_PRECISION,
        effort_penalty=EFFORT_PENALTY,
    )


class TestPreference:
    def test_stores_and_coerces_goal(self):
        p = Preference(goal=[1.0, 2.0])
        assert isinstance(p.goal, np.ndarray) or hasattr(p.goal, "shape")
        np.testing.assert_array_equal(p.goal, [1.0, 2.0])

    def test_precision_defaults_to_identity(self):
        p = Preference(goal=[1.0, 2.0])
        np.testing.assert_array_equal(p.precision, np.eye(2))

    def test_accepts_an_explicit_precision(self):
        p = Preference(goal=[0.0, 0.0], precision=[[2.0, 0.0], [0.0, 3.0]])
        np.testing.assert_array_equal(p.precision, [[2.0, 0.0], [0.0, 3.0]])

    def test_rejects_non_1d_goal(self):
        with pytest.raises(ValueError, match="1-D"):
            Preference(goal=[[1.0]])

    def test_rejects_asymmetric_precision(self):
        with pytest.raises(ValueError, match="symmetric"):
            Preference(goal=[0.0, 0.0], precision=[[1.0, 0.2], [0.9, 1.0]])

    def test_rejects_precision_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            Preference(goal=[0.0, 0.0], precision=[[1.0]])


class TestLQRSelector:
    def test_satisfies_the_action_selector_protocol(self):
        selector = LQRSelector(_controller())
        assert isinstance(selector, ActionSelector)

    def test_select_is_a_faithful_pass_through(self):
        controller = _controller()
        selector = LQRSelector(controller)
        belief = Belief(mean=[0.3, -0.2], cov=[[1.0, 0.0], [0.0, 1.0]])
        pref = Preference(goal=[1.0, 0.0])

        np.testing.assert_array_equal(
            selector.select(belief, pref),
            controller.action(belief.mean, pref.goal),
        )
