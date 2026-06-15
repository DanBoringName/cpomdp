import numpy as np
import pytest

from cpomdp.types import Belief, LinearGaussianModel


class TestBeliefs:
    def test_valid_belief_stores_what_you_passed(self):
        b = Belief(mean=[1.0, 2.0], cov=[[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(b.mean, [1.0, 2.0])
        np.testing.assert_array_equal(b.cov, [[1.0, 0.0], [0.0, 1.0]])

    def test_coerce_lists_to_float_arrays(self):
        b = Belief(mean=[0, 1], cov=[[1, 0], [0, 1]])
        assert isinstance(b.mean, np.ndarray)
        assert b.mean.dtype == np.float64

    def test_rejects_mean_not_1D(self):
        with pytest.raises(ValueError, match="1-D"):
            Belief(mean=[[0.0]], cov=[[1.0]])

    def test_rejects_cov_not_2D(self):
        with pytest.raises(ValueError, match="2-D"):
            Belief(mean=[0.0], cov=[1.0])

    def test_reject_shape_mismatch(self):
        with pytest.raises(ValueError, match="match"):
            Belief(mean=[0.0, 0.0], cov=[[1.0]])

    def test_rejects_asymmetric_cov(self):
        with pytest.raises(ValueError, match="symmetric"):
            Belief(mean=[0.0, 0.0], cov=[[1.0, 0.2], [0.9, 1.0]])

    def test_ndim_reports_state_dimension(self):
        assert Belief(mean=[0.0], cov=[[1.0]]).ndim == 1
        assert Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]).ndim == 2


# A valid 2-state / 1-observation model, reused as the baseline. Each rejection
# test overrides exactly one field with a bad value, so the test isolates one
# validation branch.
def _valid_kwargs(**overrides):
    kwargs = dict(
        dynamics=[[1.0, 0.1], [0.0, 1.0]],  # 2x2  (n=2)
        sensor_model=[[1.0, 0.0]],  # 1x2  (m=1)
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],  # 2x2
        sensor_noise=[[1.0]],  # 1x1
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
    )
    kwargs.update(overrides)
    return kwargs


class TestLinearGaussianModels:
    def test_valid_model_constructs(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.dynamics, [[1.0, 0.1], [0.0, 1.0]])
        assert m.n_states == 2
        assert m.n_observations == 1

    def test_control_is_optional(self):
        m = LinearGaussianModel(**_valid_kwargs())
        assert m.control is None
        assert m.B is None
        assert m.n_controls == 0

    def test_with_control(self):
        m = LinearGaussianModel(**_valid_kwargs(control=[[0.0], [1.0]]))  # 2x1 (p=1)
        np.testing.assert_array_equal(m.control, [[0.0], [1.0]])
        np.testing.assert_array_equal(m.B, [[0.0], [1.0]])
        assert m.n_controls == 1

    def test_letter_aliases_map_to_role_names(self):
        m = LinearGaussianModel(**_valid_kwargs())
        np.testing.assert_array_equal(m.A, m.dynamics)
        np.testing.assert_array_equal(m.C, m.sensor_model)
        np.testing.assert_array_equal(m.Q, m.dynamics_noise)
        np.testing.assert_array_equal(m.R, m.sensor_noise)

    def test_rejects_non_square_dynamics(self):
        with pytest.raises(ValueError, match="square"):
            LinearGaussianModel(
                **_valid_kwargs(dynamics=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )

    def test_rejects_sensor_model_wrong_columns(self):
        with pytest.raises(ValueError, match="columns"):
            LinearGaussianModel(**_valid_kwargs(sensor_model=[[1.0, 0.0, 0.0]]))

    def test_rejects_dynamics_noise_wrong_size(self):
        with pytest.raises(ValueError, match="dynamics_noise"):
            LinearGaussianModel(**_valid_kwargs(dynamics_noise=[[1.0]]))

    def test_rejects_sensor_noise_wrong_size(self):
        with pytest.raises(ValueError, match="sensor_noise"):
            LinearGaussianModel(**_valid_kwargs(sensor_noise=[[1.0, 0.0], [0.0, 1.0]]))

    def test_rejects_control_wrong_rows(self):
        with pytest.raises(ValueError, match="control"):
            LinearGaussianModel(**_valid_kwargs(control=[[0.0], [0.0], [0.0]]))  # 3x1

    def test_rejects_prior_wrong_dimension(self):
        bad_prior = Belief(mean=[0.0, 0.0, 0.0], cov=np.eye(3))  # 3-D, but n=2
        with pytest.raises(ValueError, match="prior"):
            LinearGaussianModel(**_valid_kwargs(prior=bad_prior))

    def test_rejects_prior_not_a_belief(self):
        with pytest.raises(TypeError, match="Belief"):
            LinearGaussianModel(**_valid_kwargs(prior=[0.0, 0.0]))
