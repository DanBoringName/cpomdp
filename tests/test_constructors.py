import numpy as np
import pytest
from numpy.typing import ArrayLike

from cpomdp.constructors import CORRECT, ConstructorSet, ModelSpec, Perturbation
from cpomdp.types import LinearGaussianModel

DT = 0.1
CONTROL = [[0.0], [DT]]


def _spec(
    *,
    control_matrix: ArrayLike | None = CONTROL,
    version: str = "spec-v1",
) -> ModelSpec:
    return ModelSpec(
        dynamics_matrix=[[1.0, DT], [0.0, 1.0]],
        observation_matrix=[[1.0, 0.0]],
        dynamics_noise=[[1e-4, 0.0], [0.0, 1e-4]],
        observation_noise=[[1e-2]],
        prior_mean=[0.0, 0.0],
        prior_cov=[[1.0, 0.0], [0.0, 1.0]],
        control_matrix=control_matrix,
        version=version,
    )


def _leaves(model: LinearGaussianModel) -> set[int]:
    return {
        id(model.dynamics_matrix),
        id(model.observation_matrix),
        id(model.dynamics_noise),
        id(model.observation_noise),
        id(model.prior.mean),
        id(model.prior.cov),
    }


# --- the spec builds models, and every build is its own object ----------------------


def test_build_returns_a_model_carrying_the_declared_parameters():
    model = _spec().build()
    assert isinstance(model, LinearGaussianModel)
    assert np.allclose(np.asarray(model.dynamics_matrix), [[1.0, DT], [0.0, 1.0]])
    assert np.allclose(np.asarray(model.observation_noise), [[1e-2]])
    assert np.allclose(np.asarray(model.prior.mean), [0.0, 0.0])


def test_two_builds_are_different_objects():
    spec = _spec()
    assert spec.build() is not spec.build()


def test_two_builds_share_no_array():
    spec = _spec()
    left, right = spec.build(), spec.build()  # both held: a freed id can be reused
    assert not _leaves(left) & _leaves(right)


def test_two_builds_are_equal_by_value():
    spec = _spec()
    left, right = spec.build(), spec.build()
    assert np.array_equal(np.asarray(left.dynamics_matrix), np.asarray(right.A))
    assert np.array_equal(np.asarray(left.observation_noise), np.asarray(right.R))


def test_a_control_free_spec_builds_a_control_free_model():
    model = _spec(control_matrix=None).build()
    assert model.control_matrix is None
    assert model.n_controls == 0


def test_the_spec_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        _spec(version="")


# --- perturbations are data, applied as a relative change ---------------------------


def test_correct_changes_nothing():
    spec = _spec()
    plain, corrected = spec.build(), spec.build(CORRECT)
    assert np.array_equal(
        np.asarray(plain.dynamics_matrix), np.asarray(corrected.dynamics_matrix)
    )


def test_a_perturbation_scales_the_named_parameter():
    spec = _spec()
    perturbed = spec.build(Perturbation("noisy", "observation_noise", 0.5))
    assert np.allclose(np.asarray(perturbed.observation_noise), [[1.5e-2]])


def test_a_perturbation_leaves_every_other_parameter_alone():
    spec = _spec()
    perturbed = spec.build(Perturbation("noisy", "observation_noise", 0.5))
    plain = spec.build()
    assert np.array_equal(
        np.asarray(perturbed.dynamics_matrix), np.asarray(plain.dynamics_matrix)
    )
    assert np.array_equal(np.asarray(perturbed.prior.cov), np.asarray(plain.prior.cov))


def test_a_zero_magnitude_on_a_named_axis_is_the_identity():
    spec = _spec()
    perturbed = spec.build(Perturbation("flat", "dynamics_matrix", 0.0))
    assert np.array_equal(
        np.asarray(perturbed.dynamics_matrix), np.asarray(spec.build().A)
    )


def test_a_control_matrix_perturbation_applies():
    spec = _spec()
    perturbed = spec.build(Perturbation("weak", "control_matrix", -0.5))
    assert np.allclose(np.asarray(perturbed.control_matrix), [[0.0], [DT * 0.5]])


def test_perturbing_an_absent_control_matrix_does_not_build():
    spec = _spec(control_matrix=None)
    with pytest.raises(ValueError, match="no control_matrix"):
        spec.build(Perturbation("weak", "control_matrix", -0.5))


def test_an_unknown_parameter_does_not_construct():
    with pytest.raises(ValueError, match="not a perturbable parameter"):
        Perturbation("odd", "prior_mean", 0.5)


def test_a_perturbation_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Perturbation("", "dynamics_noise", 0.5)


def test_a_named_parameter_needs_a_magnitude_that_leaves_a_covariance_valid():
    spec = _spec()
    with pytest.raises(ValueError, match="observation_noise"):
        spec.build(Perturbation("void", "observation_noise", -1.0))


def test_correct_names_no_parameter():
    assert CORRECT.parameter is None
    assert CORRECT.magnitude == 0.0


def test_an_unperturbed_entry_may_not_name_a_magnitude():
    with pytest.raises(ValueError, match="perturbs nothing"):
        Perturbation("odd", None, 0.5)


# --- the set is declared, versioned, and pinned by a test ---------------------------


def test_a_set_reports_its_names_in_declaration_order():
    declared = ConstructorSet(
        (CORRECT, Perturbation("noisy", "observation_noise", 0.5)), version="cross-v1"
    )
    assert declared.names == ("correct", "noisy")
    assert declared.size == 2


def test_a_set_needs_a_version():
    with pytest.raises(ValueError, match="declared and versioned"):
        ConstructorSet((CORRECT,), version="")


def test_a_set_refuses_a_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        ConstructorSet(
            (
                CORRECT,
                Perturbation("noisy", "observation_noise", 0.5),
                Perturbation("noisy", "dynamics_noise", 0.5),
            ),
            version="cross-v1",
        )


def test_a_set_with_nothing_unperturbed_does_not_construct():
    with pytest.raises(ValueError, match="unperturbed"):
        ConstructorSet(
            (Perturbation("noisy", "observation_noise", 0.5),), version="cross-v1"
        )


def test_building_a_set_pairs_every_name_with_its_model():
    declared = ConstructorSet(
        (CORRECT, Perturbation("noisy", "observation_noise", 0.5)), version="cross-v1"
    )
    built = declared.build_all(_spec())
    assert tuple(name for name, _ in built) == declared.names
    assert np.allclose(np.asarray(built[0][1].observation_noise), [[1e-2]])
    assert np.allclose(np.asarray(built[1][1].observation_noise), [[1.5e-2]])


def test_every_model_a_set_builds_is_its_own_object():
    declared = ConstructorSet(
        (CORRECT, Perturbation("same", "observation_noise", 0.0)), version="cross-v1"
    )
    left, right = (model for _, model in declared.build_all(_spec()))
    assert left is not right
    assert not _leaves(left) & _leaves(right)
