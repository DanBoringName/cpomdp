"""The shared `Warrant` vocabulary.

`SearchWarrant` had two levels: `PROVED` (3b, exhaustive enumeration) and `CORROBORATED`
(3a, a grid sample). `Warrant` adds `CERTIFIED` (3c, validated numerics over a compact
domain) and moves to `cpomdp.warrant`, where checks can reach it too. `SearchWarrant`
stays as an alias, so existing call sites keep their members and their return type.

Imports `cpomdp.warrant`, so until it lands this module is collection-red — the
`ModuleNotFoundError` naming it is the build cue.
"""

import pytest

from cpomdp.enumeration import (
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
    OpenLoopSelector,
    RecedingHorizonSelector,
    SearchWarrant,
)
from cpomdp.selection import EFESelector, Preference
from cpomdp.types import Belief, LinearGaussianModel
from cpomdp.warrant import Warrant


def _model():
    """A plain fixed-sensor model, p = 1."""
    return LinearGaussianModel(
        dynamics=[[1.0, 0.1], [0.0, 1.0]],
        sensor_model=[[1.0, 0.0]],
        dynamics_noise=[[0.1, 0.0], [0.0, 0.1]],
        sensor_noise=[[0.5]],
        prior=Belief(mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 1.0]]),
        control=[[0.0], [1.0]],
    )


def _finite_set(actions):
    return FiniteActionSet([[a] for a in actions], version="test-v1")


class TestWarrantLevels:
    def test_the_three_prover_classes(self):
        assert Warrant.PROVED.value == "PROVED"
        assert Warrant.CERTIFIED.value == "CERTIFIED"
        assert Warrant.CORROBORATED.value == "CORROBORATED"

    def test_no_fourth_level(self):
        # A fourth member is a fourth prover class: a decision, not an edit.
        assert [w.name for w in Warrant] == ["PROVED", "CERTIFIED", "CORROBORATED"]

    def test_round_trips_through_its_value(self):
        # Checks report by value, so a value read back must land on the same member.
        for level in Warrant:
            assert Warrant(level.value) is level


class TestSearchWarrantAlias:
    def test_alias_is_the_same_enum(self):
        # Not merely equal. A separate enum breaks every `is` check in the suite.
        assert SearchWarrant is Warrant

    def test_existing_members_resolve_to_the_same_objects(self):
        assert SearchWarrant.PROVED is Warrant.PROVED
        assert SearchWarrant.CORROBORATED is Warrant.CORROBORATED


class TestExistingLabelsUnchanged:
    def test_enumerated_search_is_still_proved(self):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert search.warrant is Warrant.PROVED

    def test_grid_selector_is_still_corroborated(self):
        selector = EFESelector(_model(), n_candidates=5, action_bounds=(-1.0, 1.0))
        assert selector.warrant is Warrant.CORROBORATED

    @pytest.mark.parametrize("driver", [RecedingHorizonSelector, OpenLoopSelector])
    def test_enumerated_drivers_are_still_proved(self, driver):
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        assert driver(search).warrant is Warrant.PROVED

    def test_certificate_renders_the_same_string(self):
        # This string reaches the write-up, so not a character of it may move.
        cert = CompletenessCertificate(expected=4, visited=4, warrant=Warrant.PROVED)
        assert str(cert) == "PROVED (finite set, |A|^H = 4, visited 4)"

    def test_search_still_scores(self):
        # The label is a property, so a broken search would still report PROVED.
        search = EnumeratedEfeSearch(_model(), _finite_set([-1.0, 1.0]), horizon=2)
        result = search.evaluate(
            Belief([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
            Preference(goal=[0.0], precision=[[1.0]]),
        )
        assert result.g.shape == (4,)
        assert search.warrant is Warrant.PROVED
