"""The two divergences a cell is scored by, and the shape that keeps them honest."""

from dataclasses import fields

from cpomdp.scoring import Decomposition


def test_the_type_carries_the_two_divergences_and_nothing_else():
    # Standing prohibition 1: never obtain a term by subtracting H(p*). No entropy
    # field, no estimator slot, no total. Adding one is a deliberate edit here first.
    assert [f.name for f in fields(Decomposition)] == [
        "misspecification",
        "inference_gap",
    ]
