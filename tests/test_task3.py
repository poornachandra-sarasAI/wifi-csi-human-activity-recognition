"""Task 3 support-sampling and prototype tests. Synthetic arrays only: no dataset needed."""

from __future__ import annotations

import numpy as np
import pytest

from src import fewshot

rng = np.random.default_rng(0)


def _captures():
    """3 activities x 6 captures, 2 windows per capture."""
    y = np.repeat(["FALL", "SIT", "STAND"], 12)
    capture_id = np.array([f"{a}_{i // 2}" for a, i in zip(y, np.tile(range(12), 3))])
    return capture_id, y


def test_support_is_k_distinct_captures_per_activity():
    capture_id, y = _captures()
    for k in (1, 3, 6):
        mask = fewshot.sample_support(capture_id, y, ["FALL", "STAND"], k, np.random.default_rng(1))
        assert set(y[mask]) == {"FALL", "STAND"}  # SIT withheld
        for activity in ("FALL", "STAND"):
            assert len(set(capture_id[mask & (y == activity)])) == k
        # The capture is atomic: a chosen capture brings all of its windows.
        assert np.isin(capture_id, capture_id[mask]).sum() == mask.sum()


def test_support_draw_is_reproducible_and_bounded():
    capture_id, y = _captures()
    draw = lambda seed: fewshot.sample_support(capture_id, y, ["SIT"], 2, np.random.default_rng(seed))
    np.testing.assert_array_equal(draw(5), draw(5))
    with pytest.raises(ValueError, match="K=7"):
        fewshot.sample_support(capture_id, y, ["SIT"], 7, np.random.default_rng(0))


def test_room_scaler_standardises_each_room_separately():
    z = np.concatenate([rng.normal(0, 1, (50, 4)), rng.normal(100, 9, (50, 4))])
    room = np.repeat(["A", "B"], 50)
    out = fewshot.RoomScaler(z, room)(z, room)
    for r in ("A", "B"):
        np.testing.assert_allclose(out[room == r].mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(out[room == r].std(axis=0), 1, atol=1e-3)


def test_prototypes_classify_separable_clusters():
    centres = np.eye(3) * 5
    y = np.repeat(["A", "B", "C"], 20)
    z = centres[np.repeat(range(3), 20)] + rng.normal(0, 0.3, (60, 3))
    protos = fewshot.prototypes(z, y, ["A", "B", "C"])
    labels, confidence = fewshot.predict(z, ["A", "B", "C"], protos)
    assert (labels == y).all()
    assert ((confidence > 0) & (confidence <= 1)).all()


def test_refine_moves_a_shifted_prototype_onto_its_cluster():
    """A source prototype that is off-target is pulled onto the target cluster."""
    centres = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.2]]) * 5
    unlabelled = centres[np.repeat([0, 1], 30)] + rng.normal(0, 0.2, (60, 3))
    support = centres[[1]] + rng.normal(0, 0.2, (1, 3))
    shifted = fewshot.prototypes(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.2]]),
                                 np.array(["X", "Y"]), ["X", "Y"])
    refined = fewshot.refine(shifted, support, np.array(["Y"]), unlabelled, ["X", "Y"])
    target = centres[0] / np.linalg.norm(centres[0])
    assert refined[0] @ target > shifted[0] @ target
    assert refined[0] @ target > 0.99
