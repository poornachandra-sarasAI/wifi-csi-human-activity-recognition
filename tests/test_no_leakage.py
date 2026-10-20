"""Leakage controls (challenge spec section 8).

These are the tests that protect every reported number. If any of them
fail, results computed from the current windows/ are not valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import amplitude, config, splits

cfg = config.load()
MANIFEST = cfg.path("windows") / "manifest.csv"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="run `python -m src.windowing` first to build windows/manifest.csv",
)


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST)


def test_capture_level_isolation(manifest):
    """No original capture appears on both sides of the split."""
    by_split = manifest.groupby("split")["capture_id"].apply(set)
    dev = by_split.get(splits.DEVELOPMENT, set())
    val = by_split.get(splits.LOCAL_VALIDATION, set())
    assert not (dev & val), f"{len(dev & val)} captures span both splits"


def test_every_window_belongs_to_one_capture(manifest):
    """A window is never assembled from more than one capture."""
    assert manifest["window_id"].is_unique
    per_window = manifest.groupby("window_id")["capture_id"].nunique()
    assert (per_window == 1).all()


def test_room_a_validation_is_the_supplied_set(manifest):
    """Room A's supplied validation captures are all present, and none
    of them leaked into development (spec section 9)."""
    room_a = manifest[manifest.room == "A"]
    supplied = room_a[room_a.validation_origin == "supplied"]
    assert len(supplied) == 150, f"expected 150 supplied captures, got {len(supplied)}"
    assert (supplied.split == splits.LOCAL_VALIDATION).all()
    assert supplied.groupby("activity").size().eq(25).all()


def test_room_b_validation_is_25_per_class(manifest):
    """Room B's validation set is the seed-42 draw, 25 captures per class."""
    room_b = manifest[manifest.room == "B"]
    val = room_b[room_b.split == splits.LOCAL_VALIDATION]
    assert (val.validation_origin == "seed42").all()
    assert val.groupby("activity").size().eq(25).all()


def test_room_b_draw_is_reproducible():
    """Re-running the seed-42 draw selects exactly the same captures."""
    ingest_b = pd.read_csv(cfg.path("results") / "ingest_room_B.csv")
    first = splits.assign_room_b(ingest_b, cfg)
    second = splits.assign_room_b(ingest_b, cfg)
    assert first.equals(second)


def test_windows_match_the_declared_contract(manifest):
    """Every window on disk is (150, 256) float32, finite and non-negative."""
    length = cfg["windowing"]["length"]
    sample = manifest.sample(min(40, len(manifest)), random_state=0)
    for row in sample.itertuples(index=False):
        path = cfg.path("windows") / row.split / f"{row.window_id}.npy"
        window = np.load(path)
        assert window.shape == (length, amplitude.N_SUBCARRIERS)
        assert window.dtype == np.float32
        assert np.isfinite(window).all()
        assert (window >= 0).all()


def test_manifest_covers_every_window_file(manifest):
    """No orphan .npy files left over from an earlier configuration."""
    on_disk = {p.stem for p in cfg.path("windows").rglob("*.npy")}
    assert on_disk == set(manifest["window_id"])


def test_config_hash_is_uniform(manifest):
    """All windows were produced by one configuration."""
    assert manifest["config_hash"].nunique() == 1
    assert manifest["config_hash"].iloc[0] == cfg.config_hash
