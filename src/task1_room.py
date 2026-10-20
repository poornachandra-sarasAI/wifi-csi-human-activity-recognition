"""Task 1: recorded-environment identification (Room A vs Room B).

Given a 150 x 256 CSI amplitude window, predict which room it came from.
The classifier sees only the numerical window: no filename, folder, file
extension, capture length or packet count (spec section 7, section 8).

Two settings are reported, because one of them is honest and the other is
a measurement of our own instrumentation:

  task1          global normalization. Absolute amplitude survives. Room A
                 (our decode) and Room B (supplied) differ by a global
                 factor of ~490x and Room A carries a ~4x AGC step across
                 subcarriers 137-187. A classifier will find these first.

  task1_honest   per-window normalization. Absolute level is removed, so
                 only the spectral SHAPE and the temporal dynamics remain.
                 Accuracy here is the defensible claim about environment
                 fingerprinting.

The gap between the two is the informative result, and it speaks directly
to the room-versus-session confound noted in spec section 12.

Run:  python -m src.task1_room
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config, dataset, splits


def featurize(x: np.ndarray) -> np.ndarray:
    """Per-subcarrier summary of a window: static level plus variability.

    Deliberately simple, and deliberately not capture-length dependent:
    both statistics are computed over the window's own observations, so a
    longer capture cannot be distinguished from a shorter one.
    """
    if x.ndim == 4:  # two-channel preset; keep the static channel
        x = x[..., 0]
    return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)


def evaluate(preset: str, cfg: config.Config, lengths=None) -> pd.DataFrame:
    """Train on development windows, score on the fixed validation set."""
    dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
    val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)
    lengths = lengths or cfg["task1"]["window_length_sweep"]

    rows = []
    for length in lengths:
        # Truncate to the first `length` observations of the same window.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=cfg["seed"]),
        )
        model.fit(featurize(dev.x[:, :length]), dev.room)
        predicted = model.predict(featurize(val.x[:, :length]))

        rows.append(
            {
                "experiment": "task1_room_id",
                "preset": preset,
                "window_length": length,
                "metric": "accuracy",
                "value": accuracy_score(val.room, predicted),
            }
        )
        if length == cfg["windowing"]["length"]:
            matrix = confusion_matrix(val.room, predicted, labels=["A", "B"])
            print(f"  confusion at length {length} (rows true A,B):\n{matrix}")

    return pd.DataFrame(rows)


def main() -> int:
    cfg = config.load()
    tables = []
    for preset in ("task1", "task1_honest"):
        print(f"\n=== {preset} ===")
        table = evaluate(preset, cfg)
        print(table[["window_length", "value"]].to_string(index=False))
        tables.append(table)

    results = pd.concat(tables, ignore_index=True)
    out = cfg.path("results") / "task1_room_accuracy.csv"
    results.to_csv(out, index=False)

    headline = results[results.window_length == cfg["windowing"]["length"]]
    print("\nHeadline (official 150-observation validation windows):")
    for row in headline.itertuples(index=False):
        print(f"  {row.preset:14s} accuracy = {row.value:.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
