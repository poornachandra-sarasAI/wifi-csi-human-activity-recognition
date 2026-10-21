"""Task 1 ablation: what information is the room classifier actually using?

Task 1 reaches very high accuracy, so the interesting question is not the
number but its cause. Three instrumentation differences could each explain
a near-perfect score without any environment fingerprinting:

  1. a global ~490x amplitude gap between our Room A decode and Room B's
     supplied integers,
  2. a ~4x receiver AGC step across subcarriers 135-187 in Room A only,
  3. different capture dynamics between the two recording sessions.

Normalization ladder
--------------------
    task1           all static structure kept
    task1_subband   sub-band offsets removed, within-band shape kept
    task1_honest    no static per-subcarrier structure at all

Feature-region ablation
-----------------------
    all        every subcarrier
    no_agc     subcarriers 128-191 (the AGC sub-band) dropped
    only_agc   only subcarriers 128-191
    no_nulls   802.11ac null subcarriers dropped

Run:  python -m src.task1_ablation
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config, dataset, splits, task1_room

AGC_BAND = slice(128, 192)
NULL_SUBCARRIERS = sorted(set([255, 0, 1] + list(range(123, 134))))

REGIONS = {
    "all": np.arange(256),
    "no_agc": np.array([i for i in range(256) if not 128 <= i < 192]),
    "only_agc": np.arange(128, 192),
    "no_nulls": np.array([i for i in range(256) if i not in NULL_SUBCARRIERS]),
}


def _fit_score(dev, val, columns: np.ndarray, seed: int) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    model.fit(task1_room.featurize(dev.x[:, :, columns]), dev.room)
    return accuracy_score(val.room, model.predict(task1_room.featurize(val.x[:, :, columns])))


def run(cfg: config.Config) -> pd.DataFrame:
    rows = []
    for preset in ("task1", "task1_subband", "task1_honest"):
        dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
        val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)
        for region, columns in REGIONS.items():
            accuracy = _fit_score(dev, val, columns, cfg["seed"])
            rows.append(
                {
                    "experiment": "task1_ablation",
                    "preset": preset,
                    "region": region,
                    "n_subcarriers": len(columns),
                    "metric": "accuracy",
                    "value": accuracy,
                }
            )
            print(f"  {preset:14s} {region:9s} ({len(columns):3d} sc)  acc = {accuracy:.4f}")
    return pd.DataFrame(rows)


def dynamics_diagnostic(cfg: config.Config) -> pd.DataFrame:
    """Do the two rooms differ in temporal dynamics independently of level?

    Lag-1 autocorrelation is computed per subcarrier on the per-window
    standardized windows, so it is unaffected by the amplitude gap and the
    AGC step. A large difference here means the sessions were recorded with
    different effective dynamics, which is a session confound rather than
    an environment fingerprint (spec section 12).
    """
    dev = dataset.load("task1_honest", splits.DEVELOPMENT, cfg)
    rows = []
    for room in ("A", "B"):
        x = dev.x[dev.room == room]
        centered = x - x.mean(axis=1, keepdims=True)
        numerator = (centered[:, :-1] * centered[:, 1:]).mean(axis=1)
        denominator = (centered**2).mean(axis=1) + 1e-8
        lag1 = (numerator / denominator).mean(axis=1)
        rows.append(
            {
                "room": room,
                "n_windows": len(x),
                "lag1_autocorr_mean": float(lag1.mean()),
                "lag1_autocorr_std": float(lag1.std()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    cfg = config.load()

    print("Feature-region ablation (accuracy on the fixed validation windows):")
    table = run(cfg)
    out = cfg.path("results") / "task1_ablation.csv"
    table.to_csv(out, index=False)

    print("\nPivot (rows = normalization, columns = feature region):")
    print(table.pivot(index="preset", columns="region", values="value").to_string())

    print("\nTemporal-dynamics diagnostic (level-invariant):")
    dynamics = dynamics_diagnostic(cfg)
    print(dynamics.to_string(index=False))
    dynamics.to_csv(cfg.path("results") / "task1_dynamics_diagnostic.csv", index=False)

    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
