"""Do SIT/STAND/WALK transfer across rooms? A development-only check.

Fits a linear probe on one room's SIT/STAND/WALK development windows and
scores it on the OTHER room's development windows. No local_validation
window is loaded, so this can inform Task 3's design without spending the
validation set.

Rows compared
-------------
    ssl_full / as_is                  SSL embeddings (seed 0), task2 preset
    ssl_full / per_room_standardised  same, z-scored separately per room
    handcrafted / as_is               handcrafted features, task2 preset
    handcrafted / per_room_standardised
    handcrafted_task3 / as_is         handcrafted features on the task3 preset
    within_room / *                   same probe, 5-fold CV inside one room

0.167 is the macro-F1 of predicting one class for every window.

Run:  python -m src.task2_transfer_check
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import amplitude, baselines, config, dataset, encoder, splits

SHARED = list(amplitude.SHARED)


def per_room_standardise(z: np.ndarray, room: np.ndarray) -> np.ndarray:
    out = z.copy()
    for r in np.unique(room):
        mask = room == r
        out[mask] = (z[mask] - z[mask].mean(axis=0)) / (z[mask].std(axis=0) + 1e-6)
    return out


def _probe(balanced: bool = True):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced" if balanced else None),
    )


def cross_room(z: np.ndarray, ws: dataset.WindowSet) -> dict[str, float]:
    scores = {}
    for source, target in (("B", "A"), ("A", "B")):
        fit = (ws.room == source) & np.isin(ws.y, SHARED)
        score = (ws.room == target) & np.isin(ws.y, SHARED)
        predicted = _probe().fit(z[fit], ws.y[fit]).predict(z[score])
        scores[f"{source}_to_{target}"] = f1_score(ws.y[score], predicted, average="macro")
    return scores


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = config.load()
    dev2 = dataset.load("task2", splits.DEVELOPMENT, cfg)
    dev3 = dataset.load("task3", splits.DEVELOPMENT, cfg)

    ssl = encoder.embed(encoder.load_pretrained(cfg, "full", 0), dev2.x)
    hand2 = baselines.handcrafted_features(dev2.x)
    hand3 = baselines.handcrafted_features(dev3.x)

    cases = [
        ("ssl_full", "as_is", ssl, dev2),
        ("ssl_full", "per_room_standardised", per_room_standardise(ssl, dev2.room), dev2),
        ("handcrafted", "as_is", hand2, dev2),
        ("handcrafted", "per_room_standardised", per_room_standardise(hand2, dev2.room), dev2),
        ("handcrafted_task3", "as_is", hand3, dev3),
    ]

    rows = []
    for features, variant, z, ws in cases:
        for direction, value in cross_room(z, ws).items():
            rows.append({"experiment": f"task2_transfer_check_{features}_{variant}",
                         "target_room": direction[-1], "K": "", "metric": "macro_f1",
                         "value": value})
    for room in ("A", "B"):
        mask = (dev3.room == room) & np.isin(dev3.y, SHARED)
        value = cross_val_score(_probe(balanced=False), hand3[mask], dev3.y[mask],
                                cv=5, scoring="f1_macro").mean()
        rows.append({"experiment": "task2_transfer_check_within_room_handcrafted_task3",
                     "target_room": room, "K": "", "metric": "macro_f1_cv5", "value": value})

    table = pd.DataFrame(rows)
    out = cfg.path("results") / "task2_transfer_check.csv"
    table.to_csv(out, index=False)
    print("Development-only, SIT/STAND/WALK (single-class level = 0.167):\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
