"""Task 3 diagnostics: is the transfer failure a missing signal or a wrong map?

A classifier with no signal produces diffuse confusion rows. Task 3's rows are
not diffuse - Room B sends 98% of SIT to WALK - which suggests the target
cluster structure is intact and the source prototypes are landing on it in the
wrong order.

The optimal-assignment (Hungarian) bound measures exactly that: the best
macro-F1 reachable by any one-to-one relabelling of the predicted classes.

    reported << bound    the embedding separates the activities; the
                         cross-room correspondence is wrong
    reported ~= bound    no recoverable structure; the pessimistic reading

ORACLE - NOT A RESULT
---------------------
The permutation is chosen using validation labels, so this is a diagnostic
upper bound and nothing else. Challenge spec section 8 permits a
validation-derived statistic for validation ANALYSIS; it may never be used to
fit a model or be reported as achieved performance. Every row written here
carries `oracle=True`.

Also reported: shared macro-F1 excluding WALK. Room B's WALK captures are the
only Room B activity whose files lack the `_Amp_` infix and they sit ~50% above
every other Room B activity in mean amplitude, so dropping WALK isolates how
much of the shared-activity failure that one contaminated class accounts for.

Run:  python -m src.task3_diagnostics
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src import amplitude, config

SHARED = list(amplitude.SHARED)


def _per_class_f1(matrix: np.ndarray, labels: list[str]) -> dict[str, float]:
    scores = {}
    for i, label in enumerate(labels):
        tp = matrix[i, i]
        fn = matrix[i].sum() - tp
        fp = matrix[:, i].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return scores


def bound(cfg: config.Config) -> pd.DataFrame:
    confusion = pd.read_csv(cfg.path("results") / "task3_confusion_K5.csv")
    rows = []

    for room in sorted(confusion["target_room"].unique()):
        table = (
            confusion[confusion.target_room == room]
            .pivot(index="true", columns="predicted", values="count")
            .fillna(0)
            .astype(int)
            .drop(columns=["OTHER"], errors="ignore")
        )
        labels = list(table.index)
        matrix = table.loc[labels, labels].to_numpy()

        reported = _per_class_f1(matrix, labels)
        _, assignment = linear_sum_assignment(-matrix)        # maximise the trace
        optimal = _per_class_f1(matrix[:, assignment], labels)
        mapping = {labels[i]: labels[j] for i, j in enumerate(assignment)}

        shared = [l for l in labels if l in SHARED]
        shared_no_walk = [l for l in shared if l != "WALK"]

        for metric, selection in (
            ("macro_f1", labels),
            ("shared_macro_f1", shared),
            ("shared_macro_f1_excl_walk", shared_no_walk),
        ):
            if not selection:
                continue
            rows.append({
                "target_room": room, "metric": metric,
                "reported": float(np.mean([reported[l] for l in selection])),
                "hungarian_bound": float(np.mean([optimal[l] for l in selection])),
                "oracle": True,
            })
        rows.append({
            "target_room": room, "metric": "accuracy",
            "reported": float(np.trace(matrix) / matrix.sum()),
            "hungarian_bound": float(np.trace(matrix[:, assignment]) / matrix.sum()),
            "oracle": True,
        })
        rows.append({
            "target_room": room, "metric": "optimal_label_map",
            "reported": "", "hungarian_bound": str(mapping), "oracle": True,
        })

    return pd.DataFrame(rows)


def main() -> int:
    cfg = config.load()
    table = bound(cfg)
    out = cfg.path("results") / "task3_hungarian_bound.csv"
    table.to_csv(out, index=False)

    print("Optimal-assignment (Hungarian) upper bound — ORACLE, not a result.\n")
    numeric = table[table.metric != "optimal_label_map"].copy()
    numeric["reported"] = numeric["reported"].astype(float).round(3)
    numeric["hungarian_bound"] = numeric["hungarian_bound"].astype(float).round(3)
    print(numeric.to_string(index=False))

    print("\nOptimal label map (true -> predicted cluster):")
    for row in table[table.metric == "optimal_label_map"].itertuples(index=False):
        print(f"  Room {row.target_room}: {row.hungarian_bound}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
