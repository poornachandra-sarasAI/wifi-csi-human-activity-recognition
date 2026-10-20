"""Capture-level development / validation splits (challenge spec section 9).

The original capture is the atomic unit. A capture is assigned to exactly
one split, and every window derived from it inherits that assignment, so
no capture is ever split across development and validation.

Room A: the supplied validation/ directory IS the validation set (25 per
        class, 6 classes, 150 captures). It is never moved into
        development.
Room B: no validation directory is supplied. We create one once, at the
        capture level, 25 per class, drawing from deterministically
        sorted filenames with the public seed 42.

Run:  python -m src.splits
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import config

DEVELOPMENT = "development"
LOCAL_VALIDATION = "local_validation"


def assign_room_a(ingest_a: pd.DataFrame) -> pd.Series:
    """Room A split comes straight from the supplied directory layout."""
    return np.where(ingest_a["supplied_validation"], LOCAL_VALIDATION, DEVELOPMENT)


def assign_room_b(ingest_b: pd.DataFrame, cfg: config.Config) -> pd.Series:
    """Draw Room B's validation captures once, with the public seed."""
    settings = cfg["splits"]
    per_class = settings["room_b_validation_per_class"]
    seed = settings["room_b_seed"]

    split = pd.Series(DEVELOPMENT, index=ingest_b.index, dtype=object)

    for activity in sorted(ingest_b["activity"].unique()):
        rows = ingest_b.index[ingest_b["activity"] == activity]
        # Sort candidate filenames deterministically before sampling so
        # that every team constructs the same validation set.
        candidates = sorted(ingest_b.loc[rows, "capture_id"])
        if len(candidates) < per_class:
            raise ValueError(
                f"Room B activity {activity} has only {len(candidates)} captures, "
                f"fewer than the {per_class} required for validation"
            )
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(candidates, size=per_class, replace=False).tolist())
        split.loc[ingest_b["capture_id"].isin(chosen)] = LOCAL_VALIDATION

    return split


def build(cfg: config.Config | None = None) -> pd.DataFrame:
    """Combine both rooms' ingest reports into one capture-level table."""
    cfg = cfg or config.load()
    results = cfg.path("results")

    ingest_a = pd.read_csv(results / "ingest_room_A.csv")
    ingest_b = pd.read_csv(results / "ingest_room_B.csv")

    ingest_a = ingest_a.assign(
        split=assign_room_a(ingest_a),
        validation_origin=np.where(ingest_a["supplied_validation"], "supplied", ""),
    )
    split_b = assign_room_b(ingest_b, cfg)
    ingest_b = ingest_b.assign(
        split=split_b,
        validation_origin=np.where(split_b == LOCAL_VALIDATION, "seed42", ""),
    )

    columns = [
        "capture_id",
        "room",
        "activity",
        "split",
        "validation_origin",
        "source_format",
        "source_file",
        "n_obs_clean",
    ]
    return pd.concat([ingest_a[columns], ingest_b[columns]], ignore_index=True)


def main() -> int:
    cfg = config.load()
    captures = build(cfg)

    out = cfg.path("results") / "captures.csv"
    captures.to_csv(out, index=False)

    # The Room B draw is committed separately so the seed-42 selection can
    # be audited without regenerating the pipeline.
    room_b_val = sorted(
        captures.loc[
            (captures.room == "B") & (captures.split == LOCAL_VALIDATION), "capture_id"
        ]
    )
    audit = cfg.path("results") / "room_B_validation_seed42.json"
    audit.write_text(
        json.dumps(
            {
                "seed": cfg["splits"]["room_b_seed"],
                "per_class": cfg["splits"]["room_b_validation_per_class"],
                "n_captures": len(room_b_val),
                "capture_ids": room_b_val,
            },
            indent=2,
        )
    )

    print("captures per room and split:")
    print(captures.pivot_table(index="room", columns="split", values="capture_id",
                               aggfunc="count").to_string())
    print("\ncaptures per activity and split:")
    print(captures.pivot_table(index=["room", "activity"], columns="split",
                               values="capture_id", aggfunc="count").to_string())
    print(f"\n-> {out}")
    print(f"-> {audit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
