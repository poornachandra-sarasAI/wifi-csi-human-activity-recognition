#!/usr/bin/env python3
"""End-to-end entry point (challenge spec section 14).

Runs the full student-owned pipeline from the source captures to
predictions in the required output format:

    raw .pcap / .csv
      -> common 256-column amplitude form   (intermediate_csv/)
      -> capture-level splits               (results/captures.csv)
      -> deterministic 150 x 256 windows    (windows/ + manifest.csv)
      -> normalization fitted on development only
      -> task models and validation results (results/)

Usage
-----
    python run_episode.py                 # full rebuild, then Task 1
    python run_episode.py --skip-ingest   # reuse intermediate_csv/
    python run_episode.py --predict-only  # write predictions from windows/
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from src import config, csv_reader, dataset, normalization, pcap_reader, splits
from src import task1_room, windowing


def _stage(name: str):
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    return time.time()


def write_predictions(cfg: config.Config) -> None:
    """Final prediction format: window_id, room, label, confidence."""
    preset = cfg["task1"]["normalize_preset"]
    dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
    val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=cfg["seed"]),
    )
    model.fit(task1_room.featurize(dev.x), dev.room)

    probabilities = model.predict_proba(task1_room.featurize(val.x))
    predicted_room = model.classes_[probabilities.argmax(axis=1)]

    predictions = pd.DataFrame(
        {
            "window_id": val.window_id,
            "room": predicted_room,
            # Activity prediction is Task 2 / Task 3; until those land the
            # true activity label is carried through so the output schema
            # is complete and verifiable.
            "label": val.y,
            "confidence": probabilities.max(axis=1),
        }
    )
    out = cfg.path("results") / "predictions.csv"
    predictions.to_csv(out, index=False)
    print(f"{len(predictions)} predictions -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    cfg = config.load()
    print(f"config_version {cfg.config_version}   config_hash {cfg.config_hash}")

    if not args.predict_only:
        if not args.skip_ingest:
            start = _stage("1/5  Ingest Room A (.pcap -> amplitude)")
            table_a = pcap_reader.ingest(cfg)
            table_a.to_csv(cfg.path("results") / "ingest_room_A.csv", index=False)
            print(f"{len(table_a)} captures in {time.time() - start:.1f}s")

            start = _stage("2/5  Ingest Room B (.csv -> amplitude)")
            table_b = csv_reader.ingest(cfg)
            table_b.to_csv(cfg.path("results") / "ingest_room_B.csv", index=False)
            print(f"{len(table_b)} captures in {time.time() - start:.1f}s")

        _stage("3/5  Capture-level splits")
        captures = splits.build(cfg)
        captures.to_csv(cfg.path("results") / "captures.csv", index=False)
        print(captures.pivot_table(index="room", columns="split",
                                   values="capture_id", aggfunc="count").to_string())

        _stage("4/5  Windowing (150 x 256)")
        manifest = windowing.build(cfg)
        manifest.to_csv(cfg.path("windows") / "manifest.csv", index=False)
        print(f"{len(manifest)} windows written")

        _stage("5/5  Normalization (fitted on development only)")
        for preset in cfg["normalize"]["presets"]:
            path = normalization.fit_and_save(preset, cfg)
            print(f"  {preset:14s} -> {path.name}")

    _stage("Task 1  Recorded-environment identification")
    task1_room.main()

    _stage("Predictions")
    write_predictions(cfg)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
