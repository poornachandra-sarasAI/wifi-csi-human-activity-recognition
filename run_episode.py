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
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, csv_reader, dataset, normalization, pcap_reader, splits
from src import task1_room, windowing


def _stage(name: str):
    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    return time.time()


def predict_episode(cfg: config.Config, episode_dir: Path) -> pd.DataFrame:
    """Ingest a directory of raw captures and predict each one. No labels needed.

    Accepts Room A \u002epcap and Room B \u002ecsv captures in any mixture, at any depth.
    Both normalisation presets used downstream are per-window, so nothing has to
    be fitted against this episode and no room has to be known in advance.
    """
    from src import amplitude, cleaning, csi_decode, csv_reader, predict

    paths = sorted([p for p in episode_dir.rglob("*") if p.suffix in (".pcap", ".csv")])
    if not paths:
        raise FileNotFoundError(f"no .pcap or .csv captures under {episode_dir}")

    layout = csi_decode.layout_from_config(cfg)
    raw, ids, skipped = [], [], []
    for path in paths:
        try:
            if path.suffix == ".pcap":
                frames, _ = pcap_reader.read_capture(path, cfg)
                observations = csi_decode.unpack_frames(frames, layout)
            else:
                observations = csv_reader.read_source_csv(path)
            clean, _ = cleaning.clean_capture(observations, cfg)
            window, _ = windowing.extract(clean, cfg)
        except Exception as exc:                      # a malformed capture must not stop the run
            skipped.append((path.name, repr(exc)))
            continue
        raw.append(window)
        ids.append(path.stem)

    if skipped:
        print(f"  skipped {len(skipped)} unreadable capture(s): "
              + ", ".join(n for n, _ in skipped[:5]))
    if not raw:
        raise RuntimeError("no capture in this episode could be processed")
    print(f"  {len(raw)} captures -> {len(raw)} windows of "
          f"({cfg['windowing']['length']}, {amplitude.N_SUBCARRIERS})")

    presets = cfg["normalize"]["presets"]
    def normalise(preset):
        stats = normalization.fit(preset, cfg)       # per-window scopes fit nothing
        return np.stack([normalization.apply(w, stats, presets[preset]) for w in raw])

    predictor = predict.fit(cfg)
    return predictor(normalise(cfg["task3"]["router_preset"]),
                     normalise(cfg["task2"]["normalize_preset"]),
                     np.array(ids))


def write_predictions(cfg: config.Config, episode_dir: Path | None = None) -> None:
    """Section 10 output contract: window_id, room, label, confidence."""
    from src import predict

    if episode_dir is not None:
        predictions = predict_episode(cfg, episode_dir)
    else:
        predictions = predict.predict_split(splits.LOCAL_VALIDATION, cfg)

    out = cfg.path("results") / "predictions.csv"
    predictions.to_csv(out, index=False)
    print(f"{len(predictions)} predictions -> {out}")
    print(predictions.head().to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--episode", type=Path, default=None,
                        help="directory of raw .pcap/.csv captures to predict; "
                             "labels are not required and are never read")
    args = parser.parse_args()

    cfg = config.load()
    print(f"config_version {cfg.config_version}   config_hash {cfg.config_hash}")

    if not args.predict_only and args.episode is None:
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

    if args.episode is None:
        _stage("Task 1  Recorded-environment identification")
        task1_room.main()

    _stage(f"Predictions{'' if args.episode is None else f'  (episode: {args.episode})'}")
    write_predictions(cfg, args.episode)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
