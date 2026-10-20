"""Deterministic (150, 256) window construction (challenge spec hint 7).

The official benchmark input is a fixed window of 150 consecutive valid CSI
observations by 256 subcarriers. Two rules constrain how it is built:

  * observations from different captures are never mixed into one window
  * no labels and no validation statistics are used to choose the window

Selection strategy
------------------
`center` takes the middle 150 observations of the capture. Room A captures
run 112-251 frames and Room B captures are a uniform 500, so a centred crop
avoids both the ping ramp-up at the start and any trailing silence, and it
is a pure function of the capture length - no data is inspected to pick it.

Short captures
--------------
12 of the 1084 Room A captures hold fewer than 150 observations, and three
of those are in the SUPPLIED validation set. Dropping them would silently
change the validation denominator, so they are reflect-padded up to 150
instead and flagged in the manifest via `was_padded`.

Run:  python -m src.windowing
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import amplitude, config


def window_bounds(n_obs: int, length: int, selection: str) -> tuple[int, int]:
    """Start/stop indices of the single window taken from a capture."""
    if selection == "first":
        start = 0
    elif selection == "center":
        start = max((n_obs - length) // 2, 0)
    else:
        raise ValueError(f"unknown window selection strategy: {selection!r}")
    return start, min(start + length, n_obs)


def extract(
    observations: np.ndarray, cfg: config.Config | None = None
) -> tuple[np.ndarray, dict]:
    """Take one (length, 256) window from a capture's observations."""
    cfg = cfg or config.load()
    settings = cfg["windowing"]
    length = settings["length"]

    obs = np.asarray(observations, dtype=np.float32)
    n_obs = len(obs)
    start, stop = window_bounds(n_obs, length, settings["selection"])
    window = obs[start:stop]

    was_padded = False
    if len(window) < length:
        if settings["pad_short_captures"] == "drop":
            raise ValueError(f"capture too short: {n_obs} < {length}")
        # Reflect-pad: repeats the capture's own observations rather than
        # inventing values or padding with zeros, which would look like
        # invalid CSI to the model.
        pad = length - len(window)
        window = np.pad(window, ((0, pad), (0, 0)), mode="reflect")
        was_padded = True

    meta = {
        "start_idx": int(start),
        "n_obs_in_capture": int(n_obs),
        "was_padded": was_padded,
    }
    return window.astype(np.float32), meta


def build(cfg: config.Config | None = None) -> pd.DataFrame:
    """Write one window per capture and return the manifest rows."""
    cfg = cfg or config.load()
    captures = pd.read_csv(cfg.path("results") / "captures.csv")
    windows_root = cfg.path("windows")
    config_hash = cfg.config_hash

    rows = []
    for i, capture in enumerate(captures.itertuples(index=False)):
        room_dir = cfg.path("intermediate") / f"room_{capture.room}"
        observations = amplitude.read_capture(room_dir / f"{capture.capture_id}.csv")
        window, meta = extract(observations, cfg)

        window_id = f"window_{i:05d}"
        out_dir = windows_root / capture.split
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{window_id}.npy", window)

        rows.append(
            {
                "window_id": window_id,
                "capture_id": capture.capture_id,
                "room": capture.room,
                "activity": capture.activity,
                "split": capture.split,
                "validation_origin": capture.validation_origin,
                "source_format": capture.source_format,
                "config_hash": config_hash,
                **meta,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    cfg = config.load()
    manifest = build(cfg)

    out = cfg.path("windows") / "manifest.csv"
    manifest.to_csv(out, index=False)

    print(f"{len(manifest)} windows of shape "
          f"({cfg['windowing']['length']}, {amplitude.N_SUBCARRIERS})")
    print("\nwindows per room and split:")
    print(manifest.pivot_table(index="room", columns="split", values="window_id",
                               aggfunc="count").to_string())
    print(f"\npadded (short) captures: {int(manifest.was_padded.sum())}")
    print(f"config_hash: {cfg.config_hash}")
    print(f"\nmanifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
