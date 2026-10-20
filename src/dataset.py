"""Shared window loader - the single integration point for all three tasks.

FROZEN CONTRACT. Task 2 and Task 3 build on this signature; changing it
breaks their code, so raise an issue rather than editing it in place.

    ds = dataset.load("task1", split="development")
    ds.x        (n_windows, 150, 256)        float32, normalized
    ds.y        (n_windows,)                 activity labels
    ds.room     (n_windows,)                 "A" / "B"
    ds.window_id, ds.capture_id              provenance

Windows are stored on disk UN-normalized. The preset named here is applied
at load time, so all three tasks share one windows/ directory and nobody
has to regenerate anything to change normalization.

Metadata rule (challenge spec section 8)
----------------------------------------
`room` and `activity` are labels: use them for supervision and scoring.
`capture_id`, `source_format` and `n_obs_in_capture` are provenance ONLY
and must never be used as model features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config, normalization, splits


@dataclass
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    room: np.ndarray
    window_id: np.ndarray
    capture_id: np.ndarray
    preset: str

    def __len__(self) -> int:
        return len(self.x)

    def filter(self, *, room: str | None = None, activities=None) -> "WindowSet":
        mask = np.ones(len(self), dtype=bool)
        if room is not None:
            mask &= self.room == room
        if activities is not None:
            mask &= np.isin(self.y, list(activities))
        return WindowSet(
            x=self.x[mask],
            y=self.y[mask],
            room=self.room[mask],
            window_id=self.window_id[mask],
            capture_id=self.capture_id[mask],
            preset=self.preset,
        )

    def __repr__(self) -> str:
        return (
            f"WindowSet(n={len(self)}, shape={self.x.shape[1:]}, preset={self.preset!r}, "
            f"rooms={dict(zip(*np.unique(self.room, return_counts=True)))})"
        )


def load(
    preset: str,
    split: str = splits.DEVELOPMENT,
    cfg: config.Config | None = None,
    stats: normalization.Stats | None = None,
    rooms: str | list[str] | None = None,
) -> WindowSet:
    """Load and normalize one split.

    `stats` are always fitted on development data. Pass the same Stats
    object when loading validation so the identical transform is applied
    without refitting (spec section 8).
    """
    cfg = cfg or config.load()
    preset_cfg = cfg["normalize"]["presets"][preset]

    if stats is None:
        cached = cfg.path("results") / f"norm_{preset}.npz"
        stats = (
            normalization.Stats.load(cached)
            if cached.exists()
            else normalization.fit(preset, cfg)
        )

    manifest = pd.read_csv(cfg.path("windows") / "manifest.csv")
    rows = manifest[manifest["split"] == split]
    if rooms is not None:
        wanted = [rooms] if isinstance(rooms, str) else list(rooms)
        rows = rows[rows["room"].isin(wanted)]
    if rows.empty:
        raise ValueError(f"no windows for split={split!r} rooms={rooms!r}")

    windows = []
    for row in rows.itertuples(index=False):
        raw = np.load(cfg.path("windows") / row.split / f"{row.window_id}.npy")
        windows.append(normalization.apply(raw, stats, preset_cfg, room=row.room))

    return WindowSet(
        x=np.stack(windows).astype(np.float32),
        y=rows["activity"].to_numpy(),
        room=rows["room"].to_numpy(),
        window_id=rows["window_id"].to_numpy(),
        capture_id=rows["capture_id"].to_numpy(),
        preset=preset,
    )


if __name__ == "__main__":
    cfg = config.load()
    for preset in cfg["normalize"]["presets"]:
        dev = load(preset, splits.DEVELOPMENT, cfg)
        print(f"{preset:14s} {dev}")
