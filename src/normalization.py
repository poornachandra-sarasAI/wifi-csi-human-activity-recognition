"""Leakage-safe normalization (challenge spec hint 8, section 8).

Scope decides what information survives
---------------------------------------
    global      one transform for all data     room signature PRESERVED
    per_room    fitted separately per room      room signature REMOVED
    per_window  fitted on the 150x256 array     room + slow context REMOVED

Task 1 needs the room signature kept; Task 3 needs it removed. The three
tasks therefore select different presets from configs/experiment.yaml;
they all apply the SAME transform to development and validation data,
which is what section 10 requires.

Leakage control
---------------
`fit()` filters the manifest to split == "development" itself. No caller
can hand it validation rows, because no caller hands it rows at all. The
capture ids that contributed to a Stats object are recorded inside it so
tests/test_no_leakage.py can prove the fit never touched validation.

Two measured facts shape the defaults
-------------------------------------
  * Room A amplitudes (our decode) and Room B amplitudes (supplied) differ
    by a global factor of ~490x. That gap is an artifact of the authors'
    unknown integer scaling, not a property of the rooms, so absolute
    amplitude is NOT a trustworthy room feature.
  * Room A shows a ~4x gain step across subcarriers ~137-187 (the third
    20 MHz sub-band) that Room B does not. It is a receiver AGC effect,
    constant across all Room A captures and activities.
Both survive `global` scope. See the README for how Task 1 reports around
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import amplitude, config, splits

EPS = 1e-8


@dataclass
class Stats:
    """Fitted normalization statistics plus the provenance of the fit."""

    scope: str
    center: dict[str, np.ndarray] = field(default_factory=dict)
    scale: dict[str, np.ndarray] = field(default_factory=dict)
    fitted_on_captures: tuple[str, ...] = ()
    log1p: bool = True
    robust: bool = True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": self.scope,
            "log1p": self.log1p,
            "robust": self.robust,
            "fitted_on_captures": np.array(self.fitted_on_captures, dtype=object),
        }
        for key, value in self.center.items():
            payload[f"center__{key}"] = value
        for key, value in self.scale.items():
            payload[f"scale__{key}"] = value
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "Stats":
        blob = np.load(path, allow_pickle=True)
        return cls(
            scope=str(blob["scope"]),
            center={
                k.removeprefix("center__"): blob[k] for k in blob if k.startswith("center__")
            },
            scale={
                k.removeprefix("scale__"): blob[k] for k in blob if k.startswith("scale__")
            },
            fitted_on_captures=tuple(blob["fitted_on_captures"].tolist()),
            log1p=bool(blob["log1p"]),
            robust=bool(blob["robust"]),
        )


def _center_scale(values: np.ndarray, robust: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-subcarrier location and spread."""
    if robust:
        center = np.median(values, axis=0)
        q75, q25 = np.percentile(values, [75, 25], axis=0)
        scale = q75 - q25
    else:
        center = values.mean(axis=0)
        scale = values.std(axis=0)
    return center.astype(np.float32), np.maximum(scale, EPS).astype(np.float32)


def fit(
    preset: str,
    cfg: config.Config | None = None,
    max_captures: int | None = 400,
) -> Stats:
    """Fit statistics for a preset, using DEVELOPMENT windows only."""
    cfg = cfg or config.load()
    settings = cfg["normalize"]
    scope = settings["presets"][preset]["scope"]
    log1p, robust = settings["log1p"], settings["robust"]

    stats = Stats(scope=scope, log1p=log1p, robust=robust)
    if scope == "per_window":
        # Nothing to fit: a per-window transform is a deterministic
        # function of the sample itself and so is leakage-free.
        return stats

    manifest = pd.read_csv(cfg.path("windows") / "manifest.csv")
    development = manifest[manifest["split"] == splits.DEVELOPMENT]
    if development.empty:
        raise RuntimeError("no development windows found")

    groups = {"all": development} if scope == "global" else {
        room: rows for room, rows in development.groupby("room")
    }

    used: list[str] = []
    for key, rows in groups.items():
        if max_captures is not None and len(rows) > max_captures:
            rows = rows.sample(max_captures, random_state=cfg["seed"])
        stacked = []
        for row in rows.itertuples(index=False):
            window = np.load(cfg.path("windows") / row.split / f"{row.window_id}.npy")
            stacked.append(np.log1p(window) if log1p else window)
            used.append(row.capture_id)
        pooled = np.concatenate(stacked, axis=0)
        stats.center[key], stats.scale[key] = _center_scale(pooled, robust)

    stats.fitted_on_captures = tuple(sorted(set(used)))
    return stats


def apply(
    window: np.ndarray,
    stats: Stats,
    preset_cfg: dict,
    room: str | None = None,
) -> np.ndarray:
    """Apply a fitted preset to one (150, 256) window."""
    out = np.asarray(window, dtype=np.float32)
    if stats.log1p:
        out = np.log1p(out)

    detrend = preset_cfg["detrend"]
    static = None
    if detrend in ("per_window", "two_channel"):
        static = out.mean(axis=0, keepdims=True)

    if stats.scope == "per_window":
        center, scale = _center_scale(out, stats.robust)
    else:
        key = "all" if stats.scope == "global" else room
        if key not in stats.center:
            raise KeyError(f"no fitted statistics for scope key {key!r}")
        center, scale = stats.center[key], stats.scale[key]

    scaled = (out - center) / scale

    if detrend == "none":
        return scaled.astype(np.float32)
    if detrend == "per_window":
        return (scaled - scaled.mean(axis=0, keepdims=True)).astype(np.float32)
    if detrend == "two_channel":
        dynamic = scaled - scaled.mean(axis=0, keepdims=True)
        return np.stack([scaled, dynamic], axis=-1).astype(np.float32)
    raise ValueError(f"unknown detrend mode: {detrend!r}")


def fit_and_save(preset: str, cfg: config.Config | None = None) -> Path:
    cfg = cfg or config.load()
    stats = fit(preset, cfg)
    path = cfg.path("results") / f"norm_{preset}.npz"
    stats.save(path)
    return path


def main() -> int:
    cfg = config.load()
    for preset in cfg["normalize"]["presets"]:
        path = fit_and_save(preset, cfg)
        stats = Stats.load(path)
        print(f"{preset:14s} scope={stats.scope:10s} "
              f"fitted on {len(stats.fitted_on_captures):4d} development captures "
              f"-> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
