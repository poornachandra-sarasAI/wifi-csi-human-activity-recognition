"""Room B (Experiment 3) ingestion: amplitude CSV -> common intermediate form.

Room B starts later in the pipeline than Room A: the supplied files already
contain amplitude values with 256 columns, so the work here is schema
reconciliation and quality control rather than signal reconstruction
(challenge spec hint 3).

Two file-naming facts, verified over all 1201 files:

  * 1196 files carry the header "Subcarrier_1 .. Subcarrier_256"; the
    remaining 5 (Empty_11, Lying_0, Sitting_0, Standing_0, Walking_0)
    carry a numeric header "0 .. 255". Both are headers, never data, so
    row 0 is always consumed as the header.
  * Every file is exactly 500 data rows x 256 columns.

The activity is taken from the filename for bookkeeping only. It is
recorded in the manifest as a label; it is never used as a model feature
(challenge spec section 8, "no metadata shortcuts").

Run:  python -m src.csv_reader
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import amplitude, cleaning, config

# Filename stem prefix -> canonical activity label.
_ACTIVITY_PREFIXES = {
    "Empty": "EMPTY",
    "Lying": "LYING",
    "Sitting": "SIT",
    "Standing": "STAND",
    "Walking": "WALK",
}

_STEM_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)(?:_Amp)?_(?P<index>\d+)$")


def parse_filename(path: Path) -> tuple[str, str]:
    """Return (activity, capture_id) for a Room B source file."""
    match = _STEM_RE.match(path.stem)
    if match is None:
        raise ValueError(f"unrecognised Room B filename: {path.name}")
    prefix = match.group("prefix")
    try:
        activity = _ACTIVITY_PREFIXES[prefix]
    except KeyError as exc:
        raise ValueError(f"unknown activity prefix {prefix!r} in {path.name}") from exc
    # Keep the original stem inside the capture id so every derived row can
    # be traced back to the exact source file (hint 9).
    return activity, f"roomB__{path.stem}"


def read_source_csv(path: Path, expected_rows: int | None = None) -> np.ndarray:
    """Read one Room B source file as an (n_obs, 256) float32 array."""
    frame = pd.read_csv(path, header=0)

    if frame.shape[1] != amplitude.N_SUBCARRIERS:
        raise ValueError(
            f"{path.name}: expected {amplitude.N_SUBCARRIERS} columns, "
            f"got {frame.shape[1]}"
        )
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"{path.name}: expected {expected_rows} data rows, got {len(frame)}"
        )

    values = frame.to_numpy(dtype=np.float32)
    if np.any(values < 0):
        raise ValueError(f"{path.name}: negative amplitude values present")
    return values


def ingest(cfg: config.Config | None = None, limit: int | None = None) -> pd.DataFrame:
    """Convert every Room B capture into the common intermediate form."""
    cfg = cfg or config.load()
    src_dir = cfg.path("raw_room_b")
    out_dir = cfg.path("intermediate") / "room_B"
    out_dir.mkdir(parents=True, exist_ok=True)

    expected_rows = cfg["ingest"]["room_b"]["expected_rows"]
    sources = sorted(src_dir.glob("*.csv"))
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        raise FileNotFoundError(f"no Room B CSV files under {src_dir}")

    records = []
    for path in sources:
        activity, capture_id = parse_filename(path)
        raw = read_source_csv(path, expected_rows=expected_rows)
        clean, report = cleaning.clean_capture(raw, cfg)

        n_written = amplitude.write_capture(clean, out_dir / f"{capture_id}.csv")
        records.append(
            {
                "capture_id": capture_id,
                "room": "B",
                "activity": activity,
                "source_format": "csv",
                "source_file": path.name,
                "n_obs_raw": len(raw),
                "n_obs_clean": n_written,
                **report,
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    cfg = config.load()
    table = ingest(cfg)

    out = cfg.path("results") / "ingest_room_B.csv"
    table.to_csv(out, index=False)

    print(f"Room B: {len(table)} captures -> {cfg.path('intermediate') / 'room_B'}")
    print("\nper-activity capture counts:")
    print(table["activity"].value_counts().to_string())
    print("\nobservations per capture after QC:")
    print(table["n_obs_clean"].describe()[["min", "50%", "max"]].to_string())
    rejected = table["n_obs_raw"].sum() - table["n_obs_clean"].sum()
    print(f"\nobservations rejected by QC: {rejected}")
    print(f"ingest report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
