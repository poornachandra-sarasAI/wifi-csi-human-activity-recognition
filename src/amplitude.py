"""The common intermediate amplitude representation shared by both rooms.

Once a capture is written through this module the downstream pipeline no
longer knows, or needs to know, whether it started as a .pcap or a .csv
(challenge spec hint 3).

Common form
-----------
One CSV per original capture:
    - one row per valid CSI observation, in original temporal order
    - 256 columns, named subcarrier_001 .. subcarrier_256
    - real-valued, non-negative amplitude
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_SUBCARRIERS = 256
COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, N_SUBCARRIERS + 1)]

# Canonical activity vocabulary. Room A and Room B spell the three shared
# activities differently on disk; both map onto SIT / STAND / WALK here.
SHARED = ("SIT", "STAND", "WALK")
ROOM_A_EXCLUSIVE = ("SITDOWN", "STANDUP", "FALL")
ROOM_B_EXCLUSIVE = ("EMPTY", "LYING")
ACTIVITIES = SHARED + ROOM_A_EXCLUSIVE + ROOM_B_EXCLUSIVE


def complex_to_amplitude(csi: np.ndarray) -> np.ndarray:
    """Complex CSI -> real amplitude. Shape is preserved."""
    return np.abs(csi).astype(np.float32)


def to_frame(observations: np.ndarray) -> pd.DataFrame:
    """Wrap an (n_obs, 256) float array in the common schema."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] != N_SUBCARRIERS:
        raise ValueError(
            f"expected (n_obs, {N_SUBCARRIERS}) amplitude array, got {obs.shape}"
        )
    return pd.DataFrame(obs, columns=COLUMNS)


def write_capture(observations: np.ndarray, out_path: Path) -> int:
    """Write one capture in the common form. Returns the row count."""
    frame = to_frame(observations)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, float_format="%.6g")
    return len(frame)


def read_capture(path: Path) -> np.ndarray:
    """Read a common-form capture back as an (n_obs, 256) float32 array."""
    frame = pd.read_csv(path)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path.name} does not use the common schema")
    return frame.to_numpy(dtype=np.float32)
