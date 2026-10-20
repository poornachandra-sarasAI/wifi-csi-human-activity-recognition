"""CSI quality control (challenge spec hint 4).

Every rejection rule here is applied identically to both rooms and to both
development and validation captures, and every rule reports how many
observations it removed so the report can document what was rejected and
why.

Deliberately NOT done here: wavelet denoising or any other smoothing. The
challenge notes that the source authors' reader does not call their
denoising function in the active path, and treats filtering as an optional
design choice. We keep the amplitude as reconstructed and leave the
decision about smoothing to the normalization stage, where it can be
switched per task and ablated.
"""

from __future__ import annotations

import numpy as np

from src import config


def clean_capture(
    observations: np.ndarray, cfg: config.Config | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply the QC rules to one capture.

    Returns the surviving observations in original temporal order, plus a
    per-rule count of what was removed.
    """
    cfg = cfg or config.load()
    rules = cfg["cleaning"]

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f"expected a 2-D (n_obs, n_subcarriers) array, got {obs.shape}")

    n_start = len(obs)
    keep = np.ones(n_start, dtype=bool)
    report = {
        "qc_non_finite": 0,
        "qc_all_zero": 0,
        "qc_duplicate": 0,
    }

    # Rule 1: any non-finite value makes the whole observation unusable.
    if rules["drop_non_finite"]:
        bad = ~np.isfinite(obs).all(axis=1)
        report["qc_non_finite"] = int((bad & keep).sum())
        keep &= ~bad

    # Rule 2: zero-information CSI. An all-zero row carries no channel
    # measurement; it is an extractor artifact, not a quiet channel.
    if rules["drop_all_zero_rows"]:
        bad = ~np.any(obs != 0, axis=1)
        report["qc_all_zero"] = int((bad & keep).sum())
        keep &= ~bad

    # Rule 3: an observation byte-identical to its predecessor is a repeat
    # of the same measurement rather than a new one. Only *consecutive*
    # duplicates are dropped; two identical rows far apart in a capture can
    # legitimately occur in a static channel.
    if rules["drop_duplicate_consecutive"] and n_start > 1:
        same_as_previous = np.zeros(n_start, dtype=bool)
        same_as_previous[1:] = np.all(obs[1:] == obs[:-1], axis=1)
        report["qc_duplicate"] = int((same_as_previous & keep).sum())
        keep &= ~same_as_previous

    cleaned = obs[keep]
    report["qc_total_removed"] = int(n_start - len(cleaned))
    return cleaned, report


def is_usable(n_obs: int, cfg: config.Config | None = None) -> bool:
    """Does this capture have enough surviving observations to be used?"""
    cfg = cfg or config.load()
    return n_obs >= cfg["cleaning"]["min_obs_per_capture"]
