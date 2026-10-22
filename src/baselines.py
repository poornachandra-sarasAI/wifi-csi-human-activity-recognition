"""Task 2 baselines that the self-supervised encoder must be compared against.

Both are required by the challenge (spec section 7):

  handcrafted   fixed signal statistics of each window, no learning at all
  random        the SAME encoder architecture with random weights

Handcrafted features
--------------------
Computed from channel 0 of the window alone (rule 3 of HANDOFF.md: nothing
from capture length, filenames or ids). The 256 subcarriers are averaged
into 16 groups of 16 to tame per-subcarrier noise, then per group:

    mean, std, range, skewness, kurtosis   amplitude distribution
    mean |first difference|                motion energy
    temporal-spectrum energy in 4 bands    slow -> fast fluctuation

plus the per-subcarrier temporal std (256 values), the classic CSI motion
indicator. 16 x 10 + 256 = 416 features.

Random-weight encoder
---------------------
An untrained network's BatchNorm layers start at mean 0 / variance 1, which
mis-scales activations for this input. To keep the comparison about the
WEIGHTS rather than about BatchNorm, the running statistics are recalibrated
with label-free forward passes over development windows; no weight is
changed.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from src import config, encoder

N_GROUPS = 16
N_BANDS = 4


def _static(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[..., 0] if x.ndim == 4 else x


def handcrafted_features(x: np.ndarray) -> np.ndarray:
    """(n, 150, 256[, 2]) windows -> (n, 416) float32 features."""
    w = _static(x)
    n, steps, n_sub = w.shape
    grouped = w.reshape(n, steps, N_GROUPS, n_sub // N_GROUPS).mean(axis=3)

    diff = np.abs(np.diff(grouped, axis=1)).mean(axis=1)
    spectrum = np.abs(np.fft.rfft(grouped - grouped.mean(axis=1, keepdims=True), axis=1)) ** 2
    bands = np.array_split(np.arange(1, spectrum.shape[1]), N_BANDS)  # skip DC
    band_energy = np.stack([np.log1p(spectrum[:, b].sum(axis=1)) for b in bands], axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.nan_to_num(stats.skew(grouped, axis=1))
        kurt = np.nan_to_num(stats.kurtosis(grouped, axis=1))

    features = [
        grouped.mean(axis=1),
        grouped.std(axis=1),
        np.ptp(grouped, axis=1),
        skew,
        kurt,
        diff,
        band_energy.reshape(n, -1),
        w.std(axis=1),
    ]
    return np.concatenate(features, axis=1).astype(np.float32)


@torch.no_grad()
def recalibrate_batchnorm(model: torch.nn.Module, x: np.ndarray, batch_size: int = 256) -> None:
    """Re-estimate BatchNorm running statistics from x. Weights are untouched."""
    device = next(model.parameters()).device
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            module.momentum = None  # cumulative average over all batches
    model.train()
    for i in range(0, len(x), batch_size):
        model(torch.from_numpy(np.asarray(x[i : i + batch_size], dtype=np.float32)).to(device))
    model.eval()


def random_encoder(
    x_dev: np.ndarray,
    cfg: config.Config | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> torch.nn.Module:
    """Same architecture as the SSL encoder, random weights, BatchNorm fitted on dev."""
    device = torch.device(device) if device is not None else encoder.default_device()
    model = encoder.build_encoder(cfg, seed=seed).to(device)
    recalibrate_batchnorm(model, x_dev)
    return model
