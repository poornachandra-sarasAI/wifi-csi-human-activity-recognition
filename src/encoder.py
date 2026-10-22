"""Task 2 encoder: a small 1-D temporal CNN over (150, 256, 2) CSI windows.

INTERFACE CONTRACT - Task 3 imports these, so the signatures stay stable:

    model = build_encoder(cfg)                 # untrained torch.nn.Module
    model = load_pretrained(cfg)               # weights from ssl_pretrain
    z     = embed(model, x)                    # (n, 150, 256, 2) -> (n, d) float32

Architecture
------------
The 256 subcarriers x 2 channels (static view, time-mean-removed view) of
each observation are treated as 512 input features per time step:

    1x1 conv 512 -> 64          mixes subcarriers, no temporal context
    3 dilated conv blocks       kernel 5, dilation 1/2/4, residual, 128 wide
    mean + max pool over time   length-invariant summary
    linear 256 -> 128           the embedding

About 0.27 M parameters. The development set holds only 2,010 windows, so
the encoder is kept deliberately small.

Run:  python -m src.encoder      # prints the configured architecture
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import config

N_STEPS = 150
N_SUBCARRIERS = 256
N_CHANNELS = 2

MODELS_DIR = config.REPO_ROOT / "models"


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(c_out)
        self.act = nn.GELU()
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x))) + self.skip(x)


class CsiEncoder(nn.Module):
    def __init__(
        self,
        reduce_channels: int = 64,
        hidden_channels: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4),
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.reduce = nn.Sequential(
            nn.Conv1d(N_SUBCARRIERS * N_CHANNELS, reduce_channels, 1),
            nn.BatchNorm1d(reduce_channels),
            nn.GELU(),
        )
        blocks, width = [], reduce_channels
        for dilation in dilations:
            blocks.append(_ConvBlock(width, hidden_channels, kernel_size, dilation))
            width = hidden_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(2 * hidden_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time, 256, 2) -> (batch, embed_dim)."""
        batch, steps = x.shape[:2]
        h = x.reshape(batch, steps, -1).transpose(1, 2)  # (batch, 512, time)
        h = self.blocks(self.reduce(h))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(pooled)


def build_encoder(cfg: config.Config | None = None, seed: int | None = None) -> nn.Module:
    """Untrained encoder from the `task2.encoder` config block."""
    cfg = cfg or config.load()
    settings = cfg["task2"]["encoder"]
    if seed is not None:
        torch.manual_seed(seed)
    return CsiEncoder(
        reduce_channels=settings["reduce_channels"],
        hidden_channels=settings["hidden_channels"],
        kernel_size=settings["kernel_size"],
        dilations=tuple(settings["dilations"]),
        embed_dim=settings["embed_dim"],
    )


def _as_two_channel(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        raise ValueError(
            "embed() expects the two-channel 'task2' preset, shape (n, 150, 256, 2); "
            "load windows with dataset.load('task2', ...)"
        )
    if x.ndim != 4 or x.shape[2:] != (N_SUBCARRIERS, N_CHANNELS):
        raise ValueError(f"expected (n, time, {N_SUBCARRIERS}, {N_CHANNELS}), got {x.shape}")
    return x


@torch.no_grad()
def embed(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Frozen embeddings: (n, 150, 256, 2) float array -> (n, d) float32.

    Always runs in eval mode (BatchNorm uses its running statistics), so the
    result does not depend on batch size or on which other windows are
    embedded alongside.
    """
    x = _as_two_channel(np.asarray(x, dtype=np.float32))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out = [
        model(torch.from_numpy(x[i : i + batch_size]).to(device)).cpu().numpy()
        for i in range(0, len(x), batch_size)
    ]
    model.train(was_training)
    if not out:
        return np.empty((0, model.embed_dim), dtype=np.float32)
    return np.concatenate(out).astype(np.float32)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(variant: str = "full", seed: int = 0) -> Path:
    return MODELS_DIR / f"task2_encoder_{variant}_seed{seed}.pt"


def load_pretrained(
    cfg: config.Config | None = None,
    variant: str = "full",
    seed: int = 0,
    device: str | torch.device | None = None,
) -> nn.Module:
    """Encoder with self-supervised weights, in eval mode.

    Weights are gitignored (*.pt); build them with `python -m src.ssl_pretrain`.
    """
    path = checkpoint_path(variant, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ssl_pretrain` first")
    device = torch.device(device) if device is not None else default_device()
    model = build_encoder(cfg)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = build_encoder(seed=0)
    print(model)
    print(f"\nparameters: {n_parameters(model):,}   embed_dim: {model.embed_dim}")
    sys.exit(0)
