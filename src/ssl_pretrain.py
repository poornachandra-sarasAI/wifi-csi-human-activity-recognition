"""Task 2 self-supervised pretraining: SimCLR on CSI windows, no labels.

Two augmented views of the same window are pulled together and every other
window in the batch is pushed away (NT-Xent loss). The encoder never sees an
activity or a room label: `pretrain()` takes the window array and nothing
else, which is how rule 4 of HANDOFF.md is enforced in code rather than by
convention.

Augmentations
-------------
Applied to channel 0 (the globally scaled window); channel 1 is then rebuilt
as channel 0 minus its time-mean, exactly as normalization.apply() derives
it, so the two views stay consistent.

    random temporal crop      a 60-100% slice, resized back to 150 steps
    stride-2 subsampling      random phase; keeps one of Room A's two
                              interleaved frame streams (HANDOFF section 5)
    local shuffle             permute observations within spans of 4
    subcarrier band masking   zero one contiguous run of subcarriers
    per-subcarrier offset     a gain change, since the data is log-scaled
    additive noise
    time reversal

The two adjacency-breaking augmentations (subsampling, local shuffle) are
switched off in the `no_adjacency_break` variant; comparing the variants is
the interleaving-artifact ablation.

Training uses a fixed epoch budget from the config. There is no early
stopping and no checkpoint selection, so local_validation plays no part.

Run:  python -m src.ssl_pretrain                 # every variant x seed
      python -m src.ssl_pretrain --variant full --seeds 0
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src import config, encoder


# ---------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------


def augment_settings(cfg: config.Config, variant: str) -> dict:
    """Base augmentation settings with the variant's overrides applied."""
    settings = dict(cfg["task2"]["augment"])
    settings.update(cfg["task2"]["variants"][variant] or {})
    return settings


def _source_positions(n: int, steps: int, settings: dict, gen: torch.Generator):
    """Where in the original window each output step reads from.

    Subsampling, crop + resize, local shuffle and reversal are all
    re-indexings of the time axis, so they compose into ONE fractional
    source position per output step. Returns (lo, hi, weight), each
    (n, steps): output = (1 - weight) * x[lo] + weight * x[hi].

    Interpolation happens between neighbours of the SUBSAMPLED sequence, so
    a stride-2 view never blends the two interleaved Room A frame streams.
    """
    def coins(p: float) -> torch.Tensor:
        return torch.rand(n, generator=gen) < p

    subsample = coins(settings["subsample_p"])
    stride = torch.where(subsample, 2, 1)
    phase = torch.where(subsample, torch.randint(0, 2, (n,), generator=gen), 0)
    available = (steps - phase + stride - 1) // stride  # length after subsampling

    low = settings["crop_min_fraction"]
    fraction = low + (1 - low) * torch.rand(n, generator=gen)
    length = (fraction * available).round().clamp(min=8).minimum(available)
    start = (torch.rand(n, generator=gen) * (available - length + 1)).floor()

    # Output step j -> fractional index into the cropped, subsampled sequence.
    j = torch.arange(steps, dtype=torch.float32).expand(n, steps)

    shuffle = coins(settings["local_shuffle_p"])
    span = settings["local_shuffle_span"]
    n_full = steps // span * span
    keys = torch.rand(n, n_full // span, span, generator=gen).argsort(dim=2)
    perm = (keys + torch.arange(0, n_full, span).view(1, -1, 1)).reshape(n, n_full)
    perm = torch.cat([perm, torch.arange(n_full, steps).expand(n, -1)], dim=1).float()
    j = torch.where(shuffle.unsqueeze(1), perm, j)

    reverse = coins(settings["reverse_p"])
    j = torch.where(reverse.unsqueeze(1), steps - 1 - j, j)

    u = start.unsqueeze(1) + j * ((length - 1) / (steps - 1)).unsqueeze(1)
    u_lo = u.floor()
    u_hi = (u_lo + 1).minimum(available.unsqueeze(1) - 1)
    weight = u - u_lo
    to_original = lambda k: (phase.unsqueeze(1) + stride.unsqueeze(1) * k).long()
    return to_original(u_lo), to_original(u_hi), weight


def augment(batch: torch.Tensor, settings: dict, gen: torch.Generator) -> torch.Tensor:
    """(batch, 150, 256, 2) -> one augmented view of the same shape.

    Small random choices are drawn on the CPU from `gen` so runs are
    reproducible on any device; the heavy tensors stay on batch.device.
    """
    n, steps, n_sub = batch.shape[:3]
    device = batch.device
    static = batch[..., 0]

    lo, hi, weight = (t.to(device) for t in _source_positions(n, steps, settings, gen))
    take = lambda index: static.gather(1, index.unsqueeze(2).expand(-1, -1, n_sub))
    weight = weight.unsqueeze(2)
    x = (1 - weight) * take(lo) + weight * take(hi)

    masked = torch.rand(n, generator=gen) < settings["band_mask_p"]
    width = torch.randint(1, settings["band_mask_max_width"] + 1, (n,), generator=gen)
    first = (torch.rand(n, generator=gen) * (n_sub - width + 1)).floor().long()
    columns = torch.arange(n_sub)
    band = (columns >= first.unsqueeze(1)) & (columns < (first + width).unsqueeze(1))
    keep = ~(band & masked.unsqueeze(1))
    x = x * keep.to(device, x.dtype).unsqueeze(1)

    offset = torch.randn(n, 1, n_sub, generator=gen) * settings["offset_std"]
    noise = torch.randn(n, steps, n_sub, generator=gen) * settings["noise_std"]
    x = x + offset.to(device) + noise.to(device)

    dynamic = x - x.mean(dim=1, keepdim=True)
    return torch.stack([x, dynamic], dim=-1)


# ---------------------------------------------------------------------
# Contrastive objective
# ---------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """MLP used only during pretraining and discarded afterwards."""

    def __init__(self, embed_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, projection_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR loss for two (batch, dim) projections of the same windows."""
    z = F.normalize(torch.cat([a, b]), dim=1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(float("-inf"))
    n = len(a)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def pretrain(
    x: np.ndarray,
    cfg: config.Config | None = None,
    variant: str = "full",
    seed: int = 0,
    device: str | torch.device | None = None,
    epochs: int | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, list[float]]:
    """Train an encoder on unlabelled windows. Returns (encoder, loss per epoch).

    `x` is the ONLY data argument: no labels, no room, no capture ids.
    """
    cfg = cfg or config.load()
    settings = cfg["task2"]["pretrain"]
    aug = augment_settings(cfg, variant)
    epochs = epochs if epochs is not None else settings["epochs"]
    device = torch.device(device) if device is not None else encoder.default_device()

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected (n, 150, 256, 2) windows, got {x.shape}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True  # fixed input shape: let cuDNN pick kernels
    model = encoder.build_encoder(cfg, seed=seed).to(device)
    head = ProjectionHead(model.embed_dim, settings["projection_dim"]).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=settings["weight_decay"])

    batch_size = min(settings["batch_size"], len(x))
    steps_per_epoch = max(len(x) // batch_size, 1)  # drop the ragged last batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)

    data = torch.from_numpy(x).to(device)
    gen = torch.Generator().manual_seed(seed)
    history: list[float] = []
    start = time.time()

    model.train()
    head.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        losses = []
        for step in range(steps_per_epoch):
            batch = data[order[step * batch_size : (step + 1) * batch_size].to(device)]
            view_a = augment(batch, aug, gen)
            view_b = augment(batch, aug, gen)
            loss = nt_xent(head(model(view_a)), head(model(view_b)), settings["temperature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        if verbose and (epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == epochs):
            print(f"    epoch {epoch + 1:4d}/{epochs}  loss {history[-1]:.4f}  "
                  f"({time.time() - start:.0f}s)")

    return model.eval(), history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=None,
                        help="variant name from task2.variants (repeatable; default all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override, for smoke runs")
    args = parser.parse_args()

    from src import dataset, splits

    cfg = config.load()
    variants = args.variant or list(cfg["task2"]["variants"])
    seeds = args.seeds if args.seeds is not None else cfg["task2"]["pretrain"]["seeds"]

    # Labels are loaded alongside by the shared loader and dropped right here.
    x = dataset.load(cfg["task2"]["normalize_preset"], splits.DEVELOPMENT, cfg).x
    print(f"pretraining on {len(x)} development windows, shape {x.shape[1:]}, "
          f"device {encoder.default_device()}")

    encoder.MODELS_DIR.mkdir(exist_ok=True)
    loss_path = cfg.path("results") / "task2_pretrain_loss.csv"
    curves = pd.read_csv(loss_path) if loss_path.exists() else pd.DataFrame()

    for variant in variants:
        for seed in seeds:
            print(f"\n  variant={variant} seed={seed}")
            model, history = pretrain(x, cfg, variant, seed, epochs=args.epochs)
            path = encoder.checkpoint_path(variant, seed)
            torch.save(model.state_dict(), path)
            print(f"    -> {path.relative_to(config.REPO_ROOT)}")

            curve = pd.DataFrame({
                "variant": variant, "seed": seed,
                "epoch": np.arange(1, len(history) + 1), "loss": history,
            })
            if not curves.empty:
                curves = curves[~((curves.variant == variant) & (curves.seed == seed))]
            curves = pd.concat([curves, curve], ignore_index=True)

    curves.to_csv(loss_path, index=False)
    print(f"\nloss curves -> {loss_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
