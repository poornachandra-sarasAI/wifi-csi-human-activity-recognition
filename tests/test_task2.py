"""Task 2 contract and leakage tests. Synthetic arrays only: no dataset needed."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src import baselines, config, dataset, encoder, ssl_pretrain, task2_probe

cfg = config.load()
rng = np.random.default_rng(0)


def _windows(n: int) -> np.ndarray:
    static = rng.normal(size=(n, 150, 256)).astype(np.float32)
    return np.stack([static, static - static.mean(axis=1, keepdims=True)], axis=-1)


def test_embed_contract():
    """embed: (n, 150, 256, 2) -> (n, embed_dim) float32, the interface Task 3 uses."""
    model = encoder.build_encoder(cfg, seed=0)
    z = encoder.embed(model, _windows(5))
    assert z.shape == (5, cfg["task2"]["encoder"]["embed_dim"])
    assert z.dtype == np.float32
    assert np.isfinite(z).all()


def test_embed_is_deterministic_and_batch_invariant():
    model = encoder.build_encoder(cfg, seed=0)
    x = _windows(7)
    full = encoder.embed(model, x, batch_size=7)
    np.testing.assert_allclose(full, encoder.embed(model, x, batch_size=2), atol=1e-5)
    np.testing.assert_allclose(full, encoder.embed(model, x, batch_size=7), atol=1e-6)


def test_embed_rejects_single_channel_input():
    with pytest.raises(ValueError, match="two-channel"):
        encoder.embed(encoder.build_encoder(cfg), _windows(2)[..., 0])


def test_encoder_is_small():
    """2,010 development windows: keep the encoder well under a million weights."""
    assert encoder.n_parameters(encoder.build_encoder(cfg)) < 500_000


def test_pretraining_takes_no_labels():
    """HANDOFF rule 4: the only data argument to pretrain() is the window array."""
    params = set(inspect.signature(ssl_pretrain.pretrain).parameters)
    assert params == {"x", "cfg", "variant", "seed", "device", "epochs", "verbose"}
    forbidden = set(dataset.WindowSet.__dataclass_fields__) - {"x"}
    assert not params & forbidden


def test_pretrain_smoke():
    model, history = ssl_pretrain.pretrain(
        _windows(16), cfg, seed=0, device="cpu", epochs=1, verbose=False
    )
    assert len(history) == 1 and np.isfinite(history[0])
    assert not model.training


@pytest.mark.parametrize("variant", list(cfg["task2"]["variants"]))
def test_augment_keeps_channels_consistent(variant):
    """Channel 1 must stay channel 0 minus its time-mean, as normalization builds it."""
    settings = ssl_pretrain.augment_settings(cfg, variant)
    view = ssl_pretrain.augment(torch.from_numpy(_windows(4)), settings,
                                torch.Generator().manual_seed(0))
    assert view.shape == (4, 150, 256, 2)
    static, dynamic = view[..., 0], view[..., 1]
    torch.testing.assert_close(dynamic, static - static.mean(dim=1, keepdim=True))


def test_subsampled_views_never_mix_the_two_frame_streams():
    """Room A alternates two frame streams; a stride-2 view must read one parity only."""
    settings = {**ssl_pretrain.augment_settings(cfg, "full"),
                "subsample_p": 1.0, "local_shuffle_p": 0.0}
    lo, hi, _ = ssl_pretrain._source_positions(64, 150, settings, torch.Generator().manual_seed(1))
    assert ((lo % 2) == (hi % 2)).all()
    assert ((lo % 2) == (lo[:, :1] % 2)).all()  # one phase per view
    assert lo.min() >= 0 and hi.max() <= 149


def test_positions_stay_in_range_for_every_variant():
    for variant in cfg["task2"]["variants"]:
        settings = ssl_pretrain.augment_settings(cfg, variant)
        lo, hi, w = ssl_pretrain._source_positions(256, 150, settings,
                                                   torch.Generator().manual_seed(2))
        assert lo.min() >= 0 and hi.max() <= 149
        assert (w >= 0).all() and (w < 1).all()


def test_no_adjacency_break_variant_disables_both_augmentations():
    settings = ssl_pretrain.augment_settings(cfg, "no_adjacency_break")
    assert settings["subsample_p"] == 0 and settings["local_shuffle_p"] == 0
    full = ssl_pretrain.augment_settings(cfg, "full")
    assert full["subsample_p"] > 0 and full["local_shuffle_p"] > 0


def test_nt_xent_prefers_matching_views():
    a = torch.randn(8, 16)
    assert ssl_pretrain.nt_xent(a, a, 0.2) < ssl_pretrain.nt_xent(a, torch.randn(8, 16), 0.2)


def test_handcrafted_features():
    x = _windows(3)
    features = baselines.handcrafted_features(x)
    assert features.shape == (3, 416)
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, baselines.handcrafted_features(x[..., 0]))


def test_random_encoder_keeps_its_weights():
    """BatchNorm recalibration may only touch running statistics, never weights."""
    reference = encoder.build_encoder(cfg, seed=3)
    model = baselines.random_encoder(_windows(8), cfg, seed=3, device="cpu")
    for (name, p), q in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(p.detach().cpu(), q.detach(), msg=name)


def test_probe_never_fits_on_validation(monkeypatch):
    """Every row any probe is fitted on comes from development (HANDOFF rule 1)."""
    labels = np.array(["SIT", "STAND", "WALK", "FALL", "EMPTY"] * 8)
    rooms = np.array(["A"] * 20 + ["B"] * 20)
    rooms[np.isin(labels, ["FALL"])] = "A"
    rooms[np.isin(labels, ["EMPTY"])] = "B"

    def windows(sign: float) -> dataset.WindowSet:
        return dataset.WindowSet(
            x=np.empty((40, 0)), y=labels, room=rooms,
            window_id=np.arange(40).astype(str),
            capture_id=np.array([f"c{sign}_{i}" for i in range(40)]), preset="task2",
        )

    # Column 0 marks provenance: positive = development, negative = validation.
    z_dev = np.abs(rng.normal(size=(40, 6))) + 1
    z_val = -(np.abs(rng.normal(size=(40, 6))) + 1)

    fitted = []
    real_fit = task2_probe.fit_probe

    def spy(features, labels, groups, cfg):
        fitted.append(features)
        small = dict(cfg)
        small["task2"] = {**cfg["task2"], "probe": {**cfg["task2"]["probe"], "cv_folds": 2}}
        return real_fit(features, labels, groups, config.Config(small))

    monkeypatch.setattr(task2_probe, "fit_probe", spy)
    task2_probe.evaluate(z_dev, z_val, windows(1), windows(-1), cfg)

    assert len(fitted) == 4  # activity, two cross-room directions, room
    assert all((f[:, 0] > 0).all() for f in fitted)
