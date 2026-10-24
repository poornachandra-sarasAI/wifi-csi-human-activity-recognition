"""Task 3 building blocks: capture-level support sampling and prototype classifiers.

A prototype is the mean feature vector of one activity. A window is assigned
to the activity whose prototype is nearest in cosine similarity. Shared
activities (SIT, STAND, WALK) get their prototypes from the source room's
labels; target-exclusive activities get theirs from K labelled target captures.

Per-room standardisation
------------------------
Features are z-scored with statistics of each room's own development windows.
No label is used, so this is the feature-space analogue of the `task3`
preset's per-room scaling (AdaBN-style domain alignment).

Transductive refinement
-----------------------
`refine` moves the target prototypes towards the structure of the target
room's UNLABELLED development windows with a few soft k-means steps
(semi-supervised prototypical networks, Ren et al. 2018). Support windows keep
their labels. Temperature and iteration count are fixed a priori, never tuned
on local_validation.

Metadata rule: `capture_id` is used only to sample support at capture level,
never as a feature.
"""

from __future__ import annotations

import numpy as np


def sample_support(
    capture_id: np.ndarray, y: np.ndarray, activities, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask selecting all windows of K distinct captures per activity."""
    mask = np.zeros(len(y), dtype=bool)
    for activity in activities:
        captures = np.unique(capture_id[y == activity])  # sorted, so draws are reproducible
        if len(captures) < k:
            raise ValueError(f"{activity}: only {len(captures)} captures, K={k} requested")
        mask |= np.isin(capture_id, rng.choice(captures, size=k, replace=False))
    return mask


class RoomScaler:
    """z-score features with per-room statistics fitted on unlabelled windows."""

    def __init__(self, z: np.ndarray, room: np.ndarray):
        self.stats = {r: (z[room == r].mean(axis=0), z[room == r].std(axis=0) + 1e-6)
                      for r in np.unique(room)}

    def __call__(self, z: np.ndarray, room) -> np.ndarray:
        room = np.broadcast_to(np.asarray(room), (len(z),))
        out = np.empty_like(z)
        for r in np.unique(room):
            mean, std = self.stats[r]
            out[room == r] = (z[room == r] - mean) / std
        return out


def _unit(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def prototypes(z: np.ndarray, y: np.ndarray, classes) -> np.ndarray:
    """(n_classes, d) unit-norm class means of unit-norm features."""
    u = _unit(z)
    return _unit(np.stack([u[y == c].mean(axis=0) for c in classes]))


def predict(z: np.ndarray, classes, protos: np.ndarray, temperature: float = 0.1):
    """Nearest prototype by cosine. Returns (labels, softmax confidence)."""
    logits = _unit(z) @ protos.T / temperature
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return np.asarray(classes)[p.argmax(axis=1)], p.max(axis=1)


def refine(
    protos: np.ndarray,
    z_support: np.ndarray,
    y_support: np.ndarray,
    z_unlabelled: np.ndarray,
    classes,
    temperature: float = 0.1,
    iterations: int = 10,
) -> np.ndarray:
    """Soft k-means on unlabelled target windows, initialised at `protos`.

    Labelled support windows contribute with weight 1 to their own class;
    unlabelled windows contribute their soft assignment. A class with no
    target support (a shared activity) starts from the source prototype and
    is pulled towards whichever target cluster it is nearest.
    """
    s, u = _unit(z_support), _unit(z_unlabelled)
    hard = (np.asarray(y_support)[:, None] == np.asarray(classes)[None, :]).astype(float)
    for _ in range(iterations):
        logits = u @ protos.T / temperature
        logits -= logits.max(axis=1, keepdims=True)
        soft = np.exp(logits)
        soft /= soft.sum(axis=1, keepdims=True)
        weighted = hard.T @ s + soft.T @ u
        protos = _unit(weighted / (hard.sum(axis=0) + soft.sum(axis=0))[:, None])
    return protos
