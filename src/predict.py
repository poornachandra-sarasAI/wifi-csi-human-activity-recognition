"""Deployed inference: a window in, a room and activity prediction out.

This is the path behind \\texttt{run\\_episode.py} and the challenge's Section 10
output contract (\\texttt{window\\_id, room, label, confidence}). It needs no
labels for the data being predicted, so it runs on a challenge episode supplied
by the teaching team.

How a prediction is made
------------------------
1. Room is predicted by the Task 1 classifier on the \\texttt{task1\\_honest}
   preset -- the per-window-standardised one, which does not see the raw
   inter-room amplitude gap.
2. The window is embedded by the frozen Task 2 encoder, standardised with the
   predicted room's development statistics (AdaBN), and assigned to the nearest
   class prototype of that room by cosine similarity.
3. Confidence is the softmax of those cosine similarities at the configured
   temperature.

Relationship to the Task 3 numbers
----------------------------------
This is the DEPLOYED system, not the Task 3 benchmark. Task 3 deliberately
starves the target room of labels -- K captures per target-exclusive activity,
and none at all for the shared activities -- in order to measure label
efficiency. A deployed system has no reason to discard labels it already holds,
so prototypes here are built from ALL development captures of both rooms.
Numbers produced by this module are therefore not comparable with Section 6 of
the report and must never be quoted as Task 3 results.

Run:  python -m src.predict
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import amplitude, config, dataset, encoder, fewshot, splits, task1_room

ACTIVITIES = {
    "A": list(amplitude.SHARED) + list(amplitude.ROOM_A_EXCLUSIVE),
    "B": list(amplitude.SHARED) + list(amplitude.ROOM_B_EXCLUSIVE),
}


@dataclass
class Predictor:
    """Everything fitted on development data, ready to score new windows."""

    router: object                       # room classifier, task1_honest features
    scaler: fewshot.RoomScaler           # per-room embedding standardisation
    banks: dict[str, tuple[list, np.ndarray]]
    embed: object                        # callable: windows -> embeddings
    temperature: float
    n_development: int

    def __call__(self, x_router: np.ndarray, x_encoder: np.ndarray,
                 window_id: np.ndarray) -> pd.DataFrame:
        """Predict room and activity for windows. No labels required."""
        room = self.router.predict(task1_room.featurize(x_router))

        z = self.scaler(self.embed(x_encoder), room)
        label = np.empty(len(z), dtype=object)
        confidence = np.empty(len(z), dtype=float)
        for name, (classes, protos) in self.banks.items():
            mask = room == name
            if mask.any():
                label[mask], confidence[mask] = fewshot.predict(
                    z[mask], classes, protos, self.temperature)

        return pd.DataFrame({
            "window_id": window_id,
            "room": room,
            "label": label,
            "confidence": confidence,
        })


def fit(cfg: config.Config | None = None, verbose: bool = True) -> Predictor:
    """Fit the router and the per-room prototype banks on development data."""
    cfg = cfg or config.load()

    router_dev = dataset.load(cfg["task3"]["router_preset"], splits.DEVELOPMENT, cfg)
    router = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=cfg["seed"]),
    )
    router.fit(task1_room.featurize(router_dev.x), router_dev.room)

    dev = dataset.load(cfg["task2"]["normalize_preset"], splits.DEVELOPMENT, cfg)
    model = encoder.load_pretrained(cfg, cfg["task3"]["encoder_variant"],
                                    cfg["task2"]["pretrain"]["seeds"][0])
    embed = lambda x: encoder.embed(model, x)  # noqa: E731

    z_dev = embed(dev.x)
    scaler = fewshot.RoomScaler(z_dev, dev.room)
    z_scaled = scaler(z_dev, dev.room)

    banks = {}
    for room, classes in ACTIVITIES.items():
        mask = dev.room == room
        banks[room] = (classes, fewshot.prototypes(z_scaled[mask], dev.y[mask], classes))

    if verbose:
        print(f"  router fitted on {len(router_dev)} development windows")
        for room, (classes, _) in banks.items():
            print(f"  Room {room} prototype bank: {len(classes)} activities "
                  f"from {int((dev.room == room).sum())} development windows")

    return Predictor(router=router, scaler=scaler, banks=banks, embed=embed,
                     temperature=cfg["task3"]["temperature"], n_development=len(dev))


def predict_split(split: str = splits.LOCAL_VALIDATION,
                  cfg: config.Config | None = None,
                  predictor: Predictor | None = None) -> pd.DataFrame:
    """Predict every window of a split already present in windows/."""
    cfg = cfg or config.load()
    predictor = predictor or fit(cfg)

    router_q = dataset.load(cfg["task3"]["router_preset"], split, cfg)
    encoder_q = dataset.load(cfg["task2"]["normalize_preset"], split, cfg)
    assert (router_q.window_id == encoder_q.window_id).all(), "window order mismatch"
    return predictor(router_q.x, encoder_q.x, router_q.window_id)


def main() -> int:
    cfg = config.load()
    print("Fitting the deployed predictor on development data:")
    predictor = fit(cfg)
    out = predict_split(splits.LOCAL_VALIDATION, cfg, predictor)

    path = cfg.path("results") / "predictions.csv"
    out.to_csv(path, index=False)
    print(f"\n{len(out)} predictions -> {path}")
    print(out.head().to_string(index=False))

    # Scoring is only possible because this split happens to carry labels; an
    # episode supplied by the teaching team would not, and nothing above needs them.
    manifest = pd.read_csv(cfg.path("windows") / "manifest.csv")
    truth = manifest.set_index("window_id").loc[out.window_id]
    print(f"\nagainst the known labels of this split: "
          f"room {np.mean(out.room.to_numpy() == truth.room.to_numpy()):.4f}, "
          f"activity {np.mean(out.label.to_numpy() == truth.activity.to_numpy()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
