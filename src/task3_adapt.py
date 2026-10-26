"""Task 3: cross-room few-label adaptation (challenge spec sections 7, 9, 11).

For each target room, the other room is the source. The source supplies all
of its development labels; the target supplies only K labelled development
captures per target-exclusive activity. Target SIT/STAND/WALK labels are
withheld, so those activities must be recognised from the source room.

    target A   K-labelled: SITDOWN STANDUP FALL   transferred: SIT STAND WALK
    target B   K-labelled: EMPTY LYING            transferred: SIT STAND WALK

Representations
---------------
    ssl_full      frozen Task 2 encoder, every pretraining seed
    handcrafted   Task 2's handcrafted baseline features
Both come from the `task2` preset (embed() requires it) and are then
standardised per room by fewshot.RoomScaler, which does in feature space what
the `task3` preset does to the windows.

Methods
-------
    prototype     nearest class prototype
    refined       prototypes refined by soft k-means on unlabelled
                  development windows (fewshot.refine). The proposed method.

Routing ablation
----------------
    oracle        true room: target room's scaling and prototype bank
    predicted     Task 1's room classifier (task1_honest preset) picks the
                  scaling and bank; a window routed to the source room is
                  classified among the source room's activities
    none          no routing stage: one global scaling and one bank over
                  all 8 activities

Leakage control
---------------
Support captures come only from the target room's development split and are
asserted disjoint from local_validation. Scalers, prototypes, refinement and
the router are fitted on development windows only. Every setting is fixed in
configs/experiment.yaml; nothing is tuned on local_validation. K=1 is
averaged over `support_draws` seeded draws per encoder seed.

Outputs
-------
    results/task3_per_run.csv          every score of every draw
    results/task3.csv                  section 10 schema, mean and std
    results/task3_confusion_K5.csv     summed over draws, refined, oracle
    results/task3_per_activity_K5.csv  precision / recall / F1 from that matrix
    results/task3_predictions.csv      window_id, room, label, confidence

Run:  python -m src.task3_adapt
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import amplitude, baselines, config, dataset, encoder, fewshot, splits, task1_room

SHARED = list(amplitude.SHARED)
EXCLUSIVE = {"A": list(amplitude.ROOM_A_EXCLUSIVE), "B": list(amplitude.ROOM_B_EXCLUSIVE)}
ACTIVITIES = {room: SHARED + exclusive for room, exclusive in EXCLUSIVE.items()}
OTHER = {"A": "B", "B": "A"}
ROUTINGS = ("oracle", "predicted", "none")
METHODS = ("prototype", "refined")


def route(cfg: config.Config, val_window_id: np.ndarray) -> np.ndarray:
    """Predicted room of every validation window, from Task 1's classifier."""
    preset = cfg["task3"]["router_preset"]
    dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
    val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)
    assert (val.window_id == val_window_id).all(), "router and features disagree on order"
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg["seed"])
    )
    model.fit(task1_room.featurize(dev.x), dev.room)
    return model.predict(task1_room.featurize(val.x))


def _scores(truth: np.ndarray, predicted: np.ndarray, target: str) -> dict[str, float]:
    def macro(labels):
        return f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)

    return {
        "macro_f1": macro(ACTIVITIES[target]),
        "target_exclusive_macro_f1": macro(EXCLUSIVE[target]),
        "shared_macro_f1": macro(SHARED),
    }


def adapt_and_predict(
    z_dev, z_val, dev, routed, target, support, method, cfg
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit every routing arm for one support draw; predict target validation windows.

    `z_val` and `routed` cover only the target room's validation windows.
    """
    settings = cfg["task3"]
    temperature, iterations = settings["temperature"], settings["refine_iterations"]
    source = OTHER[target]
    is_source = dev.room == source

    def bank(z, fit, classes, unlabelled):
        protos = fewshot.prototypes(z[fit], dev.y[fit], classes)
        if method == "refined":
            protos = fewshot.refine(protos, z[support], dev.y[support], z[unlabelled],
                                    classes, temperature, iterations)
        return classes, protos

    # Routed arms: each room standardised with its own statistics.
    per_room = fewshot.RoomScaler(z_dev, dev.room)
    zr = per_room(z_dev, dev.room)
    target_bank = bank(zr, (is_source & np.isin(dev.y, SHARED)) | support,
                       ACTIVITIES[target], dev.room == target)
    # The source room is fully labelled, so its bank needs no adaptation.
    source_bank = (ACTIVITIES[source],
                   fewshot.prototypes(zr[is_source], dev.y[is_source], ACTIVITIES[source]))

    # No routing: one global scaling, one bank over every activity.
    everywhere = np.full(len(z_dev), "all")
    global_scaler = fewshot.RoomScaler(z_dev, everywhere)
    zg = global_scaler(z_dev, everywhere)
    global_bank = bank(zg, is_source | support,
                       SHARED + EXCLUSIVE[source] + EXCLUSIVE[target],
                       np.ones(len(z_dev), dtype=bool))

    out = {}
    for routing in ROUTINGS:
        if routing == "none":
            out[routing] = fewshot.predict(global_scaler(z_val, "all"), *global_bank, temperature)
            continue
        rooms = np.full(len(z_val), target) if routing == "oracle" else routed
        z = per_room(z_val, rooms)
        labels = np.empty(len(z_val), dtype=object)
        confidence = np.empty(len(z_val))
        for room, (classes, protos) in ((target, target_bank), (source, source_bank)):
            mask = rooms == room
            if mask.any():
                labels[mask], confidence[mask] = fewshot.predict(z[mask], classes, protos,
                                                                 temperature)
        out[routing] = (labels, confidence)
    return out


def representations(cfg: config.Config):
    """Yield (name, seed, featurizer) for each representation under test."""
    yield "handcrafted", -1, baselines.handcrafted_features
    for seed in cfg["task2"]["pretrain"]["seeds"]:
        model = encoder.load_pretrained(cfg, cfg["task3"]["encoder_variant"], seed)
        yield "ssl_full", seed, lambda x, m=model: encoder.embed(m, x)


def summarise(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over encoder seeds and support draws, section 10 schema."""
    keys = ["representation", "method", "routing", "target_room", "K", "metric"]
    stats = per_run.groupby(keys)["value"].agg(["mean", "std"]).reset_index()
    rows = []
    for row in stats.itertuples(index=False):
        for suffix in ("mean", "std"):
            rows.append({
                "experiment": f"task3_{row.representation}_{row.method}_{row.routing}",
                "target_room": row.target_room, "K": row.K,
                "metric": f"{row.metric}_{suffix}", "value": getattr(row, suffix),
            })
    return pd.DataFrame(rows)


def main() -> int:
    cfg = config.load()
    settings = cfg["task3"]
    preset = cfg["task2"]["normalize_preset"]
    dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
    val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)
    routed = route(cfg, val.window_id)
    print(f"development {dev}\nvalidation  {val}")
    print(f"router accuracy on validation: {(routed == val.room).mean():.4f}\n")

    validation_captures = set(val.capture_id)
    rows, confusions, predictions = [], [], []
    for name, seed, featurize in representations(cfg):
        start = time.time()
        z_dev, z_val = featurize(dev.x), featurize(val.x)
        for target in ("A", "B"):
            in_target = val.room == target
            candidates = dev.room == target
            for k in settings["ks"]:
                for draw in range(settings["support_draws"]):
                    # Same draws for every representation, so comparisons are paired.
                    rng = np.random.default_rng([cfg["seed"], draw, k, ord(target)])
                    support = candidates & fewshot.sample_support(
                        dev.capture_id, dev.y, EXCLUSIVE[target], k, rng)
                    assert validation_captures.isdisjoint(dev.capture_id[support])
                    assert not np.isin(dev.y[support], SHARED).any()
                    assert len(set(dev.capture_id[support])) == k * len(EXCLUSIVE[target])

                    for method in METHODS:
                        out = adapt_and_predict(z_dev, z_val[in_target], dev, routed[in_target],
                                                target, support, method, cfg)
                        for routing, (labels, confidence) in out.items():
                            truth = val.y[in_target]
                            for metric, value in _scores(truth, labels, target).items():
                                rows.append({"representation": name, "seed": seed,
                                             "draw": draw, "method": method,
                                             "routing": routing, "target_room": target,
                                             "K": k, "metric": metric, "value": value})
                            if name == "ssl_full" and k == 5 and method == "refined":
                                if routing == "oracle":
                                    confusions.append((target, truth, labels))
                                if routing == "predicted" and seed == 0 and draw == 0:
                                    predictions.append(pd.DataFrame({
                                        "window_id": val.window_id[in_target],
                                        "room": routed[in_target],
                                        "label": labels, "confidence": confidence}))
        print(f"  {name:12s} seed {seed:2d}  ({time.time() - start:.0f}s)")

    results = cfg.path("results")
    per_run = pd.DataFrame(rows)
    per_run.to_csv(results / "task3_per_run.csv", index=False)
    summarise(per_run).to_csv(results / "task3.csv", index=False)
    pd.concat(predictions).to_csv(results / "task3_predictions.csv", index=False)

    matrix_rows, activity_rows = [], []
    for target in ("A", "B"):
        # Source-room activities (only reachable without routing) collapse to OTHER.
        labels = ACTIVITIES[target] + ["OTHER"]
        truth = np.concatenate([t for r, t, _ in confusions if r == target])
        predicted = np.concatenate([p for r, _, p in confusions if r == target])
        predicted = np.where(np.isin(predicted, ACTIVITIES[target]), predicted, "OTHER")
        matrix = confusion_matrix(truth, predicted, labels=labels)
        for i, true in enumerate(ACTIVITIES[target]):
            for j, pred in enumerate(labels):
                matrix_rows.append({"target_room": target, "true": true, "predicted": pred,
                                    "count": int(matrix[i, j])})
        p, r, f, n = precision_recall_fscore_support(
            truth, predicted, labels=ACTIVITIES[target], zero_division=0)
        for activity, *values in zip(ACTIVITIES[target], p, r, f, n):
            activity_rows.append({"target_room": target, "activity": activity,
                                  "precision": values[0], "recall": values[1],
                                  "f1": values[2], "support": int(values[3])})
    pd.DataFrame(matrix_rows).to_csv(results / "task3_confusion_K5.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(results / "task3_per_activity_K5.csv", index=False)

    pd.set_option("display.width", 200)
    table = (per_run.groupby(["representation", "method", "routing", "target_room", "metric", "K"])
             ["value"].agg(lambda v: f"{v.mean():.3f}+/-{v.std():.3f}")
             .unstack("K"))
    print("\nValidation results (mean +/- std over encoder seeds x support draws):\n")
    print(table.to_string())
    print(f"\n-> {results / 'task3.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
