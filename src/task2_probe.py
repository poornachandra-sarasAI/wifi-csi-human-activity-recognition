"""Task 2 evaluation: frozen representations scored by a linear probe.

For each representation the encoder is frozen, a linear classifier is fitted
on labelled DEVELOPMENT windows, and the fixed local_validation windows are
scored (spec section 7). Validation numbers are validation numbers (section 9).

Representations
---------------
    handcrafted              baselines.handcrafted_features
    random                   same encoder architecture, random weights
    ssl_full                 SimCLR encoder, all augmentations
    ssl_no_adjacency_break   SimCLR encoder without the two augmentations
                             that break Room A's frame interleaving

Encoders are evaluated over every pretraining seed and reported as mean and
standard deviation.

Evaluations
-----------
    activity          one 8-class probe over both rooms. Scored overall
                      (target_room "all") and on each room's validation
                      windows ("A", "B"). Headline: macro-F1.
    cross_room_shared probe fitted on SIT/STAND/WALK of one room, scored on
                      SIT/STAND/WALK of the other ("A" = fitted on B, scored
                      on A). How much activity information survives a room
                      change; a preview of Task 3's transferred classes.
    room              probe predicting the room. Diagnostic only: how much
                      room/session signal the representation carries.

Leakage control
---------------
The probe's regularisation C is chosen by stratified, capture-grouped
cross-validation on development windows only. Validation windows are
embedded and scored, never fitted on.

Outputs
-------
    results/task2_probe.csv            section 10 schema, mean and std over seeds
    results/task2_probe_per_seed.csv   every individual score, plus chosen C

Run:  python -m src.task2_probe              # needs models/ from ssl_pretrain
      python -m src.task2_probe --pretrain   # pretrain first, then evaluate
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import amplitude, baselines, config, dataset, encoder, splits


def fit_probe(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, cfg: config.Config
) -> GridSearchCV:
    """Linear probe on development data; C picked by grouped CV on development."""
    settings = cfg["task2"]["probe"]
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=settings["max_iter"], random_state=cfg["seed"]
        ),
    )
    folds = StratifiedGroupKFold(
        n_splits=settings["cv_folds"], shuffle=True, random_state=cfg["seed"]
    )
    search = GridSearchCV(
        pipeline,
        {"logisticregression__C": settings["c_grid"]},
        cv=folds,
        scoring="f1_macro",
        n_jobs=-1,
    )
    search.fit(features, labels, groups=groups)
    return search


def _scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = sorted(set(truth))
    return {
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
    }


def evaluate(
    z_dev: np.ndarray, z_val: np.ndarray, dev: dataset.WindowSet, val: dataset.WindowSet,
    cfg: config.Config,
) -> list[dict]:
    """All probe evaluations for one representation. Returns score rows."""
    rows = []

    def add(evaluation, target_room, scores, chosen_c):
        for metric, value in scores.items():
            rows.append({"evaluation": evaluation, "target_room": target_room,
                         "metric": metric, "value": float(value), "chosen_C": chosen_c})

    # Activity, one probe over both rooms.
    probe = fit_probe(z_dev, dev.y, dev.capture_id, cfg)
    predicted = probe.predict(z_val)
    chosen = probe.best_params_["logisticregression__C"]
    add("activity", "all", _scores(val.y, predicted), chosen)
    for room in ("A", "B"):
        mask = val.room == room
        add("activity", room, _scores(val.y[mask], predicted[mask]), chosen)

    # Shared activities, fitted in one room and scored in the other.
    shared = list(amplitude.SHARED)
    for source, target in (("B", "A"), ("A", "B")):
        fit_mask = (dev.room == source) & np.isin(dev.y, shared)
        score_mask = (val.room == target) & np.isin(val.y, shared)
        probe = fit_probe(z_dev[fit_mask], dev.y[fit_mask], dev.capture_id[fit_mask], cfg)
        add("cross_room_shared", target,
            _scores(val.y[score_mask], probe.predict(z_val[score_mask])),
            probe.best_params_["logisticregression__C"])

    # Room, diagnostic.
    probe = fit_probe(z_dev, dev.room, dev.capture_id, cfg)
    add("room", "all", _scores(val.room, probe.predict(z_val)),
        probe.best_params_["logisticregression__C"])
    return rows


def representations(cfg: config.Config, x_dev: np.ndarray):
    """Yield (name, seed, featurizer) for every representation under test."""
    yield "handcrafted", -1, baselines.handcrafted_features
    seeds = cfg["task2"]["pretrain"]["seeds"]
    for seed in seeds:
        model = baselines.random_encoder(x_dev, cfg, seed=seed)
        yield "random", seed, lambda x, m=model: encoder.embed(m, x)
    for variant in cfg["task2"]["variants"]:
        for seed in seeds:
            model = encoder.load_pretrained(cfg, variant=variant, seed=seed)
            yield f"ssl_{variant}", seed, lambda x, m=model: encoder.embed(m, x)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds, in the section 10 results schema."""
    grouped = per_seed.groupby(["representation", "evaluation", "target_room", "metric"])
    stats = grouped["value"].agg(["mean", "std", "count"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)

    rows = []
    for row in stats.itertuples(index=False):
        experiment = f"task2_{row.evaluation}_{row.representation}"
        for suffix in ("mean", "std"):
            rows.append({"experiment": experiment, "target_room": row.target_room, "K": "",
                         "metric": f"{row.metric}_{suffix}", "value": getattr(row, suffix)})
    return pd.DataFrame(rows)


def headline_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Readable pivot: representation x (evaluation, room, metric) as mean +/- std."""
    stats = per_seed.groupby(["representation", "evaluation", "target_room", "metric"])["value"]
    table = stats.agg(lambda v: f"{v.mean():.3f} +/- {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
    return table.unstack(["evaluation", "target_room", "metric"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", action="store_true", help="run ssl_pretrain first")
    args = parser.parse_args()

    cfg = config.load()
    if args.pretrain:
        from src import ssl_pretrain

        sys.argv = [sys.argv[0]]
        ssl_pretrain.main()

    preset = cfg["task2"]["normalize_preset"]
    dev = dataset.load(preset, splits.DEVELOPMENT, cfg)
    val = dataset.load(preset, splits.LOCAL_VALIDATION, cfg)
    print(f"development {dev}\nvalidation  {val}\n")

    rows = []
    for name, seed, featurize in representations(cfg, dev.x):
        start = time.time()
        z_dev, z_val = featurize(dev.x), featurize(val.x)
        for row in evaluate(z_dev, z_val, dev, val, cfg):
            rows.append({"representation": name, "seed": seed, "dim": z_dev.shape[1], **row})
        f1 = next(r["value"] for r in rows[::-1]
                  if r["representation"] == name and r["seed"] == seed
                  and r["evaluation"] == "activity" and r["target_room"] == "all"
                  and r["metric"] == "macro_f1")
        print(f"  {name:26s} seed {seed:2d}  dim {z_dev.shape[1]:4d}  "
              f"activity macro-F1 {f1:.3f}  ({time.time() - start:.0f}s)")

    per_seed = pd.DataFrame(rows)
    results = cfg.path("results")
    per_seed.to_csv(results / "task2_probe_per_seed.csv", index=False)
    summarise(per_seed).to_csv(results / "task2_probe.csv", index=False)

    pd.set_option("display.width", 200)
    table = headline_table(per_seed)
    print("\nValidation results (mean +/- std over seeds):\n")
    for evaluation in ("activity", "cross_room_shared", "room"):
        print(f"[{evaluation}]")
        print(table[evaluation].to_string(), "\n")
    print(f"-> {results / 'task2_probe.csv'}\n-> {results / 'task2_probe_per_seed.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
