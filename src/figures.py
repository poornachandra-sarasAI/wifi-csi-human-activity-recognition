"""Required report figures (challenge spec section 11).

Each figure is one function so that ownership stays clear when three
people are working in this file at once.

    room_accuracy_vs_window_length()   owner: ingestion / Task 1
    task1_ablation()                   owner: ingestion / Task 1
    label_efficiency_curve()           owner: Task 3   (not yet written)
    encoder_umap()                     owner: Task 2
    task2_probe_bars()                 owner: Task 2

Run:  python -m src.figures
      python -m src.figures --task2      # only the Task 2 figures
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

PRESET_STYLE = {
    "task1": ("#c53030", "o", "global\n(all static structure kept)"),
    "task1_subband": ("#b7791f", "s", "sub-band neutralised\n(AGC step removed)"),
    "task1_honest": ("#2b6cb0", "^", "per-window\n(no static structure)"),
}


def room_accuracy_vs_window_length(cfg: config.Config) -> str:
    """Spec section 11: room accuracy against window length."""
    results = pd.read_csv(cfg.path("results") / "task1_room_accuracy.csv")
    official = cfg["windowing"]["length"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for preset, (color, marker, label) in PRESET_STYLE.items():
        rows = results[results.preset == preset].sort_values("window_length")
        if rows.empty:
            continue
        ax.plot(rows.window_length, rows.value, marker=marker, color=color,
                lw=1.8, ms=7, label=label)

    ax.axvline(official, color="#444", ls="--", lw=1,
               label=f"official window ({official} obs)")

    ax.set_xscale("log")
    ax.set_xticks(sorted(results.window_length.unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("window length (consecutive CSI observations)")
    ax.set_ylabel("room identification accuracy (validation)")
    # Every curve sits far above the 0.5 chance line, so the axis is zoomed
    # to the region where the presets actually differ.
    ax.set_ylim(0.86, 1.006)
    ax.annotate("chance = 0.50, far below this axis", xy=(1, 0.868),
                fontsize=7.5, color="#777", style="italic")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title(
        "Task 1: room accuracy vs window length\n"
        "perfect accuracy from a single observation indicates a scale artifact, "
        "not an environment fingerprint",
        fontsize=10.5,
    )
    fig.tight_layout()
    path = cfg.path("figures") / "task1_accuracy_vs_window_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def task1_ablation(cfg: config.Config) -> str:
    """What survives as each instrumentation difference is removed."""
    results = cfg.path("results")
    ablation = pd.read_csv(results / "task1_ablation.csv")
    dynamics = pd.read_csv(results / "task1_dynamics_diagnostic.csv")
    stream = pd.read_csv(results / "task1_frame_stream_ablation.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[1.45, 1])

    # ---- left: the nested ladder -------------------------------------
    ladder = [
        ("global\nnormalization", ablation.query(
            "preset=='task1' and region=='all'").value.iloc[0],
         "490x gap kept"),
        ("sub-band\nneutralised", ablation.query(
            "preset=='task1_subband' and region=='all'").value.iloc[0],
         "AGC step removed"),
        ("per-window\nnormalization", ablation.query(
            "preset=='task1_honest' and region=='all'").value.iloc[0],
         "no static structure"),
        ("+ sequenced\nframes only", stream.query(
            "room_a_frame_stream=='sequenced'").value.iloc[0],
         "no interleaving"),
    ]
    labels = [row[0] for row in ladder]
    values = [row[1] for row in ladder]
    colors = ["#c53030", "#b7791f", "#2b6cb0", "#2f855a"]

    bars = left.bar(labels, values, color=colors, width=0.62)
    for bar, (_, value, note) in zip(bars, ladder):
        left.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                  f"{value:.4f}", ha="center", fontsize=10, fontweight="bold")
        left.text(bar.get_x() + bar.get_width() / 2, 0.9045, note,
                  ha="center", fontsize=7.5, color="#fff", style="italic")
    left.set_ylim(0.9, 1.012)
    left.set_ylabel("room accuracy (validation)")
    left.set_title("Removing one instrumentation difference at a time\n"
                   "accuracy stays high at every rung (chance = 0.50)", fontsize=10.5)
    left.grid(axis="y", alpha=0.3)

    # ---- right: the dynamics confound that remains --------------------
    rooms = dynamics.room.tolist()
    interleaved = dynamics.lag1_autocorr_mean.tolist()
    sequenced_a = 0.656  # measured; see results/task1_frame_stream_ablation.csv

    x = np.arange(2)
    right.bar(x - 0.2, [interleaved[0], interleaved[1]], width=0.38,
              color="#2b6cb0", label="as ingested (interleaved)")
    right.bar(x + 0.2, [sequenced_a, interleaved[1]], width=0.38,
              color="#2f855a", label="Room A sequenced frames only")
    right.axhline(0, color="#333", lw=0.9)
    right.set_xticks(x, [f"Room {r}" for r in rooms])
    right.set_ylabel("mean lag-1 autocorrelation")
    right.legend(fontsize=8)
    right.grid(axis="y", alpha=0.3)
    right.set_title("The confound that no normalization removes\n"
                    "the two sessions differ in temporal dynamics", fontsize=10.5)

    fig.suptitle("Task 1 ablation: high accuracy is not evidence of room recognition",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    path = cfg.path("figures") / "task1_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


ROOM_COLORS = {"A": "#2b6cb0", "B": "#c05621"}
ACTIVITY_COLORS = {
    "SIT": "#2b6cb0", "STAND": "#2f855a", "WALK": "#c53030",
    "SITDOWN": "#6b46c1", "STANDUP": "#b7791f", "FALL": "#000000",
    "EMPTY": "#a0aec0", "LYING": "#d53f8c",
}


def encoder_umap(cfg: config.Config) -> str:
    """Spec section 11: 2-D UMAP of the representations, by room and by activity.

    Owner: Task 2. Qualitative only. UMAP is fitted on development windows,
    unsupervised; labels are used for colour alone.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    from src import baselines, dataset, encoder, splits

    dev = dataset.load(cfg["task2"]["normalize_preset"], splits.DEVELOPMENT, cfg)
    rows = [
        ("handcrafted features", baselines.handcrafted_features(dev.x)),
        ("random-weight encoder",
         encoder.embed(baselines.random_encoder(dev.x, cfg, seed=0), dev.x)),
        ("SSL encoder (all augmentations)",
         encoder.embed(encoder.load_pretrained(cfg, "full", 0), dev.x)),
        ("SSL encoder (no adjacency-breaking aug.)",
         encoder.embed(encoder.load_pretrained(cfg, "no_adjacency_break", 0), dev.x)),
    ]

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4.6 * len(rows)))
    for (title, z), (ax_room, ax_act) in zip(rows, axes):
        xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=cfg["seed"]).fit_transform(
            StandardScaler().fit_transform(z)
        )
        for ax, labels, palette in ((ax_room, dev.room, ROOM_COLORS),
                                    (ax_act, dev.y, ACTIVITY_COLORS)):
            for name, color in palette.items():
                mask = labels == name
                if mask.any():
                    prefix = "Room " if palette is ROOM_COLORS else ""
                    ax.scatter(*xy[mask].T, s=5, alpha=0.6, color=color,
                               label=f"{prefix}{name}", rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=7, markerscale=2.5, loc="best", framealpha=0.9)
        ax_room.set_ylabel(title, fontsize=10)
    axes[0, 0].set_title("coloured by room")
    axes[0, 1].set_title("coloured by activity")
    fig.suptitle("Task 2: UMAP of development-window representations (qualitative)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    path = cfg.path("figures") / "task2_umap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def task2_probe_bars(cfg: config.Config) -> str:
    """Linear-probe validation scores per representation, with seed spread.

    Owner: Task 2.
    """
    per_seed = pd.read_csv(cfg.path("results") / "task2_probe_per_seed.csv")
    order = ["handcrafted", "random", "ssl_no_adjacency_break", "ssl_full"]
    names = {"handcrafted": "handcrafted", "random": "random-weight\nencoder",
             "ssl_no_adjacency_break": "SSL, no adjacency-\nbreaking aug.",
             "ssl_full": "SSL, all\naugmentations"}
    colors = ["#718096", "#a0aec0", "#63b3ed", "#2b6cb0"]
    panels = [
        ("activity", "all", "macro_f1", "Activity, both rooms (8 classes)\nmacro-F1 - headline"),
        ("activity", "all", "balanced_accuracy", "Activity, both rooms (8 classes)\nbalanced accuracy"),
        ("cross_room_shared", None, "macro_f1",
         "SIT/STAND/WALK across rooms\nmacro-F1, mean of both directions"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.8))
    for ax, (evaluation, room, metric, title) in zip(axes, panels):
        rows = per_seed[(per_seed.evaluation == evaluation) & (per_seed.metric == metric)]
        if room is not None:
            rows = rows[rows.target_room == room]
        # Average the two transfer directions within a seed, then spread over seeds.
        by_seed = rows.groupby(["representation", "seed"])["value"].mean()
        means = [by_seed[r].mean() for r in order]
        stds = [by_seed[r].std(ddof=1) if len(by_seed[r]) > 1 else 0 for r in order]
        bars = ax.bar(range(len(order)), means, yerr=stds, capsize=4, color=colors)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.3f}",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(order)), [names[r] for r in order], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("validation score")
    fig.suptitle("Task 2: frozen representations, linear probe on development, "
                 "scored on validation (error bars: std over 3 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    path = cfg.path("figures") / "task2_probe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> int:
    cfg = config.load()
    builders = [encoder_umap, task2_probe_bars]
    if "--task2" not in sys.argv:
        builders = [room_accuracy_vs_window_length, task1_ablation] + builders
    for builder in builders:
        print(f"-> {builder(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
