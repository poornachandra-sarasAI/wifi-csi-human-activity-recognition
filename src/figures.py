"""Required report figures (challenge spec section 11).

Each figure is one function so that ownership stays clear when three
people are working in this file at once.

    room_accuracy_vs_window_length()   owner: ingestion / Task 1
    task1_ablation()                   owner: ingestion / Task 1
    label_efficiency_curve()           owner: Task 3   (not yet written)
    encoder_umap()                     owner: Task 2   (not yet written)

Run:  python -m src.figures
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


def main() -> int:
    cfg = config.load()
    for builder in (room_accuracy_vs_window_length, task1_ablation):
        print(f"-> {builder(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
