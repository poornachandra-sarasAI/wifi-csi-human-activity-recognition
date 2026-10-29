# Cross-Room Few-Shot Adaptation — WiFi CSI

CS60055 Ubiquitous Computing · Hackathon Challenge 2

Environment-aware, label-efficient activity recognition from WiFi Channel State
Information. A student-owned Python pipeline from raw captures to predictions:
PCAP/CSV ingestion, CSI reconstruction, quality control, windowing,
normalization, and the three benchmark tasks.

---

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The dataset is **not** in this repository (challenge spec §4). Place the source
archives so the layout matches `configs/experiment.yaml`:

```
raw/experiment2/realdata/input_data/30ms/csidata/   # Room A, *.pcap
raw/experiment3/Data/                               # Room B, *.csv
```

> The source authors' `matlab code/` directory, their `Results/` folders and the
> Experiment-3 LSTM notebook are **excluded from this project**. They are not
> part of the participant package and must not be used, read or ported (§6).

## Running

```bash
python run_episode.py              # full pipeline, end to end
python run_episode.py --skip-ingest   # reuse intermediate_csv/
python -m pytest tests/ -q            # leakage + contract tests
```

Individual stages:

```bash
python -m src.pcap_reader      # Room A: .pcap  -> intermediate_csv/room_A
python -m src.csv_reader       # Room B: .csv   -> intermediate_csv/room_B
python -m src.splits           # capture-level dev / local_validation
python -m src.windowing        # -> windows/ + windows/manifest.csv
python -m src.normalization    # fit presets on development only
python -m src.task1_room       # Task 1 results
python -m src.csi_decode --resolve   # re-run the decode validation test
```

## Pipeline

```
raw (read-only, never committed)
  ├─ Room A *.pcap ─► pcap_reader ─► csi_decode ─► amplitude ─┐
  └─ Room B *.csv  ─► csv_reader ─────────────────────────────┤
                                                              ▼
                              intermediate_csv/{room_A,room_B}/<capture_id>.csv
                                    256 amplitude columns, temporal order
                                                              │
                                            cleaning ─────────┤
                                            splits   ─────────┤
                                            windowing ────────▼
              windows/{development,local_validation}/window_XXXXX.npy  (150,256) float32
                                      + windows/manifest.csv
                                                              │
                      ┌───────────────────────┼───────────────────────┐
                  Task 1                   Task 2                  Task 3
             room identification       SSL encoder            few-shot transfer
```

Normalization is **not** baked into the `.npy` files. Windows are stored
un-normalized and the preset is applied at load time by `src/dataset.py`, so all
three tasks share one `windows/` directory.

## Dataset as ingested

| | Room A (Experiment 2) | Room B (Experiment 3) |
|---|---|---|
| Source | 1,084 `.pcap` | 1,201 `.csv` |
| Development | 934 | 1,076 |
| Validation | 150 (supplied, 25 × 6) | 125 (seed 42, 25 × 5) |
| Observations per capture | 112–251 (median 221) | 500 |
| Activities | SIT, STAND, WALK, SITDOWN, STANDUP, FALL | SIT, STAND, WALK, EMPTY, LYING |

## Two findings that shape the pipeline

**1. Frames with sequence-control `0xFFFF` carry valid CSI and are kept.**
Room A frames alternate between consecutive 802.11 sequence numbers and
`0xFFFF`, the latter arriving ~0.4 ms after the former. They are *not* empty:
976–992 of their 1024 payload bytes are non-zero and their amplitude
distribution matches sequenced frames. Dropping them leaves a median of **111**
observations per capture, which puts the official 150-observation window out of
reach for **all 1,084 captures**. Keeping them gives a median of 221.

**2. The packed-CSI layout was resolved empirically, not ported.**
`python -m src.csi_decode --resolve` reproduces both arguments:

- Bit occupancy over 2.01 M words: bits 30–31 are always 0 (30 bits used), and
  bits 18–29 and 6–17 have statistically identical occupancy to three decimals —
  the signature of two 12-bit fields holding the real and imaginary parts.
  Bit 5 is always 1, so the 6-bit exponent is always negative.
- Applying the exponent reproduces the 802.11ac VHT80 null structure: amplitude
  at the known guard-band and DC-gap subcarriers drops to **0.020** of the
  data-subcarrier median, matching Room B's supplied amplitudes (**0.024**).
  Without the exponent that ratio is 0.78–1.01 and the nulls vanish.

See `figures/decode_validation.png`.

## Task 1: high accuracy is not evidence of room recognition

Three instrumentation differences could each explain a near-perfect score
without any environment fingerprinting, so Task 1 is reported as an ablation
ladder rather than a single number (`python -m src.task1_ablation`).

| Rung | What is removed | Validation accuracy |
|---|---|---|
| `task1` — global normalization | nothing; ~490× amplitude gap kept | **1.0000** |
| `task1_subband` | per-sub-band AGC offsets | **1.0000** |
| `task1_honest` — per-window | all static per-subcarrier structure | **0.9927** |
| `task1_honest` + sequenced frames only | Room A's frame interleaving | **0.9418** |

Read it in order:

1. **`task1` is already perfect at a window length of one observation.** A
   single CSI sample cannot carry temporal structure, so this is a constant
   scale offset: our Room A decode and Room B's supplied integers differ by a
   global factor of **~490×**.
2. **Neutralising the sub-band AGC step changes nothing.** Room A carries a ~4×
   gain step across subcarriers 135–187 (the third 20 MHz sub-band) that Room B
   does not, but removing it leaves accuracy at 1.0000 — the amplitude gap is
   present *within* every sub-band too.
3. **Per-window normalization erases the static profile entirely** (verified:
   median, band-2 offset and spread all become exactly 0.0) and accuracy only
   falls to 0.9927. What remains is temporal, not static.
4. **That residue is substantially Room A's frame interleaving.** Keeping the
   `0xFFFF` frames — which the 150-observation requirement forces — interleaves
   two streams with very different statistics:

   | Room A stream | lag-1 autocorrelation | std/IQR |
   |---|---|---|
   | sequenced only | +0.54 … +0.84 | 0.570 |
   | `0xFFFF` only | +0.01 … +0.13 | — |
   | interleaved (as ingested) | **−0.15 … −0.66** | 1.133 |
   | Room B | +0.035 | 0.979 |

   Ingesting only the sequenced stream (`ingest.room_a.frame_stream: sequenced`)
   drops accuracy to 0.9418.

**What Task 1 can honestly claim.** Accuracy stays above 0.94 at every rung, so
there is real discriminative signal — but every rung still leaves at least one
instrumentation difference intact. Even the strictest setting leaves Room A at
lag-1 +0.656 against Room B's +0.035, i.e. the two sessions differ in effective
sampling dynamics. This benchmark cannot separate *room* from *recording
session and pipeline*, which is precisely the confound named in §12.1. Report
the ladder, not the headline.

Figures: `figures/task1_accuracy_vs_window_length.png`, `figures/task1_ablation.png`.

## Task 2: self-supervised representation

A small 1-D temporal CNN (`src/encoder.py`, ~0.27 M parameters) is pretrained
with SimCLR on the 2,010 development windows of **both rooms, with no labels**
(`src/ssl_pretrain.py`). It is then frozen, a linear probe is fitted on
labelled development windows, and the fixed validation windows are scored
(`src/task2_probe.py`). The probe's `C` is chosen by capture-grouped 5-fold CV
on development only. Encoders are reported as mean ± std over 3 seeds.

```bash
python -m src.ssl_pretrain            # 2 variants x 3 seeds -> models/*.pt  (~45 min, GTX 1650)
python -m src.task2_probe             # probes -> results/task2_probe*.csv   (~4 min)
python -m src.figures --task2         # figures/task2_umap.png, figures/task2_probe.png
python -m src.task2_transfer_check    # development-only cross-room check (table below)
```

Setting up from the raw archive on a new machine: see `docs/setup_notes.md`.
It covers a duplicate capture in the archive that silently shifts every
`window_id`.

Validation results (8-class activity probe over both rooms):

| Representation | macro-F1 | balanced acc. | Room A macro-F1 | Room B macro-F1 |
|---|---|---|---|---|
| handcrafted features | 0.898 | 0.897 | 0.906 | 0.887 |
| random-weight encoder | 0.637 ± 0.026 | 0.664 ± 0.019 | 0.519 ± 0.013 | 0.742 ± 0.045 |
| **SSL, all augmentations** | **0.898 ± 0.008** | **0.906 ± 0.008** | 0.876 ± 0.020 | **0.933 ± 0.013** |
| SSL, no adjacency-breaking aug. | 0.850 ± 0.003 | 0.856 ± 0.004 | 0.811 ± 0.020 | 0.900 ± 0.036 |

Read it as:

1. **Pretraining helps the encoder a lot** (+0.26 over the same architecture
   with random weights) but **only matches handcrafted features overall**:
   better in Room B, worse in Room A.
2. **The interleaving ablation works.** Removing the two augmentations that
   break strict temporal adjacency (stride-2 subsampling, local shuffle)
   costs 0.048 macro-F1, more than five times the seed spread, and Room A
   loses most of it. That variant also ends at a lower contrastive loss (1.63
   vs 2.08, mean of 3 seeds): with the alternating frame streams intact, matching views is
   easier and the encoder learns less about activity.
3. **Nothing transfers across rooms.** A probe fitted on one room's
   SIT/STAND/WALK and scored on the other's sits at the single-class level
   (macro-F1 ≈ 0.17) for every representation. The `task2` preset keeps the
   global amplitude level, so each room is out of distribution for the other.
   Room identity is linearly perfect (1.000) in all four representations,
   including handcrafted and random, so the room split in the UMAP comes from
   the input, not from pretraining.
4. **Caveat, qualitative.** In every representation, Room B's SIT and STAND
   windows form an island apart from its EMPTY, LYING and WALK windows. That
   looks like a recording-session effect inside Room B, which would flatter
   Room B's scores.

**For Task 3:** `from src.encoder import load_pretrained, embed`, then
`embed(load_pretrained(cfg, "full", seed), x)` gives `(n, 128)` embeddings.
The six trained encoders are committed in `models/` (1.1 MB each), so this
works straight after cloning; `python -m src.ssl_pretrain` rebuilds them.
Three things to know first:

- `embed()` takes **`task2`-preset** windows `(n, 150, 256, 2)`. The `task3`
  preset returns `(n, 150, 256)` and is rejected. Load with `task2` and do any
  per-room normalization on the embeddings.
- **Per-room normalization alone does not make SIT/STAND/WALK transfer.**
  Development-only check, a linear probe fitted in one room and scored on the
  other room's development windows:

  | Features | B → A macro-F1 | A → B macro-F1 |
  |---|---|---|
  | SSL embeddings, as is | 0.132 | 0.156 |
  | SSL embeddings, standardised per room | 0.140 | 0.108 |
  | handcrafted, `task3` preset | 0.124 | 0.079 |

  The same probe scores 0.94 within either room, and 0.167 is the
  single-class level. Across rooms the classifier collapses onto one or two
  classes (e.g. all of Room A predicted STAND). Expect the transferred
  shared-activity curve to sit near chance unless the adaptation does more
  than rescale each room.
- `src/task1_room.py` exposes `featurize()` and `evaluate()` but no fitted
  router; the predicted-room arm has to fit its own (as `run_episode.py`'s
  `write_predictions()` does).

## Task 3: cross-room few-label adaptation

Frozen Task 2 embeddings are z-scored per room (no labels), then classified by
nearest class prototype (`src/fewshot.py`). Shared SIT/STAND/WALK prototypes come
from the source room's labels, and target-exclusive prototypes from K labelled
target captures. The `refined` variant moves the prototypes with soft k-means
over unlabelled target development windows. Support is drawn at capture level
from target development only, 10 seeded draws × 3 encoder seeds per K
(`src/task3_adapt.py`).

```bash
python -m src.task3_adapt          # ~5 min on CPU -> results/task3*.csv
python -m src.figures --task3      # label efficiency, K=5 confusion, routing ablation
```

Validation, K = 5, oracle routing, SSL + prototypes (mean ± std over 30 runs),
using encoders retrained on the `task2_notched` preset (level-free plus Nyquist notch):

| Target | all activities | target-exclusive | transferred shared |
|---|---|---|---|
| Room A | 0.211 ± 0.031 | 0.358 ± 0.060 | 0.065 ± 0.030 |
| Room B | 0.326 ± 0.044 | 0.509 ± 0.072 | 0.205 ± 0.049 |

1. **The shared activities do not transfer, and this is a verified negative
   result.** We removed the two artifacts that could explain it, the Room B WALK
   loudness offset (fix 1) and Room A's frame interleaving (fix 2). Room B shared
   F1 stays at chance (0.20) and Room A's below it. The Room B swap moved from
   SIT ↔ WALK to STAND ↔ WALK but did not go away.
2. **The few target labels work.** Target-exclusive macro-F1 rises from K = 1 to
   20 (Room A 0.29 → 0.41, Room B 0.38 → 0.59), capped because untransferred
   shared windows are absorbed into the exclusive classes.
3. **Routing is free.** Predicted routing (Task 1 router, 0.9927) matches oracle
   to within 0.001. Dropping routing costs up to 0.095.
4. **Refinement does not help**, with either representation. The earlier
   handcrafted Room B "transfer" (shared 0.236) is **retracted**: it was the
   amplitude shortcut, and it falls to 0.038 on level-free features.

Full write-up: `report/task3.md`.

## Contracts — frozen, do not edit in place

**`windows/manifest.csv`** is the only place room, activity, split and
provenance live. It is committed; the `.npy` payloads are not.

```
window_id, capture_id, room, activity, split, validation_origin,
source_format, config_hash, start_idx, n_obs_in_capture, was_padded
```

**`src/dataset.py`** is the single integration point:

```python
from src import dataset, splits
dev = dataset.load("task2", splits.DEVELOPMENT)   # dev.x, dev.y, dev.room
val = dataset.load("task2", splits.LOCAL_VALIDATION)
```

**Metadata rule (§8).** `room` and `activity` are labels — use them for
supervision and scoring. `capture_id`, `source_format`, `n_obs_in_capture` and
filenames are provenance only and must **never** be model features.

## Leakage controls (§8)

- The original capture is the atomic split unit; all windows from a capture stay
  in one split. Enforced by `tests/test_no_leakage.py`.
- Room A validation is the supplied `validation/` directory, never moved into
  development. Room B validation is drawn once at capture level, 25 per class,
  seed 42, over deterministically sorted filenames — committed for audit in
  `results/room_B_validation_seed42.json`.
- Normalization statistics are fitted on `split == development` only;
  `normalization.fit()` filters the manifest itself so no caller can pass it
  validation rows.
- Every window records the `config_hash` that produced it.

Validation results are **validation results**, not unseen-test accuracy (§9).

## Ownership

One file, one owner. Raise an issue rather than editing someone else's module.

| Area | Files |
|---|---|
| Ingestion, windowing, Task 1 | `pcap_reader` `csi_decode` `amplitude` `csv_reader` `cleaning` `windowing` `normalization` `splits` `dataset` `task1_room` |
| Task 2 — representation | `encoder` `ssl_pretrain` `baselines` `task2_probe` |
| Task 3 — transfer | `fewshot` `task3_adapt` |
| Shared | `config` `metrics` `figures` `configs/experiment.yaml` |

## Repository layout

```
raw/                     source captures (gitignored, never committed)
intermediate_csv/        common 256-column amplitude form, per capture
windows/                 (150,256) float32 windows + committed manifest.csv
src/                     flat modules, per challenge spec §5.6
configs/experiment.yaml  every parameter that changes derived data
tests/                   leakage and contract tests
results/ figures/        committed: metrics, fitted stats, figures
models/                  Task 2 trained encoders (committed, 6 x 1.1 MB)
docs/                    setup notes, Task 2 plan, run logs
report/                  technical report
```
