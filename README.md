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
report/                  technical report
```
