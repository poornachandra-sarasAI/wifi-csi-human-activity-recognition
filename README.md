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

## Task 1 is reported two ways, on purpose

Two instrumentation differences survive global normalization: Room A and Room B
amplitudes differ by a global factor of **~490×** (an artifact of the authors'
unknown integer scaling), and Room A carries a **~4× AGC step** across
subcarriers 137–187 — the third 20 MHz sub-band — that Room B does not.

| Preset | Normalization | Validation accuracy | Accuracy at window length 1 |
|---|---|---|---|
| `task1` | global | **1.0000** | **1.0000** |
| `task1_honest` | per-window | **0.9927** | 0.8764 |

`task1` is perfect from a *single observation*, which is proof it reads a
constant scale offset rather than the environment. `task1_honest` removes the
absolute level and climbs 0.876 → 0.996 as the window grows, which is the
defensible environment-fingerprint claim. The gap between the two rows is the
informative result (§12, room-versus-session confound).

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
