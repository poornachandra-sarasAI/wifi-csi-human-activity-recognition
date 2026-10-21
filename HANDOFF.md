# Handoff — Tasks 2 and 3

For the two teammates picking up self-supervised representation learning and
cross-room few-shot adaptation. The ingestion pipeline, splits, windowing,
normalization and Task 1 are done and on `main`'s sibling branch
`pipeline/ingestion-and-task1`.

**You do not need to touch raw data, PCAP decoding, splits or windowing.**
Everything you need is one function call away. Read §1–§4, then your own task's
section.

---

## 1. Get running in five minutes

```bash
git clone https://github.com/poornachandra-sarasAI/wifi-csi-human-activity-recognition
cd wifi-csi-human-activity-recognition
git checkout pipeline/ingestion-and-task1

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Put the source archives where `configs/experiment.yaml` expects them (the data is
**not** in the repo, per challenge spec §4):

```
raw/experiment2/realdata/input_data/30ms/csidata/   # Room A, *.pcap
raw/experiment3/Data/                               # Room B, *.csv
```

Then build the derived data once (~5 minutes, mostly PCAP decoding):

```bash
python run_episode.py          # raw -> intermediate_csv/ -> windows/ -> results/
python -m pytest tests/ -q     # 8 passed
```

You now have `windows/` and `windows/manifest.csv`. You will not need to run
this again unless `configs/experiment.yaml` changes.

> **Do not open `_quarantine_do_not_use/`** (sits beside the repo). It holds the
> source authors' MATLAB code, their `Results/` folders and the Experiment-3
> LSTM notebook. They are not part of the participant package and reading them
> compromises the independence claim in §6. We are required to be able to
> explain every preprocessing stage as our own work.

---

## 2. The only API you need

```python
from src import dataset, splits

dev = dataset.load("task2", splits.DEVELOPMENT)
val = dataset.load("task2", splits.LOCAL_VALIDATION)

dev.x           # (n, 150, 256) float32   — or (n, 150, 256, 2) for preset "task2"
dev.y           # (n,) activity labels: SIT STAND WALK SITDOWN STANDUP FALL EMPTY LYING
dev.room        # (n,) "A" or "B"
dev.window_id   # (n,) provenance
dev.capture_id  # (n,) provenance

dev.filter(room="A", activities=["SIT", "STAND", "WALK"])   # returns a WindowSet
len(dev)        # 2010
```

This signature is **frozen**. Tasks 2 and 3 both build on it — if you need it
changed, raise it with me rather than editing `src/dataset.py` in place.

Windows are stored on disk **un-normalized**; the preset you name is applied at
load time. That is why all three tasks share one `windows/` directory and why
you can change normalization without regenerating anything.

### Normalization presets

Pick with the first argument to `dataset.load`. Defined in
`configs/experiment.yaml` under `normalize.presets`.

| Preset | Scope | What survives | Use for |
|---|---|---|---|
| `task1` | global | everything, incl. absolute level | Task 1 baseline |
| `task1_subband` | global + sub-band offsets removed | within-band shape | Task 1 ablation |
| `task1_honest` | per-window | temporal dynamics only | Task 1 headline |
| `task2` | global, 2-channel | static + dynamic as 2 channels | **Task 2** |
| `task3` | per-room | activity, room removed | **Task 3** |

`task2` returns `(n, 150, 256, 2)`: channel 0 is the scaled window, channel 1 is
the same with the per-subcarrier time-mean removed. Use both, or slice
`x[..., 0]` if you only want the static view.

---

## 3. The data

2,285 windows of shape (150, 256), one per original capture.

| | development | local_validation |
|---|---|---|
| **Room A** (Experiment 2, `.pcap`) | 934 | 150 (supplied, 25 × 6) |
| **Room B** (Experiment 3, `.csv`) | 1,076 | 125 (seed 42, 25 × 5) |

| Room | Activity | dev | val |
|---|---|---|---|
| A | FALL | 177 | 25 |
| A | SIT | 178 | 25 |
| A | SITDOWN | 155 | 25 |
| A | STAND | 115 | 25 |
| A | STANDUP | 155 | 25 |
| A | WALK | 154 | 25 |
| B | EMPTY | 242 | 25 |
| B | LYING | 326 | 25 |
| B | SIT | 128 | 25 |
| B | STAND | 196 | 25 |
| B | WALK | 184 | 25 |

SIT, STAND and WALK exist in both rooms — that is the bridge Task 3 transfers
across. Everything else is room-exclusive.

---

## 4. Rules you must not break

These are challenge requirements, not style preferences. Breaking one
invalidates every number in the report.

1. **Never train, fit or select on `local_validation`.** It is for scoring only.
   Normalization statistics, hyperparameter choices that are *fitted* from data,
   model weights — all from `development` only. (§8)
2. **The capture is the atomic unit.** Already enforced: one window per capture,
   and `tests/test_no_leakage.py` asserts the splits are disjoint. If you ever
   generate multiple windows per capture, they all stay on one side.
3. **No metadata shortcuts.** `room` and `activity` are labels — use them for
   supervision and scoring. `capture_id`, `source_format`, `n_obs_in_capture`,
   filenames, folder names and capture length must **never** be model features.
   (§8)
4. **Task 2 pretraining uses no labels at all** — not activity, not room.
5. **Describe validation results as validation results**, never as unseen-test
   accuracy. (§9)
6. **Run `python -m pytest tests/ -q` before every push.** Eight tests; they
   also catch config drift via the provenance hash.

---

## 5. Read this before you trust anything — the interleaving artifact

This one affects both of you and is not obvious.

Room A's captures interleave **two different frame streams**. Nexmon emits one
frame carrying a real 802.11 sequence number, then ~0.4 ms later a second frame
whose sequence field reads `0xFFFF`. Both carry valid CSI. We keep both, because
dropping the `0xFFFF` stream leaves a median of 111 observations per capture and
makes the official 150-observation window unreachable for **all 1,084** Room A
captures.

The cost is that Room A's time axis alternates between two streams with very
different statistics, and Room B has nothing equivalent:

| Stream | lag-1 autocorrelation | std/IQR |
|---|---|---|
| Room A, sequenced only | +0.54 … +0.84 | 0.570 |
| Room A, `0xFFFF` only | +0.01 … +0.13 | — |
| **Room A, interleaved (what you get)** | **−0.15 … −0.66** | 1.133 |
| Room B | +0.035 | 0.979 |

**What this means for you.**

- **Task 2:** any temporal-consistency or contrastive objective will find a
  strong alternating pattern in Room A and nothing like it in Room B. An encoder
  can minimise its loss by separating the two rooms on this artifact instead of
  learning activity structure. Expect your UMAP to split cleanly by room; that
  is partly this, not partly insight. Augmentations that break strict temporal
  adjacency (random temporal subsampling, random crop with stride, shuffling
  within short spans) will suppress it. Worth an explicit ablation.
- **Task 3:** part of the A↔B domain gap you are adapting across is this
  artifact rather than the rooms. Per-room normalization (the `task3` preset)
  removes static level differences but **not** this temporal signature. If
  adaptation works far better in one direction than the other, suspect this
  before you conclude anything about room geometry.

To experiment, set `ingest.room_a.frame_stream: sequenced` in
`configs/experiment.yaml` and re-run `run_episode.py`. You get ~111 observations
per capture, reflect-padded to 150. Keep `all` for headline numbers so we all
report the same thing.

---

## 6. Task 2 — learn without labels

Owner: teammate 2. Your files: `src/encoder.py`, `src/ssl_pretrain.py`,
`src/baselines.py`, `src/task2_probe.py`.

**What the challenge asks (§7, §11):**

1. Train an encoder on development windows from **both** rooms using **no
   activity and no room labels**.
2. Freeze it. Fit a **linear** classifier on labelled development data.
3. Score the frozen representation on the fixed validation set.
4. Compare against two baselines: **handcrafted features**, and **the same
   architecture with random weights**. Both are required, not optional.
5. Figure: 2-D UMAP or t-SNE of the embeddings, coloured once by room and once
   by activity. Treat as qualitative.
6. Metrics: macro-F1 is the headline, plus balanced accuracy.

**Suggested starting point.** Contrastive (SimCLR-style) on CSI windows, with
augmentations drawn from: random temporal crop, subcarrier masking, additive
noise, time reversal, random temporal subsampling. Masked reconstruction (MAE
style) is the other obvious route. Start with whichever you can get running
end to end — a working pipeline beats a clever model that does not run.

**Interface the other two need from you:**

```python
# src/encoder.py
def build_encoder(cfg) -> torch.nn.Module: ...
def embed(model, x: np.ndarray) -> np.ndarray:   # (n, 150, 256, 2) -> (n, d)
```

Teammate 3 imports `embed` to build class prototypes on frozen features, so
settle its signature early and keep it stable. Save your trained weights
somewhere ignored by git (`*.pt` is already gitignored) and commit a short
`results/task2_*.csv` with the metrics.

**Watch for:** the dataset is small (2,010 development windows) so a large
encoder will overfit; the two rooms are imbalanced (934 vs 1,076) and so are the
activities (STAND 115 vs LYING 326) — use macro-F1, and consider class-balanced
sampling for the probe.

---

## 7. Task 3 — cross-room few-label adaptation

Owner: teammate 3. Your files: `src/fewshot.py`, `src/task3_adapt.py`.

**What the challenge asks (§7, §9, §11):**

For each transfer direction, one room is the target and the other is the source.
The source supplies all labels from its development captures. The target
supplies **only K labelled captures per target-exclusive activity**.

| Target | K-labelled target activities | Recognised with **no** target labels |
|---|---|---|
| Room A | SITDOWN, STANDUP, FALL | SIT, STAND, WALK (learned from Room B) |
| Room B | EMPTY, LYING | SIT, STAND, WALK (learned from Room A) |

- K ∈ {1, 5, 10, 20}. **K means K distinct original captures**, not K windows.
- Support captures come **only from the target room's development split**.
  Validation captures are never eligible as support.
- Shared SIT/STAND/WALK labels in the target room are withheld from support.
- Evaluate every K on the fixed validation set, both directions.
- Report **macro-F1** (primary) and **target-exclusive macro-F1** separately.

**Required outputs (§11):**

- Label-efficiency curve: macro-F1 vs K, per target room, with separate curves
  for target-labelled and transferred shared activities.
- Confusion matrix per target room at K = 5, with per-activity precision,
  recall and F1.
- **Routing ablation:** compare predicted-room routing, oracle-room, and the
  same adaptation model with no routing stage. Import Task 1 from
  `src/task1_room.py` for the predicted-room arm — that is the integration
  point between us, and `task1_honest` is the preset to route with (see §8).

**Suggested starting point.** Prototypical networks / nearest class centroid on
frozen embeddings from teammate 2's encoder, with per-room normalization (the
`task3` preset already does AdaBN-style per-domain scaling). Until the encoder
exists, prototype on raw flattened windows or simple handcrafted features so
your K-sweep machinery is ready.

**Watch for:** K = 1 is high variance — average over several random support
draws with fixed seeds and report the spread, otherwise the curve is noise.
Use `dev.capture_id` to sample support at capture level, and assert your support
sets never intersect `local_validation`.

---

## 8. What Task 1 found, and why it matters to you

Task 1 is done, but the honest conclusion is a caveat you should both inherit.

| Normalization rung | What is removed | Validation accuracy |
|---|---|---|
| `task1` global | nothing | 1.0000 |
| `task1_subband` | per-sub-band AGC offsets | 1.0000 |
| `task1_honest` per-window | all static per-subcarrier structure | 0.9927 |
| + sequenced frames only | Room A's frame interleaving | 0.9418 |

`task1` is perfect **from a single observation**, which is impossible for a real
environment fingerprint — it is reading a global ~490× amplitude gap between our
Room A decode and Room B's supplied integers. Even the strictest rung leaves the
two sessions differing in temporal dynamics (lag-1 +0.656 vs +0.035).

**Conclusion:** the benchmark cannot cleanly separate *room* from *recording
session and pipeline* (§12.1). For Task 3's routing ablation this matters
directly — room routing will look near-perfect, so the interesting comparison is
predicted-room vs **oracle**-room, where a gap of zero tells you routing is free,
not that routing is clever.

Figures: `figures/task1_accuracy_vs_window_length.png`,
`figures/task1_ablation.png`, `figures/decode_validation.png`.

---

## 9. Working together

One file, one owner. Raise an issue rather than editing someone else's module.

| Area | Files | Owner |
|---|---|---|
| Ingestion, windowing, normalization, Task 1 | `pcap_reader` `csi_decode` `amplitude` `csv_reader` `cleaning` `windowing` `normalization` `splits` `dataset` `task1_room` `task1_ablation` | me |
| Task 2 | `encoder` `ssl_pretrain` `baselines` `task2_probe` | teammate 2 |
| Task 3 | `fewshot` `task3_adapt` | teammate 3 |
| Shared | `config` `metrics` `figures` `configs/experiment.yaml` | all — small commits |

- Branch per person off `main`: `task2/<what>`, `task3/<what>`. Small, frequent
  commits — the spec says the history counts and every member must contribute
  visibly.
- `src/figures.py` has one function per required figure; add yours there and tag
  the owner in the docstring so we do not collide.
- Write metrics to `results/*.csv` using the §10 schema:
  `experiment, target_room, K, metric, value`. Final predictions use
  `window_id, room, label, confidence`.
- **Never commit data.** `raw/`, `intermediate_csv/**/*.csv` and
  `windows/**/*.npy` are gitignored. `windows/manifest.csv` **is** committed on
  purpose — it is the provenance trail.
- If you change `configs/experiment.yaml`, bump `config_version`. The hash is
  stamped into every manifest row and a test will fail if the windows on disk no
  longer match the config.

---

## 10. Still open

- Task 2 and Task 3 are not started.
- The technical report (`report/`) is empty. 8–10 pages, needs the preprocessing
  diagram, design rationale, results, the §11 figures, a contribution statement
  and an AI-use disclosure.
- Only one window per capture is generated. If Task 2 wants more training data,
  `windowing.selection: strided` is scaffolded in the config but not implemented
  — ask me rather than implementing it yourself, since it has split implications.
- The `pilot/data` amplitude ratio differs between rooms (Room A 0.98 vs Room B
  1.30). Unexplained, probably another instrumentation difference. Low priority.
