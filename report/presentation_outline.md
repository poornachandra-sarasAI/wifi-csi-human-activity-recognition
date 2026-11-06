# Presentation outline — 7–9 minute video

Team Klique · CS60055 Hackathon Challenge 2

What the challenge asks for (§14, verbatim):

> A short demonstration/presentation, **including a live run on an episode supplied by the
> teaching team**.

and separately:

> Runnable end-to-end system that can process a challenge episode and **write predictions in
> the required format**.

So the video is not just slides. A live run of `run_episode.py` producing
`window_id, room, label, confidence` has to be on screen. Everything below is built around
that constraint.

---

## The demo path (fixed and verified)

`run_episode.py` previously wrote the **ground-truth activity** into the `label` column — a
placeholder from before Tasks 2 and 3 existed. That is now fixed: `src/predict.py` performs real
inference, and `label` is a genuine prediction.

How a prediction is made, with no labels read at any point:

1. **Room** — Task 1 classifier on the `task1_honest` preset (per-window standardised, so it does
   not simply read the inter-room amplitude gap).
2. **Activity** — the frozen Task 2 encoder embeds the window, the embedding is standardised with
   the predicted room's development statistics, and the nearest class prototype of that room is
   taken by cosine similarity.
3. **Confidence** — softmax over those cosine similarities.

Verified on the fixed validation set: **room 99.27%, activity 68.4%** (it was 100% before,
because it was copying the answer). Confidence spans 0.344–1.000.

```bash
python run_episode.py --predict-only          # predict the validation split
python run_episode.py --episode <DIR>         # predict a supplied episode
```

`--episode` takes a directory of raw `.pcap` and/or `.csv` captures at any depth, in any mixture,
with any filenames. Tested on a mock episode of renamed Room A pcaps plus Room B CSVs: both CSVs
were routed to Room B and correctly called `EMPTY`. Unreadable captures are skipped with a warning
rather than killing the run. Nothing is fitted against the episode — both normalisation presets in
this path are per-window — so no labels and no prior knowledge of the room are required.

**Say this on camera:** the activity number here (68.4%) is the *deployed* system, which uses all
development labels. It is not comparable with the Task 3 numbers, which deliberately starve the
target room of labels to measure label efficiency. Do not let the two get conflated.

---

## Narrative spine

Do not walk through the three tasks like a checklist. The distinguishing feature of this
project is that **three times, a headline number turned out to be measuring our instruments
rather than the rooms, and each time we found it and said so.** That is the story, and it is
what separates this from a submission that reports 1.0000 and stops.

One-line pitch to open and close with:

> Every number that looked impressive in this benchmark turned out to be an artifact. Here is
> how we found each one, and what survived.

---

## Time budget (target 8:00, band 7:00–9:00)

| # | Segment | Speaker | Length | Running |
|---|---|---|---|---|
| 1 | Problem and what the benchmark asks | Mohan | 0:40 | 0:40 |
| 2 | The data, and why it is asymmetric | Mohan | 0:35 | 1:15 |
| 3 | Pipeline: decode → common form → windows | Mohan | 1:10 | 2:25 |
| 4 | Leakage controls | Mohan | 0:25 | 2:50 |
| 5 | Task 1 and the first artifact | Mohan | 0:50 | 3:40 |
| 6 | Task 2: self-supervised encoder | Hemant | 1:20 | 5:00 |
| 7 | Task 3: few-shot transfer and its failure | Saras | 1:25 | 6:25 |
| 8 | **Live run** | Saras | 0:55 | 7:20 |
| 9 | What we would claim, and what we would not | Mohan | 0:40 | 8:00 |

Roughly 2:40 / 1:20 / 2:20 per speaker plus the demo. Section 3 is the largest single block
because §14 names the preprocessing pipeline as a deliverable in its own right.

---

## Segment-by-segment

### 1 · Problem and what the benchmark asks — 0:40

**Say:** WiFi CSI gives device-free activity sensing without cameras or wearables. The blocker
is cross-environment shift: a model trained in one room degrades in the next. The benchmark
asks three things of two rooms recorded on the same radio — identify which room a window came
from, learn a representation with no labels, and transfer activities to a new room given only
K labelled captures of the activities unique to it.

**On screen:** title slide, then one line each for Tasks 1/2/3.

**Avoid:** background on WiFi sensing literature. No time, no marks.

---

### 2 · The data, and why it is asymmetric — 0:35

**Say:** Two rooms, same hardware — Asus RT-AC86U, BCM4366c0, 5 GHz, 80 MHz, 256 subcarriers.
But the archives are deliberately asymmetric: Room A ships as raw `.pcap` that we must decode
ourselves, Room B as amplitude `.csv` that is already half-processed. Bringing them into one
representation *is* the data-engineering problem.

**Numbers:** Room A 1,084 captures (934 dev / 150 supplied validation). Room B 1,201
(1,076 dev / 125 validation, drawn at capture level with seed 42). 2,285 windows total.
SIT, STAND and WALK exist in both rooms and are the bridge for Task 3.

**On screen:** the two-archive comparison; the dataset table from the report.

---

### 3 · Pipeline — 1:10

**On screen:** Figure 1 from the report for the whole segment.

Three things to land, in this order:

**(a) We resolved the packed-CSI layout ourselves, and can prove it.** Each 32-bit word packs
a real mantissa, an imaginary mantissa and an exponent. Rather than copy the layout, we
recovered it from the data twice over. Bit occupancy across 2.01 M words shows bits 30–31
always zero and two statistically identical 12-bit fields — the signature of a real and an
imaginary part. Applying the 6-bit exponent then reproduces the 802.11ac null-subcarrier
structure: amplitude at the known guard-band and DC positions drops to **0.020** of the data
median, against Room B's **0.024**. Without the exponent that ratio is 0.78–1.01 and the
nulls vanish.

**(b) The frame-interleaving discovery.** Room A alternates two frame streams: one with real
802.11 sequence numbers, one reading `0xFFFF` about 0.4 ms later. The `0xFFFF` frames are not
corrupt — 976–992 of 1024 payload bytes are non-zero. Dropping them halves the observation
count from a median of **221 to 111**, which puts **all 1,084 Room A captures** below the
mandated 150-observation window. So we keep both. The cost is a strong negative lag-1
autocorrelation of **−0.33** in Room A against **+0.035** in Room B — which comes back to bite
us in Tasks 1 and 2.

**(c) Windowing.** Deterministic centred 150 × 256 crop per capture, no labels consulted.
Twelve short captures reflect-padded and flagged, three of them in the official validation
set, so the validation denominator is preserved.

---

### 4 · Leakage controls — 0:25

**Say:** The capture is the atomic unit — every window from a capture stays on one side of the
split. Room A uses the supplied validation directory untouched; Room B's is drawn once at
capture level with seed 42 and committed for audit. Normalisation statistics are fitted on
development data only, and the fitting function filters the manifest itself so no caller can
hand it validation rows. Eight automated tests enforce all of this.

**On screen:** `pytest tests/ -q` passing. Keep it to one line of output.

---

### 5 · Task 1 and the first artifact — 0:50

**On screen:** the accuracy-vs-window-length figure, then the ablation ladder table.

**Say:** Room identification scores a perfect **1.0000**. But it scores 1.0000 from a *single
observation*, and one observation contains no temporal information at all — so it is reading a
constant offset, not a room. The cause is a **~490×** amplitude gap between our decoded Room A
values and Room B's supplied integers.

Then the ladder: neutralising the sub-band AGC step leaves 1.0000. Per-window standardisation,
which erases every static profile, still gives **0.9927**. Only removing Room A's frame
interleaving brings it to **0.9418** — the largest single drop.

**The line to land:** Task 1 works, but it is identifying a recording session, not a room. We
report it as a router for Task 3, not as room sensing.

---

### 6 · Task 2 — 1:20 (Hemant)

**Say:** A 1-D dilated residual CNN, 280 k parameters, pretrained with SimCLR on 2,010
development windows using no activity and no room labels. Because Room A carries the
interleaving signature, naive contrastive learning can minimise its loss by detecting that
signature instead of learning motion — so we added two adjacency-breaking augmentations:
stride-2 subsampling with random phase, and local temporal shuffling.

**Numbers (frozen encoder, linear probe, 8-class validation):**

| | Macro-F1 | Balanced acc. |
|---|---|---|
| **SimCLR SSL** | **0.924 ± 0.013** | 0.928 ± 0.014 |
| Handcrafted (416-d) | 0.893 | 0.895 |
| Random weights | 0.616 ± 0.008 | 0.637 ± 0.006 |

**Two points worth making:**

The learned representation **beats** the engineered baseline, by 3.1 points — self-supervision
on unlabelled CSI recovers more than hand-designed spectral features.

The augmentation ablation is the interesting one. Removing the adjacency-breaking augmentations
costs 2.3 points overall, and the entire loss falls on Room A (0.912 → 0.869) with Room B
untouched (0.950 → 0.951) — exactly the room with interleaving. And the ablated model reaches a
*lower* pretraining loss (**1.406 vs 1.689**): keeping the artifact makes the pretext task
easier while making the representation worse. A lower SimCLR loss is not evidence of a better
encoder.

**On screen:** the probe table, then the UMAP coloured by room and by activity. Note aloud that
the embedding separates the rooms completely — which previews Task 3's problem.

---

### 7 · Task 3 — 1:25 (Saras)

**Say:** Source room supplies all its labels; the target supplies only K labelled captures of
its exclusive activities. SIT/STAND/WALK must transfer with no target labels at all.
Prototypical networks on frozen embeddings, per-room standardisation, K ∈ {1,5,10,20},
30 runs per setting.

**The split result:**

- Target-exclusive activities scale with K: Room A **0.324 → 0.510**, Room B **0.390 → 0.588**
  from K=1 to K=20. Few-shot works.
- Shared activities do not move at all — **0.090** in Room A against chance 0.167, **0.211** in
  Room B against chance 0.200. Zero-label transfer fails.

**The honest part — tell this as a retraction.** Handcrafted features first appeared to transfer
in Room B at 0.236 shared F1. Auditing found why: Room B's WALK captures are the only files in
the archive missing the `_Amp_` filename substring, and their mean amplitude is 50% above every
other Room B class — 473 against 304–358. A scalar amplitude threshold alone separates WALK at
**ROC-AUC 0.898**. Our per-room scaler could not remove it, because it is a per-*class*
difference inside one room. We built a level-free representation, retrained every encoder,
reran everything, and the apparent transfer collapsed from 0.236 to **0.038**. We retracted it.

**Then the diagnostic.** Hungarian optimal assignment asks what the best possible relabelling
would score. Room B's shared F1 rises from 0.182 to 0.305 under an oracle permutation, and the
optimal permutation swaps SIT and WALK — in *both* directions. The clusters are there; the
correspondence between rooms is wrong. Flag clearly that this is an oracle computed from
validation labels: a diagnostic ceiling, never a result.

**Routing:** predicted routing matches oracle to within 0.006 F1; removing routing costs 14.2
points in Room B. But since Task 1 succeeds on session artifacts, routing is nearly free here.

**On screen:** label-efficiency curve → confusion matrices → per-activity table.

---

### 8 · Live run — 0:55 (required)

**Terminal, recorded unedited.**

```bash
python -m pytest tests/ -q                    # 8 leakage + contract tests
python run_episode.py --episode demo_episode/ # raw captures -> predictions
head -5 results/predictions.csv               # window_id, room, label, confidence
```

Prepare `demo_episode/` in advance: a handful of Room A `.pcap` and Room B `.csv` captures with
**neutral filenames** (`capture_01.pcap`, …), so it is visible that nothing is read from the name.
That run takes seconds.

Say explicitly that the teaching team can point `--episode` at their own directory and get the same
four columns, and that no labels are read anywhere in that path.

Then show the full rebuild command and say ingestion takes about five minutes, so it was pre-run:

```bash
python run_episode.py                         # full rebuild from raw captures
```

### 9 · What we would claim, and what we would not — 0:40

**Say:** Three honest conclusions.

Within a room, activity recognition works well — 0.924 macro-F1 from a label-free encoder.
Few-shot learning of new activities works — 0.59 at K=20 from five labelled captures.
Zero-label cross-room transfer of shared activities does not work, and the Hungarian bound
shows why: the geometry survives, the correspondence does not.

And the caveat that frames all of it: this benchmark cannot separate *room* from *recording
session and pipeline*. Every rung of our Task 1 ladder still leaves one instrumentation
difference standing. We report that as the finding rather than hiding it behind 1.0000.

Close with contributions and the AI-use disclosure — one slide, five seconds, but say it out
loud: all three members used Claude via Claude Code, and the disclosure in the report names
exactly which components were AI-drafted.

---

## If you run long

Cut in this order — none of these is separately required by §11 or §14:

1. Segment 2 down to one sentence (the dataset table can stay on screen while you talk).
2. The bit-occupancy half of 3(a); keep the null-subcarrier evidence, it is the stronger one.
3. The routing paragraph in Segment 7 down to one sentence.

**Never cut:** the live run (§14 names it), any of the five §11 figures, or the retraction —
it is the single most credible thing in the submission.

---

## Likely questions

- *"Why keep frames you call an artifact?"* → Dropping them makes the mandated 150-observation
  window unreachable for all 1,084 Room A captures. We kept them and quantified the cost.
- *"Isn't 1.0000 on Task 1 a good result?"* → It is 1.0000 from one observation, which is
  impossible for a real fingerprint. The honest number is 0.9418.
- *"Why is the Hungarian bound not your headline?"* → It uses validation labels to pick the
  permutation. §8 allows that for analysis only.
- *"Is the WALK anomaly really instrumentation?"* → Walking has the lowest temporal variance and
  the highest amplitude of any Room B class, which is backwards physically, and it is the only
  class with a different filename convention. All 209 files.

---

## Verified numbers (do not re-derive for the slides)

Every figure below was checked against the committed CSVs on 2026-11-06.

| Quantity | Value | Source |
|---|---|---|
| Captures A / B | 1,084 / 1,201 | `results/captures.csv` |
| Dev / val windows | 2,010 / 275 | `windows/manifest.csv` |
| Decode layout | n_man 12, n_exp 6, exponent applied | `decode_layout_resolution.csv` |
| Null ratio A / B | 0.020 / 0.024 | report §2.2 |
| Obs per capture, A | median 221 → 111 if `0xFFFF` dropped | `ingest_room_A.csv` |
| lag-1 autocorr A / B | −0.33 / +0.035 | `task1_dynamics_diagnostic.csv` |
| Task 1 ladder | 1.0000 / 1.0000 / 0.9927 / 0.9418 | `task1_room_accuracy.csv`, `task1_frame_stream_ablation.csv` |
| Task 1 at T=1 | 1.0000 (global), 0.8764 (per-window) | `task1_room_accuracy.csv` |
| Task 2 SSL / hand / random | 0.924 / 0.893 / 0.616 | `task2_probe.csv` |
| Task 2 ablation A / B | 0.912→0.869 / 0.950→0.951 | `task2_probe.csv` |
| Pretrain loss full / ablated | 1.689 / 1.406 | `task2_pretrain_loss.csv` |
| Task 3 K=5 B, overall/excl/shared | 0.337 / 0.525 / 0.211 | `task3.csv` |
| Task 3 K=20 exclusive A / B | 0.510 / 0.588 | `task3.csv` |
| Routing predicted / none, B | 0.343 / 0.195 | `task3.csv` |
| Hungarian shared A / B | 0.177→0.251, 0.182→0.305 | `task3_hungarian_bound.csv` |
| WALK shortcut | ROC-AUC 0.898 → 0.693 level-free | report §6.2 |
| Retracted claim | 0.236 → 0.038 | report §6.2 |
