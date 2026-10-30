# Task 3 — Cross-room few-label adaptation (draft report section)

*Draft for the shared technical report. Numbers come from `results/task3_per_run.csv`
and `results/task3.csv`; figures are `figures/task3_label_efficiency.png`,
`figures/task3_confusion_K5.png` and `figures/task3_routing.png`. All scores are on the
fixed local validation set and are validation results, not unseen-test accuracy.*

## Goal

For each target room, recognise all of its activities with all labels from the other
(source) room and only **K labelled target captures per target-exclusive activity**,
K ∈ {1, 5, 10, 20}. Target SIT/STAND/WALK labels are withheld, so those three
activities must be transferred from the source room.

| Target | K-labelled (target-exclusive) | Transferred from the source |
|---|---|---|
| Room A | SITDOWN, STANDUP, FALL | SIT, STAND, WALK (from Room B) |
| Room B | EMPTY, LYING | SIT, STAND, WALK (from Room A) |

## Method

**Features.** Frozen 128-d embeddings from the Task 2 SimCLR encoder (`full` variant,
all three pretraining seeds), and Task 2's 416-d handcrafted features as a baseline.
Both are computed from `task2_level_free`-preset windows (the preset `task2.normalize_preset`
points at; see "Revision" below), and then z-scored **per room** with statistics of that room's development windows. No label is
used in that step. It is the feature-space version of the `task3` preset's per-room
scaling (AdaBN-style alignment).

**Classifier.** Nearest class prototype by cosine similarity (softmax temperature 0.1
for confidences). Shared-activity prototypes are the class means of the source room's
labelled development windows. Target-exclusive prototypes are the means of the K
support captures.

**Adaptation (`refined`).** Ten soft k-means steps over the target room's unlabelled
development windows, initialised at the prototypes above, with support windows fixed
to their labels (semi-supervised prototypical networks). The idea: if the target room
has activity clusters that the source prototypes miss, the prototypes should move onto
them. This was chosen as the proposed method before any validation result was seen,
and plain prototypes are kept as the ablation.

**Support sampling.** K means K distinct original captures. They are drawn only from
the target room's development split, and three assertions run on every draw: the
support captures are disjoint from local_validation, contain no shared-activity label,
and number exactly K per activity. Each (encoder seed, K) is repeated over 10 seeded
support draws. The draws are identical across representations, so comparisons are
paired. Results are mean ± std over the 30 runs (handcrafted: 10).

**Routing ablation.** Three ways of deciding which room's scaling and prototype bank a
validation window uses:

- *oracle*: the true room;
- *predicted*: Task 1's classifier on the `task1_honest` preset, fitted on development
  rooms (validation accuracy 0.9927). A window routed to the source room is classified
  among the source room's activities.
- *none*: no routing stage. One global scaling and one prototype bank over all 8
  activities, built from the same labels.

Every setting (temperature, iterations, number of draws, router preset) is fixed in
`configs/experiment.yaml`. Nothing was tuned on local_validation.

## Revision: level-free rerun

mohan's review (`docs/fix1_resume_notes.md`) found that Room B WALK is about 50 % louder
than the other Room B activities, so one scalar (mean amplitude) separates it with an AUC
of 0.898. The Task 2 encoders were retrained on the `task2_level_free` preset, which takes
out each window's absolute level and keeps its spectral shape. Task 3 was then rerun
unchanged, with the same settings, draws and seeds. Every number below comes from this
rerun. The earlier version of this section reported one apparent transfer success,
handcrafted + refined on Room B (shared F1 0.236). **That claim is retracted.** On
level-free features it drops to 0.038, so it came from the amplitude shortcut and not
from activity transfer.

## Results

**Headline, K = 5, oracle routing** (macro-F1, validation):

| Representation | Method | Target A all | Target A excl. | Target A shared | Target B all | Target B excl. | Target B shared |
|---|---|---|---|---|---|---|---|
| SSL | prototype | 0.240 ± 0.043 | 0.390 ± 0.075 | 0.090 ± 0.042 | 0.337 ± 0.068 | 0.525 ± 0.077 | 0.211 ± 0.082 |
| SSL | refined | 0.241 ± 0.055 | 0.308 ± 0.076 | 0.173 ± 0.084 | 0.295 ± 0.092 | 0.466 ± 0.109 | 0.181 ± 0.127 |
| handcrafted | prototype | 0.268 ± 0.041 | 0.448 ± 0.077 | 0.089 ± 0.017 | 0.170 ± 0.032 | 0.411 ± 0.072 | 0.009 ± 0.011 |
| handcrafted | refined | 0.232 ± 0.014 | 0.330 ± 0.033 | 0.134 ± 0.021 | 0.160 ± 0.017 | 0.343 ± 0.037 | 0.038 ± 0.025 |

Before the rerun (old `task2` preset), SSL + prototype scored 0.215 / 0.277 overall and
0.011 / 0.046 shared (A / B).

Uniform random guessing gives about 0.17 macro-F1 for Room A's 6 classes and 0.20 for
Room B's 5.

**Label efficiency** (SSL + prototypes, oracle, target-exclusive macro-F1):

| K | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| Target A | 0.324 ± 0.078 | 0.390 ± 0.075 | 0.471 ± 0.059 | 0.510 ± 0.053 |
| Target B | 0.390 ± 0.093 | 0.525 ± 0.077 | 0.573 ± 0.064 | 0.588 ± 0.068 |

Transferred shared macro-F1 does not grow with K, as expected, because K adds no
shared labels. For SSL it stays in 0.09–0.20 for Room A and 0.16–0.24 for Room B.

**Routing ablation, K = 5** (SSL + prototypes, overall macro-F1):

| Routing | Target A | Target B |
|---|---|---|
| oracle | 0.240 | 0.337 |
| predicted | 0.240 | 0.343 |
| none | 0.180 | 0.195 |

## Discussion

1. **The shared activities still do not transfer.** Removing the level shortcut raised
   SSL shared F1 in Room B from 0.046 to 0.211, but that is roughly chance level. Uniform
   guessing over 5 classes gives about 0.20. In Room A it is 0.09, below chance. The
   oracle Hungarian diagnostic (`results/task3_hungarian_bound.csv`) shows the
   structure: in Room B the best label map still swaps **SIT ↔ WALK**. In the K = 5
   confusion matrix (SSL, refined), 49 % of SIT windows and 37 % of STAND windows are
   predicted WALK, and 35 % of WALK windows are predicted EMPTY. In Room A the WALK swap
   is gone (WALK maps to itself), but SIT, STAND and FALL rotate into each other, and
   34 % of SIT windows are predicted SITDOWN. So the SIT↔WALK swap is not only the
   loudness artifact. Per-room standardisation aligns the first two moments of each
   room, not the class geometry. The same activity name does not produce the same
   signal in these two recordings (Task 1 §12.1 confound: room, session and pipeline
   cannot be separated).
2. **The few labels work, but the failed transfer caps them.** Target-exclusive
   macro-F1 rises with K (Room A 0.32 → 0.51 and Room B 0.39 → 0.59 from K = 1 to 20)
   and flattens for Room B by K ≈ 10. It cannot approach the within-room probe's 0.94, because the target's shared
   windows have no nearby prototype and are absorbed into the exclusive classes,
   halving their precision. With only half of Room A's validation windows
   target-exclusive, perfect recall would still cap their F1 near 0.67.
3. **Refinement does not help overall.** For SSL it trades exclusive F1 for shared F1
   (Room A shared 0.090 → 0.173, exclusive 0.390 → 0.308) and leaves overall macro-F1
   level for Room A (0.241 vs 0.240) and lower for Room B (0.295 vs 0.337). For
   handcrafted features it is now worse everywhere. The old handcrafted Room B gain
   (0.139 → 0.346) disappeared with the level shortcut (see Revision). The method we
   chose in advance is therefore not the best one, and we report that as it is. A likely
   reason: SSL embeddings separate the room's recording sessions (Task 2 found Room B's
   SIT and STAND forming their own island), so unsupervised clusters follow session,
   not activity. We did not switch the headline method after seeing validation results.
4. **Routing is free here, and that says nothing about room recognition.** Predicted
   routing matches oracle routing to within 0.006, because the router is 99.27 %
   accurate on these two sessions (Task 1). Removing routing costs up to 0.142
   (Room B), because a single bank over 8 activities lets source-exclusive classes
   absorb target windows. As HANDOFF §8 anticipated, the informative comparison is
   predicted vs oracle, and a gap of zero means routing costs nothing, not that it is
   clever.

## Limitations

- Every number above reflects one pair of recordings. The benchmark cannot separate
  room from session and pipeline (Task 1), so "cross-room" here means "cross-session
  and cross-pipeline" as much as cross-room.
- Room A's interleaved frame streams (HANDOFF §5) are a temporal signature that
  per-room scaling does not remove. 59 % of Room A's temporal power is at the Nyquist
  bin. Notching it (fix 2 in `docs/fix1_resume_notes.md`) is the next thing to try,
  because the swap survived fix 1.
- 25 validation captures per class: one capture is 4 % of a class's recall.
- The refinement's temperature and iteration count were fixed a priori and not
  searched. A development-only protocol (holding out target development captures)
  would be the correct way to choose between methods, and we have not run it.

## Reproduce

```bash
python -m src.task3_adapt          # ~5 min on CPU; results/task3*.csv
python -m src.figures --task3      # figures/task3_*.png
python -m pytest tests/ -q         # 28 passed
```
