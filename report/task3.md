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
Both are computed from `task2`-preset windows, which `embed()` requires, and then
z-scored **per room** with statistics of that room's development windows. No label is
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

## Results

**Headline, K = 5, oracle routing** (macro-F1, validation):

| Representation | Method | Target A all | Target A excl. | Target A shared | Target B all | Target B excl. | Target B shared |
|---|---|---|---|---|---|---|---|
| SSL | prototype | 0.215 ± 0.033 | 0.418 ± 0.066 | 0.011 ± 0.019 | 0.277 ± 0.025 | 0.623 ± 0.054 | 0.046 ± 0.037 |
| SSL | refined | 0.168 ± 0.031 | 0.296 ± 0.062 | 0.040 ± 0.033 | 0.219 ± 0.046 | 0.483 ± 0.095 | 0.043 ± 0.027 |
| handcrafted | prototype | 0.201 ± 0.028 | 0.329 ± 0.054 | 0.072 ± 0.019 | 0.139 ± 0.036 | 0.348 ± 0.090 | 0.000 ± 0.000 |
| handcrafted | refined | 0.213 ± 0.034 | 0.289 ± 0.056 | 0.138 ± 0.035 | 0.346 ± 0.063 | 0.513 ± 0.078 | 0.236 ± 0.085 |

Uniform random guessing gives about 0.17 macro-F1 for Room A's 6 classes and 0.20 for
Room B's 5.

**Label efficiency** (SSL + prototypes, oracle, target-exclusive macro-F1):

| K | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| Target A | 0.307 ± 0.085 | 0.418 ± 0.066 | 0.475 ± 0.065 | 0.482 ± 0.051 |
| Target B | 0.424 ± 0.114 | 0.623 ± 0.054 | 0.659 ± 0.064 | 0.676 ± 0.055 |

Transferred shared macro-F1 stays at or below 0.06 at every K for both SSL methods.

**Routing ablation, K = 5** (SSL + prototypes, overall macro-F1):

| Routing | Target A | Target B |
|---|---|---|
| oracle | 0.215 | 0.277 |
| predicted | 0.216 | 0.279 |
| none | 0.200 | 0.203 |

## Discussion

1. **The shared activities do not transfer.** With SSL features, transferred
   SIT/STAND/WALK reach macro-F1 ≤ 0.06 at every K, which is below chance. This
   matches the development-only check in Task 2 (`results/task2_transfer_check.csv`,
   0.08–0.16). The confusion matrices (`figures/task3_confusion_K5.png`) show why this
   is not noise. The target room does have activity structure, but the source
   prototypes land on the wrong part of it. In Room B, 98 % of SIT windows are
   predicted WALK and 44 % of WALK windows are predicted SIT. Per-room standardisation
   aligns the first two moments of each room, not the class geometry. The same
   activity name does not produce the same signal in these two recordings (Task 1
   §12.1 confound: room, session and pipeline cannot be separated).
2. **The few labels work, but the failed transfer caps them.** Target-exclusive
   macro-F1 rises with K (Room B 0.42 → 0.68 from K = 1 to 20) and saturates by K ≈
   10. It cannot approach the within-room probe's 0.94, because the target's shared
   windows have no nearby prototype and are absorbed into the exclusive classes,
   halving their precision. With only half of Room A's validation windows
   target-exclusive, perfect recall would still cap their F1 near 0.67.
3. **Refinement helped handcrafted features and hurt the SSL encoder.** For
   handcrafted features, soft k-means lifts Room B from 0.139 to 0.346 overall, and
   its shared F1 from 0.000 to 0.236. That is the only setting where transfer rises
   clearly above chance. For SSL embeddings it lowers every number, so the method we
   chose in advance is not the best one, and we report that as it is. A likely reason:
   SSL embeddings separate the room's recording sessions (Task 2 found Room B's SIT
   and STAND forming their own island), so unsupervised clusters follow session, not
   activity, and refinement pulls the prototypes onto them. We did not switch the
   headline method after seeing validation results.
4. **Routing is free here, and that says nothing about room recognition.** Predicted
   routing matches oracle routing to within 0.003, because the router is 99.27 %
   accurate on these two sessions (Task 1). Removing routing costs up to 0.074
   (Room B), because a single bank over 8 activities lets source-exclusive classes
   absorb target windows. As HANDOFF §8 anticipated, the informative comparison is
   predicted vs oracle, and a gap of zero means routing costs nothing, not that it is
   clever.

## Limitations

- Every number above reflects one pair of recordings. The benchmark cannot separate
  room from session and pipeline (Task 1), so "cross-room" here means "cross-session
  and cross-pipeline" as much as cross-room.
- Room A's interleaved frame streams (HANDOFF §5) are a temporal signature that
  per-room scaling does not remove. Part of the A↔B gap is likely this artifact.
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
