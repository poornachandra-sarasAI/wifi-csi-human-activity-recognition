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
Both are computed from `task2_notched`-preset windows (what `task2.normalize_preset`
points at; see "Revisions" below) and then z-scored **per room** with statistics of
that room's development windows. No label is used in that step. It is the feature-space version of the `task3` preset's per-room
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

## Revisions: two artifact fixes

mohan's review (`docs/fix1_resume_notes.md`) found two pipeline artifacts that could
explain the SIT↔WALK swap. Both fixes were planned before their results were seen. Each
one retrains the six Task 2 encoders and reruns Task 3 unchanged, with the same
settings, draws and seeds.

- **Fix 1, `task2_level_free`.** Room B WALK is about 50 % louder than the other
  Room B activities, so one scalar separates it at AUC 0.898. Each window is divided
  by its own level.
- **Fix 2, `task2_notched`** (fix 1 plus a Nyquist notch). Room A's interleaved frame
  streams put 40 % of the temporal power of its dynamic channel at the Nyquist bin
  (Room B: 1.3 %). Each window has its period-2 component projected out per
  subcarrier, which takes it to 0.0 % in both rooms. Nothing is fitted
  (`normalization.notch_nyquist`).

The earlier version of this section reported one apparent transfer success,
handcrafted + refined on Room B (shared F1 0.236). **That claim is retracted.** After
fix 1 it drops to 0.038, so it came from the amplitude shortcut, not from transfer.

**Effect of each fix** (SSL + prototype, K = 5, oracle, validation macro-F1):

| Preset | A all | A shared | B all | B shared | Room B Hungarian map |
|---|---|---|---|---|---|
| `task2` (original) | 0.215 | 0.011 | 0.277 | 0.046 | SIT ↔ WALK |
| `task2_level_free` (fix 1) | 0.240 | 0.090 | 0.337 | 0.211 | SIT ↔ WALK |
| `task2_notched` (fix 2) | 0.211 | 0.065 | 0.326 | 0.205 | STAND ↔ WALK |

Everything below is from fix 2, which is what `results/` holds.

## Results

**Headline, K = 5, oracle routing** (macro-F1, validation, `task2_notched`):

| Representation | Method | Target A all | Target A excl. | Target A shared | Target B all | Target B excl. | Target B shared |
|---|---|---|---|---|---|---|---|
| SSL | prototype | 0.211 ± 0.031 | 0.358 ± 0.060 | 0.065 ± 0.030 | 0.326 ± 0.044 | 0.509 ± 0.072 | 0.205 ± 0.049 |
| SSL | refined | 0.174 ± 0.043 | 0.242 ± 0.048 | 0.105 ± 0.062 | 0.269 ± 0.076 | 0.423 ± 0.101 | 0.167 ± 0.128 |
| handcrafted | prototype | 0.191 ± 0.019 | 0.292 ± 0.035 | 0.090 ± 0.025 | 0.174 ± 0.043 | 0.419 ± 0.104 | 0.011 ± 0.012 |
| handcrafted | refined | 0.200 ± 0.033 | 0.232 ± 0.026 | 0.169 ± 0.043 | 0.197 ± 0.064 | 0.309 ± 0.069 | 0.123 ± 0.089 |

Uniform random guessing gives about 0.17 macro-F1 for Room A's 6 classes and 0.20 for
Room B's 5.

**Label efficiency** (SSL + prototypes, oracle):

| K | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| Target A exclusive | 0.294 ± 0.069 | 0.358 ± 0.060 | 0.403 ± 0.030 | 0.410 ± 0.040 |
| Target B exclusive | 0.378 ± 0.102 | 0.509 ± 0.072 | 0.554 ± 0.063 | 0.589 ± 0.058 |
| Target A shared | 0.078 ± 0.028 | 0.065 ± 0.030 | 0.062 ± 0.022 | 0.068 ± 0.018 |
| Target B shared | 0.206 ± 0.043 | 0.205 ± 0.049 | 0.220 ± 0.046 | 0.205 ± 0.038 |

Shared F1 does not move with K, as expected, because K adds no shared labels.

**Routing ablation, K = 5** (SSL + prototypes, overall macro-F1):

| Routing | Target A | Target B |
|---|---|---|
| oracle | 0.211 | 0.326 |
| predicted | 0.212 | 0.327 |
| none | 0.194 | 0.231 |

## Discussion

1. **The shared activities do not transfer between these two recordings, and this is
   a verified negative result.** Two artifacts that could explain the failure were
   identified, measured and removed: the Room B WALK level offset and the Room A
   interleaving. Transfer did not recover. Room B shared F1 sits at chance (0.205 vs
   0.20) and Room A's is below chance (0.065). The structure moves as artifacts are
   removed but does not disappear. In Room B the oracle Hungarian map changed from
   SIT ↔ WALK to STAND ↔ WALK, and in the K = 5 confusion matrix (SSL, refined) 47 %
   of SIT and 47 % of STAND windows are predicted WALK, while 41 % of WALK windows
   are predicted STAND. In Room A the confusion is close to uniform (no shared class
   above 30 % on its own label), so there is no consistent mapping left to exploit.
   The remaining cause is the Task 1 §12.1 confound: room, session and pipeline
   cannot be separated in this benchmark, and the same activity name does not
   produce the same signal in the two recordings. mohan's notes also show that WALK
   still differs from the rest of Room B in temporal variance and lag-1
   autocorrelation, which neither fix touches.
2. **The few labels work, but the failed transfer caps them.** Target-exclusive
   macro-F1 rises with K (Room A 0.29 → 0.41 and Room B 0.38 → 0.59 from K = 1 to 20).
   It cannot approach the within-room probe's 0.94, because the target's shared
   windows have no nearby prototype and are absorbed into the exclusive classes,
   halving their precision.
3. **Refinement does not help.** It trades exclusive F1 for shared F1 and lowers
   overall macro-F1 for SSL in both rooms (A 0.211 → 0.174, B 0.326 → 0.269). It was
   chosen as the proposed method before seeing validation results, and we report that
   as it is. A likely reason: SSL embeddings separate the room's recording sessions,
   so unsupervised clusters follow session, not activity.
4. **Neither fix was selected on validation.** Both were planned from development
   measurements before their validation results existed. Fix 2 scores slightly below
   fix 1, but switching back on that basis would be selecting on local_validation, so
   the notched preset stays the default and fix 1 is reported as an ablation.
5. **Routing is free here, and that says nothing about room recognition.** Predicted
   routing matches oracle to within 0.001, because the router is 99.27 % accurate on
   these two sessions (Task 1). Removing routing costs up to 0.095 (Room B), because a
   single bank over 8 activities lets source-exclusive classes absorb target windows.

## Limitations

- Every number above reflects one pair of recordings. The benchmark cannot separate
  room from session and pipeline (Task 1), so "cross-room" here means "cross-session
  and cross-pipeline" as much as cross-room.
- The Nyquist notch removes the period-2 signature of Room A's interleaving, not any
  other effect the interleaving might have (for example on lag-1 autocorrelation).
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
