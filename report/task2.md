# Task 2 — Self-supervised representation learning (draft report section)

*Draft for the shared technical report. Numbers come from `results/task2_probe.csv`;
figures are `figures/task2_probe.png` and `figures/task2_umap.png`. All scores are on
the fixed local validation set and are validation results, not unseen-test accuracy.*

## Goal

Learn a CSI representation from development windows of both rooms **without any
activity or room labels**, freeze it, and measure how much activity information a
linear classifier can read from it, against two required baselines.

## Input

Windows of 150 observations × 256 subcarriers, loaded with the `task2` normalization
preset: `log1p` amplitude, robust per-subcarrier scaling fitted on development windows
only, and two channels. Channel 0 is the scaled window. Channel 1 is the same window
with each subcarrier's time-mean removed, so it keeps only the temporal dynamics.

## Method

**Encoder.** A 1-D temporal CNN treats the 256 subcarriers × 2 channels as 512 input
features per time step. A 1×1 convolution reduces these to 64 channels. Three residual
dilated convolution blocks follow (kernel 5, dilations 1, 2 and 4, 128 channels), then
mean and max pooling over time and a linear layer to a 128-d embedding. The encoder has
about 0.27 M parameters, deliberately small for 2,010 training windows.

**Objective.** SimCLR: two augmented views of each window are drawn, passed through the
encoder and a 2-layer projection head (discarded after training), and trained with the
NT-Xent loss (temperature 0.2, batch 128). Optimiser: AdamW (lr 1e-3, weight decay
1e-4) with cosine decay, for a fixed 200 epochs. There is no early stopping and no
checkpoint selection, so validation data plays no part in pretraining. The training
function takes the window array as its only data argument, and a test asserts that.

**Augmentations.** All are label-free. They are applied to channel 0, and channel 1 is
then rebuilt from it, so the two views stay consistent:

- random temporal crop (60–100 %) resized back to 150 steps;
- stride-2 subsampling with a random phase;
- local shuffling within spans of 4 observations;
- masking of one contiguous run of up to 32 subcarriers;
- a per-subcarrier offset (a gain change, since the data is log-scaled);
- additive Gaussian noise;
- time reversal.

**Why the subsampling and shuffling.** Room A's captures interleave two Nexmon frame
streams whose statistics differ strongly. Lag-1 autocorrelation in Room A is −0.33 as
ingested, against +0.04 in Room B. A contrastive objective can lower its loss by keying
on this alternation instead of on activity. Stride-2 subsampling keeps a single stream:
interpolation only ever blends samples of the same parity, which is tested. Local
shuffling destroys strict adjacency. The variant `no_adjacency_break` disables both
augmentations, and comparing the two variants is the artifact ablation.

**Baselines.**

1. *Handcrafted features* (416-d), computed from the window alone. The subcarriers are
   averaged into 16 groups. For each group: mean, standard deviation, range, skewness,
   kurtosis, mean absolute first difference, and log energy in four temporal-frequency
   bands. On top of that, the temporal standard deviation of every subcarrier.
2. *Random-weight encoder*: the same architecture, untrained. Its BatchNorm running
   statistics are re-estimated from development windows, without labels and without
   changing any weight, so the comparison isolates the learned weights.

**Probe.** Standardisation followed by multinomial logistic regression with balanced
class weights. The regularisation `C` is chosen from {1e-3, …, 100} by 5-fold CV,
stratified by class and grouped by capture, on development windows only. The grid's
upper end was extended from 10 to 100 after development CV put the handcrafted
baseline's optimum at the edge. Encoders are evaluated for 3 pretraining seeds
(0, 1, 2) and reported as mean ± standard deviation.

## Results

**Activity: one 8-class probe over both rooms** (validation, n = 275):

| Representation | macro-F1 | balanced acc. | Room A macro-F1 | Room B macro-F1 |
|---|---|---|---|---|
| handcrafted features | 0.898 | 0.897 | 0.906 | 0.887 |
| random-weight encoder | 0.637 ± 0.026 | 0.664 ± 0.019 | 0.519 ± 0.013 | 0.742 ± 0.045 |
| SSL, all augmentations | **0.898 ± 0.008** | **0.906 ± 0.008** | 0.876 ± 0.020 | **0.933 ± 0.013** |
| SSL, no adjacency-breaking aug. | 0.850 ± 0.003 | 0.856 ± 0.004 | 0.811 ± 0.020 | 0.900 ± 0.036 |

**Diagnostics.** A probe fitted on SIT/STAND/WALK of one room and scored on the same
activities in the other room reaches macro-F1 between 0.13 and 0.24 in each direction.
This is at the level of predicting one class for every window (0.167), for every
representation. A room probe scores 1.000 for all four representations.

## Discussion

1. **Self-supervision recovers what the random architecture lacks, but no more than
   handcrafted features.** Pretraining raises macro-F1 from 0.637 to 0.898. The
   pretrained encoder matches the handcrafted baseline overall, beats it in Room B
   (+0.046) and trails it in Room A (−0.030). With 2,010 windows, a well-chosen set of
   signal statistics is a strong competitor.
2. **The interleaving artifact is real and the augmentations counter it.** Without the
   adjacency-breaking augmentations, macro-F1 drops by 0.048, more than five times the
   seed spread. Most of the loss is in Room A, the only room with interleaved streams
   (0.876 → 0.811). The same variant reaches a *lower* contrastive loss (1.63 vs 2.08):
   its pretext task is easier because the alternation gives views away, and the encoder
   learns less about activity as a result.
3. **The representations encode the room before the activity.** Room is linearly
   perfect in every representation, the random-weight one included, and the UMAPs split
   cleanly by room. That separation comes from the input: the `task2` preset keeps the
   absolute amplitude level, which differs about 490× between our Room A decode and Room
   B's supplied values (see Task 1). It is not something pretraining discovered.
   Activity probes do not transfer across rooms at all. Per-room normalization does not
   rescue them either: in a development-only check, a probe fitted on one room's
   SIT/STAND/WALK and scored on the other's stays at 0.08–0.16 macro-F1 with per-room
   standardised SSL embeddings, or with handcrafted features on the `task3` preset.
   The same probe scores 0.94 within either room. The shared labels therefore do not
   produce shared signals across these two recordings, which bounds what Task 3 can
   transfer without target labels.
4. **Possible session confound in Room B (qualitative).** In every representation,
   Room B's SIT and STAND windows form a cluster apart from its EMPTY, LYING and WALK
   windows. That pattern is consistent with those activities having been recorded in a
   separate session. If so, part of Room B's high scores reflects session rather than
   activity. We cannot verify this from the data supplied.

## Limitations

- One window per capture gives only 2,010 pretraining samples. Strided windowing could
  multiply that, but it is not implemented in the shared pipeline.
- Hyperparameters (architecture, augmentation strengths, epochs) were fixed a priori
  and not searched. Only the probe's `C` was selected, on development data.
- With 25 validation captures per class, one capture is 4 % of a class's recall.
  Differences under about 0.02 macro-F1 should not be over-read.
- The UMAP is qualitative and fitted on development windows only.

## Reproduce

```bash
python -m src.ssl_pretrain      # ~45 min on a GTX 1650; weights -> models/ (gitignored)
python -m src.task2_probe       # ~4 min
python -m src.figures --task2
python -m pytest tests/ -q
```
