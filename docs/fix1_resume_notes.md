# Fix 1 — in progress, paused 2026-10-26

Removing the Room B WALK amplitude shortcut from the Task 2/3 representation.
Code changes are done and verified; the re-training they require is **not**.
Nothing is committed.

## Why

Task 3 loaded the `task2` preset, whose scope is `global` — which by design
preserves absolute amplitude. Room B's WALK captures are the only Room B
activity whose files lack the `_Amp_` infix (209/209) and they sit ~50% above
every other Room B activity in mean amplitude (473 vs 304–358). A single
scalar, the capture's mean amplitude, separates WALK from the rest of Room B at
**ROC-AUC 0.898**.

`fewshot.RoomScaler` cannot remove this: it standardises per *room*, and this is
a per-*class* difference inside one room. So the offset reached the prototypes
intact, which is consistent with the observed SIT↔WALK swap.

## What changed (uncommitted, in the working tree)

| File | Change |
|---|---|
| `src/normalization.py` | new `per_window_scalar` scope + `_scalar_center_scale()` |
| `configs/experiment.yaml` | new `task2_level_free` preset; `task2.normalize_preset` points at it |
| `src/task3_diagnostics.py` | new — Hungarian upper bound, every row `oracle=True` |
| `results/task3_hungarian_bound.csv` | Hungarian bound on the **pre-fix** results |
| `results/norm_task2_level_free.npz` | fitted stats (empty: scalar scope fits nothing) |

`per_window_scalar` divides each window by one scalar location and spread, so
absolute level goes while the per-subcarrier spectral shape stays. Plain
`per_window` standardises each subcarrier separately, which destroys that shape
and collapses the `two_channel` split into two identical channels — that is why
a new scope was needed rather than reusing the existing one.

The old `task2` preset is untouched, so the committed results remain
reproducible as an ablation.

## Verified so far

- Room B static-channel level by activity, old preset: WALK **0.539** against
  SIT 0.415 / STAND 0.409 — a clear outlier. New preset: WALK **−0.261**,
  in line with EMPTY −0.260, SIT −0.201, STAND −0.209.
- Level-only shortcut for WALK: ROC-AUC **0.894 → 0.693**.
- **But** with the full handcrafted feature set WALK is still separable at
  **0.986** (was 0.977). The batch difference also lives in temporal variance
  (WALK 1.014 vs SIT 1.517) and lag-1 autocorrelation (+0.092 vs ≈0), which
  fix 1 does not touch. Temper expectations accordingly.
- 28 tests pass. The window provenance hash is unchanged (`2a323eeb9765`) —
  `normalize` is not a hashed section, so no re-ingest is needed.
- Hungarian bound on the pre-fix results (ORACLE, not a result):

  | target | metric | reported | bound |
  |---|---|---|---|
  | A | macro-F1 | 0.166 | 0.317 |
  | A | shared macro-F1 | 0.038 | 0.331 |
  | A | shared excl. WALK | 0.055 | 0.290 |
  | B | macro-F1 | 0.219 | 0.450 |
  | B | shared macro-F1 | 0.042 | 0.426 |
  | B | shared excl. WALK | 0.062 | 0.414 |

  Optimal map is SIT↔WALK in **both** directions (Room A also STAND↔STANDUP).
  Note shared-excluding-WALK is still ~0.06 reported, so SIT and STAND fail to
  transfer on their own too — WALK is not the whole story.

## What is NOT done

**The six encoders have not been re-trained on the new preset.** They are the
committed checkpoints, trained on the old `task2` preset.

Running Task 2 or Task 3 right now feeds `task2_level_free` windows into
encoders trained on `task2` windows — an input-distribution mismatch whose
results would be meaningless. Either re-train first, or temporarily set
`task2.normalize_preset: task2` back.

One encoder (`full`, seed 0) *was* re-trained on the new preset before the run
was stopped. It has been moved out of `models/` so the directory is internally
consistent, and kept at:

```
/private/tmp/claude-501/-Users-mohan-AI-Ubiquitous-Computing-Hackathon-2/7f31d82a-a32b-400e-85a0-3ea86cc5159a/scratchpad/task2_encoder_full_seed0.LEVEL_FREE_PRESET.pt
```

That path is a session scratchpad and will not survive indefinitely. It is only
1 of 6 and therefore unusable on its own; treat it as disposable.

## The compute problem to solve before resuming

Pre-training is **~60 min per encoder on this Mac, ~6 hours for all six**. The
teammate's log shows 434 s per encoder, roughly 8× faster.

Measured here (batch 128, input `(128, 150, 256, 2)`, 280k params):

```
cpu threads=4    1163 ms/step  ->  62.0 min/encoder
cpu threads=8    1179 ms/step  ->  62.9 min/encoder
mps              1129 ms/step  ->  60.2 min/encoder
```

Profile of one step: augmentation 2 × 70.8 ms = 142 ms (13%), forward +
backward + optimiser step **942 ms (87%)**. So the bottleneck is the model, not
the augmentation, and MPS gives essentially nothing.

`encoder.default_device()` only checks `torch.cuda.is_available()`, so it
returns `cpu` on Apple silicon. Adding MPS would be a correct portability fix
but, per the benchmark above, would not help.

Options when resuming, in order of preference:

1. **Run it on the machine that does 434 s/encoder** (teammate 2's, or any CUDA
   box) — `python -m src.ssl_pretrain`, ~45 min total, then re-run the
   downstream steps below.
2. Run the ~6 h job here in the background and wait.
3. Re-train only the `full` variant (3 encoders, ~3 h). Task 3 uses
   `encoder_variant: full`; the `no_adjacency_break` variant is only needed for
   Task 2's interleaving ablation, which would then be stale and must be
   re-run before the report.
4. Find out why this machine is 8× slower per step. torch here is **2.9.1 on
   Python 3.14** because the pinned `torch==2.6.0` has no wheel for this
   platform — `requirements.txt` was deliberately left alone, since the pin
   works for the rest of the team. A different build may close the gap.

## Resume checklist

```bash
python -m src.ssl_pretrain                 # all 6 encoders, on a fast machine
python -m src.task2_probe
python -m src.task2_transfer_check
python -m src.task3_adapt
python -m src.task3_diagnostics            # Hungarian bound on the NEW results
python -m src.figures
python -m pytest tests/ -q                 # expect 28 passed
```

Then compare against the pre-fix copies kept for exactly this purpose:
`/tmp/task3_prefix.csv`, `/tmp/hungarian_prefix.csv`,
`/tmp/task2_probe_prefix.csv` (also session-temporary — the committed versions
in `results/` on `main` are the durable reference).

**Expect Task 2's within-room probe to fall from ~0.90.** Part of that number
was the amplitude shortcut. Report both and explain why the drop is the correct
direction.

The question fix 1 is meant to answer: **does the SIT↔WALK swap survive?** If it
does, fix 2 (notch the Nyquist bin — 59.4% of Room A's temporal power is the
period-2 interleaving artifact) is next, and if the swap survives that too, the
pessimistic reading is real and gets written up as a negative result with a
verified cause.
