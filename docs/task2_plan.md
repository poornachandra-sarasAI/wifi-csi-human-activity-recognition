# Task 2 plan — self-supervised CSI representation

Owner: Hemant. Repo: `wifi-csi-human-activity-recognition/`, branch `task2/ssl-encoder`.
Files owned: `src/encoder.py`, `src/ssl_pretrain.py`, `src/baselines.py`, `src/task2_probe.py`
(plus `encoder_umap()` in shared `src/figures.py`, a `task2:` block in shared `configs/experiment.yaml`).

Legend: `[x]` done · `[ ]` to do · `[~]` partly done / blocked (reason given)

Ground rules carried from HANDOFF §4, checked at every phase:
- R1 nothing is fitted or selected on `local_validation` (pretraining, probe, hyperparameters)
- R2 pretraining sees `x` only — no activity, no room
- R3 no metadata as features (capture_id, length, filenames)
- R4 validation numbers are called validation numbers
- R5 `pytest tests/ -q` green before every commit

---

## Phase 0 — Prerequisites (before any Task 2 code can produce numbers)

- [x] 0.1 Create branch `task2/ssl-encoder` off `main`
- [x] 0.2 Project virtualenv **outside OneDrive** → `C:\Users\heman\.venvs\wifi-csi` (Python 3.12)
- [x] 0.3 Install pinned `requirements.txt` into it; confirm imports (numpy 2.5.4, pandas 3.0.6, sklearn 1.9.1)
- [x] 0.4 Install PyTorch with CUDA for the GTX 1650 (4 GB) + `umap-learn`; `torch 2.6.0+cu124`, `cuda.is_available() == True`, umap 0.5.12
- [x] 0.5 Add `torch` and `umap-learn` to `requirements.txt` (shared file, small commit)
- [x] 0.6 **Dataset**: `hackathon_2_data-…zip` extracted to `C:\Users\heman\csi_data\` (outside OneDrive);
      `raw/experiment2`, `raw/experiment3` are junctions into it; `intermediate_csv/room_*` and
      `windows/{development,local_validation}` are junctions to `csi_data\derived\` so ~2 GB of derived data
      does not sync. `git status` stays clean.
  - [x] found `stand/stand65(1).pcap`, a byte-identical duplicate of `stand65.pcap`, absent from the committed
        ingest list → moved to `csi_data\_set_aside_duplicates\` (kept, not deleted). **Flag to teammate 1**: the
        pipeline globs `*.pcap` and would ingest it, shifting every later `window_id`.
  - [x] nested `Experiment-2.zip`, `Experiment-3.zip`, `Experiment-3/Data/model_plot_1.png` left unopened
        (possible source-author material, HANDOFF §1)
- [x] 0.7 `python run_episode.py` → `windows/*.npy` rebuilt; config hash still `2a323eeb9765`
- [x] 0.8 `python -m pytest tests/ -q` → 8 passed
- [x] 0.9 Sanity: `dataset.load("task2", ...)` gives dev (2010, 150, 256, 2), val (275, 150, 256, 2), finite

## Phase 1 — Interface contract first (unblocks teammate 3)

- [x] 1.1 `task2:` config block — encoder width, embedding dim, augmentation params, epochs, batch, lr, temperature, seeds. (Not in the provenance hash → no `config_version` bump needed)
- [x] 1.2 `src/encoder.py`
  - [x] `build_encoder(cfg) -> nn.Module`: small 1-D temporal CNN over 150 steps; 256 subcarriers × 2 channels = 512 input features; 1×1 reduction → dilated conv blocks → global average pool → d = 128
  - [x] `embed(model, x) -> (n, d)` float32 numpy; batched, eval mode, no grad, device-agnostic, deterministic
  - [x] `load_pretrained(cfg)` — loads gitignored weights, clear error telling you to run `ssl_pretrain` if missing
  - [x] parameter count kept small (target < 0.5 M) because there are only 2,010 dev windows
- [x] 1.3 Unit tests (no data needed): output shape, determinism, batch-size invariance, accepts 2-channel input
- [x] 1.4 Commit "Task 2 encoder interface" — tell teammate 3 the signature is settled

## Phase 2 — Self-supervised pretraining (`src/ssl_pretrain.py`)

- [x] 2.1 Augmentations (numpy/torch, per sample, seeded)
  - [x] random temporal crop + resize back to 150
  - [x] random temporal subsampling (stride 2, random phase) — **breaks Room A's A/B frame interleave**
  - [x] local shuffle within short spans — also breaks strict adjacency
  - [x] subcarrier band masking
  - [x] additive Gaussian noise, per-subcarrier gain jitter
  - [x] time reversal
- [x] 2.2 SimCLR NT-Xent loss with projection head (discarded after pretraining)
- [x] 2.3 Training loop: dev windows of **both rooms**, `x` only (R2 enforced by the function signature + a test); fixed epoch budget from config (no early stopping on validation — R1); AdamW + cosine schedule; seeded
- [x] 2.4 Save weights to `models/task2_encoder_<variant>_seed<k>.pt` (gitignored) + loss curve to `results/task2_pretrain_loss.csv`
- [x] 2.5 Smoke test on synthetic arrays (runs without the dataset)
- [x] 2.6 Real pretraining run — 6 runs, ~7.5 min each; final loss full 2.08, no_adjacency_break 1.63 (3-seed means)

## Phase 3 — Required baselines (`src/baselines.py`)

- [x] 3.1 Handcrafted features, computed from the window only (R3): per-subcarrier std, mean |first difference|, temporal-spectrum band energies, range; pooled per sub-band to keep dimension modest
- [x] 3.2 Random-weight encoder: same `build_encoder`, same seeds, untrained → `embed`
- [x] 3.3 Unit tests: feature shape, no NaNs, no dependence on anything but `x`

## Phase 4 — Linear probe and metrics (`src/task2_probe.py`)

- [x] 4.1 Freeze representation → `StandardScaler` + multinomial `LogisticRegression(class_weight="balanced")`
- [x] 4.2 Regularisation `C` chosen by 5-fold CV **on development only** (R1)
- [x] 4.3 Score on fixed `local_validation`: **macro-F1** (headline) + balanced accuracy — joint 8-class, plus per-room breakdown
- [x] 4.4 3 seeds for SSL and random encoders → mean ± std
- [x] 4.5 Diagnostic: linear room-probe accuracy on each representation (how much room/session the embedding carries)
- [x] 4.6 Write `results/task2_probe.csv` in §10 schema `experiment, target_room, K, metric, value` (+ per-seed detail file)
- [x] 4.7 Real numbers — needs Phase 0.6

## Phase 5 — Interleaving-artifact ablation (HANDOFF §5)

- [x] 5.1 Variant `full`: all augmentations
- [x] 5.2 Variant `no_adjacency_break`: without subsampling / local shuffle
- [x] 5.3 Compare the two variants — kept inside `results/task2_probe.csv` (rows `ssl_no_adjacency_break`) rather than a separate file; room separability is 1.000 for every representation, so the effect shows in activity F1
- [x] 5.4 Real numbers — needs Phase 0.6

## Phase 6 — Figures (`src/figures.py`, owner tag "Task 2")

- [x] 6.1 `encoder_umap()`: 2-D UMAP of embeddings, coloured by room and by activity (qualitative), SSL vs random vs handcrafted
- [x] 6.2 `task2_probe_bars()`: macro-F1 / balanced accuracy per representation with seed spread
- [x] 6.3 Render PNGs into `figures/` — needs Phase 0.6

## Phase 7 — Integration and hand-back

- [x] 7.1 `python -m src.task2_probe` runs pretrain → baselines → probe → CSVs → figures end to end
- [x] 7.2 `tests/test_task2.py` green alongside the existing 8 leakage tests
- [x] 7.3 README: Task 2 section (how to run, results table, caveats) incl. the note for teammate 3 on `embed` / `load_pretrained` (HANDOFF.md left to its owner)
- [x] 7.4 Draft report section `report/task2.md` (method, baselines, results, artifact ablation, limitations)
- [x] 7.5 Small, separate commits on `task2/ssl-encoder`
- [x] 7.6 Push + PR — PR #4 (Task 2) and PR #5 (corrected cross-room advice) merged into `main`

---

## Status log

- **0.7–0.9** `run_episode.py` reproduces teammate 1's results: manifest byte-identical, Task 1 ladder identical
  (1.0000 / 1.0000 / 0.9927). Only differences: Windows path separators in `source_file` (`fall\fall10.pcap` instead of `fall/fall10.pcap`;
  `pcap_reader` should use `.as_posix()`) and float32 rounding (≤ 2.4e-7) in `norm_*.npz` → committed versions kept.
  Tests: 8/8 leakage + 15/15 Task 2.
- **Compute**: lab server `gpu77` checked — Tesla K40m (Kepler), unsupported by current PyTorch (`cuda False`);
  not useful. Kaggle not needed: after vectorising augmentations, all 6 pretraining runs take ~45 min locally.
- **Probe**: dev-only CV for handcrafted features chose C = 10 at the grid edge → grid extended to 100
  (development evidence only). Parallel CV disabled (Windows page-file crash from torch DLLs in workers).
- Commits so far on `task2/ssl-encoder`: config/requirements, encoder, pretraining, baselines, probe, figures, tests, probe fix.
- **Results (validation, 8-class probe, macro-F1)**: handcrafted 0.898 · random 0.637 ± 0.026 ·
  **SSL full 0.898 ± 0.008** · SSL no-adjacency-break 0.850 ± 0.003. Ablation gap 0.048 (mostly Room A).
  Cross-room SIT/STAND/WALK at single-class level for all; room probe 1.000 for all.
- Figures: `figures/task2_umap.png`, `figures/task2_probe.png`. Draft write-up: `report/task2.md`.
- PR #4 merged: all Task 2 code, results, figures, README section, report draft on `main`.
- **Task 3 readiness check (development data only)**: SIT/STAND/WALK do not transfer across rooms —
  0.08–0.19 macro-F1 across rooms vs 0.94 within, with or without per-room normalization. README advice
  corrected in PR #5 (merged). Reproduce with `python -m src.task2_transfer_check`.
- Shared for teammates (branch `task2/share-artifacts`): trained encoder weights in `models/`, run logs,
  this plan, Windows setup notes, the transfer-check script and CSV. Not shared: raw data, `intermediate_csv/`,
  `windows/*.npy` (dataset-derived; challenge §4 / HANDOFF).
