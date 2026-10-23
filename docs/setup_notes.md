# Setup notes — from raw archive to Task 2 results

How the Task 2 numbers on `main` were produced, including two traps in the course
archive that are not obvious from the README.

## 1. Environment

```bash
python -m venv <somewhere outside OneDrive/Dropbox>   # CUDA torch is ~2.5 GB
<venv>/Scripts/pip install -r requirements.txt        # Linux/macOS: <venv>/bin/pip
```

`requirements.txt` pins `torch==2.6.0`, which from PyPI is the **CPU** build on
Windows. For an NVIDIA GPU, install the CUDA build first:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Used here: Python 3.12, torch 2.6.0+cu124, GTX 1650 (4 GB), driver 555.99.
Pretraining peaks below 1 GB of GPU memory. Old Kepler GPUs (e.g. Tesla K40m)
are not supported by current PyTorch.

## 2. Data — two traps in `hackathon_2_data-*.zip`

The archive unpacks to `hackathon_2_data/Experiment-2/...` and
`hackathon_2_data/Experiment-3/Data/`. The config expects lowercase names
without the hyphen:

```
raw/experiment2/realdata/input_data/30ms/csidata/   <- hackathon_2_data/Experiment-2/realdata/input_data/30ms/csidata/
raw/experiment3/Data/                               <- hackathon_2_data/Experiment-3/Data/
```

Copy or rename into place, or link. On Windows, a directory junction keeps
the data out of the repo and out of cloud sync:

```powershell
New-Item -ItemType Junction -Path raw\experiment2 -Target <unzipped>\hackathon_2_data\Experiment-2
New-Item -ItemType Junction -Path raw\experiment3 -Target <unzipped>\hackathon_2_data\Experiment-3
```

**Trap 1: a duplicate capture.** `Experiment-2/.../csidata/stand/stand65(1).pcap`
is a byte-identical copy of `stand65.pcap`. The committed manifest does not
include it, but `pcap_reader` globs `*.pcap` and would ingest it. That adds a
1,085th Room A capture and shifts every later `window_id`, so the manifest
no longer matches. **Move it out of the folder before running the pipeline.**

**Trap 2: material we must not read.** The archive also holds
`Experiment-2.zip`, `Experiment-3.zip` and `Experiment-3/Data/model_plot_1.png`.
These were left unopened because they may contain source-author material,
which HANDOFF §1 quarantines. Nothing in the pipeline needs them; the CSV reader
only globs `*.csv`.

## 3. Build and verify

```bash
python run_episode.py        # ~14 min on a laptop; intermediate_csv/, windows/, Task 1
python -m pytest tests/ -q   # 23 passed (8 leakage + 15 Task 2)
```

Expected: `config_hash 2a323eeb9765`, an unchanged `windows/manifest.csv`, and
the Task 1 ladder 1.0000 / 1.0000 / 0.9927. On Windows two committed files show
spurious diffs afterwards. Restore them with `git checkout -- results/`:

- `results/ingest_room_A.csv` and `results/captures.csv`: backslashes in
  `source_file` (`pcap_reader` uses `str(path)`, not `.as_posix()`);
- `results/norm_*.npz`: float32 rounding differences up to 2.4e-7.

## 4. Task 2

The trained encoders are committed in `models/` (6 files, 1.1 MB each), so
`load_pretrained()` works straight after cloning. Retraining is optional:

```bash
python -m src.ssl_pretrain           # ~45 min on a GTX 1650; overwrites models/*.pt
python -m src.task2_probe            # ~4 min; results/task2_probe*.csv
python -m src.figures --task2        # figures/task2_umap.png, figures/task2_probe.png
python -m src.task2_transfer_check   # development-only cross-room check
```

Logs of the runs behind the committed numbers are in `docs/run_logs/`.

On Windows the probe's cross-validation runs serially on purpose: parallel
joblib workers each re-import torch's CUDA DLLs and exhaust the page file
(`WinError 1455`).
