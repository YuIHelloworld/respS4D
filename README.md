# RespS4D

Minimal training code for **RespS4D**, a pure-motion model for facial-video
respiratory waveform estimation. The model combines an appearance-weighted
multi-tap frame-difference stem, a vertical optical-flow cue, and a
respiration-adapted bidirectional diagonal **S4D** temporal operator whose modes
are initialized inside the breathing band `[0.10, 0.50] Hz`.

This repository contains everything needed to reproduce the main model's
training under three random seeds on COHFACE, plus the resulting weights.

## Layout

```
model.py                RespS4D model (encoder + respiratory branch + quality head)
encoder.py              Motion stem, temporal pyramid, spatial-attn / S4D blocks
s4d_layer.py            Diagonal S4D kernel (respiration-band / resp_log init, bidirectional)
losses.py               NegPearson + spectral + amplitude loss (Eq. 13)
dataset.py              COHFACE farneback_y cache loader + 5-fold subject split
train.py                5-fold cross-validation trainer
preprocess_cohface.py   COHFACE AVI -> 128x128 RGB HDF5 cache
preprocess_flow.py      Vertical Farneback optical-flow cache
configs/                seed42 / seed1337 / seed2025 training configs
checkpoints/            trained weights, 3 seeds x 5 folds (state_dict)
predictions/seed42/     published 30 s evaluation outputs for the seed-42 model
```

## Setup

```bash
pip install -r requirements.txt   # torch>=2.0, numpy, scipy, h5py, opencv, tqdm, pyyaml
```

## Data

Download COHFACE (40 subjects x 4 sessions, 20 fps RGB + synchronized chest-belt
respiration) and build the two HDF5 caches once:

```bash
python preprocess_cohface.py --data_root /path/to/COHFACE \
    --out caches/cache_cohface_128.h5 --workers 6

python preprocess_flow.py --rgb_cache caches/cache_cohface_128.h5 \
    --out caches/cache_cohface_128_farneback_flow5.h5 --flow_scale 5.0 --workers 6
```

## Train (3-seed reproduction of the main model)

```bash
python train.py --config configs/seed42.yaml     # published headline
python train.py --config configs/seed1337.yaml
python train.py --config configs/seed2025.yaml
```

Each run performs subject-disjoint 5-fold cross-validation (8 test subjects per
fold), training on 8 s windows with AdamW (lr 1e-4, weight decay 3e-4), linear
warmup + cosine decay, batch size 10, up to 50 epochs, early stopping on
validation Pearson (patience 10). The fold-to-subject split is pinned by
`split_seed = 42` across all three seeds, so only the training/init seed varies.
Best weights per fold are written to
`checkpoints/<run_name>_fold{k}_best.pt` as a plain `state_dict`.

Add `--fold_id k` to run a single fold.

## Loading a released checkpoint

```python
import yaml, torch
from model import RespS4D

cfg = yaml.safe_load(open("configs/seed42.yaml"))
model = RespS4D(**cfg["model"])
model.load_state_dict(torch.load(
    "checkpoints/abl_s4d_clean_nospatial_appear_gate_fold0_best.pt",
    map_location="cpu"))
model.eval()
# input: video [B, T, 4, H, W] = RGB/255 concatenated with vertical flow v_y
# output: respiratory waveform [B, T]
```

The three seeds map to the checkpoint prefixes:

| config | run_name (checkpoint prefix) |
|---|---|
| `configs/seed42.yaml` | `abl_s4d_clean_nospatial_appear_gate` |
| `configs/seed1337.yaml` | `abl_s4d_clean_nospatial_appear_gate_seed1337` |
| `configs/seed2025.yaml` | `abl_s4d_clean_nospatial_appear_gate_seed2025` |

## Prediction outputs (seed 42)

`predictions/seed42/fold{0-4}/` contains the published 30 s Shao-protocol
evaluation outputs of the seed-42 model on its held-out test subjects:

- `windows_subj*_sess*.csv` — per 30 s window metrics (`resp_pearson`, `rr_*`, ...)
- `waveforms_subj*_sess*.csv` — reconstructed predicted vs. reference waveform
- `summary.csv` — per-session summary

`predictions/seed42/pooled_metrics.csv` pools all folds:

| scope | n_windows | wave PCC (mean / median) |
|---|---|---|
| pooled 5-fold | 190 | **0.8589 / 0.9136** |

The pooled waveform PCC is the paper's headline number. The RR-MAE column in
these CSVs is from the pre-fix rate tracker; the paper reports the corrected
RR-MAE (0.358 bpm).

## License

MIT (see `LICENSE`).
