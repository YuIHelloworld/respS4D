# RespS4D

Official PyTorch implementation of the paper:

**RespS4D: Respiratory-Frequency State Space Modeling for Facial Video Respiratory Waveform Estimation**

*Zehan Zhang, Tao Wang, Chenhao Wu, Yonggang Tong, Zhipei Huang*, Feng Qiu, Wei Huang, Xiaoyong Tao, Ye Fan, Fei Qin*

School of Electronic, Electrical and Communication Engineering, University of Chinese Academy of Sciences

## Overview

**RespS4D** formulates contactless respiratory waveform measurement from facial RGB video as a narrowband, quasi-periodic weak-signal recovery problem:

- **Motion Frontend**: Combines multi-tap frame differences, vertical Farneback optical flow (νy​), and appearance-guided spatial gating to capture subtle breathing displacement.
- **Respiration-Adapted S4D Operator**: Parameterizes diagonal state-space modes initialized log-spaced inside the breathing band `[0.10, 0.50] Hz`, forming a learnable resonator bank with a bidirectional kernel for zero-phase-lag continuous waveform recovery.

## 🛠️ Environment Setup

```
git clone https://github.com/YulHelloworld/respS4D.git
cd respS4D
pip install -r requirements.txt
```

> **Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, `numpy`, `scipy`, `h5py`, `opencv-python`, `tqdm`, `pyyaml`.

## 📦 Data Preparation

Download the [COHFACE](https://www.idiap.ch/en/dataset/cohface) dataset (40 subjects, 4 sessions each, RGB video + chest-belt respiration), then build the video and flow caches:

```
# 1. Video frame cache (128x128)
python preprocess_cohface.py --data_root /path/to/COHFACE --out caches/cache_cohface_128.h5 --workers 6

# 2. Vertical Farneback optical flow cache
python preprocess_flow.py --rgb_cache caches/cache_cohface_128.h5 --out caches/cache_cohface_128_farneback_flow5.h5 --flow_scale 5.0 --workers 6
```

## 🚀 Training & Testing

The evaluation follows a subject-disjoint 5-fold cross-validation protocol (28 train / 4 val / 8 test subjects per fold):

```
# Train headline model (Seed 42)
python train.py --config configs/seed42.yaml

# Train a single fold (e.g., Fold 0)
python train.py --config configs/seed42.yaml --fold_id 0

# Multi-seed reproduction
python train.py --config configs/seed1337.yaml
python train.py --config configs/seed2025.yaml
```

Trained weights will be saved to `checkpoints/`. Predicted waveforms and window evaluation metrics are exported to `predictions/`.

## 💡 Quick Inference

```
import yaml
import torch
from model import RespS4D

# Load model architecture & weights
cfg = yaml.safe_load(open("configs/seed42.yaml"))
model = RespS4D(**cfg["model"])
model.load_state_dict(torch.load("checkpoints/abl_s4d_clean_nospatial_appear_gate_fold0_best.pt", map_location="cpu"))
model.eval()

# input:  [B, T, 4, H, W] -> RGB normalized to [0, 1] concatenated with vertical flow v_y
# output: [B, T] -> continuous respiratory waveform
dummy_input = torch.randn(1, 160, 4, 128, 128)
with torch.no_grad():
    waveform_pred = model(dummy_input)
```

## 📑 Citation

If you find this code or paper useful in your research, please cite:

```
@article{zhang2026resps4d,
  title={RespS4D: Respiratory-Frequency State Space Modeling for Facial Video Respiratory Waveform Estimation},
  author={Zhang, Zehan and Wang, Tao and Wu, Chenhao and Tong, Yonggang and Huang, Zhipei and Qiu, Feng and Huang, Wei and Tao, Xiaoyong and Fan, Ye and Qin, Fei},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
