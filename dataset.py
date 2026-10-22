"""COHFACE data loading for RespS4D (pure-motion, farneback_y).

Reads the two HDF5 caches produced by preprocess_cohface.py and
precompute_flow.py:

    RGB  cache : /cohface/{subj}/{sess}/{frames[N,H,W,3] uint8, pulse[N], resp[N]}
    flow cache : /cohface/{subj}/{sess}/flow[N,H,W,1|2] float16  (vertical v_y)

Each item is a video tensor [T, 4, H, W] = (RGB / 255) concatenated with the
vertical optical-flow channel v_y, plus the windowed reference respiration.
Signals in the cache are already resampled to 20 fps, trimmed, and robustly
z-scored, so no further signal processing is applied here.
"""

import os
import random
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_FPS = 20.0


def _flow_to_vy(flow: np.ndarray) -> np.ndarray:
    """Cached flow [T,H,W,1|2] -> vertical channel v_y [T,H,W,1] float32."""
    flow = flow.astype(np.float32, copy=False)
    if flow.shape[-1] == 1:
        return flow
    return flow[..., 1:2]


class COHFACEMotionDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        flow_cache_path: str,
        clips: List[Tuple[int, int]],
        frame_len: int = 160,
        stride: int = 80,
        augment: bool = False,
    ):
        self.cache_path = cache_path
        self.flow_cache_path = flow_cache_path
        self.frame_len = frame_len
        self.augment = augment

        self._frames: dict = {}
        self._flow: dict = {}
        self._resp: dict = {}
        self.windows: List[Tuple[int, int, int]] = []

        with h5py.File(cache_path, "r") as h5, h5py.File(flow_cache_path, "r") as fh5:
            for subj, sess in clips:
                key = f"cohface/{subj}/{sess}"
                if key not in h5 or key not in fh5:
                    continue
                grp = h5[key]
                n = int(grp.attrs.get("n_frames", grp["frames"].shape[0]))
                self._frames[(subj, sess)] = grp["frames"][()]        # [N,H,W,3] uint8
                self._resp[(subj, sess)] = grp["resp"][()].astype(np.float32)
                self._flow[(subj, sess)] = _flow_to_vy(fh5[key]["flow"][()])
                for start in range(0, n - frame_len + 1, stride):
                    self.windows.append((subj, sess, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        subj, sess, start = self.windows[idx]
        T = self.frame_len
        sl = slice(start, start + T)

        rgb = self._frames[(subj, sess)][sl].astype(np.float32) / 255.0   # [T,H,W,3]
        vy = self._flow[(subj, sess)][sl]                                 # [T,H,W,1]
        frames = np.concatenate([rgb, vy], axis=-1)                       # [T,H,W,4]
        video = torch.from_numpy(frames).permute(0, 3, 1, 2)             # [T,4,H,W]
        resp_gt = torch.from_numpy(self._resp[(subj, sess)][sl])

        if self.augment:
            if random.random() > 0.5:               # horizontal flip (v_y sign kept)
                video = torch.flip(video, dims=[-1])
            b = random.uniform(0.85, 1.15)          # clip-level brightness jitter (RGB only)
            video[:, :3] = torch.clamp(video[:, :3] * b, 0.0, 1.0)

        return video, resp_gt


def list_subjects(cache_path: str) -> List[int]:
    with h5py.File(cache_path, "r") as h5:
        return sorted(int(s) for s in h5["cohface"].keys())


def list_clips(cache_path: str) -> List[Tuple[int, int]]:
    clips = []
    with h5py.File(cache_path, "r") as h5:
        for subj in sorted(h5["cohface"].keys(), key=int):
            for sess in sorted(h5[f"cohface/{subj}"].keys(), key=int):
                clips.append((int(subj), int(sess)))
    return clips


def make_kfold_splits(
    cache_path: str,
    n_folds: int = 5,
    val_from_train: int = 4,
    seed: int = 42,
) -> List[Tuple[List, List, List]]:
    """Subject-disjoint K-fold splits (identical to the released protocol).

    40 subjects shuffled with Random(seed); each fold holds out a contiguous
    block of `n // n_folds` subjects for test, the next `val_from_train`
    (wrapping) for validation, and the rest for training.
    """
    subjects = list_subjects(cache_path)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    fold_size = n // n_folds
    all_clips = list_clips(cache_path)

    folds = []
    for k in range(n_folds):
        test_subjs = set(subjects[k * fold_size:(k + 1) * fold_size])
        val_subjs = set(
            subjects[(k * fold_size + fold_size + i) % n] for i in range(val_from_train)
        )
        train_subjs = set(s for s in subjects if s not in test_subjs and s not in val_subjs)
        train_clips = [(s, ss) for s, ss in all_clips if s in train_subjs]
        val_clips = [(s, ss) for s, ss in all_clips if s in val_subjs]
        test_clips = [(s, ss) for s, ss in all_clips if s in test_subjs]
        folds.append((train_clips, val_clips, test_clips))
    return folds
