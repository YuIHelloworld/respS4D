"""Build the COHFACE RGB HDF5 cache used for training.

For each session: decode the AVI to 128x128 RGB uint8 frames (plain resize),
resample the 256 Hz PPG / chest-belt respiration to 20 fps, drop the first 40
frames (sensor-settling spike), and robustly z-score. Output:

    /cohface/{subj}/{sess}/{frames[N,128,128,3] uint8, pulse[N], resp[N]}

Usage:
    python preprocess_cohface.py --data_root /path/to/COHFACE \
        --out caches/cache_cohface_128.h5 --workers 6
"""

import argparse
import multiprocessing as mp
import os
import sys
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
from scipy.signal import resample_poly

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

RESAMPLE_UP = 5           # 256 * 5 / 64 = 20 fps
RESAMPLE_DN = 64
SIGNAL_TRIM_FRAMES = 40   # drop first 2 s @ 20 fps


def _robust_zscore(x: np.ndarray, mad_thresh: float = 10.0) -> np.ndarray:
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) + 1e-8
    clipped = np.clip(x, median - mad_thresh * mad, median + mad_thresh * mad)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-8)


def list_clips(root: str):
    clips = []
    cohface_dir = os.path.join(root, "cohface")
    for subj in sorted(os.listdir(cohface_dir)):
        if not subj.isdigit():
            continue
        subj_dir = os.path.join(cohface_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            if sess.isdigit() and os.path.isdir(os.path.join(subj_dir, sess)):
                clips.append((int(subj), int(sess)))
    return clips


def _load_clip(task: Tuple[str, int, int, int]) -> Optional[Tuple]:
    root, subj, sess, img_size = task
    video_path = os.path.join(root, "cohface", str(subj), str(sess), "data.avi")
    hdf5_path = os.path.join(root, "cohface", str(subj), str(sess), "data.hdf5")
    if not (os.path.exists(video_path) and os.path.exists(hdf5_path)):
        print(f"  [SKIP] subj={subj} sess={sess}: missing files", file=sys.stderr)
        return None

    with h5py.File(hdf5_path, "r") as f:
        pulse_raw = f["pulse"][()].astype(np.float64)
        resp_raw = f["respiration"][()].astype(np.float64)
    pulse_20 = resample_poly(pulse_raw, RESAMPLE_UP, RESAMPLE_DN).astype(np.float32)
    resp_20 = resample_poly(resp_raw, RESAMPLE_UP, RESAMPLE_DN).astype(np.float32)
    pulse_20 = _robust_zscore(pulse_20[SIGNAL_TRIM_FRAMES:])
    resp_20 = _robust_zscore(resp_20[SIGNAL_TRIM_FRAMES:])

    cap = cv2.VideoCapture(video_path)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_clip = min(n_vid, len(pulse_20))
    frames = []
    for _ in range(n_clip):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (img_size, img_size):
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"  [SKIP] subj={subj} sess={sess}: no frames decoded", file=sys.stderr)
        return None

    frames_arr = np.stack(frames, axis=0)
    N = min(len(frames_arr), n_clip)
    return subj, sess, frames_arr[:N], pulse_20[:N], resp_20[:N]


def main():
    ap = argparse.ArgumentParser(description="COHFACE -> 128x128 RGB HDF5 cache")
    ap.add_argument("--data_root", required=True, help="COHFACE root (contains cohface/)")
    ap.add_argument("--out", default="caches/cache_cohface_128.h5")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--compress", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    clips = list_clips(args.data_root)
    tasks = [(args.data_root, s, ss, args.img_size) for s, ss in clips]
    print(f"Processing {len(tasks)} sessions -> {args.out}")

    with h5py.File(args.out, "w") as h5:
        if args.workers <= 1:
            results = map(_load_clip, tasks)
        else:
            pool = mp.get_context("spawn").Pool(args.workers)
            results = pool.imap_unordered(_load_clip, tasks)
        for res in tqdm(results, total=len(tasks), desc="COHFACE"):
            if res is None:
                continue
            subj, sess, frames_arr, pulse_20, resp_20 = res
            grp = h5.require_group(f"cohface/{subj}/{sess}")
            N = len(frames_arr)
            kw = {"compression": "gzip", "compression_opts": 1} if args.compress else {}
            grp.create_dataset("frames", data=frames_arr,
                               chunks=(min(32, N), args.img_size, args.img_size, 3), **kw)
            grp.create_dataset("pulse", data=pulse_20)
            grp.create_dataset("resp", data=resp_20)
            grp.attrs["n_frames"] = N
        if args.workers > 1:
            pool.close()
            pool.join()
    print("Done.")


if __name__ == "__main__":
    main()
