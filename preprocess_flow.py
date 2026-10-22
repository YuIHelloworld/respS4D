"""Precompute the vertical Farneback optical-flow cache from the RGB cache.

For each session, dense Farneback flow is computed between adjacent frames; the
vertical component v_y (channel 1) is scaled, clipped to [-1, 1], stored float16:

    /cohface/{subj}/{sess}/flow[N,128,128,1] float16

Usage:
    python precompute_flow.py --rgb_cache caches/cache_cohface_128.h5 \
        --out caches/cache_cohface_128_farneback_flow5.h5 --flow_scale 5.0 --workers 6
"""

import argparse
import multiprocessing as mp
import os
import time

import cv2
import h5py
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def _farneback_vy(frames_u8: np.ndarray, flow_scale: float) -> np.ndarray:
    """RGB uint8 [T,H,W,3] -> vertical flow v_y [T,H,W,1] float16 (first frame 0)."""
    T, H, W, _ = frames_u8.shape
    vy = np.zeros((T, H, W, 1), dtype=np.float32)
    prev_gray = None
    for t in range(T):
        gray = cv2.cvtColor(frames_u8[t], cv2.COLOR_RGB2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            vy[t, ..., 0] = flow[..., 1]
        prev_gray = gray
    return np.clip(vy / float(flow_scale), -1.0, 1.0).astype(np.float16)


def _compute_one(args):
    rgb_cache, subj, sess, flow_scale = args
    with h5py.File(rgb_cache, "r") as h5:
        frames = h5[f"cohface/{subj}/{sess}/frames"][()]
    return subj, sess, _farneback_vy(frames, flow_scale)


def _list_sessions(rgb_cache):
    with h5py.File(rgb_cache, "r") as h5:
        return [(int(s), int(ss))
                for s in sorted(h5["cohface"].keys(), key=int)
                for ss in sorted(h5[f"cohface/{s}"].keys(), key=int)
                if "frames" in h5[f"cohface/{s}/{ss}"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flow_scale", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sessions = _list_sessions(args.rgb_cache)
    tasks = [(args.rgb_cache, s, ss, args.flow_scale) for s, ss in sessions]
    print(f"Computing Farneback v_y for {len(tasks)} sessions -> {args.out}")

    t0 = time.time()
    with h5py.File(args.out, "w") as dst:
        dst.attrs["flow_scale"] = float(args.flow_scale)
        if args.workers <= 1:
            results = map(_compute_one, tasks)
        else:
            pool = mp.get_context("spawn").Pool(args.workers)
            results = pool.imap_unordered(_compute_one, tasks)
        for subj, sess, flow in tqdm(results, total=len(tasks), desc="Farneback"):
            grp = dst.require_group(f"cohface/{subj}/{sess}")
            grp.create_dataset("flow", data=flow, dtype=np.float16,
                               chunks=(min(32, flow.shape[0]), flow.shape[1], flow.shape[2], 1))
            grp.attrs["n_frames"] = int(flow.shape[0])
        if args.workers > 1:
            pool.close()
            pool.join()
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
