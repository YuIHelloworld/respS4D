"""Train RespS4D on COHFACE with subject-disjoint 5-fold cross-validation.

Reproduces the paper's headline model under a chosen random seed:

    python train.py --config configs/seed42.yaml
    python train.py --config configs/seed1337.yaml
    python train.py --config configs/seed2025.yaml

The fold -> subject split is pinned by `split_seed` (42) so the three seeds are
apples-to-apples; only the training/init `seed` varies. Best weights per fold
are written to `save_dir/<run_name>_fold{k}_best.pt` as a plain state_dict.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import yaml
from torch import optim
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import COHFACEMotionDataset, make_kfold_splits
from losses import RespS4DLoss
from model import RespS4D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int) -> LambdaLR:
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def pearson_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean per-window Pearson correlation over a batch."""
    p = pred.float() - pred.float().mean(-1, keepdim=True)
    t = target.float() - target.float().mean(-1, keepdim=True)
    num = (p * t).sum(-1)
    den = p.norm(dim=-1) * t.norm(dim=-1) + 1e-8
    return (num / den).mean().item()


def run_fold(cfg, fold_name, train_clips, val_clips, device):
    ds_kw = dict(
        cache_path=cfg["cache_path"],
        flow_cache_path=cfg["flow_cache_path"],
        frame_len=cfg["frame_len"],
    )
    train_ds = COHFACEMotionDataset(clips=train_clips, stride=cfg["stride_train"],
                                    augment=True, **ds_kw)
    val_ds = COHFACEMotionDataset(clips=val_clips, stride=cfg["stride_eval"],
                                  augment=False, **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])

    model = RespS4D(**cfg["model"]).to(device)
    loss_fn = RespS4DLoss(alpha=cfg["alpha"], beta_init=cfg["beta_init"],
                          lam_amp=cfg["lam_amp"], eta=cfg["eta"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg["warmup_epochs"], cfg["epochs"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    best_ckpt = os.path.join(cfg["save_dir"], f"{fold_name}_best.pt")
    best_val = -float("inf")
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for video, resp_gt in train_loader:
            video, resp_gt = video.to(device), resp_gt.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                r_hat, _ = model(video)
                loss, _ = loss_fn(r_hat, resp_gt, epoch=epoch, total_epochs=cfg["epochs"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            optimizer.step()
        scheduler.step()

        model.eval()
        corrs = []
        with torch.no_grad():
            for video, resp_gt in val_loader:
                video, resp_gt = video.to(device), resp_gt.to(device)
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    r_hat, _ = model(video)
                corrs.append(pearson_batch(r_hat, resp_gt))
        val_pearson = float(np.mean(corrs)) if corrs else float("nan")
        lr = optimizer.param_groups[0]["lr"]
        print(f"[{fold_name}] epoch {epoch:2d}  val_pearson={val_pearson:.4f}  lr={lr:.2e}",
              flush=True)

        if val_pearson > best_val:
            best_val = val_pearson
            patience = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"[{fold_name}] early stop at epoch {epoch} "
                      f"(best val_pearson={best_val:.4f})", flush=True)
                break

    print(f"[{fold_name}] DONE  best val_pearson={best_val:.4f}  -> {best_ckpt}", flush=True)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a seed config YAML")
    ap.add_argument("--fold_id", type=int, default=None, help="run only this fold (0-4)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config={args.config}  seed={cfg['seed']}  split_seed={cfg['split_seed']}  "
          f"run_name={cfg['run_name']}  device={device}")

    folds = make_kfold_splits(cfg["cache_path"], n_folds=cfg["n_folds"],
                              val_from_train=cfg["val_from_train"], seed=cfg["split_seed"])
    results = {}
    for k, (train_clips, val_clips, _test_clips) in enumerate(folds):
        if args.fold_id is not None and k != args.fold_id:
            continue
        fold_name = f"{cfg['run_name']}_fold{k}"
        results[k] = run_fold(cfg, fold_name, train_clips, val_clips, device)

    if results:
        vals = list(results.values())
        print(f"\n5-fold val_pearson mean={np.mean(vals):.4f} "
              f"std={np.std(vals):.4f}  folds={results}")


if __name__ == "__main__":
    main()
