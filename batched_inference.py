#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load a saved CNNONT model (.pth / state_dict) and evaluate on NPZ-batched data
loaded exactly like your training script:
- Each dataset item is one saved batch (.npz) under bucket_*/batch_*.npz
- NPZ contains: X [B,L], Y [B,L] with IGNORE_INDEX padding, lengths [B]
- We report:
    * precision/recall/F1 for TARGET_CLASS (label 1)
    * full confusion matrix (fast, vectorized)
"""

import glob
import argparse
import numpy as np
import torch
import model as md
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Dataset: each item is one saved batch (.npz)
# ----------------------------
class NPZBatchDataset(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(glob.glob(f"{root_dir}/bucket_*/batch_*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz batches found under: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True)
        X = torch.from_numpy(data["X"]).unsqueeze(1)  # [B,1,L]
        Y = torch.from_numpy(data["Y"]).long()        # [B,L]
        lengths = torch.from_numpy(data["lengths"]).long()
        return X, Y, lengths


def eval_target_metrics(model, loader, g, device, target_class=1, ignore_index=-100):
    """Precision/recall/F1 for target_class over all non-padding positions."""
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for X, Y, _ in loader:
            X = X.squeeze(0).to(device, non_blocking=True)  # [B,1,L]
            Y = Y.squeeze(0).to(device, non_blocking=True)  # [B,L]

            logits = model(X, g)            # [B,C,L]
            pred = logits.argmax(dim=1)     # [B,L]

            mask = (Y != ignore_index)
            y = Y[mask]
            p = pred[mask]

            # print("Done. Computing metrics...")

            tp += int(((p == target_class) & (y == target_class)).sum().item())
            fp += int(((p == target_class) & (y != target_class)).sum().item())
            fn += int(((p != target_class) & (y == target_class)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def confusion_matrix_stream(model, loader, g, device, num_classes=3, ignore_index=-100, use_amp=False):
    """Fast confusion matrix: cm[true, pred] accumulated over non-padding positions."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():
        for X, Y, _ in loader:
            X = X.squeeze(0).to(device, non_blocking=True)  # [B,1,L]
            Y = Y.squeeze(0).to(device, non_blocking=True)  # [B,L]

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(X, g)
            else:
                logits = model(X, g)

            pred = logits.argmax(dim=1)  # [B,L]

            mask = (Y != ignore_index)
            y = Y[mask].to(torch.long)     # [N]
            p = pred[mask].to(torch.long)  # [N]

            idx = y * num_classes + p
            cm += torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)

    return cm.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root dir with bucket_*/batch_*.npz (e.g. .../test)")
    ap.add_argument("--model", required=True, help="Path to .pth state_dict (best or last)")
    ap.add_argument("--g0", type=float, default=50.0)
    ap.add_argument("--g1", type=float, default=25.0)
    ap.add_argument("--target_class", type=int, default=1)
    ap.add_argument("--ignore_index", type=int, default=-100)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--use_amp", action="store_true", help="Use autocast during eval (CUDA only)")
    args = ap.parse_args()

    print("torch cuda build:", torch.version.cuda)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device == "cpu":
        torch.set_num_threads(8)
    print("device:", device)

    ds = NPZBatchDataset(args.root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = md.CNNONT(2).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Global features
    g = torch.tensor([[args.g0, args.g1]], dtype=torch.float32, device=device)

    print(f"Loaded model from: {args.model}")
    # Metrics
    p1, r1, f1 = eval_target_metrics(
        model, loader, g, device,
        target_class=args.target_class,
        ignore_index=args.ignore_index
    )
    print(f"SET={args.root} | class={args.target_class} P={p1:.4f} R={r1:.4f} F1={f1:.4f}")

    cm = confusion_matrix_stream(
        model, loader, g, device,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        use_amp=args.use_amp
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
