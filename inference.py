#!/usr/bin/env python3
"""
Inference script: run a trained CNNONT model on ORIGINAL per-read pile TSVs.

Input per-read files:
- One file per read in PILES_DIR
- Filename: PREFIX + <MIDDLE> + SUFFIX  (e.g. minimizer_piles_multi_<...>.csv)
- Content: TSV with header, columns:
    multiplicity   gt
  (even if extension is .csv)

What it does:
- Loads model checkpoint (state_dict)
- Iterates all pile files in PILES_DIR (or a subset if you provide a read list)
- Runs inference per file (no padding needed)
- Writes per-read predictions to OUT_DIR, one file per read:
    <original_filename_without_ext>.pred.tsv
  with columns: multiplicity, gt, pred
- Accumulates and prints confusion matrix over all non-padding positions

Notes:
- Assumes labels are 0/1/2 and class "1" is your important middle class.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import model as md


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = md.CNNONT(2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def iter_pile_files(piles_dir: Path, prefix: str, suffix: str):
    for p in piles_dir.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix):
            yield p


def update_confusion(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    # y_true/y_pred are 1D numpy int arrays
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piles_dir", required=True, help="Directory with per-read pile files")
    ap.add_argument("--out_dir", required=True, help="Output directory for per-read prediction files")
    ap.add_argument("--checkpoint", required=True, help="Path to model .pth (state_dict)")
    ap.add_argument("--sep", default="\t", help="Separator for pile files (default: tab)")
    ap.add_argument("--feature_col", default="multiplicity")
    ap.add_argument("--label_col", default="gt")
    ap.add_argument("--prefix", default="minimizer_piles_multi_")
    ap.add_argument("--suffix", default=".csv")
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--g0", type=float, default=50.0, help="Global feature 0")
    ap.add_argument("--g1", type=float, default=25.0, help="Global feature 1")
    ap.add_argument("--max_reads", type=int, default=0, help="If >0, process only first N reads")
    args = ap.parse_args()

    piles_dir = Path(args.piles_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)
    print("torch cuda build:", torch.version.cuda)

    model = load_model(args.checkpoint, device)

    g = torch.tensor([[args.g0, args.g1]], dtype=torch.float32, device=device)

    cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

    processed = 0
    for fpath in iter_pile_files(piles_dir, args.prefix, args.suffix):
        # load pile
        df = pd.read_csv(fpath, sep=args.sep)
        if args.feature_col not in df.columns or args.label_col not in df.columns:
            raise RuntimeError(
                f"Missing required columns in {fpath.name}. "
                f"Have: {list(df.columns)} expected: {args.feature_col}, {args.label_col}"
            )

        x = df[args.feature_col].to_numpy(dtype=np.float32, copy=False)
        y = df[args.label_col].to_numpy(dtype=np.int64, copy=False)

        # shape to [B=1, C_in=1, L]
        X = torch.from_numpy(x).to(device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = model(X, g)               # [1, num_classes, L]
            pred = torch.argmax(logits, dim=1) # [1, L]
            pred_np = pred.squeeze(0).cpu().numpy().astype(np.int64)

        # save per-read predictions
        out_name = f"{fpath.stem}.pred.tsv"
        out_path = out_dir / out_name

        out_df = pd.DataFrame({
            args.feature_col: x,
            args.label_col: y,
            "pred": pred_np
        })
        out_df.to_csv(out_path, sep="\t", index=False)

        # update confusion matrix
        update_confusion(cm, y, pred_np, args.num_classes)

        processed += 1
        if processed % 200 == 0:
            print(f"Processed {processed} reads...")

        if args.max_reads and processed >= args.max_reads:
            break

    print("\nDone. Processed reads:", processed)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()
