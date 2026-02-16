#!/usr/bin/env python3
"""
Add ground-truth labels to per-read k-mer multiplicity sketches.

Input per-read files are expected to be TSVs with two columns:
  1) multiplicity (int)
  2) hash (int; 64-bit k-mer hash)

For each row, assign:
  gt = 0  if hash in hom_kmers (present in both haplotypes; "diploid"/shared)
  gt = 1  if hash in het_kmers (present in exactly one haplotype; "haploid"/unique)
  gt = 2  otherwise ("error"/not in reference sets)

The output files are written as TSV with columns:
  multiplicity, gt

Example:
  python add_gt_labels.py \
    --hom-kmers /path/hom_kmers.npy \
    --het-kmers /path/het_kmers.npy \
    --input-dir /path/piles_60k \
    --output-dir /path/piles_60k_gt \
    --ext .csv \
    --sep $'\\t' \
    --clip-max 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add GT labels to per-read (multiplicity, hash) sketch files."
    )
    p.add_argument("--hom-kmers", required=True, type=Path, help="Path to hom_kmers.npy")
    p.add_argument("--het-kmers", required=True, type=Path, help="Path to het_kmers.npy")
    p.add_argument("--input-dir", required=True, type=Path, help="Directory with per-read files")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to write labeled files")
    p.add_argument("--ext", default=".csv", help="File extension to process (default: .csv)")
    p.add_argument("--sep", default="\t", help="Input/output separator (default: tab)")
    p.add_argument("--clip-max", type=int, default=200, help="Clip multiplicity to [0, clip-max]")
    p.add_argument("--clip-min", type=int, default=0, help="Clip multiplicity to [clip-min, clip-max]")
    p.add_argument("--quiet", action="store_true", help="Reduce logging")
    return p.parse_args()


def load_kmer_set(path: Path) -> set[int]:
    arr = np.load(path, allow_pickle=False)
    # Ensure Python ints for set membership.
    # If arr is uint64, converting to Python int preserves value.
    return set(int(x) for x in arr)


def main() -> int:
    args = parse_args()

    if not args.hom_kmers.exists():
        print(f"ERROR: hom-kmers file not found: {args.hom_kmers}", file=sys.stderr)
        return 1
    if not args.het_kmers.exists():
        print(f"ERROR: het-kmers file not found: {args.het_kmers}", file=sys.stderr)
        return 1
    if not args.input_dir.exists():
        print(f"ERROR: input-dir not found: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("Loading k-mer sets...")
    hom_kmers = load_kmer_set(args.hom_kmers)
    het_kmers = load_kmer_set(args.het_kmers)

    if not args.quiet:
        print(f"Loaded hom_kmers: {len(hom_kmers):,}")
        print(f"Loaded het_kmers: {len(het_kmers):,}")
        print(f"Processing files in: {args.input_dir}")

    files = sorted(args.input_dir.glob(f"*{args.ext}"))
    if not files:
        print(f"WARNING: No files matching *{args.ext} found in {args.input_dir}", file=sys.stderr)
        return 0

    processed = 0
    for i, fpath in enumerate(files, start=1):
        out_path = args.output_dir / fpath.name

        # Read input: multiplicity, hash
        df = pd.read_csv(
            fpath,
            sep=args.sep,
            header=None,
            names=["multiplicity", "hash"],
            dtype={"multiplicity": np.int64, "hash": object},  # hash may exceed int64 depending on encoding
        )

        # Convert hash to Python int if possible
        # (If hashes are stored as strings, this will parse them; if already numeric, it is cheap.)
        df["hash"] = df["hash"].map(lambda x: int(x))

        # Clip multiplicities
        df["multiplicity"] = df["multiplicity"].clip(lower=args.clip_min, upper=args.clip_max)

        # Vectorized labeling:
        # Start as 2 (error), then overwrite with 0/1 where appropriate.
        gt = np.full(len(df), 2, dtype=np.int8)

        is_hom = df["hash"].isin(hom_kmers).to_numpy()
        is_het = (~is_hom) & df["hash"].isin(het_kmers).to_numpy()

        gt[is_hom] = 0
        gt[is_het] = 1
        df["gt"] = gt

        # Drop hash column
        df = df.drop(columns=["hash"])

        df.to_csv(out_path, sep=args.sep, index=False)

        processed += 1
        if not args.quiet and (i == 1 or i % 25 == 0 or i == len(files)):
            print(f"[{i}/{len(files)}] wrote {out_path}")

    if not args.quiet:
        print(f"Done. Processed {processed} file(s). Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
