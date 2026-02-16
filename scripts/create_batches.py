#!/usr/bin/env python3
"""
SAFE batching script WITH train/val/test split that filters out reads without pile files.

Why this version:
- Some reads in read_lengths.csv have no corresponding pile file on disk
  (common with chimeras / filtered reads / naming differences).
- Instead of crashing mid-run, we:
    1) index all pile files on disk
    2) keep only reads whose UUID exists in the pile index
    3) (optional) also verify resolvable pile path for each read (costs more CPU)
    4) split train/val/test at READ level (no leakage)
    5) batch + pad + save .npz

Also avoids:
- OSError: [Errno 36] File name too long (we never construct long paths)

Assumptions:
- lengths TSV has columns: read, pile_length
- pile files exist in PILES_DIR and are named:
    minimizer_piles_multi_<MIDDLE>.csv
  where <MIDDLE> starts with the UUID (before first ';')
- file content is TSV-with-header created by pandas with columns: multiplicity, gt
  (even if extension is .csv). Set SEP accordingly.

Outputs:
- OUT_ROOT/train/... NPZ batches + metadata CSV
- OUT_ROOT/val  /... NPZ batches + metadata CSV
- OUT_ROOT/test /... NPZ batches + metadata CSV
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IGNORE_INDEX = -100

# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
READ_LENGTHS_CSV = "" # TSV with columns: read, pile_length (no header), can be generated with seqkit
PILES_DIR = "" # Directory with per-read pile files, named like: minimizer_piles_multi_<MIDDLE>.csv
OUT_ROOT = "" # Output directory for NPZ batches; will create subdirs train/ val/ test/

PREFIX = "minimizer_piles_multi_" ## standard Draven output prefix; change if your files differ
SUFFIX = ".csv"

SEP = "\t"  # pile file separator (TSV content with header); change to "," if truly CSV

FEATURE_COL = "multiplicity"
LABEL_COL = "gt"

# Split fractions
TEST_SIZE = 0.20
VAL_SIZE  = 0.10   # fraction of total data to reserve for validation
RANDOM_STATE = 42
STRATIFY_BY_LENGTH = True

TOKEN_BUDGET = 150_000
BUCKET_EDGES = (2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000)

MAX_BATCH_SIZE = None
STRICT_LENGTH_CHECK = False

VERIFY_RESOLVE_FOR_EACH_READ = True


# -----------------------------
# Normalization + indexing
# -----------------------------
def normalize_read(s: str) -> str:
    return str(s).strip().strip("'").strip('"')


def uuid_key(full_read: str) -> str:
    return normalize_read(full_read).split(";", 1)[0]


def index_piles(piles_dir: str | Path, prefix: str, suffix: str):
    """
    Build:
    - by_middle: middle string (filename without prefix/suffix) -> path
    - by_uuid:   uuid -> list of candidate paths
    """
    piles_dir = Path(piles_dir)
    by_middle: dict[str, Path] = {}
    by_uuid: dict[str, list[Path]] = {}

    for p in piles_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue

        middle = name[len(prefix):-len(suffix)]
        by_middle[middle] = p

        u = middle.split(";", 1)[0]
        by_uuid.setdefault(u, []).append(p)

    if not by_middle:
        raise RuntimeError(f"No pile files matched prefix={prefix} suffix={suffix} in {piles_dir}")

    return by_middle, by_uuid


def resolve_pile_path(full_read: str, by_middle: dict[str, Path], by_uuid: dict[str, list[Path]],
                      prefix: str, suffix: str) -> Path | None:
    """
    Resolve pile file path for a given df read string.
    Tries:
      1) exact full match (middle == full_read)
      2) uuid-only match if exactly one candidate
      3) if multiple candidates: prefer the one whose middle is a prefix of full_read;
         else choose shortest filename as last resort.
    """
    r = normalize_read(full_read)

    if r in by_middle:
        return by_middle[r]

    u = uuid_key(r)
    cands = by_uuid.get(u, [])
    if not cands:
        return None

    if len(cands) == 1:
        return cands[0]

    best = None
    best_score = -1
    for p in cands:
        middle = p.name[len(prefix):-len(suffix)]
        if r.startswith(middle):
            score = len(middle)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best = p

    if best is not None and best_score > 0:
        return best

    return sorted(cands, key=lambda p: len(p.name))[0]


# -----------------------------
# Load per-read pile file
# -----------------------------
def load_pile_file(pile_path: Path,
                   feature_col: str = FEATURE_COL,
                   label_col: str = LABEL_COL,
                   feature_dtype=np.float32,
                   label_dtype=np.int16) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(pile_path, sep=SEP, usecols=[feature_col, label_col])
    x = df[feature_col].to_numpy(dtype=feature_dtype, copy=False)
    y = df[label_col].to_numpy(dtype=label_dtype, copy=False)
    return x, y


# -----------------------------
# Bucketing + padding + saving
# -----------------------------
def assign_bucket(pile_len: int, edges: list[int]) -> int:
    for b, e in enumerate(edges):
        if pile_len <= e:
            return b
    return len(edges)


def pad_and_save_batch(X_list: list[np.ndarray],
                       Y_list: list[np.ndarray],
                       read_ids: list[str],
                       bucket_id: int,
                       save_path: Path) -> dict:
    lengths = np.array([len(x) for x in X_list], dtype=np.int32)
    Lmax = int(lengths.max())
    B = len(X_list)

    X = np.zeros((B, Lmax), dtype=np.float32)
    Y = np.full((B, Lmax), IGNORE_INDEX, dtype=np.int16)

    for i, (x, y) in enumerate(zip(X_list, Y_list)):
        L = len(x)
        X[i, :L] = x
        Y[i, :L] = y

    np.savez_compressed(
        save_path,
        X=X,
        Y=Y,
        lengths=lengths,
        read_ids=np.array(read_ids, dtype=object),
        bucket=np.int32(bucket_id),
        Lmax=np.int32(Lmax),
    )

    sum_lengths = int(lengths.sum())
    pad_ratio = float((B * Lmax) / max(1, sum_lengths))
    return {
        "file": str(save_path),
        "bucket": int(bucket_id),
        "batch_size": int(B),
        "Lmin": int(lengths.min()),
        "Lmax": int(Lmax),
        "sum_lengths": sum_lengths,
        "pad_ratio": pad_ratio,
    }


def build_batched_npz(read_df: pd.DataFrame,
                      by_middle: dict[str, Path],
                      by_uuid: dict[str, list[Path]],
                      out_dir: str | Path,
                      token_budget: int = TOKEN_BUDGET,
                      bucket_edges: tuple[int, ...] = BUCKET_EDGES,
                      max_batch_size: int | None = MAX_BATCH_SIZE,
                      strict_length_check: bool = STRICT_LENGTH_CHECK) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_df.copy()
    required = {"read", "pile_length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"read_df missing required columns: {missing}")

    edges = list(bucket_edges)
    df["read_norm"] = df["read"].astype(str).apply(normalize_read)
    df["bucket"] = df["pile_length"].astype(int).apply(lambda L: assign_bucket(int(L), edges))

    meta_rows: list[dict] = []
    batch_id = 0

    for b in sorted(df["bucket"].unique()):
        sub = df[df["bucket"] == b].sort_values("pile_length")
        if sub.empty:
            continue

        bucket_dir = out_dir / f"bucket_{b:02d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)

        cur: list[tuple[str, int]] = []  # (read_norm, expected_len)
        cur_sum = 0

        def flush(rows: list[tuple[str, int]]):
            nonlocal batch_id
            X_list, Y_list, read_ids = [], [], []

            for read_norm, expected_len in rows:
                fpath = resolve_pile_path(read_norm, by_middle, by_uuid, PREFIX, SUFFIX)
                if fpath is None:
                    raise FileNotFoundError(
                        f"Unexpected: could not resolve pile file for UUID={uuid_key(read_norm)}\n"
                        f"Read: {read_norm}"
                    )

                x, y = load_pile_file(fpath)

                if strict_length_check and len(x) != expected_len:
                    raise ValueError(
                        f"Length mismatch for UUID={uuid_key(read_norm)}\n"
                        f"File: {fpath}\n"
                        f"Loaded: {len(x)} expected pile_length: {expected_len}\n"
                        f"Read: {read_norm}"
                    )

                X_list.append(x)
                Y_list.append(y)
                read_ids.append(read_norm)

            save_path = bucket_dir / f"batch_{batch_id:06d}.npz"
            meta_rows.append(pad_and_save_batch(X_list, Y_list, read_ids, b, save_path))
            batch_id += 1

        for _, row in sub.iterrows():
            read_norm = row["read_norm"]
            L_expected = int(row["pile_length"])

            if cur and (cur_sum + L_expected > token_budget):
                flush(cur)
                cur, cur_sum = [], 0

            cur.append((read_norm, L_expected))
            cur_sum += L_expected

            if max_batch_size is not None and len(cur) >= max_batch_size:
                flush(cur)
                cur, cur_sum = [], 0

        if cur:
            flush(cur)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "batches_metadata.csv", index=False)
    return meta_df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if TEST_SIZE + VAL_SIZE >= 1.0:
        raise SystemExit("TEST_SIZE + VAL_SIZE must be < 1.0")

    df = pd.read_csv(READ_LENGTHS_CSV, sep="\t", names=["read", "pile_length"])
    if "read" not in df.columns or "pile_length" not in df.columns:
        raise SystemExit("Index TSV must contain columns: read, pile_length")

    print("Indexing pile files...")
    by_middle, by_uuid = index_piles(PILES_DIR, PREFIX, SUFFIX)
    print("Indexed pile files:", len(by_middle), "unique UUIDs:", len(by_uuid))

    # Normalize + add UUID
    df["read_norm"] = df["read"].astype(str).apply(normalize_read)
    df["uuid"] = df["read_norm"].apply(uuid_key)

    # Filter: keep only reads whose UUID exists in pile index
    before = len(df)
    df = df[df["uuid"].isin(by_uuid.keys())].copy()
    after = len(df)
    print(f"Filtered reads by UUID presence: {before} -> {after} (dropped {before-after})")

    # Optional: ensure we can resolve a specific file path for each read
    if VERIFY_RESOLVE_FOR_EACH_READ:
        ok = []
        for r in df["read_norm"].values:
            ok.append(resolve_pile_path(r, by_middle, by_uuid, PREFIX, SUFFIX) is not None)
        df = df[np.array(ok, dtype=bool)].copy()
        print(f"Filtered reads by resolvable file path: now {len(df)} reads")

    if len(df) == 0:
        raise SystemExit("After filtering, no reads remain. Check PILES_DIR/PREFIX/SUFFIX/SEP.")

    # Stratify split by length bucket
    edges = list(BUCKET_EDGES)
    if STRATIFY_BY_LENGTH:
        df["len_bucket"] = df["pile_length"].astype(int).apply(lambda L: assign_bucket(int(L), edges))
        strat_all = df["len_bucket"]
    else:
        strat_all = None

    # 1) Split off TEST
    trainval_df, test_df = train_test_split(
        df.drop(columns=["read_norm", "uuid"], errors="ignore"),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=strat_all
    )

    # 2) Split TRAIN vs VAL from remaining
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)  # convert to fraction of trainval

    if STRATIFY_BY_LENGTH:
        tmp = trainval_df.copy()
        tmp["len_bucket"] = tmp["pile_length"].astype(int).apply(lambda L: assign_bucket(int(L), edges))
        strat_tv = tmp["len_bucket"]
    else:
        strat_tv = None

    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_rel,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=strat_tv
    )

    out_root = Path(OUT_ROOT)
    train_out = out_root / "train"
    val_out = out_root / "val"
    test_out = out_root / "test"

    print(f"Split sizes | train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    print("Building TRAIN batches...")
    train_meta = build_batched_npz(train_df, by_middle, by_uuid, train_out)

    print("Building VAL batches...")
    val_meta = build_batched_npz(val_df, by_middle, by_uuid, val_out)

    print("Building TEST batches...")
    test_meta = build_batched_npz(test_df, by_middle, by_uuid, test_out)

    print("Done.")
    print("Train batches:", len(train_meta), "->", train_out / "batches_metadata.csv")
    print("Val   batches:", len(val_meta),   "->", val_out / "batches_metadata.csv")
    print("Test  batches:", len(test_meta),  "->", test_out / "batches_metadata.csv")
