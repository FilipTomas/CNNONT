import pandas as pd
import numpy as np


def classify_gt_kmers(pat_kmers, mat_kmers):
    shared_kmers = pat_kmers.intersection(mat_kmers)

    unique_pat_kmers = pat_kmers - shared_kmers
    unique_mat_kmers = mat_kmers - shared_kmers
    het_kmers = unique_pat_kmers.union(unique_mat_kmers)
    hom_kmers = shared_kmers
    total_kmers = hom_kmers.union(het_kmers)

    return het_kmers, hom_kmers, total_kmers

def load_and_preprocess_piles(path):
    k_mer_piles = pd.read_csv(path, sep="\t", header=None, names=["read", "pile", "ids"])
    k_mer_piles["pile"] = (
    k_mer_piles["pile"]
      .apply(np.fromstring,    # vectorised C‑routine
             sep=",",          # delimiter
             dtype=np.uint32)
    )
    
    k_mer_piles["ids"] = (
    k_mer_piles["ids"]
      .apply(np.fromstring,    # vectorised C‑routine
             sep=",",          # delimiter
             dtype=np.uint64)
    )

    return k_mer_piles


def preprocess_pile(piles, name):
    pile = piles.loc[piles.read == name, "pile"].values[0]
    print(f"Preprocessing pile for read {name}, length {len(pile)}")
   # pile = np.clip(window_mins_nonoverlap(pile), None, 100)
    pile = np.clip(pile, None, 100)
    return pile, len(pile)


def label_piles(piles, name, het_kmers, hom_kmers):
    ids = piles.loc[piles.read == name, "ids"].values[0]
    classifications = ["H" if x in het_kmers else ("D" if x in hom_kmers else "E") for x in ids]
    return classifications

def window_median_threshold(arr, w=30, thresh=50*1.5, keep_tail=True):
    arr = np.asarray(arr, dtype=float)

    # mask out values above threshold
    arr = np.where(arr > thresh, np.nan, arr)

    n = (len(arr) // w) * w
    meds = np.nanmedian(arr[:n].reshape(-1, w), axis=1)

    if keep_tail and n < len(arr):
        meds = np.concatenate([meds, [np.nanmedian(arr[n:])]])
    return meds

def expand_median(meds, pile):
    expanded = np.repeat(meds[:-1], 30)
    if len(expanded) < len(pile):
        expanded = np.concatenate([expanded, [meds[-1]]*(len(pile)-len(expanded))])

    return expanded

def class_to_int(labels):
    int_labels = []
    for label in labels:
        if label == "D":
            int_labels.append(0)
        elif label == "H":
            int_labels.append(1)
        else:
            int_labels.append(2)
    return int_labels