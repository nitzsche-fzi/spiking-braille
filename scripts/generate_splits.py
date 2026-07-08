"""Generate the canonical 5-fold train/val/test splits for the Braille dataset.

Produces ``braille_dataset/splits.json``, the file the dataset loader reads at
runtime. Kept as reference and to experiment with different splits. Not intended to be rerun by users.

Per class (each of 27 classes has exactly 200 samples):
  * 30 samples (15%) -> ``test``               (fixed once)
  * 5 disjoint folds of 30 samples (15%) each -> ``val1`` .. ``val5``
  * remaining 20 samples per class -> always in train
  * ``trainK`` = all samples - test - valK   = 140 per class (70%)

The split is stratified by class (val and test get equal samples per class).
Within each class, the per-sample assignment is a deterministic shuffle keyed
by ``--seed`` (default 0). Labels are identical across the 4 threshold files,
so the produced splits work for any threshold.

Usage:
    python scripts/generate_splits.py \\
        --pkl data/Braille/data_braille_letters_th_1.pkl \\
        --out braille_dataset/splits.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Dict, List


N_FOLDS = 5
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15  # per fold


def build_splits(labels: List[str], seed: int = 0) -> Dict[str, List[int]]:
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_label[lbl].append(i)

    counts = Counter(labels)
    if len(set(counts.values())) != 1:
        raise ValueError(f"Expected uniform class counts, got {counts}")
    per_class = next(iter(counts.values()))
    n_test = round(TEST_FRACTION * per_class)
    n_val = round(VAL_FRACTION * per_class)
    if n_test + N_FOLDS * n_val > per_class:
        raise ValueError(
            f"Per class need >= {n_test + N_FOLDS * n_val} samples, have {per_class}"
        )

    rng = random.Random(seed)
    test_idx: List[int] = []
    val_idx: List[List[int]] = [[] for _ in range(N_FOLDS)]

    for lbl in sorted(by_label):
        idxs = list(by_label[lbl])
        rng.shuffle(idxs)
        test_idx.extend(idxs[:n_test])
        for k in range(N_FOLDS):
            start = n_test + k * n_val
            val_idx[k].extend(idxs[start : start + n_val])

    test_set = set(test_idx)
    all_idx = set(range(len(labels)))
    splits: Dict[str, List[int]] = {"test": sorted(test_idx)}
    for k in range(N_FOLDS):
        val_set = set(val_idx[k])
        if val_set & test_set:
            raise AssertionError("val/test overlap")
        train_set = all_idx - test_set - val_set
        splits[f"train{k + 1}"] = sorted(train_set)
        splits[f"val{k + 1}"] = sorted(val_set)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True, help="Path to one threshold's pickle.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    labels = list(data["letter"])
    print(f"Loaded {len(labels)} labels, {len(set(labels))} classes.")

    splits = build_splits(labels, seed=args.seed)

    # Sanity report
    for name, idxs in splits.items():
        cls_counts = Counter(labels[i] for i in idxs)
        print(
            f"  {name:>7}: n={len(idxs)}  per-class min/max="
            f"{min(cls_counts.values())}/{max(cls_counts.values())}"
        )

    payload = {
        "schema_version": 1,
        "n_samples": len(labels),
        "n_classes": len(set(labels)),
        "n_folds": N_FOLDS,
        "test_fraction": TEST_FRACTION,
        "val_fraction": VAL_FRACTION,
        "seed": args.seed,
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
