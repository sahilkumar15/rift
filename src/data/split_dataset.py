# Path: iganer/rift/data/split_dataset.py
# Status: NEW
"""Deterministic train/val split by index."""
import random
def split_indices(n, val_frac=0.1, seed=42):
    idx=list(range(n)); random.Random(seed).shuffle(idx)
    k=int(n*(1-val_frac)); return idx[:k], idx[k:]
