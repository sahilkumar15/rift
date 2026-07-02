# Path: src/data/datamodule.py
"""Bundles dataset + loaders.

DDP behavior:
  * Training uses DistributedSampler with shuffling.
  * Validation uses an exact non-padding distributed sampler so full validation
    reports the real CSV size instead of a padded number.
"""

from __future__ import annotations

import math
import os
from typing import Iterator, Optional

from .ffpp_dataset import FFPPDataset


def _is_ddp_env():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _dist_rank_world():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def _parse_max_items(v):
    if v is None:
        return None

    if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "false", "0"):
        return None

    try:
        v = int(v)
    except Exception:
        return None

    return v if v > 0 else None


def _maybe_subset(ds, max_items):
    max_items = _parse_max_items(max_items)

    if max_items is None:
        return ds

    from torch.utils.data import Subset

    n = min(len(ds), max_items)
    return Subset(ds, list(range(n)))


def _identity_gap_mode(ds):
    base = getattr(ds, "dataset", ds)
    return getattr(base, "identity_gap_mode", "unknown")


class ExactDistributedEvalSampler:
    """DDP eval sampler with no padding and no duplicate validation rows."""

    def __init__(self, dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None):
        self.dataset = dataset

        if num_replicas is None or rank is None:
            rank0, world0, _ = _dist_rank_world()
            self.rank = rank0 if rank is None else int(rank)
            self.num_replicas = world0 if num_replicas is None else int(num_replicas)
        else:
            self.rank = int(rank)
            self.num_replicas = int(num_replicas)

        self.num_samples = len(range(self.rank, len(self.dataset), self.num_replicas))

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        return None


def build_dataloaders(cfg):
    try:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
    except Exception:
        raise RuntimeError("torch required.")

    from .transforms import build_transform
    from PIL import Image

    def loader(p):
        return Image.open(p).convert("RGB")

    image_size = cfg.get("image_size", 256)

    tf_train = build_transform(image_size, train=bool(cfg.get("train_aug", False)))
    tf_eval = build_transform(image_size, train=False)

    train_csv = cfg.get("train_csv")
    val_csv = cfg.get("val_csv") or train_csv

    strict = cfg.get("strict_identity_gap", False)

    if not train_csv:
        return _synthetic_loaders(cfg)

    common = dict(
        image_loader=loader,
        strict_identity_gap=strict,
        data_root=cfg.get("data_root") or cfg.get("root_path") or cfg.get("root"),
        path_prefix_from=cfg.get("path_prefix_from"),
        path_prefix_to=cfg.get("path_prefix_to"),
        path_rewrites=cfg.get("path_rewrites"),
        check_files=cfg.get("check_files", True),
        check_limit=cfg.get("check_limit", 0),
    )

    train = FFPPDataset(train_csv, transform=tf_train, **common)
    val = FFPPDataset(val_csv, transform=tf_eval, **common)

    train = _maybe_subset(train, cfg.get("max_items"))
    val = _maybe_subset(val, cfg.get("val_max_items", cfg.get("max_items")))

    ddp = _is_ddp_env()

    if ddp:
        rank, world, _ = _dist_rank_world()

        train_sampler = DistributedSampler(
            train,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

        val_sampler = ExactDistributedEvalSampler(
            val,
            num_replicas=world,
            rank=rank,
        )
    else:
        train_sampler = None
        val_sampler = None

    def coll(b):
        return b

    num_workers = int(cfg.get("num_workers", 0) or 0)
    batch_size = int(cfg.get("batch_size", 8))

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        collate_fn=coll,
        persistent_workers=(num_workers > 0 and bool(cfg.get("persistent_workers", True))),
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2) or 2)

    train_loader = DataLoader(
        train,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )

    if ddp and int(os.environ.get("RANK", "0")) == 0:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        print(
            f"[ddp datamodule] world={world} "
            f"train_total={len(train)} val_total={len(val)} "
            f"per_rank_train≈{math.ceil(len(train) / world)} "
            f"per_rank_val_exact≈{math.ceil(len(val) / world)} "
            f"batch_per_rank={batch_size}"
        )

    return train_loader, val_loader, _identity_gap_mode(train)


def _synthetic_loaders(cfg):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class Synth(Dataset):
        def __init__(self, n=32, size=256):
            self.n = n
            self.size = size

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {
                "image": torch.rand(3, self.size, self.size) * 2.0 - 1.0,
                "donor": None,
                "label": i % 2,
                "sample": None,
            }

    def coll(b):
        return b

    ds = Synth(size=cfg.get("image_size", 256))

    return (
        DataLoader(ds, batch_size=cfg.get("batch_size", 8), collate_fn=coll),
        DataLoader(ds, batch_size=cfg.get("batch_size", 8), collate_fn=coll),
        "proxy",
    )
