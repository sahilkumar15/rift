# Path: src/data/datamodule.py
"""Bundles dataset + loaders.

DDP behavior:
  If launched with torchrun, this uses DistributedSampler so each GPU sees a
  different shard of the same dataset.

Example:
  torchrun --nproc_per_node=4 train_rift_rl.py ...
    rank 0 -> shard 0
    rank 1 -> shard 1
    rank 2 -> shard 2
    rank 3 -> shard 3
"""

from __future__ import annotations

import os

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

    if isinstance(v, str):
        if v.strip().lower() in ("", "none", "null", "false", "0"):
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

    try:
        from torch.utils.data import Subset
    except Exception:
        return ds

    n = min(len(ds), max_items)
    return Subset(ds, list(range(n)))


def _identity_gap_mode(ds):
    # If Subset, original dataset is ds.dataset.
    base = getattr(ds, "dataset", ds)
    return getattr(base, "identity_gap_mode", "unknown")


def build_dataloaders(cfg):
    try:
        import torch  # noqa: F401
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

    train = FFPPDataset(
        train_csv,
        transform=tf_train,
        **common,
    )

    val = FFPPDataset(
        val_csv,
        transform=tf_eval,
        **common,
    )

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

        val_sampler = DistributedSampler(
            val,
            num_replicas=world,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    def coll(b):
        return b

    num_workers = int(cfg.get("num_workers", 0) or 0)

    loader_kwargs = dict(
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=coll,
        persistent_workers=(num_workers > 0),
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
        print(
            f"[ddp datamodule] world={os.environ.get('WORLD_SIZE')} "
            f"train_total={len(train)} val_total={len(val)} "
            f"per_rank_train≈{len(train_loader.dataset) // int(os.environ.get('WORLD_SIZE', '1'))}"
        )

    return (
        train_loader,
        val_loader,
        _identity_gap_mode(train),
    )


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
