"""Bundles dataset + loaders.

For real CIFT/RIFT runs, pass data.train_csv and data.val_csv.

Path handling:
  If the CSV contains placeholder /data/ffpp paths, pass:

    data.data_root=/actual/ffpp/root

  or:

    data.path_prefix_from=/data/ffpp
    data.path_prefix_to=/actual/ffpp/root
"""
from __future__ import annotations

from .ffpp_dataset import FFPPDataset


def build_dataloaders(cfg):
    try:
        import torch  # noqa: F401
        from torch.utils.data import DataLoader
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

    def coll(b):
        return b

    return (
        DataLoader(
            train,
            batch_size=cfg.get("batch_size", 8),
            shuffle=True,
            num_workers=cfg.get("num_workers", 0),
            collate_fn=coll,
        ),
        DataLoader(
            val,
            batch_size=cfg.get("batch_size", 8),
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            collate_fn=coll,
        ),
        train.identity_gap_mode,
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
