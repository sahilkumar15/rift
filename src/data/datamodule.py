# Path: src/data/datamodule.py
# Status: NEW
"""Bundles dataset + loaders. Falls back to a synthetic dataset when data.root is null,
so the pipeline is runnable for smoke tests without FF++."""
from __future__ import annotations
from .ffpp_dataset import FFPPDataset
def build_dataloaders(cfg):
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception:
        raise RuntimeError("torch required (Katz).")
    from .transforms import build_transform
    from PIL import Image
    def loader(p): return Image.open(p).convert("RGB")
    tf=build_transform(cfg.get("image_size",256))
    if not cfg.get("train_csv"):
        return _synthetic_loaders(cfg)
    train=FFPPDataset(cfg["train_csv"], image_loader=loader, transform=tf,
                      strict_identity_gap=cfg.get("strict_identity_gap",False))
    val=FFPPDataset(cfg["val_csv"], image_loader=loader, transform=tf,
                    strict_identity_gap=cfg.get("strict_identity_gap",False))
    def coll(b): return b
    return (DataLoader(train,batch_size=cfg.get("batch_size",8),shuffle=True,collate_fn=coll),
            DataLoader(val,batch_size=cfg.get("batch_size",8),shuffle=False,collate_fn=coll),
            train.identity_gap_mode)
def _synthetic_loaders(cfg):
    import torch
    from torch.utils.data import DataLoader, Dataset
    class Synth(Dataset):
        def __init__(self,n=32,size=256): self.n=n; self.size=size
        def __len__(self): return self.n
        def __getitem__(self,i):
            return {"image":torch.rand(3,self.size,self.size),
                    "label":i%2, "sample":None}
    def coll(b): return b
    ds=Synth(size=cfg.get("image_size",256))
    return (DataLoader(ds,batch_size=cfg.get("batch_size",8),collate_fn=coll),
            DataLoader(ds,batch_size=cfg.get("batch_size",8),collate_fn=coll),
            "proxy")
