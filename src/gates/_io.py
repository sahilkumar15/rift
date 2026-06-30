# Path: iganer/rift/gates/_io.py
# Status: NEW
"""Tiny IO helpers for the Phase-0 gate scripts. torch/PIL guarded so the module
imports without them (sandbox); the loaders themselves require torch+PIL on Katz.

Convention: CIFT consumes BCHW float images in [-1, 1] (see GuideNet._preprocess_img).
"""
from __future__ import annotations


def load_image_minus1_1(path: str, size: int = 256, device: str = "cuda"):
    """Load an image as (1,3,size,size) float in [-1,1] on `device`."""
    import torch
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype="float32") / 255.0          # [0,1] HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    t = t * 2.0 - 1.0                                        # -> [-1,1]
    return t.to(device)


def load_mask(path: str, like, size: int = 256):
    """Load a soft mask as (1,1,H,W) in [0,1], resized to match `like`'s H,W."""
    import torch
    from PIL import Image
    import numpy as np
    H, W = int(like.shape[-2]), int(like.shape[-1])
    m = Image.open(path).convert("L").resize((W, H))
    arr = np.asarray(m, dtype="float32") / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)     # (1,1,H,W)
    return t.to(like.device)
