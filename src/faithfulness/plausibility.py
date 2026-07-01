# Path: src/faithfulness/plausibility.py
"""Plausibility = IoU between explanation mask and annotation mask.

Important:
  Plausibility is NOT faithfulness.
  A mask can overlap human annotations and still be non-causal for the detector.
"""

from __future__ import annotations


def plausibility_iou(pred_mask, gt_mask, thresh=0.5):
    """Return IoU(pred_mask, gt_mask) after thresholding.

    Supports torch tensors and numpy arrays.
    If tensor shapes differ, pred_mask is resized to gt_mask's spatial shape.
    """
    try:
        import torch
        import torch.nn.functional as F

        p = pred_mask
        g = gt_mask

        if not torch.is_tensor(p):
            p = torch.as_tensor(p)
        if not torch.is_tensor(g):
            g = torch.as_tensor(g)

        p = p.float()
        g = g.float()

        if p.dim() == 2:
            p = p.unsqueeze(0).unsqueeze(0)
        elif p.dim() == 3:
            p = p.unsqueeze(1)

        if g.dim() == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.dim() == 3:
            g = g.unsqueeze(1)

        if p.shape[-2:] != g.shape[-2:]:
            p = F.interpolate(p, size=g.shape[-2:], mode="bilinear", align_corners=False)

        if p.shape[0] != g.shape[0]:
            if p.shape[0] == 1:
                p = p.repeat(g.shape[0], 1, 1, 1)
            elif g.shape[0] == 1:
                g = g.repeat(p.shape[0], 1, 1, 1)
            else:
                n = min(p.shape[0], g.shape[0])
                p = p[:n]
                g = g[:n]

        pb = (p > thresh).float()
        gb = (g > thresh).float()

        inter = (pb * gb).flatten(1).sum(dim=1)
        union = ((pb + gb) > 0).float().flatten(1).sum(dim=1)

        return float((inter / (union + 1e-8)).mean().item())

    except Exception:
        import numpy as np

        p = np.asarray(pred_mask)
        g = np.asarray(gt_mask)

        p = np.squeeze(p)
        g = np.squeeze(g)

        if p.shape != g.shape:
            try:
                from PIL import Image

                p_img = Image.fromarray(p.astype("float32"))
                p_img = p_img.resize((g.shape[-1], g.shape[-2]), resample=Image.BILINEAR)
                p = np.asarray(p_img)
            except Exception:
                raise ValueError(f"Mask shapes differ and resize failed: pred={p.shape}, gt={g.shape}")

        pb = (p > thresh).astype(float)
        gb = (g > thresh).astype(float)

        inter = (pb * gb).sum()
        union = ((pb + gb) > 0).astype(float).sum()

        return float(inter / (union + 1e-8))
