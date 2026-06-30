# Path: src/faithfulness/plausibility.py
# Status: NEW
"""Plausibility = IoU between explanation mask and annotation mask (if present)."""
from __future__ import annotations
def plausibility_iou(pred_mask, gt_mask, thresh=0.5):
    try:
        import torch
        p = (pred_mask > thresh).float(); g = (gt_mask > thresh).float()
        inter = (p*g).sum(); union = ((p+g) > 0).float().sum()
        return float((inter/(union+1e-8)).item())
    except Exception:
        import numpy as np
        p = (np.asarray(pred_mask) > thresh).astype(float)
        g = (np.asarray(gt_mask) > thresh).astype(float)
        inter = (p*g).sum(); union = ((p+g) > 0).sum()
        return float(inter/(union+1e-8))
