# Path: iganer/rift/utils/visualization.py
# Status: NEW
"""Save audit panels: original | explanation | necessity | sufficiency. torch/PIL guarded."""
from __future__ import annotations
import os
def save_audit_panel(image, expl_map, nec_img, suf_img, metrics, out_path):
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        print("[viz] PIL/numpy missing; skipping panel"); return None
    def to_np(t):
        try:
            import torch
            if hasattr(t, "detach"): t = t.detach().cpu().float()
            a = t.numpy() if hasattr(t, "numpy") else t
        except Exception:
            a = t
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[0] in (1, 3): a = a.transpose(1, 2, 0)
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        if a.ndim == 2: a = np.stack([a]*3, -1)
        if a.shape[-1] == 1: a = np.repeat(a, 3, -1)
        return (a*255).astype("uint8")
    panels = [to_np(image), to_np(expl_map), to_np(nec_img), to_np(suf_img)]
    h = min(p.shape[0] for p in panels)
    panels = [p[:h] for p in panels]
    strip = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(strip).save(out_path)
    return out_path
