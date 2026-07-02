# Path: src/interventions/interventions.py
# Status: NEW
"""
interventions.py — necessity/sufficiency masking + GATE-1 validity probe.

Necessity: remove the cited region (zero/mean/blur), recompute evidence.
Sufficiency: keep ONLY the cited region, mask the rest, recompute evidence.

GATE-1 (gate1_intervention_validity) is the experiment EVERYTHING depends on:
does masking a cited pixel region actually move the evidence measurably and in
the right direction? If masking barely changes Δ/logit, every RIFT number is
noise and the paper must pivot. Run gate-1 on ~50 images BEFORE any training.

torch paths are guarded so the file imports without torch (sandbox); run for
real on Katz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Dict

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_binary(mask, topk_frac: float):
    """Convert a soft or grid mask to a binary intervention mask.

    RIFT policies produce accumulated 0/1 grid masks. For those masks the
    intended intervention region is exactly the selected cells, not an arbitrary
    top-k subset. For soft maps from Grad-CAM/external explainers, keep the
    top-k fraction per sample.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch required at runtime (Katz).")

    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # Hard policy masks are already binary after nearest-neighbor upsampling.
    # Treat every positive selected cell as part of the intervention.
    with torch.no_grad():
        hard_like = bool(((mask <= 1e-6) | (mask >= 1.0 - 1e-6)).all().item())
    if hard_like:
        return (mask > 1e-6).float()

    flat = mask.flatten(1)
    k = max(1, min(flat.shape[1], int(round(float(topk_frac) * flat.shape[1]))))
    thresh = flat.topk(k, dim=1).values[:, -1:].clamp(min=1e-12)
    return (flat >= thresh).view_as(mask).float()


def apply_necessity(image, mask, mode: str = "blur", topk_frac: float = 0.1):
    """Remove cited region: image where mask=0 stays, mask=1 region is destroyed."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required at runtime (Katz).")
    m = _to_binary(mask, topk_frac)
    if mode == "zero":
        return image * (1 - m)
    if mode == "mean":
        mean = image.mean(dim=(2, 3), keepdim=True)
        return image * (1 - m) + mean * m
    if mode == "blur":
        k = 15
        blurred = F.avg_pool2d(image, k, stride=1, padding=k // 2)
        return image * (1 - m) + blurred * m
    raise ValueError(mode)


def apply_sufficiency(image, mask, mode: str = "blur", topk_frac: float = 0.1):
    """Keep ONLY cited region: invert the necessity operation."""
    if not _HAS_TORCH:
        raise RuntimeError("torch required at runtime (Katz).")
    m = _to_binary(mask, topk_frac)
    if mode == "zero":
        return image * m
    if mode == "mean":
        mean = image.mean(dim=(2, 3), keepdim=True)
        return image * m + mean * (1 - m)
    if mode == "blur":
        k = 15
        blurred = F.avg_pool2d(image, k, stride=1, padding=k // 2)
        return image * m + blurred * (1 - m)
    raise ValueError(mode)


def mask_area(mask, topk_frac: float = 0.1) -> float:
    m = _to_binary(mask, topk_frac)
    return m.flatten(1).mean(dim=1).mean().item()


@dataclass
class Gate1Report:
    n: int
    mean_necessity_drop: float       # how much evidence falls when cited region removed
    mean_sufficiency_retained: float # how much remains when only cited region kept
    mean_random_drop: float          # SAME test with RANDOM masks (control)
    separation: float                # necessity_drop - random_drop ; must be >> 0
    verdict: str

    def passed(self, min_sep: float = 0.15) -> bool:
        return self.separation >= min_sep


def gate1_intervention_validity(
    images,                              # iterable of (B,3,H,W) tensors
    cited_masks,                         # matching iterable of (B,1,H,W) cited maps
    evidence_fn: Callable,               # x -> scalar evidence (Δ.value or logit)
    mode: str = "blur",
    topk_frac: float = 0.1,
    rng_seed: int = 0,
) -> Gate1Report:
    """
    THE precondition test. For each batch: compute E0, E after removing cited region
    (necessity), E keeping only cited (sufficiency), and E after removing a RANDOM
    region of equal area (control). If cited-removal drops evidence much more than
    random-removal, interventions are valid and RIFT is measurable. Else: pivot.
    """
    if not _HAS_TORCH:
        raise RuntimeError("Run gate-1 on Katz with torch live.")
    torch.manual_seed(rng_seed)
    nec_drops, suf_rets, rand_drops, n = [], [], [], 0
    for img, cm in zip(images, cited_masks):
        e0 = float(evidence_fn(img))
        if abs(e0) < 1e-6:
            continue
        e_nec = float(evidence_fn(apply_necessity(img, cm, mode, topk_frac)))
        e_suf = float(evidence_fn(apply_sufficiency(img, cm, mode, topk_frac)))
        rand = torch.rand_like(cm)
        e_rand = float(evidence_fn(apply_necessity(img, rand, mode, topk_frac)))
        nec_drops.append((e0 - e_nec) / (abs(e0) + 1e-8))
        suf_rets.append(e_suf / (e0 + 1e-8))
        rand_drops.append((e0 - e_rand) / (abs(e0) + 1e-8))
        n += img.shape[0]
    if not nec_drops:
        return Gate1Report(0, 0, 0, 0, 0, "NO valid samples (all evidence ~0)")
    mean = lambda a: sum(a) / len(a)
    nd, sr, rd = mean(nec_drops), mean(suf_rets), mean(rand_drops)
    sep = nd - rd
    verdict = (
        f"PASS: cited-removal drops evidence {nd:.3f} vs random {rd:.3f} "
        f"(sep={sep:.3f}). Interventions are valid -> RIFT measurable."
        if sep >= 0.15 else
        f"FAIL: cited-removal ({nd:.3f}) ~ random ({rd:.3f}), sep={sep:.3f}. "
        f"Pixel masks don't move evidence specifically -> RIFT numbers would be noise. PIVOT."
    )
    return Gate1Report(n, nd, sr, rd, sep, verdict)
