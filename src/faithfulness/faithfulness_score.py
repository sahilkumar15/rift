# Path: src/faithfulness/faithfulness_score.py
"""RIFT causal-faithfulness score.

RIFT treats an explanation as a falsifiable causal claim.

Given evidence E0 on the original image:
  Necessity:
    remove the cited region.
    If the evidence collapses, the cited region was necessary.

  Sufficiency:
    keep only the cited region.
    If the evidence remains, the cited region was sufficient.

Faithfulness is the harmonic mean of necessity and sufficiency, so a method must
satisfy both. This penalizes plausible-but-non-causal masks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

EPS = 1e-8


def _to_float(x) -> float:
    try:
        import torch

        if torch.is_tensor(x):
            return float(x.detach().float().mean().item())
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return float(np.asarray(x).mean())
    except Exception:
        pass

    return float(x)


def _clip01(x: float) -> float:
    x = _to_float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def necessity(e0: float, e_nec: float, floor: float = 0.0) -> float:
    """Normalized evidence drop after removing the cited region.

    High means the cited region was necessary.
    """
    e0 = _to_float(e0)
    e_nec = _to_float(e_nec)
    floor = _to_float(floor)

    denom = (e0 - floor) + EPS

    if denom <= EPS:
        return 0.0

    return _clip01((e0 - e_nec) / denom)


def sufficiency(e0: float, e_suf: float, floor: float = 0.0) -> float:
    """Normalized evidence retained when only the cited region is kept.

    High means the cited region was sufficient.
    """
    e0 = _to_float(e0)
    e_suf = _to_float(e_suf)
    floor = _to_float(floor)

    denom = (e0 - floor) + EPS

    if denom <= EPS:
        return 0.0

    return _clip01((e_suf - floor) / denom)


def harmonic(nec: float, suf: float) -> float:
    """Harmonic mean. Zero if either necessity or sufficiency is zero."""
    nec = _to_float(nec)
    suf = _to_float(suf)

    if nec <= 0.0 or suf <= 0.0:
        return 0.0

    return float(2.0 * nec * suf / (nec + suf + EPS))


@dataclass
class FaithfulnessComponents:
    necessity_delta: float
    sufficiency_delta: float
    faithfulness_delta: float

    necessity_logit: float
    sufficiency_logit: float
    faithfulness_logit: float

    mask_area: float
    identity_preservation: float
    perceptual_distance: float
    plausibility_iou: Optional[float]

    rift_score: float
    identity_gap_mode: str

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items()}





















def compute_rift_score(
    *,
    e0_delta: float, e_nec_delta: float, e_suf_delta: float, delta_floor: float = 0.0,
    e0_logit: float, e_nec_logit: float, e_suf_logit: float, logit_floor: float = 0.0,
    mask_area: float,
    identity_preservation: float = 1.0,
    perceptual_distance: float = 0.0,
    plausibility_iou: Optional[float] = None,
    identity_gap_mode: str = "proxy",
    weights: Optional[Dict[str, float]] = None,
) -> FaithfulnessComponents:
    """
    Compute RIFT causal-faithfulness score.

    objective:
      harmonic     = harmonic mean of necessity and sufficiency
      necessity    = necessity-only ablation
      sufficiency  = sufficiency-only ablation
      none/off     = no faithfulness objective
    """
    w = {
        "w_delta": 1.0,
        "w_logit": 0.5,
        "w_sparsity": 0.3,
        "w_identity": 0.3,
        "w_perceptual": 0.2,
        "w_plausibility": 0.0,
        "objective": "harmonic",
    }

    if weights:
        w.update(weights)

    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
    suf_d = sufficiency(e0_delta, e_suf_delta, delta_floor)

    nec_l = necessity(e0_logit, e_nec_logit, logit_floor)
    suf_l = sufficiency(e0_logit, e_suf_logit, logit_floor)

    objective = str(w.get("objective", "harmonic")).lower()

    if objective in ("necessity", "necessity_only", "nec"):
        faith_d = nec_d
        faith_l = nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d = suf_d
        faith_l = suf_l
    elif objective in ("none", "off"):
        faith_d = 0.0
        faith_l = 0.0
    else:
        faith_d = harmonic(nec_d, suf_d)
        faith_l = harmonic(nec_l, suf_l)

    # HONESTY GUARD: proxy mode cannot claim mechanism faithfulness.
    if identity_gap_mode != "true":
        w["w_delta"] = 0.0

    reward = float(w["w_delta"]) * faith_d + float(w["w_logit"]) * faith_l
    reward -= float(w["w_sparsity"]) * mask_area
    reward -= float(w["w_identity"]) * (1.0 - identity_preservation)
    reward -= float(w["w_perceptual"]) * perceptual_distance

    if plausibility_iou is not None:
        reward += float(w["w_plausibility"]) * plausibility_iou

    return FaithfulnessComponents(
        necessity_delta=nec_d,
        sufficiency_delta=suf_d,
        faithfulness_delta=faith_d,
        necessity_logit=nec_l,
        sufficiency_logit=suf_l,
        faithfulness_logit=faith_l,
        mask_area=mask_area,
        identity_preservation=identity_preservation,
        perceptual_distance=perceptual_distance,
        plausibility_iou=plausibility_iou,
        rift_score=float(reward),
        identity_gap_mode=identity_gap_mode,
    )
