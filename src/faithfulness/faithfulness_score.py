# Path: src/faithfulness/faithfulness_score.py
"""RIFT causal-faithfulness score — single source of truth.

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

IMPORTANT CONTRACTS

1. Evidence must be non-negative. Raw detector logits are signed, so callers
   must map logits through logit_to_evidence() exactly once before scoring.

2. Faithfulness of a "why fake" explanation is undefined when the detector has
   no fake evidence. min_evidence gates these samples out and valid_frac_*
   reports how many samples contributed.

3. With fixed H4 and forbid_revisit=True, selected_cells and mask_area are
   constants by design. Sparsity only becomes a learnable objective when
   allow_stop=True or horizon/mask size can vary.

4. Proxy identity-gap mode can never earn delta credit. This prevents reporting
   donor-gap faithfulness when real donor evidence is not available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

EPS = 1e-8

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def logit_to_evidence(x):
    """Map a signed detector logit to non-negative evidence.

    Softplus is monotone, strictly positive, and preserves large-logit ordering
    better than sigmoid saturation.
    """
    if _HAS_TORCH and torch.is_tensor(x):
        return F.softplus(x.float())

    import math
    xf = float(x)
    return math.log1p(math.exp(-abs(xf))) + max(xf, 0.0)


def _default_weights() -> Dict[str, object]:
    return {
        "w_delta": 1.0,
        "w_logit": 0.5,
        "w_sparsity": 0.3,
        "w_identity": 0.0,
        "w_perceptual": 0.0,
        "w_plausibility": 0.0,
        "objective": "harmonic",
        # sparsity shaping
        "sparsity_mode": "linear",  # linear | hinge
        "area_lo": 0.02,
        "area_hi": 0.35,
        # evidence validity gating
        "min_evidence": 0.0,
        # degenerate-mask guards; important when allow_stop=True
        "min_selected_cells": 1.0,
        "w_min_cells": 0.10,
        "empty_mask_penalty": 0.25,
    }


def _to_float(x) -> float:
    try:
        import torch as _t
        if _t.is_tensor(x):
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
    """Normalized evidence drop after removing the cited region."""
    e0 = _to_float(e0)
    e_nec = _to_float(e_nec)
    floor = _to_float(floor)
    denom = (e0 - floor) + EPS
    if denom <= EPS:
        return 0.0
    return _clip01((e0 - e_nec) / denom)


def sufficiency(e0: float, e_suf: float, floor: float = 0.0) -> float:
    """Normalized evidence retained when only the cited region is kept."""
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


def _sparsity_penalty_scalar(area: float, w: Dict[str, object]) -> float:
    ws = float(w["w_sparsity"])
    if str(w.get("sparsity_mode", "linear")).lower() == "hinge":
        lo = float(w.get("area_lo", 0.02))
        hi = float(w.get("area_hi", 0.35))
        return ws * (max(0.0, lo - area) + max(0.0, area - hi))
    return ws * area


def _size_penalty_scalar(area: float, selected_cells: Optional[float], w: Dict[str, object]) -> tuple[float, float]:
    if selected_cells is None:
        selected_cells = 0.0 if area <= 1e-6 else float("inf")
    selected_cells = _to_float(selected_cells)
    min_cells = float(w.get("min_selected_cells", 1.0) or 0.0)
    if min_cells > 0 and selected_cells != float("inf"):
        small = max(0.0, (min_cells - selected_cells) / max(min_cells, EPS))
    else:
        small = 0.0
    small_penalty = float(w.get("w_min_cells", 0.0) or 0.0) * small
    empty_penalty = float(w.get("empty_mask_penalty", 0.0) or 0.0) if selected_cells <= 0 or area <= 1e-6 else 0.0
    return small_penalty, empty_penalty


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
    e0_delta: float,
    e_nec_delta: float,
    e_suf_delta: float,
    delta_floor: float = 0.0,
    e0_logit: float,
    e_nec_logit: float,
    e_suf_logit: float,
    logit_floor: float = 0.0,
    mask_area: float,
    identity_preservation: float = 1.0,
    perceptual_distance: float = 0.0,
    plausibility_iou: Optional[float] = None,
    selected_cells: Optional[float] = None,
    identity_gap_mode: str = "proxy",
    weights: Optional[Dict[str, float]] = None,
) -> FaithfulnessComponents:
    """Scalar RIFT score.

    Logit inputs must already be non-negative evidence. Use
    logit_to_evidence(raw_logit) before calling this function.
    """
    w = _default_weights()
    if weights:
        w.update(weights)

    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
    suf_d = sufficiency(e0_delta, e_suf_delta, delta_floor)
    nec_l = necessity(e0_logit, e_nec_logit, logit_floor)
    suf_l = sufficiency(e0_logit, e_suf_logit, logit_floor)

    objective = str(w.get("objective", "harmonic")).lower()
    if objective in ("necessity", "necessity_only", "nec"):
        faith_d, faith_l = nec_d, nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d, faith_l = suf_d, suf_l
    elif objective in ("none", "off"):
        faith_d, faith_l = 0.0, 0.0
    else:
        faith_d = harmonic(nec_d, suf_d)
        faith_l = harmonic(nec_l, suf_l)

    min_ev = float(w.get("min_evidence", 0.0) or 0.0)
    if min_ev > 0.0:
        if _to_float(e0_delta) <= min_ev:
            faith_d = 0.0
        if _to_float(e0_logit) <= min_ev:
            faith_l = 0.0

    w_delta = float(w["w_delta"]) if str(identity_gap_mode) == "true" else 0.0
    reward = w_delta * faith_d + float(w["w_logit"]) * faith_l

    area = _to_float(mask_area)
    reward -= _sparsity_penalty_scalar(area, w)
    small_penalty, empty_penalty = _size_penalty_scalar(area, selected_cells, w)
    reward -= small_penalty + empty_penalty
    reward -= float(w["w_identity"]) * (1.0 - _to_float(identity_preservation))
    reward -= float(w["w_perceptual"]) * _to_float(perceptual_distance)
    if plausibility_iou is not None:
        reward += float(w["w_plausibility"]) * _to_float(plausibility_iou)

    return FaithfulnessComponents(
        necessity_delta=nec_d,
        sufficiency_delta=suf_d,
        faithfulness_delta=faith_d,
        necessity_logit=nec_l,
        sufficiency_logit=suf_l,
        faithfulness_logit=faith_l,
        mask_area=area,
        identity_preservation=_to_float(identity_preservation),
        perceptual_distance=_to_float(perceptual_distance),
        plausibility_iou=plausibility_iou,
        rift_score=float(reward),
        identity_gap_mode=identity_gap_mode,
    )


def _nec_t(e0, e1, floor: float = 0.0):
    denom = (e0 - float(floor)) + EPS
    raw = (e0 - e1) / denom.clamp_min(EPS)
    return torch.where(denom <= EPS, torch.zeros_like(raw), raw.clamp(0.0, 1.0))


def _suf_t(e0, e1, floor: float = 0.0):
    denom = (e0 - float(floor)) + EPS
    raw = (e1 - float(floor)) / denom.clamp_min(EPS)
    return torch.where(denom <= EPS, torch.zeros_like(raw), raw.clamp(0.0, 1.0))


def _harmonic_t(a, b):
    h = 2.0 * a * b / (a + b + EPS)
    return torch.where((a > 0) & (b > 0), h, torch.zeros_like(h))


def _sparsity_penalty_t(area, w):
    ws = float(w["w_sparsity"])
    if str(w.get("sparsity_mode", "linear")).lower() == "hinge":
        lo = float(w.get("area_lo", 0.02))
        hi = float(w.get("area_hi", 0.35))
        return ws * ((lo - area).clamp_min(0.0) + (area - hi).clamp_min(0.0))
    return ws * area


def _size_penalty_t(mask_area, selected_cells, w):
    if selected_cells is None:
        # Backward-compatible fallback: if the caller only supplies an area,
        # penalize true empty masks but do not invent a too-small-cell count for
        # positive-area masks. BatchedRIFTEnv passes exact selected_cells.
        min_cells_default = float(w.get("min_selected_cells", 1.0) or 1.0)
        selected_cells = torch.where(
            mask_area <= 1e-6,
            torch.zeros_like(mask_area),
            torch.full_like(mask_area, min_cells_default),
        )
    selected_cells = selected_cells.to(mask_area.device).float()
    min_cells = float(w.get("min_selected_cells", 1.0) or 0.0)
    if min_cells > 0:
        too_small = torch.clamp((min_cells - selected_cells) / max(min_cells, EPS), min=0.0)
    else:
        too_small = torch.zeros_like(mask_area)
    small_mask_penalty = float(w.get("w_min_cells", 0.0) or 0.0) * too_small
    empty_mask_penalty = float(w.get("empty_mask_penalty", 0.0) or 0.0) * ((selected_cells <= 0) | (mask_area <= 1e-6)).float()
    return selected_cells, small_mask_penalty, empty_mask_penalty


def compute_rift_score_tensor(
    *,
    e0_delta,
    e_nec_delta,
    e_suf_delta,
    e0_logit,
    e_nec_logit,
    e_suf_logit,
    mask_area,
    identity_gap_mode: str,
    identity_preservation=None,
    perceptual_distance=None,
    plausibility_iou=None,
    selected_cells=None,
    weights: Optional[Dict[str, float]] = None,
):
    """Vectorized RIFT reward with explicit component logging.

    All logit inputs must already be non-negative evidence. Returns
    (reward, comps), where tensor comps are per-sample except valid_frac_*.
    """
    assert _HAS_TORCH, "compute_rift_score_tensor requires torch."

    w = _default_weights()
    if weights:
        w.update(weights)

    nec_d = _nec_t(e0_delta, e_nec_delta)
    suf_d = _suf_t(e0_delta, e_suf_delta)
    nec_l = _nec_t(e0_logit, e_nec_logit)
    suf_l = _suf_t(e0_logit, e_suf_logit)

    objective = str(w.get("objective", "harmonic")).lower()
    if objective in ("necessity", "necessity_only", "nec"):
        faith_d, faith_l = nec_d, nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d, faith_l = suf_d, suf_l
    elif objective in ("none", "off"):
        faith_d = torch.zeros_like(nec_d)
        faith_l = torch.zeros_like(nec_l)
    else:
        faith_d = _harmonic_t(nec_d, suf_d)
        faith_l = _harmonic_t(nec_l, suf_l)

    min_ev = float(w.get("min_evidence", 0.0) or 0.0)
    valid_d = e0_delta > min_ev
    valid_l = e0_logit > min_ev
    faith_d = torch.where(valid_d, faith_d, torch.zeros_like(faith_d))
    faith_l = torch.where(valid_l, faith_l, torch.zeros_like(faith_l))

    w_delta = float(w["w_delta"]) if str(identity_gap_mode) == "true" else 0.0
    reward_delta_component = w_delta * faith_d
    reward_logit_component = float(w["w_logit"]) * faith_l
    sparsity_penalty = _sparsity_penalty_t(mask_area, w)
    selected_cells, small_mask_penalty, empty_mask_penalty = _size_penalty_t(mask_area, selected_cells, w)

    reward = reward_delta_component + reward_logit_component - sparsity_penalty - small_mask_penalty - empty_mask_penalty

    if identity_preservation is not None:
        reward = reward - float(w["w_identity"]) * (1.0 - identity_preservation)
    if perceptual_distance is not None:
        reward = reward - float(w["w_perceptual"]) * perceptual_distance
    if plausibility_iou is not None:
        reward = reward + float(w["w_plausibility"]) * plausibility_iou

    comps = {
        "rift_score": reward,
        "faithfulness_delta": faith_d,
        "faithfulness_logit": faith_l,
        "necessity_delta": nec_d,
        "sufficiency_delta": suf_d,
        "necessity_logit": nec_l,
        "sufficiency_logit": suf_l,
        "reward_delta_component": reward_delta_component,
        "reward_logit_component": reward_logit_component,
        "dense_delta": reward_delta_component,
        "dense_logit": reward_logit_component,
        "sparsity_penalty": sparsity_penalty,
        "small_mask_penalty": small_mask_penalty,
        "empty_mask_penalty": empty_mask_penalty,
        "mask_area": mask_area,
        "selected_cells_tensor": selected_cells,
        "valid_frac_delta": valid_d.float().mean(),
        "valid_frac_logit": valid_l.float().mean(),
    }
    return reward, comps
