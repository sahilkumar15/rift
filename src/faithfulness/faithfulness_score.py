# Path: src/faithfulness/faithfulness_score.py
# Status: NEW
"""
faithfulness_score.py — the causal-faithfulness math, with ZERO model dependencies.

Definitions (all on a single sample, given pre-/post-intervention evidence):

  Let  E0   = evidence on the untouched image  (Δ for mechanism mode, or detector
              logit for logit mode).
       E_nec = evidence after the EXPLANATION region is REMOVED (necessity test).
       E_suf = evidence after only the explanation region is KEPT (sufficiency test).
       E_real_baseline = evidence on a genuine/real reference (Δ≈0 floor), optional.

Necessity  (did removing the cited evidence destroy the signal?):
    nec = clip( (E0 - E_nec) / (E0 - floor + eps), 0, 1 )
    high  -> removing the cited region collapses the signal -> region was necessary.

Sufficiency (does the cited region alone preserve the signal?):
    suf = clip( (E_suf - floor) / (E0 - floor + eps), 0, 1 )
    high  -> the cited region alone sustains the signal -> region was sufficient.

These are deliberately normalised to [0,1] and direction-agnostic to whether the
underlying evidence is a distance (Δ, higher=more forged) — caller passes the
correctly-signed scalars. faithfulness = harmonic mean of (nec, suf) so a method
must be BOTH necessary and sufficient to score high (penalises the MARE failure
mode of citing a plausible-but-non-causal superset).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict

EPS = 1e-8


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def necessity(e0: float, e_nec: float, floor: float = 0.0) -> float:
    denom = (e0 - floor) + EPS
    if denom <= EPS:                      # signal already at floor -> nothing to remove
        return 0.0
    return _clip01((e0 - e_nec) / denom)


def sufficiency(e0: float, e_suf: float, floor: float = 0.0) -> float:
    denom = (e0 - floor) + EPS
    if denom <= EPS:
        return 0.0
    return _clip01((e_suf - floor) / denom)


def harmonic(nec: float, suf: float) -> float:
    if nec <= 0.0 or suf <= 0.0:
        return 0.0
    return 2.0 * nec * suf / (nec + suf + EPS)


@dataclass
class FaithfulnessComponents:
    # mechanism (Δ) channel
    necessity_delta: float
    sufficiency_delta: float
    faithfulness_delta: float          # harmonic mean on Δ
    # detector-logit channel (works even in proxy mode; weaker claim)
    necessity_logit: float
    sufficiency_logit: float
    faithfulness_logit: float
    # constraints / costs
    mask_area: float                   # fraction of image cited (sparsity); lower better
    identity_preservation: float       # 1 = identity untouched by intervention
    perceptual_distance: float         # LPIPS-style; lower better
    plausibility_iou: Optional[float]  # vs annotation mask if available, else None
    # final
    rift_score: float
    identity_gap_mode: str             # "true" | "proxy" — propagated for honesty

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
    Combine the two channels + constraints into one configurable scalar.
    `weights` keys (all optional; sensible defaults):
        w_delta, w_logit, w_sparsity, w_identity, w_perceptual, w_plausibility
    In proxy mode w_delta is forced to 0 and the score leans on the logit channel,
    so you never get credit for a Δ you couldn't actually measure.
    """
    w = {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
    }
    if weights:
        w.update(weights)

    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
    suf_d = sufficiency(e0_delta, e_suf_delta, delta_floor)
    faith_d = harmonic(nec_d, suf_d)

    nec_l = necessity(e0_logit, e_nec_logit, logit_floor)
    suf_l = sufficiency(e0_logit, e_suf_logit, logit_floor)
    faith_l = harmonic(nec_l, suf_l)

    # HONESTY GUARD: proxy mode cannot claim mechanism faithfulness.
    if identity_gap_mode != "true":
        w["w_delta"] = 0.0

    reward = w["w_delta"] * faith_d + w["w_logit"] * faith_l
    reward -= w["w_sparsity"] * mask_area
    reward -= w["w_identity"] * (1.0 - identity_preservation)
    reward -= w["w_perceptual"] * perceptual_distance
    if plausibility_iou is not None:
        reward += w["w_plausibility"] * plausibility_iou

    return FaithfulnessComponents(
        necessity_delta=nec_d, sufficiency_delta=suf_d, faithfulness_delta=faith_d,
        necessity_logit=nec_l, sufficiency_logit=suf_l, faithfulness_logit=faith_l,
        mask_area=mask_area, identity_preservation=identity_preservation,
        perceptual_distance=perceptual_distance, plausibility_iou=plausibility_iou,
        rift_score=float(reward), identity_gap_mode=identity_gap_mode,
    )
