# Path: src/audit/audit_runner.py
# Status: NEW
"""Run necessity/sufficiency audit for a set of (detector x explainer) on a dataset.
Produces per-sample rows + aggregate leaderboard rows. torch-guarded at call time."""
from __future__ import annotations
from typing import List, Dict
from ..interventions.interventions import apply_necessity, apply_sufficiency, mask_area
from ..faithfulness.faithfulness_score import compute_rift_score
from ..faithfulness.plausibility import plausibility_iou

def audit_one(image, adapter, explainer, *, intervention_mode="blur", topk_frac=0.12,
              donor=None, source_id=None, target_id=None, gt_mask=None, reward_weights=None):
    import torch
    mask=explainer.explain(image, adapter)
    with torch.no_grad():
        g0=adapter.identity_gap(image, donor=donor, source_id=source_id, target_id=target_id)
        l0=float(adapter.predict_logits(image).mean().item())
        nec=apply_necessity(image, mask, intervention_mode, topk_frac)
        suf=apply_sufficiency(image, mask, intervention_mode, topk_frac)
        gn=adapter.identity_gap(nec, donor=donor, source_id=source_id, target_id=target_id)
        gs=adapter.identity_gap(suf, donor=donor, source_id=source_id, target_id=target_id)
        ln=float(adapter.predict_logits(nec).mean().item())
        ls=float(adapter.predict_logits(suf).mean().item())
    comp=compute_rift_score(
        e0_delta=g0.value, e_nec_delta=gn.value, e_suf_delta=gs.value,
        e0_logit=l0, e_nec_logit=ln, e_suf_logit=ls,
        mask_area=mask_area(mask, topk_frac),
        plausibility_iou=(plausibility_iou(mask, gt_mask) if gt_mask is not None else None),
        identity_gap_mode=g0.mode.value, weights=reward_weights or {})
    row=comp.to_dict(); row["explainer"]=explainer.name
    return row, mask, nec, suf

def aggregate(rows: List[Dict]) -> Dict:
    if not rows: return {}
    keys=[k for k,v in rows[0].items() if isinstance(v,(int,float))]
    agg={k: sum(r[k] for r in rows if r.get(k) is not None)/max(1,len(rows)) for k in keys}
    agg["explainer"]=rows[0].get("explainer"); agg["n"]=len(rows)
    agg["identity_gap_mode"]=rows[0].get("identity_gap_mode")
    return agg
