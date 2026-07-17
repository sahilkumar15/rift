# Path: ablations/audit_batch.py
# Status: NEW
"""Batch-major auditing: load once, explain N ways, one mega-forward.

THE OPTIMISATION
----------------
Variant-major (the naive loop):
    for variant in variants:          # 7
        for batch in dataset:         # 13.4k images re-read EVERY variant
            e0   = forward(image)     # recomputed 7x, identical every time
            forward(nec); forward(suf)
  -> 7 x N decodes, 21 forwards per image.

Batch-major (this module):
    for batch in dataset:             # N decodes, ONCE, in DataLoader workers
        e0 = forward(image)           # ONCE - e0 does not depend on the variant
        masks = {v: explain(image) for v in variants}
        forward(cat([nec(m), suf(m) for m in masks]))   # ONE call, chunked
  -> N decodes, 1 + 2*V forwards per image.

For V=7: 21 -> 15 forwards (1.4x) and 7x fewer decodes. The decode saving
dominates in practice because PIL was the serial bottleneck.

Explainers that expose a COMPLETE evidence cache (PolicyExplainer already ran
the interventions internally during its rollout) are excluded from the mega
stack entirely - re-running their interventions would be pure waste.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

METRIC_KEYS = (
    "necessity_delta", "sufficiency_delta", "faithfulness_delta",
    "necessity_logit", "sufficiency_logit", "faithfulness_logit",
    "mask_area", "rift_score",
)


def audit_batch(
    *,
    adapter,
    image,
    donor,
    explainers: Dict[Any, Any],
    order: List[Any],
    intervention_mode: str,
    topk_frac: float,
    forward_batch_size: int,
    grid: int,
    weights: Dict[str, Any],
    dead: Dict[Any, str],
) -> Tuple[Dict[Any, Dict[str, List[float]]], Dict[Any, List[float]], Dict[Any, List[float]], str]:
    """Score every explainer in `order` on one batch.

    Returns (per_key_metrics, per_key_valid_delta, per_key_valid_logit, mode).
    """
    import torch

    from ablations.lib.explainers import logit_to_evidence, predict_evidence
    from src.faithfulness.faithfulness_score import compute_rift_score_tensor
    from src.interventions.interventions import apply_necessity, apply_sufficiency

    batch = int(image.shape[0])
    min_ev = float(weights.get("min_evidence", 0.0) or 0.0)

    # ---- e0 ONCE for the whole batch, shared by every variant ---------------
    raw0, gap0, mode, _ = predict_evidence(
        adapter, image, donor, max_batch=forward_batch_size, return_features=False
    )
    e0_g = gap0.float().view(-1)
    e0_l = logit_to_evidence(raw0).float().view(-1)

    # ---- masks for every variant on the already-loaded batch ----------------
    # ---- FAULT ISOLATION -------------------------------------------------
    # Batch-major runs every explainer inside ONE loop, so an exception from any
    # single explainer used to abort the whole sweep and mark EVERY row failed -
    # e.g. a GradCAM gradient error stamped its message onto 'Random cells',
    # which does not use gradients at all. A failing explainer is now recorded in
    # `dead` and skipped for this and all later batches; the others continue.
    masks: Dict[Any, Any] = {}
    cached: Dict[Any, Any] = {}
    for key in list(order):
        if key in dead:
            continue
        ex = explainers[key]
        try:
            masks[key] = ex.explain(image, adapter, donor=donor)
            if hasattr(ex, "cached_original_evidence"):
                c = ex.cached_original_evidence(image)
                if c is not None and bool(c.get("complete", False)):
                    cached[key] = c
        except Exception as exc:
            dead[key] = f"{type(exc).__name__}: {exc}"
            masks.pop(key, None)
            cached.pop(key, None)

    order = [k for k in order if k in masks]
    if not order:
        return {}, {}, {}, "proxy"

    # ---- ONE mega-forward for every variant that still needs interventions --
    need = [k for k in order if k not in cached]
    results: Dict[Any, Dict[str, Any]] = {}

    if need:
        pieces = []
        for key in need:
            pieces.append(apply_necessity(image, masks[key], intervention_mode, topk_frac))
            pieces.append(apply_sufficiency(image, masks[key], intervention_mode, topk_frac))
        stacked = torch.cat(pieces, dim=0)
        tiled = None
        if donor is not None:
            tiled = donor.repeat(len(pieces), *([1] * (donor.dim() - 1)))
        raw, gaps, _, _ = predict_evidence(
            adapter, stacked, tiled, max_batch=forward_batch_size, return_features=False
        )
        ev = logit_to_evidence(raw)
        for i, key in enumerate(need):
            s = 2 * i * batch
            results[key] = {
                "gap_nec": gaps[s: s + batch], "gap_suf": gaps[s + batch: s + 2 * batch],
                "logit_nec": ev[s: s + batch], "logit_suf": ev[s + batch: s + 2 * batch],
            }

    for key, c in cached.items():
        results[key] = {
            "gap_nec": c["gap_nec"].float().view(-1), "gap_suf": c["gap_suf"].float().view(-1),
            "logit_nec": c["logit_nec"].float().view(-1), "logit_suf": c["logit_suf"].float().view(-1),
        }

    # ---- score ---------------------------------------------------------------
    out_metrics: Dict[Any, Dict[str, List[float]]] = {}
    out_vd: Dict[Any, List[float]] = {}
    out_vl: Dict[Any, List[float]] = {}

    vd = (e0_g > min_ev).float().cpu().view(-1).tolist()
    vl = (e0_l > min_ev).float().cpu().view(-1).tolist()

    for key in order:
        m = masks[key]
        binary = (m.float() > 1e-6).float()
        area = binary.flatten(1).mean(dim=1)
        sel = binary.flatten(1).sum(dim=1) / float(binary.shape[-1] * binary.shape[-2]) * float(grid * grid)
        r = results[key]
        _, comps = compute_rift_score_tensor(
            e0_delta=e0_g, e_nec_delta=r["gap_nec"], e_suf_delta=r["gap_suf"],
            e0_logit=e0_l, e_nec_logit=r["logit_nec"], e_suf_logit=r["logit_suf"],
            mask_area=area, selected_cells=sel,
            identity_gap_mode=mode, weights=weights,
        )
        out_metrics[key] = {
            k: comps[k].detach().float().cpu().view(-1).tolist() for k in METRIC_KEYS
        }
        out_vd[key] = list(vd)
        out_vl[key] = list(vl)

    return out_metrics, out_vd, out_vl, str(mode)
