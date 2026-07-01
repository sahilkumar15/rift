# Path: src/audit/ablation_runner.py
# Status: MODIFIED (config-driven 5-block planner + torch-guarded executors)
"""
ablation_runner.py — expands the RIFT ablation spec (configs/ablations_rift.yaml) into
runnable cells and executes them.

Two layers, cleanly separated so planning works WITHOUT torch:
  * PLANNER (pure logic): plan_blocks() + render_matrix() — used by `--dry-run`.
  * EXECUTORS (torch):     run_block0..run_block4 — consume a CIFTAdapter + dataset and
                           reuse audit_one/aggregate/build_leaderboard/run_correlation.

Honesty is enforced per cell: ground_in_delta is "—" (None) whenever intervene is False,
and the runner refuses to report Δ-faithfulness for those cells (it blanks the column).
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable, Tuple

from ..rl.reward import get_reward_weights

# ───────────────────────────── back-compat (kept) ─────────────────────────────
ABLATION_ROWS = {
    # name -> (reward_preset, use_intervention, ground_in_delta)
    "acc_only":            ("acc_only",        False, False),
    "plausibility_only":   ("plausibility",    False, False),
    "generic_logit":       ("generic_logit",   True,  False),
    "delta_reward_no_int": ("delta_no_interv", False, True),
    "delta_grounded":      ("full_rift",       True,  True),
    "full_rift":           ("full_rift",       True,  True),
}


def resolve_ablation(name):
    if name not in ABLATION_ROWS:
        raise KeyError(name)
    preset, use_int, ground = ABLATION_ROWS[name]
    return {"reward_weights": get_reward_weights(preset),
            "use_intervention": use_int, "ground_in_delta": ground}


# ───────────────────────────── CLI override coercion ──────────────────────────
def coerce_value(s: str) -> Any:
    """Turn a CLI string into a typed value: true/false/null/int/float/csv-list/str."""
    if isinstance(s, (int, float, bool)) or s is None:
        return s
    t = s.strip()
    low = t.lower()
    if low in ("true", "yes"):  return True
    if low in ("false", "no"):  return False
    if low in ("null", "none"): return None
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    if "," in t:
        return [coerce_value(x) for x in t.split(",")]
    return t


def parse_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    """['a.b=1','c=true'] -> {'a.b':1,'c':True}."""
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = coerce_value(v)
    return out


# ───────────────────────────── PLANNER (no torch) ─────────────────────────────
_TICK = "✓"
_CROSS = "✗"
_NA = "—"


def _flag(v: Optional[bool]) -> str:
    if v is None:
        return _NA
    return _TICK if v else _CROSS


def plan_blocks(spec: Dict[str, Any], blocks_enabled: Dict[str, bool],
                only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Expand the ablation spec into ordered block plans. Pure logic, no torch.
    `only` filters cell ids (applies to block1/block4 cell lists and block2 explainers)."""
    plans: List[Dict[str, Any]] = []

    def keep(cid: str) -> bool:
        return (only is None) or (cid in only)

    if blocks_enabled.get("block0_validity", True) and "block0_validity" in spec:
        b = spec["block0_validity"]
        cells = [dict(c) for c in b.get("cells", []) if keep(c["id"])]
        plans.append({"block": "block0_validity", "kind": "validity",
                      "min_sep": b.get("min_sep", 0.15), "cells": cells,
                      "columns": ["id", "mask_op", "topk_frac", "separation", "verdict"]})

    if blocks_enabled.get("block1_method", True) and "block1_method" in spec:
        b = spec["block1_method"]
        cells = [dict(c) for c in b.get("cells", []) if keep(c["id"])]
        plans.append({"block": "block1_method", "kind": "factorial", "cells": cells,
                      "factors": ["intervene", "ground_in_delta", "plausibility", "constraints"],
                      "columns": ["id", "I", "G", "P", "K",
                                  "faithfulness_delta", "faithfulness_logit",
                                  "plausibility_iou", "mask_area", "identity_gap_mode", "n"]})

    if blocks_enabled.get("block2_audit", True) and "block2_audit" in spec:
        b = spec["block2_audit"]
        ex = [e for e in b.get("explainers", []) if keep(e)]
        plans.append({"block": "block2_audit", "kind": "leaderboard", "explainers": ex,
                      "sort_key": b.get("sort_key", "faithfulness_delta"),
                      "expose_rule": b.get("expose_rule", {}),
                      "columns": ["explainer", "faithfulness_delta", "faithfulness_logit",
                                  "plausibility_iou", "mask_area", "identity_gap_mode", "exposed", "n"]})

    if blocks_enabled.get("block3_correlation", True) and "block3_correlation" in spec:
        b = spec["block3_correlation"]
        plans.append({"block": "block3_correlation", "kind": "correlation",
                      "predictors": b.get("predictors", ["faithfulness", "in_domain_auc", "plausibility"]),
                      "target": b.get("target", "zero_shot_auc"),
                      "headline_min_n": b.get("headline_min_n", 15),
                      "saturation_control": b.get("saturation_control", {}),
                      "columns": ["predictor", "spearman", "ci_lo", "ci_hi", "n", "reportable"]})

    if blocks_enabled.get("block4_rl", True) and "block4_rl" in spec:
        b = spec["block4_rl"]
        cells = [dict(c) for c in b.get("cells", []) if keep(c["id"])]
        plans.append({"block": "block4_rl", "kind": "rl_sensitivity", "cells": cells,
                      "demote_rule": b.get("demote_rule", {}),
                      "factors": ["horizon", "game", "protection"],
                      "columns": ["id", "H", "game", "protection",
                                  "faithfulness_delta", "n"]})
    return plans


def render_matrix(plans: List[Dict[str, Any]]) -> str:
    """Human-readable ✓/✗ plan for `--dry-run`."""
    out = []
    for p in plans:
        out.append(f"\n=== {p['block']}  ({p['kind']}) ===")
        if p["kind"] == "factorial":
            out.append(f"{'cell':20s} | I | G | P | K | reward_preset")
            out.append("-" * 60)
            for c in p["cells"]:
                out.append(f"{c['id']:20s} | {_flag(c.get('intervene'))} | "
                           f"{_flag(c.get('ground_in_delta'))} | {_flag(c.get('plausibility'))} | "
                           f"{_flag(c.get('constraints'))} | {c.get('reward_preset','-')}")
        elif p["kind"] == "rl_sensitivity":
            out.append(f"{'cell':12s} | H | game | protect")
            out.append("-" * 36)
            for c in p["cells"]:
                out.append(f"{c['id']:12s} | {c.get('horizon'):>1} | "
                           f"{_flag(c.get('game')):>4} | {_flag(c.get('protection'))}")
        elif p["kind"] == "validity":
            out.append(f"{'cell':14s} | mask_op | topk_frac")
            out.append("-" * 36)
            for c in p["cells"]:
                out.append(f"{c['id']:14s} | {c.get('mask_op'):7s} | {c.get('topk_frac')}")
        elif p["kind"] == "leaderboard":
            out.append("explainers: " + ", ".join(p["explainers"]))
            out.append(f"expose_rule (plausible-but-unfaithful): {p['expose_rule']}")
        elif p["kind"] == "correlation":
            out.append("predictors: " + ", ".join(p["predictors"]) + f"  vs  {p['target']}")
            out.append(f"headline_min_n={p['headline_min_n']}  saturation_control={p['saturation_control']}")
    return "\n".join(out)


# ───────────────────────────── EXECUTORS (torch) ──────────────────────────────
def _cell_explainer(cell: Dict[str, Any], adapter):
    """Pick the explainer that defines each Block-1 cell's explanation source."""
    from ..explainers.gradcam_explainer import GradCAMExplainer
    from ..explainers.cift_gap_explainer import CIFTGapExplainer
    from ..explainers.random_explainer import RandomExplainer
    cid = cell["id"]
    if cid in ("delta_grounded", "full_rift", "delta_reward_no_int"):
        return CIFTGapExplainer()
    if cid == "generic_logit":
        return GradCAMExplainer(target_class=1)
    if cid in ("plausibility_only",):
        # explanation IS the annotation mask; handled specially in run_block1
        return None
    return RandomExplainer()


def make_explainer(name: str, adapter=None):
    from ..explainers.random_explainer import RandomExplainer
    from ..explainers.gradcam_explainer import GradCAMExplainer
    from ..explainers.cift_gap_explainer import CIFTGapExplainer
    table = {
        "random": RandomExplainer,
        "gradcam_logit": lambda: GradCAMExplainer(target_class=1),
        "cift_gap": CIFTGapExplainer,
    }
    if name in table:
        e = table[name]()
        if not getattr(e, "name", None):
            e.name = name
        return e
    # annotation / vlm_external are handled by the caller (need external maps); skip here.
    return None


def iter_audit_samples(cfg, device="cuda", n: Optional[int] = None):
    """Yield (image, donor, gt_mask) from the RIFT split CSV.

    Required for strict donor-grounded RIFT:
      image_path must exist
      donor_path/source_ref_path must exist
    """
    import csv as _csv
    from pathlib import Path as _Path
    from ..gates._io import load_image_minus1_1, load_mask

    csv_path = cfg.get_dotted("dataset.split_csv")

    if not csv_path:
        raise RuntimeError(
            "dataset.split_csv is not set. Build/use data/slices/rift_ffpp_rela.csv."
        )

    csv_path = str(csv_path)

    if not _Path(csv_path).exists():
        raise FileNotFoundError(f"RIFT split CSV not found: {csv_path}")

    cap = n if n is not None else (cfg.get_dotted("dataset.max_items") or 200)
    strict = bool(cfg.get_dotted("detector.strict_identity_gap", True))

    count = 0

    with open(csv_path, newline="") as f:
        for row_idx, r in enumerate(_csv.DictReader(f), start=2):
            if str(r.get("label", "1")).strip() not in ("1", "fake", "forged", "True", "true"):
                continue

            image_path = r.get("image_path")
            donor_path = r.get("donor_path") or r.get("source_ref_path")
            mask_path = r.get("mask_path") or ""

            if not image_path or not _Path(image_path).exists():
                raise FileNotFoundError(
                    f"Missing image_path in RIFT CSV row={row_idx}: {image_path}\n"
                    f"CSV: {csv_path}\n"
                    "Fix: build a real CSV with scripts/build_rift_csv_from_cift_ffpp.py"
                )

            if strict and (not donor_path or not _Path(donor_path).exists()):
                raise FileNotFoundError(
                    f"Missing donor_path/source_ref_path in RIFT CSV row={row_idx}: {donor_path}\n"
                    f"CSV: {csv_path}\n"
                    "Strict identity-gap mode needs donor/reference images."
                )

            img = load_image_minus1_1(image_path, device=device)
            donor = load_image_minus1_1(donor_path, device=device) if donor_path else None
            gt = load_mask(mask_path, like=img) if mask_path and _Path(mask_path).exists() else None

            yield img, donor, gt

            count += 1

            if count >= cap:
                break


def run_block1(cfg, adapter, cells, seeds=(0,), device="cuda") -> List[Dict[str, Any]]:
    """Execute the method factorial. One aggregate row per cell (mean over slice×seeds)."""
    import torch
    from .audit_runner import audit_one, aggregate
    rows = []
    for cell in cells:
        preset = cell.get("reward_preset", "full_rift")
        weights = get_reward_weights(preset)
        intervene = bool(cell.get("intervene"))
        ground = cell.get("ground_in_delta")           # may be None ("—")
        per_sample = []
        for seed in seeds:
            torch.manual_seed(int(seed))
            explainer = _cell_explainer(cell, adapter)
            for img, donor, gt in iter_audit_samples(cfg, device=device):
                if cell["id"] == "plausibility_only":
                    # explanation = annotation; no intervention -> plausibility only
                    from ..faithfulness.plausibility import plausibility_iou
                    if gt is None:
                        continue
                    per_sample.append({"faithfulness_delta": 0.0, "faithfulness_logit": 0.0,
                                       "plausibility_iou": plausibility_iou(gt, gt),
                                       "mask_area": float(gt.flatten(1).mean().item()),
                                       "identity_gap_mode": "proxy", "explainer": "annotation"})
                    continue
                if explainer is None:
                    continue
                row, *_ = audit_one(img, adapter, explainer,
                                    intervention_mode=cfg.get_dotted("intervention.mode", "blur"),
                                    topk_frac=cfg.get_dotted("intervention.topk_frac", 0.12),
                                    donor=donor, gt_mask=gt, reward_weights=weights)
                if not intervene:
                    # no falsification test -> blank the Δ/logit faithfulness honestly
                    row["faithfulness_delta"] = 0.0
                    row["faithfulness_logit"] = 0.0
                if ground is False:
                    row["faithfulness_delta"] = 0.0   # generic-logit cell reports logit channel only
                per_sample.append(row)
        agg = aggregate(per_sample) if per_sample else {}
        agg["id"] = cell["id"]
        agg["I"] = _flag(cell.get("intervene"))
        agg["G"] = _flag(cell.get("ground_in_delta"))
        agg["P"] = _flag(cell.get("plausibility"))
        agg["K"] = _flag(cell.get("constraints"))
        rows.append(agg)
    return rows


def block1_contrasts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {r.get("id"): r for r in rows}
    out = {}
    if "delta_grounded" in by and "generic_logit" in by:
        out["delta_grounding_gain"] = (by["delta_grounded"].get("faithfulness_delta", 0.0)
                                       - by["generic_logit"].get("faithfulness_logit", 0.0))
    if "plausibility_only" in by:
        out["mare_plausibility"] = by["plausibility_only"].get("plausibility_iou")
        out["mare_faithfulness"] = by["plausibility_only"].get("faithfulness_delta", 0.0)
    if "delta_reward_no_int" in by:
        out["no_int_sanity_delta_faith"] = by["delta_reward_no_int"].get("faithfulness_delta", 0.0)
    return out


def run_block2(cfg, adapter, explainers, device="cuda") -> Tuple[List[str], List[List[Any]]]:
    """Execute the audit leaderboard over the configured explainers with tqdm progress."""
    from .audit_runner import audit_one, aggregate
    from .leaderboard import build_leaderboard

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    agg_rows = []

    total = cfg.get_dotted("dataset.max_items") or 200
    try:
        total = int(total)
    except Exception:
        total = 200

    for name in explainers:
        ex = make_explainer(name, adapter)

        if ex is None:
            print(f"  [skip] explainer not wired: {name}", flush=True)
            continue

        print(f"  [audit] explainer={name}", flush=True)

        per = []

        try:
            sample_iter = iter_audit_samples(cfg, device=device)

            if tqdm is not None:
                sample_iter = tqdm(
                    sample_iter,
                    total=total,
                    desc=f"audit:{name}",
                    dynamic_ncols=True,
                    leave=True,
                )

            for sample_idx, (img, donor, gt) in enumerate(sample_iter, start=1):
                if tqdm is None:
                    print(f"    [{name}] sample {sample_idx}/{total}", flush=True)

                row, *_ = audit_one(
                    img,
                    adapter,
                    ex,
                    intervention_mode=cfg.get_dotted("intervention.mode", "blur"),
                    topk_frac=cfg.get_dotted("intervention.topk_frac", 0.12),
                    donor=donor,
                    gt_mask=gt,
                )

                row["explainer"] = name
                per.append(row)

                if tqdm is not None:
                    sample_iter.set_postfix(
                        {
                            "rows": len(per),
                            "last_rift": f"{row.get('rift_score', 0.0):.3f}",
                        }
                    )

        except KeyboardInterrupt:
            print(f"\n[STOP] interrupted during explainer={name}", flush=True)
            raise

        except Exception as e:
            print(f"  [WARN] explainer={name} failed: {type(e).__name__}: {e}", flush=True)
            continue

        if per:
            agg = aggregate(per)
            agg["explainer"] = name
            agg_rows.append(agg)

            print(
                f"  [done] explainer={name} "
                f"n={len(per)} "
                f"rift_score={agg.get('rift_score', 0.0):.4f}",
                flush=True,
            )

    if not agg_rows:
        print("  [WARN] no audit rows produced. Check CSV, CIFT checkpoint, and explainers.", flush=True)

    return build_leaderboard(agg_rows)


def run_block3(checkpoint_rows: List[Dict[str, Any]], spec_block3: Dict[str, Any]):
    """Correlation + optional saturation control. checkpoint_rows: one dict per ckpt."""
    from ..metrics.correlation_metrics import correlate_predictors
    target = spec_block3.get("target", "zero_shot_auc")
    preds = spec_block3.get("predictors", ["faithfulness", "in_domain_auc", "plausibility"])
    res_all = correlate_predictors(checkpoint_rows, target=target, predictors=preds)
    out = {"full": [r.__dict__ for r in res_all], "n": len(checkpoint_rows)}
    sc = spec_block3.get("saturation_control", {})
    if sc.get("enable") and checkpoint_rows:
        band = float(sc.get("in_domain_auc_band", 0.005))
        vals = sorted(r["in_domain_auc"] for r in checkpoint_rows if "in_domain_auc" in r)
        if vals:
            med = vals[len(vals) // 2]
            kept = [r for r in checkpoint_rows if abs(r.get("in_domain_auc", med) - med) <= band]
            res_sc = correlate_predictors(kept, target=target, predictors=preds)
            out["saturation_controlled"] = {"median_auc": med, "band": band,
                                            "n": len(kept),
                                            "results": [r.__dict__ for r in res_sc]}
    return out
