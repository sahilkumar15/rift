#!/usr/bin/env python3
# Path: ablations/run_table1.py
# Status: NEW
"""TABLE 1 - component ablation x dataset generalization.

DESIGN CONTRACTS (each exists to kill a specific reviewer objection)
--------------------------------------------------------------------
1. AREA IS CONSTANT ACROSS ROWS. Every explainer is forced to exactly
   eval.cells grid cells via GridTopKExplainer. rift_score is monotone in
   mask_area, so any row allowed a bigger budget wins for free. Soft baselines
   (Grad-CAM, CIFT-Delta saliency) were previously binarized at topk_frac=0.12
   = 2x the policy's 0.0625. The runner ASSERTS area equality and fails loudly.
   -> kills "your baseline had half the budget".

2. EVERY NUMBER IS ANCHORED TO RANDOM. gap_vs_random is computed per dataset
   from the row-0 control at identical area and geometry. A faithfulness score
   is meaningless in absolute terms.
   -> kills "0.15 compared to what?".

3. EVIDENCE VALIDITY IS GATED AND REPORTED. Faithfulness is undefined when the
   detector has no fake evidence (e0 <= min_evidence). Those samples are
   EXCLUDED from the conditional mean and counted in valid_frac. Ungated, OOD
   datasets where CIFT is weak contribute structural zeros and you would report
   "faithfulness collapses OOD" when you measured "CIFT saw nothing".
   -> kills "your OOD drop is a detection artifact".

4. DONOR PROVENANCE IS A COLUMN. FF++ has frame-aligned ground-truth donors;
   CelebDF/DiffSwap only have retrieved same-identity references (which CIFT
   itself uses at test time, paper C.6/Table 10). The ffpp_c23_retrieved
   control isolates donor quality from generalization.
   -> kills "is the OOD drop worse donors or worse transfer?".

5. UNCERTAINTY IS REPORTED. Bootstrap CI over samples; stochastic rows also
   re-run across eval.seeds. Deterministic rows (argmax policies, gradients)
   are run once - re-running them would be a fake CI.
   -> kills "single seed, no error bars".

6. PROXY MODE CANNOT EARN DELTA CREDIT. If a dataset yields
   identity_gap_mode != 'true', the delta columns are reported as NaN rather
   than 0, and the row is flagged. w_delta is already forced to 0 by the
   canonical scorer; NaN prevents a zero from being averaged as if it were a
   measurement.
   -> kills "you reported Delta faithfulness without a donor".

EFFICIENCY
----------
Original evidence per batch is computed once and shared. Necessity+sufficiency
images go through a single OOM-adaptive predict_evidence call. Deterministic
rows skip seed loops. Per-sample scalars accumulate on CPU as float32. The
bootstrap is chunked (never allocates n_boot x n).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ablations.eval_table123 import _iter_batches, _quiet_stdio, write_csv
from ablations.gate1_validity import bootstrap_ci
from ablations.lib.explainers import (
    CausalSelectExplainer,
    PolicyExplainer,
    logit_to_evidence,
    predict_evidence,
)

TICK, CROSS = "✓", "✗"


def tick(v) -> str:
    return TICK if bool(v) else CROSS


def _count_fake_rows(path: str) -> int:
    fake = {"1", "fake", "forged", "True", "true"}
    try:
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames and "label" in rd.fieldnames:
                return sum(1 for r in rd if str(r.get("label", "1")).strip() in fake)
    except Exception:
        return 0
    return 0


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def build_explainer(variant: Dict[str, Any], cfg_t1: Dict[str, Any], device: str, seed: int):
    """Construct an explainer forced onto the shared cell budget."""
    from src.explainers.cift_gap_explainer import CIFTGapExplainer
    from src.explainers.gradcam_explainer import GradCAMExplainer
    from src.explainers.grid_topk import GridTopKExplainer
    from src.explainers.random_cell_explainer import RandomCellExplainer

    ev = cfg_t1["eval"]
    cells = int(ev["cells"])
    grid = int(ev.get("grid", 8))
    fbs = int(ev.get("forward_batch_size", 32))
    kind = variant["kind"]

    if kind == "random_cell":
        return RandomCellExplainer(cells=cells, grid=grid, seed=seed)

    if kind == "gradcam":
        base = GradCAMExplainer(
            target_class=1, strict=bool(ev.get("strict_gradcam", True))
        )
        return GridTopKExplainer(base, cells=cells, grid=grid)

    if kind == "cift_delta":
        return GridTopKExplainer(CIFTGapExplainer(), cells=cells, grid=grid)

    if kind == "causal_select":
        b = variant.get("base")
        if b == "gradcam":
            base = GradCAMExplainer(
                target_class=1, strict=bool(ev.get("strict_gradcam", True))
            )
        elif b == "cift_delta":
            base = CIFTGapExplainer()
        else:
            raise RuntimeError(f"Unknown causal_select base={b}")
        # CausalSelect already emits a hard grid mask of exactly `horizon` cells.
        return CausalSelectExplainer(
            base,
            channel=variant.get("channel", "delta"),
            grid=grid,
            horizon=cells,
            candidate_pool=int(ev.get("candidate_pool", 16)),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(cells) / float(grid * grid),
            forward_batch_size=fbs,
        )

    if kind == "policy":
        from ablations.lib.manifest import policy_ckpt

        key = variant["policy"]
        pol = cfg_t1["policies"][key]
        ckpt = policy_ckpt(cfg_t1, key)
        if not Path(ckpt).exists():
            raise FileNotFoundError(
                f"Policy checkpoint missing for row '{variant['variant']}': {ckpt}\n"
                f"Train it:  ROW={key} GPUS=0,1 BATCH=256 EPOCHS=30 "
                f"bash ablations/scripts/train_table123_row.sh"
            )
        defaults = cfg_t1["policy_defaults"]
        if int(pol["horizon"]) != cells:
            raise RuntimeError(
                f"Row '{variant['variant']}' uses policy {key} with horizon="
                f"{pol['horizon']}, but eval.cells={cells}. Table 1 requires "
                "every row at the SAME mask budget. Fix the horizon or eval.cells."
            )
        return PolicyExplainer(
            ckpt,
            grid=grid,
            hidden=int(defaults.get("hidden", 256)),
            feat_dim=int(defaults.get("feat_dim", 1024)),
            horizon=int(pol["horizon"]),
            reward_preset=str(pol["reward_preset"]),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(cells) / float(grid * grid),
            device=device,
            forward_batch_size=fbs,
            allow_stop=bool(pol.get("allow_stop", False)),
            min_cells=int(pol.get("min_cells", 1)),
            max_cells=pol.get("max_cells", pol.get("horizon", cells)),
            force_min_cells=bool(pol.get("force_min_cells", True)),
            forbid_revisit=bool(pol.get("forbid_revisit", True)),
            state_blind=bool(pol.get("state_blind", False)),
        )

    raise RuntimeError(f"Unknown variant kind={kind}")


def audit_one(*, cfg, adapter, explainer, cfg_t1, device, desc):
    """Return per-sample metric lists for one (dataset, variant, seed)."""
    import torch
    from tqdm import tqdm

    from src.audit.ablation_runner import iter_audit_samples
    from src.faithfulness.faithfulness_score import compute_rift_score_tensor
    from src.interventions.interventions import apply_necessity, apply_sufficiency
    from src.rl.reward import get_reward_weights

    ev = cfg_t1["eval"]
    mode = str(ev.get("intervention_mode", "blur"))
    cells = int(ev["cells"])
    grid = int(ev.get("grid", 8))
    topk = float(cells) / float(grid * grid)
    fbs = int(ev.get("forward_batch_size", 32))
    bs = max(1, int(ev.get("batch_size", 8)))
    max_items = int(ev["_max_items_resolved"])
    min_ev = float(ev.get("min_evidence", 0.0) or 0.0)

    weights = dict(get_reward_weights(str(ev.get("reward_preset", "full_rift"))))
    weights["min_evidence"] = min_ev

    keys = (
        "necessity_delta", "sufficiency_delta", "faithfulness_delta",
        "necessity_logit", "sufficiency_logit", "faithfulness_logit",
        "mask_area", "rift_score",
    )
    acc = {k: [] for k in keys}
    valid_d, valid_l = [], []
    modes = set()
    n = 0

    it = _iter_batches(iter_audit_samples(cfg, device=device, n=max_items), bs)
    bar = tqdm(total=max_items, desc=desc, unit="img", file=sys.stderr,
               dynamic_ncols=True, mininterval=1.0)
    try:
        for image, donor, _ in it:
            mask = explainer.explain(image, adapter, donor=donor)
            nec_img = apply_necessity(image, mask, mode, topk)
            suf_img = apply_sufficiency(image, mask, mode, topk)

            cached = None
            if hasattr(explainer, "cached_original_evidence"):
                cached = explainer.cached_original_evidence(image)

            if cached is not None and bool(cached.get("complete", False)):
                b = int(image.shape[0])
                e0_g = cached["gap"].to(image.device).float().view(-1)
                e0_l = cached["logit"].to(image.device).float().view(-1)
                g_nec, g_suf = cached["gap_nec"].float().view(-1), cached["gap_suf"].float().view(-1)
                l_nec, l_suf = cached["logit_nec"].float().view(-1), cached["logit_suf"].float().view(-1)
                emode = str(cached.get("mode", "proxy"))
            else:
                stack = torch.cat([image, nec_img, suf_img], dim=0)
                dstack = torch.cat([donor] * 3, dim=0) if donor is not None else None
                raw, gaps, emode, _ = predict_evidence(
                    adapter, stack, dstack, max_batch=fbs, return_features=False
                )
                b = int(image.shape[0])
                lev = logit_to_evidence(raw)
                e0_g, g_nec, g_suf = gaps[:b], gaps[b:2 * b], gaps[2 * b:]
                e0_l, l_nec, l_suf = lev[:b], lev[b:2 * b], lev[2 * b:]

            modes.add(str(emode))

            binary = (mask.float() > 1e-6).float()
            area = binary.flatten(1).mean(dim=1)
            sel = binary.flatten(1).sum(dim=1) / float(binary.shape[-1] * binary.shape[-2]) * float(grid * grid)

            _, comps = compute_rift_score_tensor(
                e0_delta=e0_g, e_nec_delta=g_nec, e_suf_delta=g_suf,
                e0_logit=e0_l, e_nec_logit=l_nec, e_suf_logit=l_suf,
                mask_area=area, selected_cells=sel,
                identity_gap_mode=emode, weights=weights,
            )

            vd = (e0_g > min_ev).float().cpu().view(-1).tolist()
            vl = (e0_l > min_ev).float().cpu().view(-1).tolist()
            valid_d.extend(vd)
            valid_l.extend(vl)
            for k in keys:
                acc[k].extend(comps[k].detach().float().cpu().view(-1).tolist())

            n += int(image.shape[0])
            bar.update(int(image.shape[0]))
    finally:
        bar.close()

    if n == 0:
        raise RuntimeError(
            "0 samples evaluated. Check the dataset eval_csv, donor paths, and max_items."
        )

    emode_final = "true" if modes == {"true"} else (sorted(modes)[0] if modes else "proxy")
    return {"per_sample": acc, "valid_delta": valid_d, "valid_logit": valid_l,
            "n": n, "mode": emode_final}


def _conditional(values: List[float], valid: List[float]) -> List[float]:
    """Keep only samples where the evidence channel was defined."""
    return [v for v, ok in zip(values, valid) if ok > 0.5]


def aggregate(runs: List[Dict[str, Any]], cfg_t1: Dict[str, Any]) -> Dict[str, Any]:
    """Pool per-sample values across seeds; conditional means + bootstrap CI."""
    ev = cfg_t1["eval"]
    nboot = int(ev.get("bootstrap", 2000))

    pooled = {k: [] for k in runs[0]["per_sample"]}
    vd, vl = [], []
    for r in runs:
        for k, v in r["per_sample"].items():
            pooled[k].extend(v)
        vd.extend(r["valid_delta"])
        vl.extend(r["valid_logit"])

    mode = "true" if all(r["mode"] == "true" for r in runs) else runs[0]["mode"]
    delta_ok = mode == "true"

    fd = _conditional(pooled["faithfulness_delta"], vd)
    fl = _conditional(pooled["faithfulness_logit"], vl)
    nd = _conditional(pooled["necessity_delta"], vd)
    sd = _conditional(pooled["sufficiency_delta"], vd)

    lo, hi = bootstrap_ci(fd, n_boot=nboot, seed=0) if (delta_ok and fd) else (float("nan"),) * 2

    nan = float("nan")
    return {
        "faith_delta": _mean(fd) if delta_ok else nan,
        "faith_delta_ci_lo": lo if delta_ok else nan,
        "faith_delta_ci_hi": hi if delta_ok else nan,
        "nec_delta": _mean(nd) if delta_ok else nan,
        "suf_delta": _mean(sd) if delta_ok else nan,
        "faith_logit": _mean(fl),
        "mask_area": _mean(pooled["mask_area"]),
        "rift_score": _mean(pooled["rift_score"]),
        "valid_frac_delta": _mean(vd),
        "valid_frac_logit": _mean(vl),
        "identity_gap_mode": mode,
        "n": sum(r["n"] for r in runs),
        "n_seeds": len(runs),
        "_fd_samples": fd,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Table 1: components x datasets.")
    ap.add_argument("--config", default="ablations/configs/table1.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", default=None, help="comma list; default = all")
    ap.add_argument("--variants", default=None, help="comma list of ids; default = all")
    ap.add_argument("--max-items", default=None, help="int | full")
    ap.add_argument("--intervention-mode", default=None, choices=["blur", "mean", "zero"])
    ap.add_argument("--cells", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--forward-batch-size", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--skip-ok-existing", action="store_true")
    ap.add_argument("--verbose-model-load", action="store_true")
    args = ap.parse_args()

    t1 = yaml.safe_load(Path(args.config).read_text())
    ev = t1["eval"]
    if args.intervention_mode:
        ev["intervention_mode"] = args.intervention_mode
    if args.cells is not None:
        ev["cells"] = int(args.cells)
    if args.batch_size:
        ev["batch_size"] = int(args.batch_size)
    if args.forward_batch_size:
        ev["forward_batch_size"] = int(args.forward_batch_size)
    if args.max_items:
        ev["max_items"] = args.max_items
    out_dir = args.output_dir or t1["output_dir"]
    mode_tag = str(ev.get("intervention_mode", "blur"))
    out_dir = os.path.join(out_dir, mode_tag)
    os.makedirs(out_dir, exist_ok=True)

    ds_keys = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(t1["datasets"])
    var_ids = {int(x) for x in args.variants.split(",")} if args.variants else None
    variants = [v for v in t1["variants"] if var_ids is None or int(v["id"]) in var_ids]
    variants.sort(key=lambda v: int(v["id"]))
    if not any(v["kind"] == "random_cell" for v in variants):
        print("[warn] no random_cell row selected -> gap_vs_random will be NaN.", flush=True)

    import torch

    torch.manual_seed(int(ev["seeds"][0]))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    from src.adapters.cift_adapter import CIFTAdapter
    from src.utils.config import load_config, merge_overrides

    print(f"Loading CIFT: {t1['cift']['ckpt']}", flush=True)
    adapter = CIFTAdapter(
        ckpt_path=t1["cift"]["ckpt"], device=args.device,
        backbone=t1["cift"].get("backbone", "convnextv2_base"),
        strict_identity_gap=True, cift_root=t1["cift"]["root"],
        config_path=t1["cift"].get("config", "configs/diffusionfake_mixed.yaml"),
    )
    with _quiet_stdio(not args.verbose_model_load):
        adapter.load_detector()

    grid = int(ev.get("grid", 8))
    expected_area = float(ev["cells"]) / float(grid * grid)
    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(out_dir, "table1_component_x_dataset.csv")

    for ds_key in ds_keys:
        if ds_key not in t1["datasets"]:
            raise RuntimeError(f"Unknown dataset '{ds_key}'. Have: {list(t1['datasets'])}")
        ds = t1["datasets"][ds_key]
        eval_csv = ds["eval_csv"]
        if not Path(eval_csv).exists():
            print(f"\n[SKIP] {ds_key}: eval_csv not found -> {eval_csv}", flush=True)
            print("       Build it: python scripts/build_rift_csv_generic.py --help", flush=True)
            continue

        resolved = _count_fake_rows(eval_csv) if str(ev.get("max_items")) in ("full", "all", "auto") else int(ev["max_items"])
        ev["_max_items_resolved"] = max(1, resolved)

        base = load_config(t1["base_config"])
        cfg = merge_overrides(base, {
            "device": args.device,
            "detector.cift_root": t1["cift"]["root"],
            "detector.cift_ckpt": t1["cift"]["ckpt"],
            "detector.cift_config": t1["cift"].get("config", "configs/diffusionfake_mixed.yaml"),
            "detector.backbone": t1["cift"].get("backbone", "convnextv2_base"),
            "detector.strict_identity_gap": True,
            "dataset.split_csv": eval_csv,
            "dataset.max_items": int(ev["_max_items_resolved"]),
            "dataset.shard_id": 0, "dataset.shard_count": 1,
            "intervention.mode": ev.get("intervention_mode", "blur"),
            "intervention.topk_frac": expected_area,
        })

        print(f"\n{'=' * 78}\nDATASET {ds_key} ({ds['display']})  role={ds['role']}  "
              f"donor={ds['donor_type']}  n={ev['_max_items_resolved']}\n{'=' * 78}", flush=True)

        ds_rows = []
        for v in variants:
            seeds = list(ev["seeds"]) if v.get("stochastic") else [int(ev["seeds"][0])]
            print(f"\n[{ds_key}] id={v['id']} {v['variant']}  seeds={len(seeds)}", flush=True)
            row = {
                "Dataset": ds["display"], "dataset_key": ds_key, "Role": ds["role"],
                "Donor": ds["donor_type"], "ID": v["id"], "Variant": v["variant"],
                "Mask source": v.get("mask_source", ""),
                "ΔG": tick(v.get("delta_g")), "NS": tick(v.get("ns")), "RP": tick(v.get("rp")),
            }
            try:
                runs = []
                for s in seeds:
                    ex = build_explainer(v, t1, args.device, s)
                    runs.append(audit_one(cfg=cfg, adapter=adapter, explainer=ex,
                                          cfg_t1=t1, device=args.device,
                                          desc=f"{ds_key}:{v['id']}:{s}"))
                    if getattr(ex, "used_delta_fallback", False):
                        row["gradcam_delta_fallback"] = True
                agg = aggregate(runs, t1)
                fd_samples = agg.pop("_fd_samples")
                row["_fd_samples"] = fd_samples

                area = agg["mask_area"]
                if not math.isclose(area, expected_area, abs_tol=2e-3):
                    raise RuntimeError(
                        f"AREA MISMATCH: row '{v['variant']}' produced mask_area="
                        f"{area:.4f}, expected {expected_area:.4f}. Table 1 requires "
                        "identical budget across rows; an unmatched row is "
                        "uninterpretable. Wrap the explainer in GridTopKExplainer."
                    )
                for k, val in agg.items():
                    row[k] = round(val, 4) if isinstance(val, float) else val
                row["status"], row["error"] = "ok", ""
                print(f"  ok  faithΔ={row['faith_delta']}  faith_logit={row['faith_logit']}  "
                      f"area={row['mask_area']}  valid_frac_Δ={row['valid_frac_delta']}  "
                      f"mode={row['identity_gap_mode']}", flush=True)
            except Exception as exc:
                row.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "n": 0})
                row["_fd_samples"] = []
                print(f"  FAILED: {row['error']}", flush=True)

            ds_rows.append(row)
            write_csv(csv_path, [{k: x for k, x in r.items() if not k.startswith("_")}
                                 for r in rows + ds_rows])

        # ---- anchor every row of this dataset to its own random control ----
        rnd = next((r for r in ds_rows if r["ID"] == 0 and r["status"] == "ok"), None)
        for r in ds_rows:
            if rnd and r["status"] == "ok" and isinstance(r.get("faith_delta"), float):
                if not math.isnan(r["faith_delta"]) and not math.isnan(rnd["faith_delta"]):
                    r["gap_vs_random"] = round(r["faith_delta"] - rnd["faith_delta"], 4)
                    r["ratio_vs_random"] = (
                        round(r["faith_delta"] / rnd["faith_delta"], 2)
                        if abs(rnd["faith_delta"]) > 1e-6 else float("inf")
                    )
                else:
                    r["gap_vs_random"] = float("nan")
            else:
                r["gap_vs_random"] = float("nan")
        rows.extend(ds_rows)
        write_csv(csv_path, [{k: x for k, x in r.items() if not k.startswith("_")} for r in rows])

    # ---------------- Gate 2 verdict: row 6 (RIFT full) vs row 5 (logit-only)
    gate2 = []
    for ds_key in {r["dataset_key"] for r in rows}:
        sub = {r["ID"]: r for r in rows if r["dataset_key"] == ds_key and r["status"] == "ok"}
        full, lg = sub.get(6), sub.get(5)
        if not (full and lg):
            continue
        try:
            d = float(full["faith_delta"]) - float(lg["faith_delta"])
        except Exception:
            continue
        if math.isnan(d):
            continue
        diff = [a - b for a, b in zip(full.get("_fd_samples", []), lg.get("_fd_samples", []))]
        lo, hi = bootstrap_ci(diff, n_boot=int(ev.get("bootstrap", 2000)), seed=1) if len(diff) > 1 else (float("nan"),) * 2
        gate2.append({"dataset": ds_key, "delta_full_minus_logit": round(d, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                      "verdict": "PASS" if (d >= 0.03 and lo == lo and lo > 0) else "FAIL"})

    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    write_csv(csv_path, clean)
    Path(os.path.join(out_dir, "table1_summary.json")).write_text(json.dumps({
        "generated_at": time.time(), "intervention_mode": mode_tag,
        "cells": ev["cells"], "expected_area": expected_area,
        "min_evidence": ev.get("min_evidence"), "gate2": gate2, "rows": clean,
    }, indent=2, default=str))

    print(f"\n[wrote] {csv_path}")
    print(f"[wrote] {os.path.join(out_dir, 'table1_summary.json')}")
    if gate2:
        print("\nGATE 2 (RIFT full - RIFT logit-only, needs >= 0.03 and CI>0):")
        for g in gate2:
            print(f"  {g['dataset']:22s} Δ={g['delta_full_minus_logit']:+.4f} "
                  f"[{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}]  {g['verdict']}")

    return 2 if any(r["status"] != "ok" for r in clean) else 0


if __name__ == "__main__":
    raise SystemExit(main())
