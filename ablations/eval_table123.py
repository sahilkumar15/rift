#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Any, Dict, List

from ablations.lib.manifest import load_manifest, policy_ckpt
from ablations.lib.explainers import CausalSelectExplainer, PolicyExplainer, sigmoid_mean, gap_value

TICK = "✓"
CROSS = "✗"


def tick(v) -> str:
    return TICK if bool(v) else CROSS


def read_existing_csv(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            out[str(r.get("ID", ""))] = dict(r)
        return out


def _count_csv_rows(path: str) -> int:
    try:
        with open(path, "rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def _parse_max_items_arg(v, *, eval_csv: str | None = None):
    if v is None:
        return None
    ss = str(v).strip()
    if ss.lower() in ("full", "all"):
        n = _count_csv_rows(eval_csv or "")
        return n if n > 0 else None
    if ss.lower() in ("", "none", "null", "0"):
        return None
    return int(ss)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not rows:
        return

    cols = []

    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def gap_mode(res) -> str:
    m = getattr(res, "mode", "proxy")
    return str(getattr(m, "value", m))


def mean(rows, key: str) -> float:
    vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / max(1, len(vals))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return v


def _make_tqdm(iterable, *, total=None, desc="", unit="it"):
    """Use tqdm when available; otherwise return the plain iterable."""
    try:
        from tqdm.auto import tqdm

        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            leave=False,
            mininterval=1.0,
        )
    except Exception:
        return iterable


def make_explainer(row: Dict[str, Any], manifest: Dict[str, Any], device: str):
    from src.explainers.random_explainer import RandomExplainer
    from src.explainers.gradcam_explainer import GradCAMExplainer
    from src.explainers.cift_gap_explainer import CIFTGapExplainer

    kind = row["kind"]
    ev = manifest["eval"]
    pd = manifest["policy_defaults"]

    if kind == "random":
        return RandomExplainer()

    if kind == "gradcam":
        return GradCAMExplainer(target_class=1)

    if kind == "cift_delta":
        return CIFTGapExplainer()

    if kind == "causal_select":
        base_name = row.get("base")

        if base_name == "gradcam":
            base = GradCAMExplainer(target_class=1)
        elif base_name == "cift_delta":
            base = CIFTGapExplainer()
        else:
            raise RuntimeError(f"Unknown causal_select base={base_name}")

        return CausalSelectExplainer(
            base,
            channel=row.get("channel", "delta"),
            grid=int(pd.get("grid", 8)),
            horizon=int(manifest["policies"].get("full_h4", {}).get("horizon", 4)),
            candidate_pool=int(ev.get("candidate_pool", 16)),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(ev.get("topk_frac", 0.12)),
        )

    if kind == "policy":
        key = row["policy"]
        p = manifest["policies"][key]
        ckpt = policy_ckpt(manifest, key)
        print(f"  policy={key} ckpt={ckpt}", flush=True)

        return PolicyExplainer(
            ckpt,
            grid=int(pd.get("grid", 8)),
            hidden=int(pd.get("hidden", 256)),
            feat_dim=int(pd.get("feat_dim", 1024)),
            horizon=int(p["horizon"]),
            reward_preset=str(p["reward_preset"]),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(ev.get("topk_frac", 0.12)),
            device=device,
        )

    raise RuntimeError(f"Unknown explainer kind={kind}")


def audit_explainer(
    cfg,
    adapter,
    explainer,
    manifest: Dict[str, Any],
    device: str,
    progress_desc: str = "eval",
) -> Dict[str, Any]:
    import torch

    from src.audit.ablation_runner import iter_audit_samples
    from src.faithfulness.faithfulness_score import compute_rift_score
    from src.interventions.interventions import apply_necessity, apply_sufficiency, mask_area
    from src.rl.reward import get_reward_weights

    ev = manifest["eval"]
    mode = str(ev.get("intervention_mode", "blur"))
    topk = float(ev.get("topk_frac", 0.12))
    max_items = int(ev.get("max_items", 512))
    weights = get_reward_weights("full_rift")

    sample_rows = []

    sample_iter = iter_audit_samples(cfg, device=device, n=max_items)
    sample_iter = _make_tqdm(
        sample_iter,
        total=max_items,
        desc=progress_desc,
        unit="img",
    )

    for img, donor, gt in sample_iter:
        # Important: mask generation is outside no_grad because Grad-CAM needs gradients.
        mask = explainer.explain(img, adapter, donor=donor)

        with torch.no_grad():
            g0 = adapter.identity_gap(img, donor=donor)
            l0 = sigmoid_mean(adapter.predict_logits(img))

            nec_img = apply_necessity(img, mask, mode, topk)
            suf_img = apply_sufficiency(img, mask, mode, topk)

            gn = adapter.identity_gap(nec_img, donor=donor)
            gs = adapter.identity_gap(suf_img, donor=donor)
            ln = sigmoid_mean(adapter.predict_logits(nec_img))
            ls = sigmoid_mean(adapter.predict_logits(suf_img))

            comp = compute_rift_score(
                e0_delta=gap_value(g0),
                e_nec_delta=gap_value(gn),
                e_suf_delta=gap_value(gs),
                e0_logit=l0,
                e_nec_logit=ln,
                e_suf_logit=ls,
                mask_area=mask_area(mask, topk),
                identity_gap_mode=gap_mode(g0),
                weights=weights,
            )

        sample_rows.append(comp.to_dict())

        if hasattr(sample_iter, "set_postfix") and len(sample_rows) % 10 == 0:
            sample_iter.set_postfix(
                n=len(sample_rows),
                rift=f"{mean(sample_rows, 'rift_score'):.4f}",
                mask=f"{mean(sample_rows, 'mask_area'):.4f}",
            )

    if not sample_rows:
        raise RuntimeError(
            "No samples were evaluated. Check dataset.split_csv, donor_path/source_ref_path, "
            "detector.strict_identity_gap, and MAX_ITEMS."
        )

    keys = [
        "necessity_delta",
        "sufficiency_delta",
        "faithfulness_delta",
        "necessity_logit",
        "sufficiency_logit",
        "faithfulness_logit",
        "mask_area",
        "rift_score",
    ]

    out = {k: mean(sample_rows, k) for k in keys}
    out["n"] = len(sample_rows)

    return out


def row_prefix(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "ID": row["id"],
        "Variant": row["variant"],
    }

    if "mask_source" in row:
        out["Mask source"] = row["mask_source"]

    if "delta_g" in row:
        out["ΔG"] = tick(row.get("delta_g"))
        out["NS"] = tick(row.get("ns"))
        out["RP"] = tick(row.get("rp"))

    if "necessity" in row:
        out["Necessity"] = tick(row.get("necessity"))
        out["Sufficiency"] = tick(row.get("sufficiency"))
        out["Sparsity"] = tick(row.get("sparsity"))

    if "horizon" in row:
        out["Horizon"] = row.get("horizon")

    return out


def run_table(
    table_key: str,
    table_spec: Dict[str, Any],
    manifest: Dict[str, Any],
    cfg,
    adapter,
    device: str,
    *,
    existing: Dict[str, Dict[str, Any]] | None = None,
    skip_ok_existing: bool = False,
) -> List[Dict[str, Any]]:
    rows_out = []
    existing = existing or {}
    target_n = int(manifest.get("eval", {}).get("max_items", 0) or 0)

    for row in table_spec["rows"]:
        print(f"\n[{table_key}] ID={row['id']} {row['variant']}", flush=True)

        out = row_prefix(row)

        old = existing.get(str(row["id"]))
        if skip_ok_existing and old and str(old.get("status", "")).lower() == "ok":
            try:
                old_n = int(float(old.get("n", 0)))
            except Exception:
                old_n = 0
            if target_n <= 0 or old_n >= target_n:
                print(f"  SKIP existing ok n={old_n}", flush=True)
                rows_out.append(old)
                continue

        try:
            ex = make_explainer(row, manifest, device)
            metrics = audit_explainer(
                cfg,
                adapter,
                ex,
                manifest,
                device,
                progress_desc=f"{table_key}:ID={row['id']} {row['variant']}",
            )

            out["Nec Δ ↑"] = fmt(metrics["necessity_delta"])
            out["Suf Δ ↑"] = fmt(metrics["sufficiency_delta"])
            out["Faith Δ ↑"] = fmt(metrics["faithfulness_delta"])
            out["Faith logit ↑"] = fmt(metrics["faithfulness_logit"])
            out["Mask area ↓"] = fmt(metrics["mask_area"])
            out["RIFT score ↑"] = fmt(metrics["rift_score"])
            out["n"] = int(metrics["n"])
            out["status"] = "ok"
            out["error"] = ""

            print(
                f"  ok n={out['n']} faithΔ={out['Faith Δ ↑']} "
                f"mask={out['Mask area ↓']} rift={out['RIFT score ↑']}",
                flush=True,
            )

        except Exception as e:
            out["n"] = 0
            out["status"] = "FAILED"
            out["error"] = f"{type(e).__name__}: {e}"

            print(f"  FAILED: {out['error']}", flush=True)

        rows_out.append(out)

    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-config", default="ablations/configs/table123_rift.yaml")
    ap.add_argument("--tables", default="table1_component,table2_objective,table3_horizon")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-items", default=None)
    ap.add_argument("--skip-ok-existing", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.ablation_config)

    parsed_max = _parse_max_items_arg(args.max_items, eval_csv=manifest.get("data", {}).get("eval_csv"))
    if parsed_max is not None:
        manifest["eval"]["max_items"] = int(parsed_max)

    seed = int(manifest.get("eval", {}).get("seed", 3407))
    random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass

    from src.adapters.cift_adapter import CIFTAdapter
    from src.utils.config import load_config, merge_overrides

    cfg = load_config(manifest["base_config"])

    cfg = merge_overrides(
        cfg,
        {
            "device": args.device,
            "detector.cift_root": manifest["cift"]["root"],
            "detector.cift_ckpt": manifest["cift"]["ckpt"],
            "detector.cift_config": manifest["cift"].get("config", "configs/diffusionfake_mixed.yaml"),
            "detector.backbone": manifest["cift"].get("backbone", "convnextv2_base"),
            "detector.strict_identity_gap": True,
            "dataset.split_csv": manifest["data"]["eval_csv"],
            "dataset.max_items": int(manifest["eval"].get("max_items", 512)),
            "intervention.mode": manifest["eval"].get("intervention_mode", "blur"),
            "intervention.topk_frac": float(manifest["eval"].get("topk_frac", 0.12)),
        },
    )

    print(f"Loading CIFT checkpoint: {manifest['cift']['ckpt']}", flush=True)

    adapter = CIFTAdapter(
        ckpt_path=manifest["cift"]["ckpt"],
        device=args.device,
        backbone=manifest["cift"].get("backbone", "convnextv2_base"),
        strict_identity_gap=True,
        cift_root=manifest["cift"]["root"],
        config_path=manifest["cift"].get("config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    out_dir = manifest["eval"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    wanted = [x.strip() for x in args.tables.split(",") if x.strip()]
    combined = []

    for key in wanted:
        if key not in manifest["tables"]:
            raise RuntimeError(f"Unknown table key {key}. Available: {list(manifest['tables'])}")

        spec = manifest["tables"][key]

        print(f"\n{'=' * 80}\n{key}\n{'=' * 80}", flush=True)

        out_path = os.path.join(out_dir, spec["filename"])
        existing = read_existing_csv(out_path)
        rows = run_table(
            key,
            spec,
            manifest,
            cfg,
            adapter,
            args.device,
            existing=existing,
            skip_ok_existing=bool(args.skip_ok_existing),
        )

        write_csv(out_path, rows)

        print(f"[wrote] {out_path}", flush=True)

        for r in rows:
            combined.append({"table": key, **r})

    combined_path = os.path.join(out_dir, "combined_tables_1_2_3.csv")
    write_csv(combined_path, combined)

    print(f"\n[done] combined: {combined_path}", flush=True)

    import time
    history_path = os.path.join(out_dir, "metrics_history.csv")
    hist_rows = []
    run_ts = int(time.time())
    for r in combined:
        hist_rows.append({"run_ts": run_ts, "max_items": manifest["eval"].get("max_items"), **r})
    if hist_rows:
        exists = os.path.exists(history_path)
        cols = []
        for r in hist_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(history_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerows(hist_rows)
        print(f"[history] appended: {history_path}", flush=True)

    failed = [r for r in combined if r.get("status") != "ok"]

    if failed:
        print(f"[warn] {len(failed)} rows failed. Open the CSV and read the error column.", flush=True)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
