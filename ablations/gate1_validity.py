#!/usr/bin/env python3
# Path: ablations/gate1_validity.py
# Status: NEW
"""GATE 1 - intervention validity.

THE QUESTION
------------
Does masking the CITED region move CIFT's donor identity-gap Delta measurably
more than masking a RANDOM region of IDENTICAL AREA and IDENTICAL GEOMETRY?

If it does not, the necessity/sufficiency interventions are not probing the
mechanism, and every downstream RIFT number in Tables 1-3 is noise. This gate
runs BEFORE any claim is made.

VERDICT RULE
------------
    separation = mean_over_samples( nec_delta_cited - nec_delta_random )
    PASS if separation >= 0.15 AND the bootstrap 95% CI lower bound > 0.

WHAT ELSE THIS EMITS (and why you need it)
------------------------------------------
  * sufficiency separation. If sufficiency is saturated (every mask looks
    "sufficient" because blur preserves low-frequency identity structure), the
    sufficiency separation collapses toward 0 while necessity separation stays
    healthy. That is a diagnosis, not a failure: it tells you the harmonic mean
    is being carried by necessity alone and the blur intervention needs an
    alternative arm.
  * the logit channel alongside the delta channel. If delta separation exceeds
    logit separation, that is direct supporting evidence for Gate 2 (the
    mechanistic claim that Delta-grounding beats logit-grounding).
  * paired Cohen's d. Reviewers ask for effect size, not just a mean gap.

EFFICIENCY
----------
Original evidence is computed ONCE per batch and reused for the cited mask and
all R random draws. Every intervention image for the batch goes through a single
predict_evidence() call (which is already OOM-adaptive and chunked), with the
donor tensor tiled to match. Everything runs under inference_mode. Per-sample
scalars are accumulated on CPU as float32; at n=13453 with 12 tracked scalars
that is well under 1 MB. The bootstrap is chunked so the (n_boot x n) index
matrix is never materialized.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ablations.eval_table123 import (
    _iter_batches,
    _parse_max_items_arg,
    _quiet_stdio,
    make_explainer,
    write_csv,
)
from ablations.lib.explainers import logit_to_evidence, predict_evidence
from ablations.lib.manifest import load_manifest

GATE1_SEPARATION_THRESHOLD = 0.15


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: List[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return float("nan")
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - ddof)
    return float(var ** 0.5)


def paired_cohens_d(diffs: List[float]) -> float:
    """Paired effect size: mean(diff) / sd(diff). Conventional bar: d >= 0.8."""
    sd = _std(diffs, ddof=1)
    if not sd or sd != sd or sd <= 0:
        return float("nan")
    return _mean(diffs) / sd


def bootstrap_ci(
    values: List[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    chunk: int = 100,
) -> tuple:
    """Percentile bootstrap CI of the mean, chunked to bound peak memory.

    Never allocates the full (n_boot x n) resample index matrix; draws `chunk`
    resamples at a time, so peak extra memory is O(chunk * n).
    """
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        size = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(size, n))
        means[done: done + size] = arr[idx].mean(axis=1)
        done += size

    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def _components_for_mask(
    *,
    adapter,
    image,
    donor,
    masks: List[Any],
    e0_gap,
    e0_logit,
    evidence_mode: str,
    intervention_mode: str,
    topk_frac: float,
    forward_batch_size: int,
    weights: Dict[str, Any],
):
    """Score every mask in `masks` against shared original evidence.

    All necessity/sufficiency images for the whole batch are concatenated into a
    SINGLE predict_evidence call. Returns a list of component dicts, one per
    input mask, aligned to `masks`.
    """
    import torch

    from src.faithfulness.faithfulness_score import compute_rift_score_tensor
    from src.interventions.interventions import apply_necessity, apply_sufficiency

    batch = int(image.shape[0])
    pieces = []
    for mask in masks:
        pieces.append(apply_necessity(image, mask, intervention_mode, topk_frac))
        pieces.append(apply_sufficiency(image, mask, intervention_mode, topk_frac))

    stacked = torch.cat(pieces, dim=0)
    tiled_donor = None
    if donor is not None:
        # torch.cat above lays out blocks of size `batch`; tiling the donor
        # len(pieces) times keeps every block aligned to its own sample.
        tiled_donor = donor.repeat(len(pieces), *([1] * (donor.dim() - 1)))

    raw, gaps, _, _ = predict_evidence(
        adapter,
        stacked,
        tiled_donor,
        max_batch=forward_batch_size,
        return_features=False,
    )
    evidence = logit_to_evidence(raw)

    outputs = []
    for i, mask in enumerate(masks):
        start = 2 * i * batch
        gap_nec = gaps[start: start + batch]
        gap_suf = gaps[start + batch: start + 2 * batch]
        logit_nec = evidence[start: start + batch]
        logit_suf = evidence[start + batch: start + 2 * batch]

        binary = (mask.float() > 1e-6).float()
        area = binary.flatten(1).mean(dim=1)
        cells = binary.flatten(1).sum(dim=1) / float(
            binary.shape[-1] * binary.shape[-2]
        ) * 64.0

        _, comps = compute_rift_score_tensor(
            e0_delta=e0_gap,
            e_nec_delta=gap_nec,
            e_suf_delta=gap_suf,
            e0_logit=e0_logit,
            e_nec_logit=logit_nec,
            e_suf_logit=logit_suf,
            mask_area=area,
            selected_cells=cells,
            identity_gap_mode=evidence_mode,
            weights=weights,
        )
        comps = dict(comps)
        # Unnormalized raw move, for readers who distrust the normalization.
        comps["raw_delta_move"] = (e0_gap - gap_nec)
        comps["raw_logit_move"] = (e0_logit - logit_nec)
        outputs.append(comps)

    return outputs


def run_gate1_for_row(
    *,
    cfg,
    adapter,
    manifest: Dict[str, Any],
    device: str,
    row: Dict[str, Any],
    cells: int,
    n_random: int,
    seed: int,
    label: str,
) -> Dict[str, Any]:
    import torch
    from tqdm import tqdm

    from src.audit.ablation_runner import iter_audit_samples
    from src.explainers.random_cell_explainer import RandomCellExplainer
    from src.rl.reward import get_reward_weights

    eval_cfg = manifest["eval"]
    intervention_mode = str(eval_cfg.get("intervention_mode", "blur"))
    topk_frac = float(eval_cfg.get("topk_frac", 0.12))
    max_items = int(eval_cfg.get("max_items", 512))
    batch_size = max(1, int(eval_cfg.get("batch_size", 4)))
    forward_batch_size = max(1, int(eval_cfg.get("forward_batch_size", 32)))
    grid = int(manifest.get("policy_defaults", {}).get("grid", 8))

    weights = dict(get_reward_weights("full_rift"))
    min_evidence = float(eval_cfg.get("min_evidence", 0.0) or 0.0)
    weights["min_evidence"] = min_evidence

    cited_explainer = make_explainer(row, manifest, device)
    random_explainers = [
        RandomCellExplainer(cells=cells, grid=grid, seed=seed + 1000 * r)
        for r in range(n_random)
    ]

    track = {
        key: []
        for key in (
            "nec_delta_cited",
            "nec_delta_random",
            "suf_delta_cited",
            "suf_delta_random",
            "faith_delta_cited",
            "faith_delta_random",
            "nec_logit_cited",
            "nec_logit_random",
            "faith_logit_cited",
            "faith_logit_random",
            "raw_delta_move_cited",
            "raw_delta_move_random",
            "area_cited",
            "area_random",
        )
    }

    sample_iter = iter_audit_samples(cfg, device=device, n=max_items)
    bar = tqdm(
        total=max_items,
        desc=f"gate1:{label}",
        unit="img",
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=1.0,
    )

    count = 0
    try:
        for image, donor, _ in _iter_batches(sample_iter, batch_size):
            cited_mask = cited_explainer.explain(image, adapter, donor=donor)

            cached = None
            if hasattr(cited_explainer, "cached_original_evidence"):
                cached = cited_explainer.cached_original_evidence(image)

            if cached is not None:
                e0_gap = cached["gap"].to(image.device).float().view(-1)
                e0_logit = cached["logit"].to(image.device).float().view(-1)
                evidence_mode = str(cached.get("mode", "proxy"))
            else:
                raw0, gap0, evidence_mode, _ = predict_evidence(
                    adapter,
                    image,
                    donor,
                    max_batch=forward_batch_size,
                    return_features=False,
                )
                e0_gap = gap0.float().view(-1)
                e0_logit = logit_to_evidence(raw0).float().view(-1)

            if evidence_mode != "true":
                raise RuntimeError(
                    "Gate 1 audits the donor identity gap and requires "
                    "identity_gap_mode='true'. Got "
                    f"'{evidence_mode}'. Use an eval CSV with donor pairs "
                    "(FF++ c23) and detector.strict_identity_gap=True."
                )

            masks = [cited_mask] + [
                ex.explain(image, adapter, donor=donor) for ex in random_explainers
            ]

            comps = _components_for_mask(
                adapter=adapter,
                image=image,
                donor=donor,
                masks=masks,
                e0_gap=e0_gap,
                e0_logit=e0_logit,
                evidence_mode=evidence_mode,
                intervention_mode=intervention_mode,
                topk_frac=topk_frac,
                forward_batch_size=forward_batch_size,
                weights=weights,
            )

            cited = comps[0]
            randoms = comps[1:]

            def _stack_mean(key):
                vals = torch.stack([r[key].float().view(-1) for r in randoms], dim=0)
                return vals.mean(dim=0)

            pairs = [
                ("nec_delta_cited", cited["necessity_delta"]),
                ("nec_delta_random", _stack_mean("necessity_delta")),
                ("suf_delta_cited", cited["sufficiency_delta"]),
                ("suf_delta_random", _stack_mean("sufficiency_delta")),
                ("faith_delta_cited", cited["faithfulness_delta"]),
                ("faith_delta_random", _stack_mean("faithfulness_delta")),
                ("nec_logit_cited", cited["necessity_logit"]),
                ("nec_logit_random", _stack_mean("necessity_logit")),
                ("faith_logit_cited", cited["faithfulness_logit"]),
                ("faith_logit_random", _stack_mean("faithfulness_logit")),
                ("raw_delta_move_cited", cited["raw_delta_move"]),
                ("raw_delta_move_random", _stack_mean("raw_delta_move")),
                ("area_cited", cited["mask_area"]),
                ("area_random", _stack_mean("mask_area")),
            ]
            for key, tensor in pairs:
                track[key].extend(
                    tensor.detach().float().cpu().view(-1).tolist()
                )

            count += int(image.shape[0])
            bar.update(int(image.shape[0]))
    finally:
        bar.close()

    if count == 0:
        raise RuntimeError(
            "Gate 1 evaluated 0 samples. Check dataset.split_csv, donor paths, "
            "and eval.max_items."
        )

    nec_diffs = [
        c - r for c, r in zip(track["nec_delta_cited"], track["nec_delta_random"])
    ]
    suf_diffs = [
        c - r for c, r in zip(track["suf_delta_cited"], track["suf_delta_random"])
    ]
    faith_diffs = [
        c - r for c, r in zip(track["faith_delta_cited"], track["faith_delta_random"])
    ]
    logit_diffs = [
        c - r for c, r in zip(track["faith_logit_cited"], track["faith_logit_random"])
    ]

    separation = _mean(nec_diffs)
    ci_lo, ci_hi = bootstrap_ci(nec_diffs, seed=seed)
    effect = paired_cohens_d(nec_diffs)

    area_c = _mean(track["area_cited"])
    area_r = _mean(track["area_random"])
    area_matched = abs(area_c - area_r) < 1e-4

    passed = bool(
        separation >= GATE1_SEPARATION_THRESHOLD
        and ci_lo == ci_lo
        and ci_lo > 0.0
        and area_matched
    )

    return {
        "label": label,
        "cells": cells,
        "n": count,
        "n_random_draws": n_random,
        "area_cited": round(area_c, 6),
        "area_random": round(area_r, 6),
        "area_matched": area_matched,
        "nec_delta_cited": round(_mean(track["nec_delta_cited"]), 4),
        "nec_delta_random": round(_mean(track["nec_delta_random"]), 4),
        "separation_nec_delta": round(separation, 4),
        "separation_ci_lo": round(ci_lo, 4),
        "separation_ci_hi": round(ci_hi, 4),
        "cohens_d_paired": round(effect, 4),
        "suf_delta_cited": round(_mean(track["suf_delta_cited"]), 4),
        "suf_delta_random": round(_mean(track["suf_delta_random"]), 4),
        "separation_suf_delta": round(_mean(suf_diffs), 4),
        "faith_delta_cited": round(_mean(track["faith_delta_cited"]), 4),
        "faith_delta_random": round(_mean(track["faith_delta_random"]), 4),
        "separation_faith_delta": round(_mean(faith_diffs), 4),
        "faith_logit_cited": round(_mean(track["faith_logit_cited"]), 4),
        "faith_logit_random": round(_mean(track["faith_logit_random"]), 4),
        "separation_faith_logit": round(_mean(logit_diffs), 4),
        "delta_beats_logit": bool(_mean(faith_diffs) > _mean(logit_diffs)),
        "raw_delta_move_cited": round(_mean(track["raw_delta_move_cited"]), 4),
        "raw_delta_move_random": round(_mean(track["raw_delta_move_random"]), 4),
        "threshold": GATE1_SEPARATION_THRESHOLD,
        "verdict": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 1: cited-vs-random intervention validity."
    )
    parser.add_argument(
        "--ablation-config",
        default="ablations/configs/table123_rift.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--rows",
        default="full_h4",
        help="comma-separated policy keys from the manifest to audit",
    )
    parser.add_argument(
        "--max-items",
        default=None,
        help="integer, or full/all/auto for every eligible fake row",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--forward-batch-size", type=int, default=None)
    parser.add_argument(
        "--n-random",
        type=int,
        default=4,
        help="independent random-cell draws averaged per sample",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose-model-load", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.ablation_config)
    manifest.setdefault("eval", {})

    parsed_max = _parse_max_items_arg(
        args.max_items,
        eval_csv=manifest.get("data", {}).get("eval_csv"),
        yaml_value=manifest.get("eval", {}).get("max_items", 512),
    )
    if parsed_max is not None:
        manifest["eval"]["max_items"] = int(parsed_max)

    manifest["eval"].setdefault("batch_size", 4)
    manifest["eval"].setdefault("forward_batch_size", 32)
    if args.batch_size is not None:
        manifest["eval"]["batch_size"] = max(1, int(args.batch_size))
    if args.forward_batch_size is not None:
        manifest["eval"]["forward_batch_size"] = max(1, int(args.forward_batch_size))

    seed = int(
        args.seed
        if args.seed is not None
        else manifest.get("eval", {}).get("seed", 3407)
    )

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
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
            "detector.cift_config": manifest["cift"].get(
                "config", "configs/diffusionfake_mixed.yaml"
            ),
            "detector.backbone": manifest["cift"].get("backbone", "convnextv2_base"),
            "detector.strict_identity_gap": True,
            "dataset.split_csv": manifest["data"]["eval_csv"],
            "dataset.max_items": int(manifest["eval"].get("max_items", 512)),
            "dataset.shard_id": 0,
            "dataset.shard_count": 1,
            "intervention.mode": manifest["eval"].get("intervention_mode", "blur"),
            "intervention.topk_frac": float(
                manifest["eval"].get("topk_frac", 0.12)
            ),
        },
    )

    print(f"Loading CIFT checkpoint: {manifest['cift']['ckpt']}", flush=True)
    adapter = CIFTAdapter(
        ckpt_path=manifest["cift"]["ckpt"],
        device=args.device,
        backbone=manifest["cift"].get("backbone", "convnextv2_base"),
        strict_identity_gap=True,
        cift_root=manifest["cift"]["root"],
        config_path=manifest["cift"].get(
            "config", "configs/diffusionfake_mixed.yaml"
        ),
    )
    with _quiet_stdio(not args.verbose_model_load):
        adapter.load_detector()

    output_dir = args.output_dir or os.path.join(
        manifest["eval"].get("output_dir", "experiments/ablations/gates"),
    )
    os.makedirs(output_dir, exist_ok=True)

    wanted = [x.strip() for x in args.rows.split(",") if x.strip()]
    results: List[Dict[str, Any]] = []

    for key in wanted:
        if key not in manifest["policies"]:
            raise RuntimeError(
                f"Unknown policy row {key}. Available: {list(manifest['policies'])}"
            )
        policy = manifest["policies"][key]
        horizon = int(policy["horizon"])
        cells = int(policy.get("max_cells", horizon))

        row = {
            "id": key,
            "variant": policy["run_name"],
            "kind": "policy",
            "policy": key,
        }

        print(f"\n{'=' * 78}\nGATE 1  row={key}  cells={cells}\n{'=' * 78}", flush=True)
        try:
            result = run_gate1_for_row(
                cfg=cfg,
                adapter=adapter,
                manifest=manifest,
                device=args.device,
                row=row,
                cells=cells,
                n_random=int(args.n_random),
                seed=seed,
                label=key,
            )
            result["status"] = "ok"
            result["error"] = ""
        except Exception as exc:
            result = {
                "label": key,
                "cells": cells,
                "n": 0,
                "verdict": "ERROR",
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {result['error']}", flush=True)

        results.append(result)

        if result.get("status") == "ok":
            print(
                f"  separation(nec Δ) = {result['separation_nec_delta']:.4f} "
                f"[{result['separation_ci_lo']:.4f}, {result['separation_ci_hi']:.4f}]  "
                f"d={result['cohens_d_paired']:.3f}  -> {result['verdict']}",
                flush=True,
            )
            print(
                f"  separation(suf Δ) = {result['separation_suf_delta']:.4f}  "
                f"(near 0 => sufficiency saturated under "
                f"{manifest['eval'].get('intervention_mode', 'blur')})",
                flush=True,
            )
            print(
                f"  Δ-sep {result['separation_faith_delta']:.4f} vs "
                f"logit-sep {result['separation_faith_logit']:.4f}  "
                f"-> Δ beats logit: {result['delta_beats_logit']}",
                flush=True,
            )

        csv_path = os.path.join(output_dir, "gate1_validity.csv")
        write_csv(csv_path, results)

    csv_path = os.path.join(output_dir, "gate1_validity.csv")
    json_path = os.path.join(output_dir, "gate1_verdict.json")
    Path(json_path).write_text(
        json.dumps(
            {
                "threshold": GATE1_SEPARATION_THRESHOLD,
                "generated_at": time.time(),
                "intervention_mode": manifest["eval"].get("intervention_mode"),
                "n_random_draws": int(args.n_random),
                "seed": seed,
                "rows": results,
                "all_pass": all(r.get("verdict") == "PASS" for r in results),
            },
            indent=2,
        )
    )
    print(f"\n[wrote] {csv_path}\n[wrote] {json_path}", flush=True)

    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        return 2
    return 0 if all(r.get("verdict") == "PASS" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
