#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ablations.lib.manifest import load_manifest, policy_ckpt
from ablations.lib.explainers import (
    CausalSelectExplainer,
    PolicyExplainer,
    logit_to_evidence,
    predict_evidence,
)

TICK = "✓"
CROSS = "✗"
FAKE_LABELS = {"1", "fake", "forged", "True", "true"}


def tick(value) -> str:
    return TICK if bool(value) else CROSS


def read_existing_csv(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="") as handle:
        return {
            str(row.get("ID", "")): dict(row)
            for row in csv.DictReader(handle)
        }


def _count_eligible_csv_rows(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "label" in reader.fieldnames:
                return sum(
                    1
                    for row in reader
                    if str(row.get("label", "1")).strip() in FAKE_LABELS
                )
    except Exception:
        pass
    try:
        with open(path, "rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        return 0


def _parse_max_items_arg(
    value,
    *,
    eval_csv: Optional[str] = None,
    yaml_value: Any = None,
) -> Optional[int]:
    if value is None:
        value = yaml_value
    if value is None:
        return None

    text = str(value).strip()
    low = text.lower()
    if low in ("full", "all", "auto", "0"):
        count = _count_eligible_csv_rows(eval_csv or "")
        return count if count > 0 else None
    if low in ("", "none", "null"):
        return None
    return int(float(text))


def _shard_total(max_items: int, shard_id: int, shard_count: int) -> int:
    if shard_count <= 1:
        return int(max_items)
    if max_items <= 0:
        return 0
    return max(0, (max_items - 1 - shard_id) // shard_count + 1)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return

    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _atomic_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, target)


@contextlib.contextmanager
def _quiet_stdio(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _iter_batches(
    iterable: Iterable[Tuple[Any, Any, Any]],
    batch_size: int,
) -> Iterator[Tuple[Any, Any, List[Any]]]:
    import torch

    images = []
    donors = []
    masks = []

    def flush():
        if not images:
            return None
        image_batch = torch.cat(images, dim=0)
        donor_batch = None
        if all(d is not None for d in donors):
            donor_batch = torch.cat(donors, dim=0)
        elif any(d is not None for d in donors):
            raise RuntimeError("A batch mixed donor-present and donor-missing samples.")
        result = image_batch, donor_batch, list(masks)
        images.clear()
        donors.clear()
        masks.clear()
        return result

    for image, donor, gt in iterable:
        images.append(image)
        donors.append(donor)
        masks.append(gt)
        if len(images) >= batch_size:
            result = flush()
            if result is not None:
                yield result

    result = flush()
    if result is not None:
        yield result


def _mask_area_tensor(mask, topk_frac: float):
    import torch

    mask = mask.float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    with torch.no_grad():
        hard_like = bool(
            ((mask <= 1e-6) | (mask >= 1.0 - 1e-6)).all().item()
        )

    if hard_like:
        binary = (mask > 1e-6).float()
    else:
        flat = mask.flatten(1)
        k = max(
            1,
            min(
                flat.shape[1],
                int(round(float(topk_frac) * flat.shape[1])),
            ),
        )
        threshold = flat.topk(k, dim=1).values[:, -1:].clamp(min=1e-12)
        binary = (flat >= threshold).view_as(mask).float()

    return binary.flatten(1).mean(dim=1)


def row_prefix(row: Dict[str, Any]) -> Dict[str, Any]:
    output = {
        "ID": row["id"],
        "Variant": row["variant"],
    }

    if "mask_source" in row:
        output["Mask source"] = row["mask_source"]

    if "delta_g" in row:
        output["ΔG"] = tick(row.get("delta_g"))
        output["NS"] = tick(row.get("ns"))
        output["RP"] = tick(row.get("rp"))

    if "necessity" in row:
        output["Necessity"] = tick(row.get("necessity"))
        output["Sufficiency"] = tick(row.get("sufficiency"))
        output["Sparsity"] = tick(row.get("sparsity"))

    if "horizon" in row:
        output["Horizon"] = row.get("horizon")

    return output


def make_explainer(row: Dict[str, Any], manifest: Dict[str, Any], device: str):
    from src.explainers.cift_gap_explainer import CIFTGapExplainer
    from src.explainers.gradcam_explainer import GradCAMExplainer
    from src.explainers.random_cell_explainer import RandomCellExplainer
    from src.explainers.random_explainer import RandomExplainer

    kind = row["kind"]
    eval_cfg = manifest["eval"]
    defaults = manifest["policy_defaults"]
    forward_batch_size = int(eval_cfg.get("forward_batch_size", 32))

    if kind == "random_cell":
        # Area- AND geometry-matched control. cells must equal the horizon
        # of the policy this row is a control for.
        return RandomCellExplainer(
            cells=int(row.get("cells", 4)),
            grid=int(defaults.get("grid", 8)),
            seed=int(eval_cfg.get("seed", 3407)) + int(row.get("id", 0) or 0)
            if str(row.get("id", "")).isdigit()
            else int(eval_cfg.get("seed", 3407)),
        )

    if kind == "random":
        # LEGACY: scattered-pixel mask. NOT geometry-matched to a grid
        # policy; under blur it is a near-identity intervention. Kept only
        # for backward compatibility. Use kind: random_cell.
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
            grid=int(defaults.get("grid", 8)),
            horizon=int(
                manifest["policies"].get("full_h4", {}).get("horizon", 4)
            ),
            candidate_pool=int(eval_cfg.get("candidate_pool", 16)),
            intervention_mode=str(eval_cfg.get("intervention_mode", "blur")),
            topk_frac=float(eval_cfg.get("topk_frac", 0.12)),
            forward_batch_size=forward_batch_size,
        )

    if kind == "policy":
        key = row["policy"]
        policy = manifest["policies"][key]
        checkpoint = policy_ckpt(manifest, key)
        print(f"  policy={key} ckpt={checkpoint}", flush=True)

        return PolicyExplainer(
            checkpoint,
            grid=int(defaults.get("grid", 8)),
            hidden=int(defaults.get("hidden", 256)),
            feat_dim=int(defaults.get("feat_dim", 1024)),
            horizon=int(policy["horizon"]),
            reward_preset=str(policy["reward_preset"]),
            intervention_mode=str(
                eval_cfg.get("intervention_mode", "blur")
            ),
            topk_frac=float(
                eval_cfg.get("topk_frac", 0.12)
            ),
            device=device,
            forward_batch_size=forward_batch_size,
            allow_stop=bool(
                policy.get("allow_stop", False)
            ),
            min_cells=int(
                policy.get("min_cells", 1)
            ),
            max_cells=policy.get(
                "max_cells",
                policy.get("horizon", 1),
            ),
            force_min_cells=bool(
                policy.get("force_min_cells", True)
            ),
            forbid_revisit=bool(
                policy.get("forbid_revisit", True)
            ),
            state_blind=bool(
                policy.get("state_blind", False)
            ),
        )

    raise RuntimeError(f"Unknown explainer kind={kind}")


def audit_explainer(
    cfg,
    adapter,
    explainer,
    manifest: Dict[str, Any],
    device: str,
    *,
    progress_desc: str,
    disable_tqdm: bool = False,
    on_progress=None,
) -> Dict[str, Any]:
    import torch
    from tqdm import tqdm

    from src.audit.ablation_runner import iter_audit_samples
    from src.faithfulness.faithfulness_score import compute_rift_score_tensor
    from src.interventions.interventions import apply_necessity, apply_sufficiency
    from src.rl.reward import get_reward_weights

    eval_cfg = manifest["eval"]
    mode = str(eval_cfg.get("intervention_mode", "blur"))
    topk = float(eval_cfg.get("topk_frac", 0.12))
    max_items = int(eval_cfg.get("max_items", 512))
    shard_id = int(eval_cfg.get("shard_id", 0) or 0)
    shard_count = int(eval_cfg.get("shard_count", 1) or 1)
    batch_size = max(1, int(eval_cfg.get("batch_size", 4)))
    forward_batch_size = max(1, int(eval_cfg.get("forward_batch_size", 32)))
    weights = get_reward_weights("full_rift")

    metric_keys = [
        "necessity_delta",
        "sufficiency_delta",
        "faithfulness_delta",
        "necessity_logit",
        "sufficiency_logit",
        "faithfulness_logit",
        "mask_area",
        "rift_score",
    ]
    sums = {key: 0.0 for key in metric_keys}
    count = 0

    total = _shard_total(max_items, shard_id, shard_count)
    sample_iter = iter_audit_samples(cfg, device=device, n=max_items)
    batch_iter = _iter_batches(sample_iter, batch_size)

    bar = tqdm(
        total=total,
        desc=progress_desc,
        unit="img",
        dynamic_ncols=True,
        leave=True,
        mininterval=1.0,
        file=sys.stderr,
        disable=disable_tqdm,
    )

    try:
        for image, donor, _ in batch_iter:
            mask = explainer.explain(image, adapter, donor=donor)
            nec_image = apply_necessity(image, mask, mode, topk)
            suf_image = apply_sufficiency(image, mask, mode, topk)

            cached = None
            if hasattr(explainer, "cached_original_evidence"):
                cached = explainer.cached_original_evidence(image)

            if cached is not None and bool(cached.get("complete", False)):
                batch = int(image.shape[0])
                e0_gap = cached["gap"].to(image.device).float().view(-1)
                e0_logit = cached["logit"].to(image.device).float().view(-1)
                gap_nec = cached["gap_nec"].to(image.device).float().view(-1)
                gap_suf = cached["gap_suf"].to(image.device).float().view(-1)
                logit_nec = cached["logit_nec"].to(image.device).float().view(-1)
                logit_suf = cached["logit_suf"].to(image.device).float().view(-1)
                evidence_mode = str(cached.get("mode", "proxy"))
            elif cached is not None:
                all_images = torch.cat([nec_image, suf_image], dim=0)
                all_donors = (
                    torch.cat([donor, donor], dim=0)
                    if donor is not None
                    else None
                )
                raw_logits, gaps, evidence_mode, _ = predict_evidence(
                    adapter,
                    all_images,
                    all_donors,
                    max_batch=forward_batch_size,
                )
                batch = int(image.shape[0])
                e0_gap = cached["gap"].to(image.device).float().view(-1)
                e0_logit = cached["logit"].to(image.device).float().view(-1)
                gap_nec = gaps[:batch]
                gap_suf = gaps[batch:]
                logit_ev = logit_to_evidence(raw_logits)
                logit_nec = logit_ev[:batch]
                logit_suf = logit_ev[batch:]
                evidence_mode = str(cached.get("mode", evidence_mode))
            else:
                all_images = torch.cat([image, nec_image, suf_image], dim=0)
                all_donors = (
                    torch.cat([donor, donor, donor], dim=0)
                    if donor is not None
                    else None
                )
                raw_logits, gaps, evidence_mode, _ = predict_evidence(
                    adapter,
                    all_images,
                    all_donors,
                    max_batch=forward_batch_size,
                )
                batch = int(image.shape[0])
                logit_ev = logit_to_evidence(raw_logits)
                e0_gap = gaps[:batch]
                gap_nec = gaps[batch: 2 * batch]
                gap_suf = gaps[2 * batch:]
                e0_logit = logit_ev[:batch]
                logit_nec = logit_ev[batch: 2 * batch]
                logit_suf = logit_ev[2 * batch:]

            area = _mask_area_tensor(mask, topk)
            _, components = compute_rift_score_tensor(
                e0_delta=e0_gap,
                e_nec_delta=gap_nec,
                e_suf_delta=gap_suf,
                e0_logit=e0_logit,
                e_nec_logit=logit_nec,
                e_suf_logit=logit_suf,
                mask_area=area,
                identity_gap_mode=evidence_mode,
                weights=weights,
            )

            for key in metric_keys:
                value = components[key]
                sums[key] += float(value.detach().float().sum().item())

            count += batch
            bar.update(batch)

            if count % max(batch_size, 16) == 0:
                bar.set_postfix(
                    rift=f"{sums['rift_score'] / max(1, count):.4f}",
                    mask=f"{sums['mask_area'] / max(1, count):.4f}",
                    refresh=False,
                )

            if on_progress is not None:
                on_progress(count, total)
    finally:
        bar.close()

    if count == 0:
        raise RuntimeError(
            "No samples were evaluated. Check dataset.split_csv, donor paths, "
            "strict_identity_gap, shard_id/shard_count, and MAX_ITEMS."
        )

    output = {key: sums[key] / count for key in metric_keys}
    output["n"] = count
    return output


def run_table(
    table_key: str,
    table_spec: Dict[str, Any],
    manifest: Dict[str, Any],
    cfg,
    adapter,
    device: str,
    *,
    existing: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_ok_existing: bool = False,
    checkpoint_path: Optional[str] = None,
    disable_tqdm: bool = False,
    progress_file: Optional[str] = None,
    overall_offset: int = 0,
    overall_total: int = 0,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    existing = existing or {}
    target_n = _shard_total(
        int(manifest.get("eval", {}).get("max_items", 0) or 0),
        int(manifest.get("eval", {}).get("shard_id", 0) or 0),
        int(manifest.get("eval", {}).get("shard_count", 1) or 1),
    )

    for row_index, row in enumerate(table_spec["rows"]):
        print(f"\n[{table_key}] ID={row['id']} {row['variant']}", flush=True)
        output = row_prefix(row)
        row_offset = overall_offset + row_index * target_n

        def update_progress(done: int, total: int):
            _atomic_json(
                progress_file,
                {
                    "table": table_key,
                    "row_id": str(row["id"]),
                    "variant": row["variant"],
                    "row_done": int(done),
                    "row_total": int(total),
                    "overall_done": int(row_offset + done),
                    "overall_total": int(overall_total),
                    "updated_at": time.time(),
                },
            )

        old = existing.get(str(row["id"]))
        if skip_ok_existing and old and str(old.get("status", "")).lower() == "ok":
            try:
                old_n = int(float(old.get("n", 0)))
            except Exception:
                old_n = 0
            if target_n <= 0 or old_n >= target_n:
                print(f"  SKIP existing ok n={old_n}", flush=True)
                rows_out.append(old)
                update_progress(target_n, target_n)
                if checkpoint_path:
                    write_csv(checkpoint_path, rows_out)
                continue

        try:
            explainer = make_explainer(row, manifest, device)
            metrics = audit_explainer(
                cfg,
                adapter,
                explainer,
                manifest,
                device,
                progress_desc=f"{table_key}:ID={row['id']} {row['variant']}",
                disable_tqdm=disable_tqdm,
                on_progress=update_progress,
            )

            output["Nec Δ ↑"] = fmt(metrics["necessity_delta"])
            output["Suf Δ ↑"] = fmt(metrics["sufficiency_delta"])
            output["Faith Δ ↑"] = fmt(metrics["faithfulness_delta"])
            output["Faith logit ↑"] = fmt(metrics["faithfulness_logit"])
            output["Mask area ↓"] = fmt(metrics["mask_area"])
            output["RIFT score ↑"] = fmt(metrics["rift_score"])
            output["n"] = int(metrics["n"])
            output["status"] = "ok"
            output["error"] = ""

            print(
                f"  ok n={output['n']} faithΔ={output['Faith Δ ↑']} "
                f"mask={output['Mask area ↓']} rift={output['RIFT score ↑']}",
                flush=True,
            )
            update_progress(target_n, target_n)

        except Exception as exc:
            output["n"] = 0
            output["status"] = "FAILED"
            output["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {output['error']}", flush=True)
            update_progress(target_n, target_n)

        rows_out.append(output)

        # Critical durability fix: completed rows survive interruption.
        if checkpoint_path:
            write_csv(checkpoint_path, rows_out)

    return rows_out


def _append_history(path: str, rows: List[Dict[str, Any]]) -> None:
    old_rows: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, newline="") as handle:
            old_rows = list(csv.DictReader(handle))
    write_csv(path, old_rows + rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-config",
        default="ablations/configs/table123_rift.yaml",
    )
    parser.add_argument(
        "--tables",
        default="table1_component,table2_objective,table3_horizon",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-items",
        default=None,
        help="integer, or full/all/auto for all eligible fake validation rows",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--forward-batch-size", type=int, default=None)
    parser.add_argument("--skip-ok-existing", action="store_true")
    parser.add_argument(
        "--shard-id",
        type=int,
        default=int(os.environ.get("SHARD_ID", 0)),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=int(os.environ.get("SHARD_COUNT", 1)),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override eval.output_dir; used by parallel shards",
    )
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--verbose-model-load", action="store_true")
    args = parser.parse_args()

    if args.shard_count < 1:
        raise RuntimeError(f"Invalid shard_count={args.shard_count}")
    if args.shard_id < 0 or args.shard_id >= args.shard_count:
        raise RuntimeError(
            f"Invalid shard_id={args.shard_id}, shard_count={args.shard_count}"
        )

    manifest = load_manifest(args.ablation_config)
    manifest.setdefault("eval", {})

    parsed_max = _parse_max_items_arg(
        args.max_items,
        eval_csv=manifest.get("data", {}).get("eval_csv"),
        yaml_value=manifest.get("eval", {}).get("max_items", 512),
    )
    if parsed_max is not None:
        manifest["eval"]["max_items"] = int(parsed_max)

    if args.batch_size is not None:
        manifest["eval"]["batch_size"] = max(1, int(args.batch_size))
    else:
        manifest["eval"].setdefault("batch_size", 4)

    if args.forward_batch_size is not None:
        manifest["eval"]["forward_batch_size"] = max(
            1,
            int(args.forward_batch_size),
        )
    else:
        manifest["eval"].setdefault("forward_batch_size", 32)

    if args.output_dir:
        manifest["eval"]["output_dir"] = args.output_dir

    manifest["eval"]["shard_id"] = int(args.shard_id)
    manifest["eval"]["shard_count"] = int(args.shard_count)

    seed = int(manifest.get("eval", {}).get("seed", 3407))
    random.seed(seed + args.shard_id)

    import torch

    torch.manual_seed(seed + args.shard_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + args.shard_id)
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
                "config",
                "configs/diffusionfake_mixed.yaml",
            ),
            "detector.backbone": manifest["cift"].get(
                "backbone",
                "convnextv2_base",
            ),
            "detector.strict_identity_gap": True,
            "dataset.split_csv": manifest["data"]["eval_csv"],
            "dataset.max_items": int(manifest["eval"].get("max_items", 512)),
            "dataset.shard_id": int(args.shard_id),
            "dataset.shard_count": int(args.shard_count),
            "intervention.mode": manifest["eval"].get(
                "intervention_mode",
                "blur",
            ),
            "intervention.topk_frac": float(
                manifest["eval"].get("topk_frac", 0.12)
            ),
        },
    )

    wanted = [x.strip() for x in args.tables.split(",") if x.strip()]
    row_count = sum(len(manifest["tables"][key]["rows"]) for key in wanted)
    shard_n = _shard_total(
        int(manifest["eval"].get("max_items", 0) or 0),
        args.shard_id,
        args.shard_count,
    )
    overall_total = row_count * shard_n

    _atomic_json(
        args.progress_file,
        {
            "table": "loading",
            "row_id": "",
            "variant": "Loading CIFT",
            "row_done": 0,
            "row_total": shard_n,
            "overall_done": 0,
            "overall_total": overall_total,
            "updated_at": time.time(),
        },
    )

    print(
        f"[eval] max_items={manifest['eval'].get('max_items')} "
        f"shard={args.shard_id}/{args.shard_count} device={args.device} "
        f"batch={manifest['eval']['batch_size']} "
        f"forward_batch={manifest['eval']['forward_batch_size']} "
        f"output_dir={manifest['eval'].get('output_dir')}",
        flush=True,
    )
    print(f"Loading CIFT checkpoint: {manifest['cift']['ckpt']}", flush=True)

    adapter = CIFTAdapter(
        ckpt_path=manifest["cift"]["ckpt"],
        device=args.device,
        backbone=manifest["cift"].get("backbone", "convnextv2_base"),
        strict_identity_gap=True,
        cift_root=manifest["cift"]["root"],
        config_path=manifest["cift"].get(
            "config",
            "configs/diffusionfake_mixed.yaml",
        ),
    )

    try:
        with _quiet_stdio(not args.verbose_model_load):
            adapter.load_detector()
    except Exception:
        print("[error] CIFT model loading failed.", file=sys.stderr, flush=True)
        raise

    output_dir = manifest["eval"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    combined: List[Dict[str, Any]] = []
    offset = 0

    for key in wanted:
        if key not in manifest["tables"]:
            raise RuntimeError(
                f"Unknown table key {key}. Available: {list(manifest['tables'])}"
            )

        spec = manifest["tables"][key]
        print(f"\n{'=' * 80}\n{key}\n{'=' * 80}", flush=True)

        output_path = os.path.join(output_dir, spec["filename"])
        existing = read_existing_csv(output_path)
        rows = run_table(
            key,
            spec,
            manifest,
            cfg,
            adapter,
            args.device,
            existing=existing,
            skip_ok_existing=bool(args.skip_ok_existing),
            checkpoint_path=output_path,
            disable_tqdm=bool(args.no_tqdm),
            progress_file=args.progress_file,
            overall_offset=offset,
            overall_total=overall_total,
        )
        write_csv(output_path, rows)
        print(f"[wrote] {output_path}", flush=True)

        for row in rows:
            combined.append({"table": key, **row})

        offset += len(spec["rows"]) * shard_n

    combined_path = os.path.join(output_dir, "combined_tables_1_2_3.csv")
    write_csv(combined_path, combined)
    print(f"\n[done] combined: {combined_path}", flush=True)

    history_path = os.path.join(output_dir, "metrics_history.csv")
    run_ts = int(time.time())
    history_rows = [
        {
            "run_ts": run_ts,
            "max_items": manifest["eval"].get("max_items"),
            "shard_id": args.shard_id,
            "shard_count": args.shard_count,
            **row,
        }
        for row in combined
    ]
    if history_rows:
        _append_history(history_path, history_rows)
        print(f"[history] updated: {history_path}", flush=True)

    _atomic_json(
        args.progress_file,
        {
            "table": "done",
            "row_id": "",
            "variant": "Complete",
            "row_done": shard_n,
            "row_total": shard_n,
            "overall_done": overall_total,
            "overall_total": overall_total,
            "updated_at": time.time(),
        },
    )

    failed = [row for row in combined if row.get("status") != "ok"]
    if failed:
        print(
            f"[warn] {len(failed)} rows failed. Read the CSV error column.",
            flush=True,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
