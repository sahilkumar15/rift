#!/usr/bin/env python3
"""Paper-safe Table 6: matched-area intervention validity for RIFT.

Primary question
----------------
Does removing a CIFT-Delta cited region reduce true donor-grounded identity-gap
more than removing a random region of exactly the same grid area?

The evaluator:
  * uses only forged FF++ rows with a real donor/reference image;
  * obtains the cited map from CIFTGapExplainer;
  * converts cited and random controls to exactly k cells on an 8x8 grid;
  * applies blur, mean, noise, and zero necessity interventions;
  * computes normalized Delta necessity drop in [0, 1];
  * stores per-frame paired results;
  * summarizes at video level with a paired cluster-bootstrap 95% CI.

The Gate-1 pass is based on the primary operators blur/mean/noise. Zero is a
robustness-only operator because black patches can be out-of-distribution.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

FAKE_LABELS = {"1", "fake", "forged", "true"}
PRIMARY_OPERATORS = ("blur", "mean", "noise")
ALL_OPERATORS = ("blur", "mean", "noise", "zero")
EPS = 1e-8


def _atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _parse_max_items(value: str | int | None, csv_path: str) -> int:
    if value is None:
        return 2048
    text = str(value).strip().lower()
    if text in {"full", "all", "auto", "0"}:
        count = 0
        with open(csv_path, newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("label", "1")).strip().lower() in FAKE_LABELS:
                    count += 1
        return count
    parsed = int(float(text))
    if parsed <= 0:
        raise ValueError(f"max-items must be positive or full, got {value!r}")
    return parsed


def _resolve_existing_path(raw: str | None, *, csv_path: str) -> str:
    if not raw:
        return ""
    expanded = os.path.expanduser(os.path.expandvars(str(raw).strip()))
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(os.path.dirname(csv_path), expanded))


def _infer_video_id(row: Dict[str, str], image_path: str) -> str:
    for key in (
        "video_id",
        "clip_id",
        "sequence_id",
        "source_video",
        "target_video",
        "video",
    ):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value

    path = Path(image_path)
    parent = path.parent.name or path.stem
    manipulation = str(row.get("manipulation_type", "") or "").strip()
    return f"{manipulation}:{parent}" if manipulation else parent


def _evenly_spaced(items: Sequence[Tuple[int, Dict[str, str]]], cap: int):
    if cap <= 0 or len(items) <= cap:
        return list(items)
    if cap == 1:
        return [items[len(items) // 2]]
    positions = [round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)]
    return [items[pos] for pos in positions]


def _iter_rows(
    csv_path: str,
    *,
    max_items: int,
    shard_id: int,
    shard_count: int,
    frames_per_video: int,
    seed: int,
) -> Iterable[Tuple[int, Dict[str, str]]]:
    """Video-balanced deterministic sampling before GPU sharding.

    FF++ CSVs are often grouped by video. Taking the first N rows would therefore
    give many correlated frames from only a few videos. This selector keeps at
    most `frames_per_video` evenly spaced frames per video, then round-robins
    across videos so a small audit slice covers as many independent videos as
    possible. Set frames_per_video=0 to disable the per-video cap.
    """
    groups: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    eligible_index = 0
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("label", "1")).strip().lower() not in FAKE_LABELS:
                continue
            raw_image = str(row.get("image_path", "") or "")
            video_id = _infer_video_id(row, raw_image)
            groups[video_id].append((eligible_index, row))
            eligible_index += 1

    if not groups:
        return

    selected_by_video: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for video_id, items in groups.items():
        items = sorted(items, key=lambda item: str(item[1].get("image_path", "")))
        selected_by_video[video_id] = _evenly_spaced(items, int(frames_per_video))

    video_ids = sorted(selected_by_video)
    rng = random.Random(int(seed))
    rng.shuffle(video_ids)

    selected: List[Tuple[int, Dict[str, str]]] = []
    round_index = 0
    while len(selected) < max_items:
        added = False
        for video_id in video_ids:
            items = selected_by_video[video_id]
            if round_index < len(items):
                selected.append(items[round_index])
                added = True
                if len(selected) >= max_items:
                    break
        if not added:
            break
        round_index += 1

    for selected_index, item in enumerate(selected):
        if selected_index % shard_count == shard_id:
            yield item


def _normalize_soft_map(mask, *, like):
    import torch
    import torch.nn.functional as F

    if mask is None:
        raise RuntimeError("CIFTGapExplainer returned no mask")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != like.shape[0]:
        if mask.shape[0] == 1:
            mask = mask.repeat(like.shape[0], 1, 1, 1)
        else:
            raise RuntimeError(
                f"Mask batch {mask.shape[0]} does not match image batch {like.shape[0]}"
            )
    mask = mask.detach().to(like.device).float().abs()
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    if mask.shape[-2:] != like.shape[-2:]:
        mask = F.interpolate(mask, size=like.shape[-2:], mode="bilinear", align_corners=False)
    flat = mask.flatten(1)
    lo = flat.amin(dim=1, keepdim=True)
    hi = flat.amax(dim=1, keepdim=True)
    flat = (flat - lo) / (hi - lo + EPS)
    return flat.view_as(mask)


def _topk_grid_mask(soft_mask, *, grid: int, cells: int, image_hw: Tuple[int, int]):
    import torch
    import torch.nn.functional as F

    if cells < 1 or cells > grid * grid:
        raise ValueError(f"cells must be in [1, {grid * grid}], got {cells}")
    pooled = F.adaptive_avg_pool2d(soft_mask, (grid, grid)).flatten(1)
    indices = pooled.topk(cells, dim=1).indices
    grid_mask_flat = torch.zeros(
        soft_mask.shape[0], grid * grid, device=soft_mask.device, dtype=soft_mask.dtype
    )
    grid_mask_flat.scatter_(1, indices, 1.0)
    grid_mask = grid_mask_flat.view(soft_mask.shape[0], 1, grid, grid)
    return F.interpolate(grid_mask, size=image_hw, mode="nearest")


def _random_grid_mask(like, *, grid: int, cells: int, seed: int):
    import torch
    import torch.nn.functional as F

    generator = torch.Generator(device=like.device)
    generator.manual_seed(int(seed))
    order = torch.randperm(grid * grid, generator=generator, device=like.device)
    selected = order[:cells]
    grid_mask = torch.zeros(1, 1, grid, grid, device=like.device, dtype=like.dtype)
    grid_mask.view(-1)[selected] = 1.0
    return F.interpolate(grid_mask, size=like.shape[-2:], mode="nearest")


def _apply_necessity_operator(image, mask, operator: str, *, noise_seed: int):
    import torch
    import torch.nn.functional as F

    operator = str(operator).lower()
    m = (mask > 0.5).to(image.dtype)
    if operator == "blur":
        kernel = 15
        replacement = F.avg_pool2d(image, kernel, stride=1, padding=kernel // 2)
    elif operator == "mean":
        replacement = image.mean(dim=(2, 3), keepdim=True).expand_as(image)
    elif operator == "zero":
        replacement = torch.zeros_like(image)
    elif operator == "noise":
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(noise_seed))
        mean = image.mean(dim=(2, 3), keepdim=True)
        std = image.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(0.05)
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        )
        replacement = (mean + std * noise).clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unknown operator: {operator}")
    return image * (1.0 - m) + replacement * m


def _gap_tensor(adapter, images, donors, *, forward_batch_size: int):
    import torch

    outputs = []
    modes = []
    for start in range(0, images.shape[0], forward_batch_size):
        x = images[start : start + forward_batch_size]
        d = donors[start : start + forward_batch_size]
        if hasattr(adapter, "identity_gap_tensor"):
            gap, mode = adapter.identity_gap_tensor(x, donor=d)
            outputs.append(gap.detach().float().view(-1))
            modes.append(str(mode))
        else:
            vals = []
            for j in range(x.shape[0]):
                result = adapter.identity_gap(x[j : j + 1], donor=d[j : j + 1])
                mode = str(getattr(getattr(result, "mode", "proxy"), "value", getattr(result, "mode", "proxy")))
                vals.append(float(result.value))
                modes.append(mode)
            outputs.append(torch.tensor(vals, device=x.device, dtype=torch.float32))
    if any(mode != "true" for mode in modes):
        raise RuntimeError(
            "Table 6 requires true donor-grounded identity gap, but adapter returned "
            f"modes={sorted(set(modes))}"
        )
    return torch.cat(outputs, dim=0)


def _necessity_drop(e0: float, e1: float) -> float:
    if e0 <= EPS:
        return 0.0
    return max(0.0, min(1.0, (float(e0) - float(e1)) / (float(e0) + EPS)))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _cluster_bootstrap_ci(
    video_diffs: Dict[str, float], *, n_bootstrap: int, seed: int
) -> Tuple[float, float]:
    values = list(video_diffs.values())
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    boot = []
    n = len(values)
    for _ in range(int(n_bootstrap)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _mean(values: Sequence[float]) -> float:
    return sum(float(v) for v in values) / max(1, len(values))


def _load_sample_rows(pattern: str) -> List[Dict[str, str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No Table 6 shard CSVs match: {pattern}")
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Merged Table 6 sample rows are empty")
    return rows


def summarize(
    *,
    sample_glob: str,
    output_dir: str,
    bootstrap: int,
    seed: int,
    min_ratio: float,
    primary_operators: Sequence[str],
) -> int:
    rows = _load_sample_rows(sample_glob)
    by_operator: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_operator[row["operator"]].append(row)

    summary_rows: List[Dict[str, Any]] = []
    operator_passes: Dict[str, bool] = {}

    for op in ALL_OPERATORS:
        op_rows = by_operator.get(op, [])
        if not op_rows:
            continue
        per_video: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {"cited": [], "random": [], "diff": []}
        )
        for row in op_rows:
            video = row["video_id"]
            cited = float(row["cited_delta_drop"])
            random_drop = float(row["random_delta_drop"])
            per_video[video]["cited"].append(cited)
            per_video[video]["random"].append(random_drop)
            per_video[video]["diff"].append(cited - random_drop)

        video_cited = {v: _mean(d["cited"]) for v, d in per_video.items()}
        video_random = {v: _mean(d["random"]) for v, d in per_video.items()}
        video_diff = {v: _mean(d["diff"]) for v, d in per_video.items()}

        cited_mean = _mean(list(video_cited.values()))
        random_mean = _mean(list(video_random.values()))
        diff_mean = _mean(list(video_diff.values()))
        ratio = cited_mean / max(random_mean, EPS)
        ci_low, ci_high = _cluster_bootstrap_ci(
            video_diff,
            n_bootstrap=bootstrap,
            seed=seed + sum(ord(ch) for ch in op),
        )
        passed = bool(ci_low > 0.0 and ratio >= float(min_ratio))
        operator_passes[op] = passed

        area_values = [float(row["area"]) for row in op_rows]
        random_draws = max(int(float(row["random_draws"])) for row in op_rows)
        summary_rows.append(
            {
                "Operator": op,
                "Area": f"{_mean(area_values):.4f}",
                "Cited Δ drop ↑": f"{cited_mean:.4f}",
                "Random Δ drop ↓": f"{random_mean:.4f}",
                "Difference ↑": f"{diff_mean:.4f}",
                "Ratio ↑": f"{ratio:.2f}×",
                "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "Videos": len(per_video),
                "Frames": len(op_rows),
                "Random draws": random_draws,
                "Pass": "✓" if passed else "✗",
            }
        )

    required = [op for op in primary_operators if op in operator_passes]
    overall_pass = (
        len(required) >= 3 and all(operator_passes[op] for op in required)
    )

    output = Path(output_dir)
    table_path = output / "table6_intervention_validity.csv"
    _atomic_write_csv(table_path, summary_rows)

    merged_path = output / "table6_per_sample_merged.csv"
    _atomic_write_csv(merged_path, rows)

    status = {
        "overall_pass": overall_pass,
        "primary_operators": list(primary_operators),
        "operator_passes": operator_passes,
        "pass_rule": {
            "paired_cluster_ci_lower_gt_zero": True,
            "min_ratio": float(min_ratio),
            "minimum_primary_operators": 3,
        },
        "frames": len({row["global_index"] for row in rows}),
        "videos": len({row["video_id"] for row in rows}),
        "sample_glob": sample_glob,
    }
    _atomic_write_text(output / "table6_gate_status.json", json.dumps(status, indent=2) + "\n")

    header = [
        "| Operator | Area | Cited Δ drop ↑ | Random Δ drop ↓ | Difference ↑ | Ratio ↑ | 95% CI | Videos | Frames | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    body = [
        "| {Operator} | {Area} | {Cited Δ drop ↑} | {Random Δ drop ↓} | {Difference ↑} | {Ratio ↑} | {95% CI} | {Videos} | {Frames} | {Pass} |".format(**row)
        for row in summary_rows
    ]
    verdict = "PASS" if overall_pass else "FAIL"
    markdown = "\n".join(
        [
            "# Table 6 — Intervention validity and robustness",
            "",
            *header,
            *body,
            "",
            f"**Gate-1 verdict: {verdict}.** Primary operators are blur, mean, and noise. Zero is robustness-only.",
            "",
        ]
    )
    _atomic_write_text(output / "table6_intervention_validity.md", markdown)

    print(markdown, flush=True)
    print(f"[wrote] {table_path}", flush=True)
    print(f"[wrote] {merged_path}", flush=True)
    print(f"[wrote] {output / 'table6_gate_status.json'}", flush=True)
    return 0 if overall_pass else 3


def evaluate(args: argparse.Namespace) -> int:
    import torch
    from tqdm import tqdm

    from ablations.lib.manifest import load_manifest
    from src.adapters.cift_adapter import CIFTAdapter
    from src.explainers.cift_gap_explainer import CIFTGapExplainer
    from src.gates._io import load_image_minus1_1

    manifest = load_manifest(args.ablation_config)
    csv_path = str(args.csv or manifest["data"]["eval_csv"])
    max_items = _parse_max_items(args.max_items, csv_path)
    operators = tuple(op.strip().lower() for op in args.operators.split(",") if op.strip())
    unknown = sorted(set(operators) - set(ALL_OPERATORS))
    if unknown:
        raise ValueError(f"Unsupported operators {unknown}; choose from {ALL_OPERATORS}")

    if args.shard_count < 1 or not (0 <= args.shard_id < args.shard_count):
        raise ValueError(
            f"Invalid shard {args.shard_id}/{args.shard_count}"
        )

    print(
        "\n".join(
            [
                "=" * 72,
                "RIFT Table 6 — matched-area intervention validity",
                f"config          : {args.ablation_config}",
                f"csv             : {csv_path}",
                f"device          : {args.device}",
                f"shard           : {args.shard_id}/{args.shard_count}",
                f"max_items       : {max_items} global forged frames",
                f"frames/video    : {args.frames_per_video or 'ALL'}",
                f"grid/cells/area : {args.grid}/{args.cells}/{args.cells / (args.grid * args.grid):.4f}",
                f"operators       : {','.join(operators)}",
                f"random_draws    : {args.random_draws}",
                f"forward_batch   : {args.forward_batch_size}",
                "=" * 72,
            ]
        ),
        flush=True,
    )

    adapter = CIFTAdapter(
        ckpt_path=manifest["cift"]["ckpt"],
        device=args.device,
        backbone=manifest["cift"].get("backbone", "convnextv2_base"),
        strict_identity_gap=True,
        cift_root=manifest["cift"]["root"],
        config_path=manifest["cift"].get("config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()
    explainer = CIFTGapExplainer()

    rows_out: List[Dict[str, Any]] = []
    iterator = list(
        _iter_rows(
            csv_path,
            max_items=max_items,
            shard_id=args.shard_id,
            shard_count=args.shard_count,
            frames_per_video=args.frames_per_video,
            seed=args.seed,
        )
    )
    progress = tqdm(
        iterator,
        desc=f"Table6 shard {args.shard_id + 1}/{args.shard_count}",
        unit="img",
        position=args.shard_id,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.5,
        smoothing=0.05,
        file=sys.stdout,
        disable=False,
    )

    for global_index, row in progress:
        image_path = _resolve_existing_path(row.get("image_path"), csv_path=csv_path)
        donor_path = _resolve_existing_path(
            row.get("donor_path") or row.get("source_ref_path"),
            csv_path=csv_path,
        )
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image_path at global index {global_index}: {image_path}")
        if not donor_path or not Path(donor_path).is_file():
            raise FileNotFoundError(
                f"Table 6 requires donor_path/source_ref_path at global index {global_index}: {donor_path}"
            )

        image = load_image_minus1_1(image_path, device=args.device)
        donor = load_image_minus1_1(donor_path, device=args.device)
        video_id = _infer_video_id(row, image_path)

        cited_soft = explainer.explain(image, adapter, donor=donor)
        cited_soft = _normalize_soft_map(cited_soft, like=image)
        cited_mask = _topk_grid_mask(
            cited_soft,
            grid=args.grid,
            cells=args.cells,
            image_hw=(int(image.shape[-2]), int(image.shape[-1])),
        )

        random_masks = [
            _random_grid_mask(
                image,
                grid=args.grid,
                cells=args.cells,
                seed=args.seed + global_index * 10007 + draw * 97,
            )
            for draw in range(args.random_draws)
        ]

        perturbed = [image]
        descriptors: List[Tuple[str, str, int]] = [("original", "original", -1)]
        for op_index, operator in enumerate(operators):
            perturbed.append(
                _apply_necessity_operator(
                    image,
                    cited_mask,
                    operator,
                    noise_seed=args.seed + global_index * 1009 + op_index,
                )
            )
            descriptors.append((operator, "cited", -1))
            for draw, random_mask in enumerate(random_masks):
                perturbed.append(
                    _apply_necessity_operator(
                        image,
                        random_mask,
                        operator,
                        noise_seed=args.seed + global_index * 1009 + op_index,
                    )
                )
                descriptors.append((operator, "random", draw))

        batch = torch.cat(perturbed, dim=0)
        donor_batch = donor.repeat(batch.shape[0], 1, 1, 1)
        gaps = _gap_tensor(
            adapter,
            batch,
            donor_batch,
            forward_batch_size=args.forward_batch_size,
        ).detach().cpu().tolist()
        e0 = float(gaps[0])
        if e0 <= args.min_evidence:
            continue

        cited_by_op: Dict[str, float] = {}
        random_by_op: Dict[str, List[float]] = defaultdict(list)
        for desc, gap in zip(descriptors[1:], gaps[1:]):
            operator, kind, _draw = desc
            drop = _necessity_drop(e0, float(gap))
            if kind == "cited":
                cited_by_op[operator] = drop
            else:
                random_by_op[operator].append(drop)

        area = float(cited_mask.mean().item())
        expected_area = args.cells / float(args.grid * args.grid)
        if abs(area - expected_area) > 1e-6:
            raise RuntimeError(
                f"Mask area mismatch: got {area}, expected {expected_area}"
            )

        for operator in operators:
            cited_drop = cited_by_op[operator]
            random_drop = _mean(random_by_op[operator])
            rows_out.append(
                {
                    "global_index": global_index,
                    "video_id": video_id,
                    "image_path": image_path,
                    "operator": operator,
                    "grid": args.grid,
                    "selected_cells": args.cells,
                    "area": f"{area:.8f}",
                    "e0_delta": f"{e0:.8f}",
                    "cited_delta_drop": f"{cited_drop:.8f}",
                    "random_delta_drop": f"{random_drop:.8f}",
                    "paired_difference": f"{cited_drop - random_drop:.8f}",
                    "ratio": f"{cited_drop / max(random_drop, EPS):.8f}",
                    "random_draws": args.random_draws,
                    "identity_gap_mode": "true",
                }
            )

        if rows_out:
            recent = rows_out[-len(operators) :]
            progress.set_postfix(
                diff=f"{_mean([float(r['paired_difference']) for r in recent]):.3f}",
                area=f"{area:.4f}",
            )

    if not rows_out:
        raise RuntimeError(
            "No valid Table 6 samples were produced. Check donor paths and min-evidence."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table6_samples.csv"
    _atomic_write_csv(out_path, rows_out)
    print(f"[wrote] {out_path} rows={len(rows_out)}", flush=True)
    return 0


def selftest() -> int:
    import torch

    image = torch.zeros(1, 3, 32, 32)
    soft = torch.arange(32 * 32, dtype=torch.float32).view(1, 1, 32, 32)
    cited = _topk_grid_mask(soft, grid=8, cells=4, image_hw=(32, 32))
    random_mask = _random_grid_mask(image, grid=8, cells=4, seed=7)
    assert abs(float(cited.mean().item()) - 4 / 64) < 1e-7
    assert abs(float(random_mask.mean().item()) - 4 / 64) < 1e-7
    assert set(torch.unique(cited).tolist()).issubset({0.0, 1.0})
    assert set(torch.unique(random_mask).tolist()).issubset({0.0, 1.0})

    synthetic = {
        "v1": 0.10,
        "v2": 0.20,
        "v3": 0.15,
        "v4": 0.18,
        "v5": 0.12,
    }
    lo, hi = _cluster_bootstrap_ci(synthetic, n_bootstrap=500, seed=1)
    assert lo > 0 and hi > lo
    assert abs(_necessity_drop(1.0, 0.7) - 0.3) < 1e-6
    print("[ok] Table 6 selftest passed")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RIFT Table 6 intervention validity")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--ablation-config", default="ablations/configs/table123_rift.yaml"
    )
    parser.add_argument("--csv", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", default="2048")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", required=False, default="experiments/ablations/rift_table6")
    parser.add_argument("--frames-per-video", type=int, default=0)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--operators", default="blur,mean,noise,zero")
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--forward-batch-size", type=int, default=32)
    parser.add_argument("--min-evidence", type=float, default=1e-6)
    parser.add_argument("--sample-glob", default="")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--primary-operators", default="blur,mean,noise")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.selftest:
        return selftest()
    if args.summarize_only:
        if not args.sample_glob:
            raise SystemExit("--sample-glob is required with --summarize-only")
        primary = tuple(
            op.strip().lower()
            for op in args.primary_operators.split(",")
            if op.strip()
        )
        return summarize(
            sample_glob=args.sample_glob,
            output_dir=args.output_dir,
            bootstrap=args.bootstrap,
            seed=args.seed,
            min_ratio=args.min_ratio,
            primary_operators=primary,
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
