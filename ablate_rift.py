# Path: ablate_rift.py
# Status: NEW
"""
ablate_rift.py — single entrypoint that runs RIFT's ablation blocks.

Dispatch:
  --mode dry-run     : print the ✓/✗ cell plan for the enabled blocks (NO torch, NO model).
  --mode ablations   : run the enabled blocks (Block 0/1/2/4 over the audit slice; Block 3
                       over per-checkpoint rows). Writes per-cell CSV/JSON + a combined table.
  --block N          : restrict to one block (0..4); default = all enabled in config.
  --only a,b         : restrict to a subset of cell ids / explainers.
  --seeds 0,1,2      : multi-seed (mean±std).

Examples:
  python ablate_rift.py -c configs/rift_general.yaml --mode dry-run
  python ablate_rift.py -c configs/rift_general.yaml --mode ablations --seeds 0,1,2 \
      --cift-root /scratch/.../ImageDifussionFake detector.cift_ckpt=/path/cift.ckpt \
      dataset.split_csv=data/slices/example_ffpp_forged.csv
  python ablate_rift.py -c configs/rift_general.yaml --mode ablations --block 1 \
      --only generic_logit,delta_grounded
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys

from iganer.rift.utils.config import load_config, merge_overrides
from iganer.rift.audit.ablation_runner import (
    parse_overrides, plan_blocks, render_matrix,
)

BLOCK_KEYS = {
    0: "block0_validity", 1: "block1_method", 2: "block2_audit",
    3: "block3_correlation", 4: "block4_rl",
}


def _enabled_blocks(cfg, block_arg):
    cfgb = dict(cfg.get_dotted("ablation.blocks", {}) or {})
    enabled = {k: bool(v) for k, v in cfgb.items()}
    for k in BLOCK_KEYS.values():
        enabled.setdefault(k, True)
    if block_arg is not None:
        want = BLOCK_KEYS[int(block_arg)]
        enabled = {k: (k == want) for k in enabled}
    return enabled


def _write_table(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="RIFT ablation runner")
    ap.add_argument("-c", "--config", default="configs/rift_general.yaml")
    ap.add_argument("--spec", default=None, help="override ablation spec path")
    ap.add_argument("--mode", default="dry-run",
                    choices=["dry-run", "ablations"])
    ap.add_argument("--block", default=None, help="restrict to one block 0..4")
    ap.add_argument("--only", default=None, help="comma list of cell ids/explainers")
    ap.add_argument("--seeds", default=None, help="comma list, overrides config")
    ap.add_argument("--cift-root", dest="cift_root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ov = parse_overrides(args.overrides)
    if args.cift_root:
        ov["detector.cift_root"] = args.cift_root
    if args.device:
        ov["device"] = args.device
    cfg = merge_overrides(cfg, ov)

    spec_path = args.spec or cfg.get_dotted("ablation.spec", "configs/ablations_rift.yaml")
    spec = load_config(spec_path)

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    enabled = _enabled_blocks(cfg, args.block)
    plans = plan_blocks(spec, enabled, only=only)

    # ── DRY RUN (no torch) ────────────────────────────────────────────────────
    if args.mode == "dry-run":
        print("RIFT ablation plan (dry-run; no model touched)")
        print("config :", args.config)
        print("spec   :", spec_path)
        print("blocks :", [k for k, v in enabled.items() if v])
        print(render_matrix(plans))
        print("\n[dry-run] OK. Re-run with --mode ablations on Katz (needs --cift-root + "
              "detector.cift_ckpt + dataset.split_csv) to execute.")
        return 0

    # ── REAL RUN (torch + CIFT) ───────────────────────────────────────────────
    try:
        import torch  # noqa
    except Exception:
        print("[ablate_rift] torch is not installed in this environment. Install the runtime "
              "extras on Katz:  pip install -e \".[runtime]\"   Then re-run.", file=sys.stderr)
        return 2

    ckpt = cfg.get_dotted("detector.cift_ckpt")
    cift_root = cfg.get_dotted("detector.cift_root")
    if not ckpt or not cift_root:
        print("[ablate_rift] need detector.cift_ckpt and detector.cift_root (or --cift-root) "
              "for a real run.", file=sys.stderr)
        return 2

    from iganer.rift.adapters.cift_adapter import CIFTAdapter
    from iganer.rift.audit.ablation_runner import (
        run_block1, block1_contrasts, run_block2, run_block3,
    )
    from iganer.rift.audit.leaderboard import build_leaderboard  # noqa

    device = cfg.get_dotted("device", "cuda")
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else (cfg.get_dotted("ablation.seeds") or [0]))

    adapter = CIFTAdapter(
        ckpt_path=ckpt, device=device,
        backbone=cfg.get_dotted("detector.backbone", "convnextv2_base"),
        strict_identity_gap=bool(cfg.get_dotted("detector.strict_identity_gap", True)),
        cift_root=cift_root,
        config_path=cfg.get_dotted("detector.cift_config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    out_dir = cfg.get_dotted("output.per_cell_dir", "outputs/cells")
    os.makedirs(out_dir, exist_ok=True)
    table_rows = []

    for p in plans:
        blk = p["block"]
        print(f"\n══ running {blk} ══")
        try:
            if blk == "block1_method":
                rows = run_block1(cfg, adapter, p["cells"], seeds=seeds, device=device)
                for r in rows:
                    r["block"] = blk
                table_rows.extend(rows)
                contrasts = block1_contrasts(rows)
                with open(os.path.join(out_dir, "block1_contrasts.json"), "w") as f:
                    json.dump(contrasts, f, indent=2)
                print("  contrasts:", contrasts)

            elif blk == "block2_audit":
                cols, table = run_block2(cfg, adapter, p["explainers"], device=device)
                rule = p.get("expose_rule", {})
                minp = rule.get("min_plausibility", 0.6)
                maxf = rule.get("max_faithfulness", 0.35)
                for r in table:
                    d = dict(zip(cols, r))
                    d.setdefault("plausibility_iou", None)
                    exposed = (d.get("plausibility_iou") or 0) >= minp and \
                              (d.get("faithfulness_delta") or 0) <= maxf
                    d["exposed"] = exposed
                    d["block"] = blk
                    table_rows.append(d)
                _write_table(cfg.get_dotted("output.leaderboard_csv", "outputs/leaderboard.csv"),
                             [r for r in table_rows if r.get("block") == blk])

            elif blk == "block3_correlation":
                rows = _load_checkpoint_rows(cfg)
                if not rows:
                    print("  [skip] block3 needs per-checkpoint rows. Provide a CSV via "
                          "output.correlation_input or checkpoints.*; see README.")
                else:
                    res = run_block3(rows, spec.get("block3_correlation", {}))
                    with open(cfg.get_dotted("output.correlation_json", "outputs/correlation.json"), "w") as f:
                        json.dump(res, f, indent=2, default=str)
                    print("  correlation written.")

            elif blk == "block4_rl":
                print("  [note] Block 4 (RL horizon sweep) trains the repair policy per cell; "
                      "launch via scripts/run_rift.sh --mode train with rl.horizon overrides. "
                      "The cell plan is recorded below for the table.")
                for c in p["cells"]:
                    table_rows.append({"block": blk, "id": c["id"], "H": c["horizon"],
                                       "game": c["game"], "protection": c["protection"],
                                       "faithfulness_delta": None, "n": 0})

            elif blk == "block0_validity":
                print("  [note] Block 0 is the Gate-1 separation test; run scripts/run_rift.sh "
                      "--mode gates (or rift-gate1) for the authoritative PASS/FAIL.")
        except Exception as e:  # robust: one block failing never kills the others
            print(f"  [ERROR in {blk}] {type(e).__name__}: {e}", file=sys.stderr)

    table_path = cfg.get_dotted("output.table_csv", "outputs/table_rift.csv")
    _write_table(table_path, table_rows)
    print(f"\n[ablate_rift] combined table -> {table_path}  ({len(table_rows)} rows)")
    return 0


def _load_checkpoint_rows(cfg):
    """Block-3 input: a CSV with faithfulness,in_domain_auc,plausibility,zero_shot_auc."""
    path = cfg.get_dotted("output.correlation_input")
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k in ("faithfulness", "in_domain_auc", "plausibility", "zero_shot_auc"):
                if r.get(k) not in (None, ""):
                    row[k] = float(r[k])
            if "zero_shot_auc" in row:
                rows.append(row)
    return rows


if __name__ == "__main__":
    sys.exit(main())
