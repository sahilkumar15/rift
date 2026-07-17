#!/usr/bin/env python3
# Path: ablations/preflight_table1.py
# Status: NEW
"""Check every prerequisite for a full Table 1 run BEFORE burning GPU hours.

A full Table 1 is ~54k image decodes x 15 forwards through an 859M model x
however many intervention modes you sweep. Discovering at hour three that
logit_h4 was never trained, or that the CelebDF CSV has no donor column, is an
expensive way to learn it. This runs in seconds and tells you exactly what is
missing and how to fix it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

OK, BAD, WARN = "  OK  ", " MISS ", " WARN "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ablations/configs/table1.yaml")
    args = ap.parse_args()

    t1 = yaml.safe_load(Path(args.config).read_text())
    problems, warnings_ = [], []

    print("=" * 78)
    print("TABLE 1 PREFLIGHT")
    print("=" * 78)

    print("\nCIFT checkpoint")
    ck = t1["cift"]["ckpt"]
    if Path(ck).exists():
        print(f"{OK} {ck}")
    else:
        print(f"{BAD} {ck}")
        problems.append("CIFT checkpoint missing")

    print("\nPolicy checkpoints")
    sys.path.insert(0, ".")
    try:
        from ablations.lib.manifest import policy_ckpt
        for key in t1["policies"]:
            try:
                p = policy_ckpt(t1, key)
                if Path(p).exists():
                    print(f"{OK} {key:12s} -> {p}")
                else:
                    print(f"{BAD} {key:12s} -> {p}")
                    problems.append(
                        f"Policy '{key}' not trained. Run:\n"
                        f"     ROW={key} GPUS=4,5,6,7 BATCH=256 EPOCHS=30 "
                        f"PPO_EPOCHS=3 bash ablations/scripts/train_table123_row.sh"
                    )
            except Exception as exc:
                print(f"{BAD} {key:12s} -> {exc}")
                problems.append(f"Policy '{key}': {exc}")
    except Exception as exc:
        print(f"{WARN} could not resolve policy paths: {exc}")

    print("\nDatasets")
    from ablations.lib.fast_loader import FAKE_LABELS
    import csv as _csv

    usable = []
    for key, ds in t1["datasets"].items():
        p = ds["eval_csv"]
        if not Path(p).exists():
            print(f"{BAD} {key:22s} eval_csv not found: {p}")
            problems.append(
                f"Dataset '{key}' CSV missing: {p}\n"
                f"     Build a RIFT-schema CSV with a donor_path column."
            )
            continue
        try:
            with open(p, newline="") as fh:
                rd = _csv.DictReader(fh)
                cols = rd.fieldnames or []
                n_fake = 0
                n_donor = 0
                for r in rd:
                    if str(r.get("label", "1")).strip() in FAKE_LABELS:
                        n_fake += 1
                        d = r.get("donor_path") or r.get("source_ref_path") or ""
                        if d:
                            n_donor += 1
        except Exception as exc:
            print(f"{BAD} {key:22s} unreadable: {exc}")
            problems.append(f"Dataset '{key}' unreadable")
            continue

        has_donor_col = ("donor_path" in cols) or ("source_ref_path" in cols)
        if not has_donor_col:
            print(f"{BAD} {key:22s} n_fake={n_fake:6d}  NO donor column -> Delta is unmeasurable")
            problems.append(
                f"Dataset '{key}' has no donor_path column. Without a donor the "
                "adapter returns proxy mode, w_delta is forced to 0, and every "
                "Delta column is meaningless."
            )
            continue
        frac = (n_donor / n_fake) if n_fake else 0.0
        tag = OK if frac > 0.99 else WARN
        print(f"{tag} {key:22s} n_fake={n_fake:6d}  donor_frac={frac:.3f}  "
              f"role={ds['role']:9s} donor={ds['donor_type']}")
        if 0 < frac <= 0.99:
            warnings_.append(f"Dataset '{key}': only {frac:.1%} of fake rows have a donor.")
        if frac > 0:
            usable.append((key, n_fake))

    print("\nControl coverage")
    roles = {k: v["role"] for k, v in t1["datasets"].items()}
    if "control" not in roles.values():
        warnings_.append(
            "No `role: control` dataset. Without FF++-with-retrieved-donors you "
            "cannot separate 'worse OOD transfer' from 'worse donor quality'."
        )
        print(f"{WARN} no donor-quality control dataset configured")
    else:
        print(f"{OK} donor-quality control present")

    ids = {int(v["id"]) for v in t1["variants"]}
    if 0 not in ids:
        problems.append("No random_cell row (id 0) -> gap_vs_random is undefined.")
    if not {5, 6} <= ids:
        warnings_.append("Rows 5 and 6 (Gate 2 arms) are not both present.")
    print(f"{OK} variants configured: {sorted(ids)}")

    print("\nEstimated work (per intervention mode)")
    V = len(t1["variants"])
    seeds = len(t1["eval"].get("seeds", [0]))
    extra = sum(seeds - 1 for v in t1["variants"] if v.get("stochastic"))
    total_imgs = sum(n for _, n in usable)
    if total_imgs:
        masks = V + extra
        print(f"       usable images : {total_imgs:,}")
        print(f"       masks/image   : {masks}")
        print(f"       decodes       : {total_imgs:,}  (batch-major; variant-major would be {total_imgs * masks:,})")
        print(f"       forwards/image: {1 + 2 * masks}")
        print(f"       total forwards: {total_imgs * (1 + 2 * masks):,}")

    print("\n" + "=" * 78)
    if problems:
        print(f"BLOCKERS ({len(problems)}) - the full run WILL fail:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    if warnings_:
        print(f"\nWARNINGS ({len(warnings_)}) - run will complete, reviewers will ask:\n")
        for i, w in enumerate(warnings_, 1):
            print(f"  {i}. {w}")
    if not problems and not warnings_:
        print("ALL CLEAR - safe to launch the full run.")
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
