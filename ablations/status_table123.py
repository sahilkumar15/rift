#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def load_yaml(p):
    return yaml.safe_load(open(p)) or {}


def ckpt_path(m, key):
    p = m["policies"][key]
    c = str(p.get("ckpt", "auto"))
    if c and c != "auto":
        return Path(c)
    return Path(m["root_dir"]) / p["run_name"] / "ckpt" / "latest.pth"


def ckpt_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"

    try:
        import torch
        from src.rl.policy import GridPolicy

        st = torch.load(path, map_location="cpu", weights_only=False)
        sd = st.get("policy", st)
        sd = {str(k)[7:] if str(k).startswith("module.") else str(k): v for k, v in sd.items()}

        cur = GridPolicy().state_dict()

        bad = []
        for k, v in sd.items():
            if k in cur and tuple(v.shape) != tuple(cur[k].shape):
                bad.append(f"{k}:ckpt{tuple(v.shape)}!=code{tuple(cur[k].shape)}")

        missing = [k for k in cur if k not in sd]

        if bad:
            return False, "shape_mismatch " + "; ".join(bad[:3])
        if missing:
            return False, "missing_keys " + ",".join(missing[:3])

        return True, "ok"
    except Exception as e:
        return False, type(e).__name__ + ": " + str(e)


def read_eval_rows(m):
    out = {}
    out_dir = Path(m["eval"]["output_dir"])

    for tkey, tspec in m.get("tables", {}).items():
        p = out_dir / tspec["filename"]
        if not p.exists():
            continue

        for r in csv.DictReader(open(p)):
            out[(tkey, str(r.get("ID")))] = r

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ablations/configs/table123_rift.yaml")
    ap.add_argument("--print-missing-train", action="store_true")
    ap.add_argument("--print-missing-eval", action="store_true")
    args = ap.parse_args()

    m = load_yaml(args.config)
    eval_rows = read_eval_rows(m)

    missing_train = []

    for key in m.get("policies", {}):
        ok, msg = ckpt_ok(ckpt_path(m, key))
        if not ok:
            missing_train.append(key)

        if not args.print_missing_train and not args.print_missing_eval:
            print(f"policy {key:16s} ckpt={'OK' if ok else 'BAD'} {msg}")

    missing_eval = []

    for tkey, tspec in m.get("tables", {}).items():
        for row in tspec.get("rows", []):
            r = eval_rows.get((tkey, str(row["id"])))
            if not r or str(r.get("status", "")).lower() != "ok":
                missing_eval.append((tkey, str(row["id"]), row["variant"]))

            if not args.print_missing_train and not args.print_missing_eval:
                status = r.get("status") if r else "MISSING"
                n = r.get("n") if r else ""
                print(f"eval {tkey:18s} id={row['id']:<3} {status} n={n} {row['variant']}")

    if args.print_missing_train:
        print(" ".join(missing_train))

    if args.print_missing_eval:
        print(",".join(sorted(set(t for t, _, _ in missing_eval))))


if __name__ == "__main__":
    main()
