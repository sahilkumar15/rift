#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 2

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"CSV: {csv_path}")
    print(f"samples: {len(rows)}")

    if not rows:
        print("[ERROR] CSV has zero rows")
        return 2

    missing = 0

    for i, r in enumerate(rows[: args.limit], start=2):
        image = r.get("image_path", "")
        donor = r.get("donor_path") or r.get("source_ref_path") or ""

        image_ok = bool(image) and Path(image).exists()
        donor_ok = bool(donor) and Path(donor).exists()

        print(f"row={i} image_ok={image_ok} donor_ok={donor_ok}")
        print(f"  image: {image}")
        print(f"  donor: {donor}")

        if not image_ok or (args.strict and not donor_ok):
            missing += 1

    if missing:
        print(f"[FAIL] {missing}/{min(args.limit, len(rows))} checked rows are missing paths.")
        return 2

    print("[OK] CSV paths are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
