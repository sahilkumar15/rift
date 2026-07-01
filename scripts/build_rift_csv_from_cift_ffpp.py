#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(d: Path):
    out = []
    if not d.exists() or not d.is_dir():
        return out

    for ext in IMAGE_EXTS:
        out.extend(sorted(d.glob(f"*{ext}")))

    return [p for p in out if p.is_file()]


def parse_pair(name: str):
    m = re.match(r"^(\d+)[_\-](\d+)$", name)

    if not m:
        return None

    return m.group(1), m.group(2)


def unique(paths):
    seen = set()
    out = []

    for p in paths:
        p = Path(p)
        try:
            k = str(p.resolve())
        except Exception:
            k = str(p)

        if k not in seen:
            seen.add(k)
            out.append(Path(k))

    return out


def find_method_roots(root: Path, method: str, compression: str):
    candidates = [
        root / "manipulated_sequences" / method / compression / "frames",
        root / "manipulated_sequences" / method / compression / "images",
        root / "manipulated_sequences" / method / compression,
        root / method / compression / "frames",
        root / method / compression / "images",
        root / method / compression,
        root / method / "frames",
        root / method / "images",
        root / method,
    ]

    candidates += list(root.glob(f"**/{method}/{compression}/frames"))
    candidates += list(root.glob(f"**/{method}/{compression}/images"))
    candidates += list(root.glob(f"**/{method}/{compression}"))

    return [p for p in unique(candidates) if p.exists() and p.is_dir()]


def find_pair_dirs(method_root: Path):
    pair_dirs = []

    for d in sorted(method_root.iterdir()) if method_root.exists() else []:
        if d.is_dir() and parse_pair(d.name) and list_images(d):
            pair_dirs.append(d)

    if not pair_dirs:
        for d in method_root.glob("**/*"):
            if d.is_dir() and parse_pair(d.name) and list_images(d):
                pair_dirs.append(d)

    return unique(pair_dirs)


def find_source_dir(root: Path, source_id: str, compression: str):
    candidates = [
        root / "original_sequences" / "youtube" / compression / "frames" / source_id,
        root / "original_sequences" / "youtube" / compression / "images" / source_id,
        root / "original_sequences" / "youtube" / compression / source_id,
        root / "original_sequences" / "youtube" / "raw" / "frames" / source_id,
        root / "original_sequences" / "youtube" / "raw" / "images" / source_id,
        root / "youtube" / compression / "frames" / source_id,
        root / "youtube" / compression / "images" / source_id,
        root / "youtube" / "frames" / source_id,
        root / "youtube" / "images" / source_id,
        root / "donor_ref" / source_id,
        root / "original" / source_id,
    ]

    for d in candidates:
        if d.exists() and d.is_dir() and list_images(d):
            return d

    for d in root.glob(f"**/{source_id}"):
        if not d.is_dir() or not list_images(d):
            continue

        s = str(d).lower()

        if "original" in s or "youtube" in s or "donor" in s:
            return d

    return None


def choose_donor(source_dir: Path, forged_img: Path):
    exact = source_dir / forged_img.name

    if exact.exists():
        return exact

    for ext in IMAGE_EXTS:
        p = source_dir / f"{forged_img.stem}{ext}"

        if p.exists():
            return p

    imgs = list_images(source_dir)

    if imgs:
        return imgs[0]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp-root", default="/scratch/sahil/projects/img_deepfake/datasets/ffpp")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--out-csv", default="data/slices/rift_ffpp_rela.csv")
    ap.add_argument("--methods", default="Deepfakes,Face2Face,FaceSwap,NeuralTextures")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--max-per-pair", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.ffpp_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"FF++ root not found: {root}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []

    print(f"[INFO] FF++ root: {root}")
    print(f"[INFO] compression: {args.compression}")
    print(f"[INFO] methods: {methods}")

    for method in methods:
        method_roots = find_method_roots(root, method, args.compression)
        print(f"[INFO] method={method} roots={len(method_roots)}")

        for method_root in method_roots:
            pair_dirs = find_pair_dirs(method_root)
            print(f"[INFO]   {method_root} pair_dirs={len(pair_dirs)}")

            for pair_dir in pair_dirs:
                parsed = parse_pair(pair_dir.name)

                if not parsed:
                    continue

                source_id, target_id = parsed
                source_dir = find_source_dir(root, source_id, args.compression)

                if source_dir is None:
                    print(f"[WARN] no source/donor dir for source_id={source_id}; skipping {pair_dir}")
                    continue

                for forged_img in list_images(pair_dir)[: args.max_per_pair]:
                    donor_img = choose_donor(source_dir, forged_img)

                    if donor_img is None:
                        continue

                    rows.append(
                        {
                            "image_path": str(forged_img.resolve()),
                            "label": "1",
                            "source_id": source_id,
                            "target_id": target_id,
                            "manipulation_type": method,
                            "mask_path": "",
                            "donor_path": str(donor_img.resolve()),
                            "metadata_json": json.dumps(
                                {
                                    "ffpp_root": str(root),
                                    "method_root": str(method_root),
                                    "pair_dir": str(pair_dir),
                                    "source_dir": str(source_dir),
                                }
                            ),
                        }
                    )

                    if len(rows) >= args.max_items:
                        break

                if len(rows) >= args.max_items:
                    break

            if len(rows) >= args.max_items:
                break

        if len(rows) >= args.max_items:
            break

    if not rows:
        print("[ERROR] Could not find forged+donor pairs.")
        print("Inspect your FF++ tree with:")
        print(f"find {root} -maxdepth 6 -type d | head -300")
        print(f"find {root} -type f \\( -name '*.png' -o -name '*.jpg' \\) | head -100")
        raise SystemExit(2)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "label",
        "source_id",
        "target_id",
        "manipulation_type",
        "mask_path",
        "donor_path",
        "metadata_json",
    ]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote {out}")
    print(f"[OK] rows={len(rows)}")

    for r in rows[:5]:
        print("image:", r["image_path"])
        print("donor:", r["donor_path"])
        print("---")


if __name__ == "__main__":
    main()
