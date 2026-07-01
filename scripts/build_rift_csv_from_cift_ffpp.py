#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(d: Path):
    if not d.exists() or not d.is_dir():
        return []

    out = []

    for ext in IMAGE_EXTS:
        out.extend(sorted(d.glob(f"*{ext}")))

    return [p for p in out if p.is_file()]


def parse_pair(name: str):
    m = re.match(r"^(\d+)[_\-](\d+)$", name)

    if not m:
        return None

    return m.group(1), m.group(2)


def choose_existing_dir(candidates):
    """Return the first existing directory. Do not scan parent if images exists."""
    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    return None


def method_root(ffpp_root: Path, method: str, compression: str):
    # Prefer exact CIFT-style path.
    return choose_existing_dir(
        [
            ffpp_root / "manipulated_sequences" / method / compression / "images",
            ffpp_root / "manipulated_sequences" / method / compression / "frames",
            ffpp_root / "manipulated_sequences" / method / compression,
        ]
    )


def original_root(ffpp_root: Path, compression: str):
    return choose_existing_dir(
        [
            ffpp_root / "original_sequences" / "youtube" / compression / "images",
            ffpp_root / "original_sequences" / "youtube" / compression / "frames",
            ffpp_root / "original_sequences" / "youtube" / compression,
        ]
    )


def find_pair_dirs(root: Path):
    out = []

    # Normal FF++ layout:
    #   manipulated_sequences/Deepfakes/c23/images/000_003/*.png
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir() and parse_pair(d.name) and list_images(d):
            out.append(d)

    return out


def source_dir_from_original_root(orig_root: Path, source_id: str):
    d = orig_root / source_id

    if d.exists() and d.is_dir() and list_images(d):
        return d

    # Fallback only if direct path is not present.
    for cand in sorted(orig_root.glob(f"**/{source_id}")):
        if cand.exists() and cand.is_dir() and list_images(cand):
            return cand

    return None


def donor_frame_stem(source_id: str, forged_img: Path):
    # forged stem: 000_003_0015
    # donor stem : 000_0015
    m = re.search(r"_(\d+)$", forged_img.stem)

    if m:
        return f"{source_id}_{m.group(1)}"

    return None


def choose_donor(source_dir: Path, source_id: str, forged_img: Path):
    stem = donor_frame_stem(source_id, forged_img)

    if stem:
        for ext in IMAGE_EXTS:
            p = source_dir / f"{stem}{ext}"

            if p.exists():
                return p

    imgs = list_images(source_dir)

    if imgs:
        return imgs[0]

    return None


def build_rows(args):
    # Important:
    # Do NOT call .resolve().
    # .resolve() can turn logical c23 symlink paths into raw paths.
    ffpp_root = Path(args.ffpp_root).expanduser()

    if not ffpp_root.exists():
        raise FileNotFoundError(f"FF++ root not found: {ffpp_root}")

    orig_root = original_root(ffpp_root, args.compression)

    if orig_root is None:
        raise FileNotFoundError(
            f"Could not find original youtube {args.compression} root under {ffpp_root}"
        )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[INFO] FF++ root      : {ffpp_root}")
    print(f"[INFO] compression    : {args.compression}")
    print(f"[INFO] original root  : {orig_root}")
    print(f"[INFO] methods        : {methods}")
    print(f"[INFO] max_items      : {args.max_items}")
    print(f"[INFO] max_per_pair   : {args.max_per_pair}")

    rows = []
    rng = random.Random(args.seed)

    for method in methods:
        mroot = method_root(ffpp_root, method, args.compression)

        if mroot is None:
            print(f"[WARN] missing method root for method={method}, compression={args.compression}")
            continue

        pair_dirs = find_pair_dirs(mroot)

        print(f"[INFO] method={method} root={mroot} pair_dirs={len(pair_dirs)}")

        if args.shuffle:
            rng.shuffle(pair_dirs)

        for pair_dir in pair_dirs:
            parsed = parse_pair(pair_dir.name)

            if not parsed:
                continue

            source_id, target_id = parsed
            src_dir = source_dir_from_original_root(orig_root, source_id)

            if src_dir is None:
                print(f"[WARN] missing source donor dir source_id={source_id} pair={pair_dir}")
                continue

            forged_imgs = list_images(pair_dir)

            if args.shuffle:
                rng.shuffle(forged_imgs)

            forged_imgs = forged_imgs[: args.max_per_pair]

            for forged_img in forged_imgs:
                donor_img = choose_donor(src_dir, source_id, forged_img)

                if donor_img is None:
                    continue

                # Use absolute logical path, not resolved symlink target.
                image_path = str(forged_img.absolute())
                donor_path = str(donor_img.absolute())

                rows.append(
                    {
                        "image_path": image_path,
                        "label": "1",
                        "source_id": source_id,
                        "target_id": target_id,
                        "manipulation_type": method,
                        "mask_path": "",
                        "donor_path": donor_path,
                        "metadata_json": json.dumps(
                            {
                                "ffpp_root": str(ffpp_root),
                                "compression": args.compression,
                                "method_root": str(mroot),
                                "pair_dir": str(pair_dir),
                                "source_dir": str(src_dir),
                                "builder": "rift_strict_c23_no_resolve",
                            }
                        ),
                    }
                )

                if args.max_items > 0 and len(rows) >= args.max_items:
                    return rows

    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_split(rows, train_csv: Path, val_csv: Path, val_frac: float, seed: int):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    n_val = int(round(len(rows) * val_frac))

    val = rows[:n_val]
    train = rows[n_val:]

    write_csv(train, train_csv)
    write_csv(val, val_csv)

    print(f"[OK] train rows={len(train)} -> {train_csv}")
    print(f"[OK] val rows={len(val)} -> {val_csv}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ffpp-root", default="/scratch/sahil/projects/img_deepfake/datasets/ffpp")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--methods", default="Deepfakes,Face2Face,FaceSwap,NeuralTextures")
    ap.add_argument("--max-items", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-per-pair", type=int, default=50)

    ap.add_argument("--out-csv", default="data/slices/rift_ffpp_rela_c23_full.csv")
    ap.add_argument("--make-split", action="store_true")
    ap.add_argument("--train-csv", default="data/slices/rift_ffpp_train_c23_full.csv")
    ap.add_argument("--val-csv", default="data/slices/rift_ffpp_val_c23_full.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)

    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--shuffle", action="store_true")

    args = ap.parse_args()

    rows = build_rows(args)

    if not rows:
        raise SystemExit("[ERROR] no rows built")

    write_csv(rows, Path(args.out_csv))

    print(f"[OK] wrote rows={len(rows)} -> {args.out_csv}")

    if args.make_split:
        write_split(
            rows,
            Path(args.train_csv),
            Path(args.val_csv),
            args.val_frac,
            args.seed,
        )

    print("[preview]")

    for r in rows[:5]:
        print("image:", r["image_path"])
        print("donor:", r["donor_path"])
        print("---")


if __name__ == "__main__":
    main()
