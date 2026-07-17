#!/usr/bin/env python3
# Path: scripts/build_rift_csv_v2.py
# Status: NEW
"""Build RIFT-schema CSVs for the Table 1 dataset axis.

RIFT schema:
    image_path,label,source_id,target_id,manipulation_type,mask_path,donor_path,metadata_json

Delta is a DONOR identity gap. No donor -> adapter returns proxy mode -> the
canonical scorer forces w_delta=0 -> every Delta column is meaningless. So every
row of every dataset needs a resolvable donor_path.

Per CIFT (paper C.6 / Table 10), on test benchmarks the donor is a RETRIEVED
SAME-IDENTITY REFERENCE, not necessarily the ground-truth swap donor. That is
what makes CelebDF/DiffSwap auditable at all, and it is the sanctioned protocol.

MODES
-----
retrieved : FF++ donor-quality CONTROL. Reads an existing RIFT CSV whose donors
            are frame-ALIGNED ground truth (forged 000_003_0015 -> donor
            000_0015) and swaps each for a DIFFERENT random frame of the SAME
            source identity. Same images, same identities, weaker donors.
            Purpose: if FF++-retrieved matches FF++-GT, then any CelebDF/DiffSwap
            drop is generalization, not donor quality. Without this control that
            objection has no answer.

generic   : Build from an arbitrary detection split CSV (CelebDF, DiffSwap, ...)
            by parsing identities out of the fake path and retrieving a
            same-identity frame from a real-frames root. Layout is declared via
            --id-regex / --donor-glob rather than hardcoded, because these
            datasets are extracted differently on every cluster.

ALWAYS DRY-RUN FIRST. --dry-run resolves nothing to disk; it reports what it
found, what it would emit, and every row it could NOT resolve. A builder that
guesses a donor convention silently produces confidently wrong Delta numbers on
every row, which is strictly worse than having no column at all.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FAKE_VALUES = {"1", "fake", "forged", "true", "True"}
RIFT_FIELDS = [
    "image_path", "label", "source_id", "target_id",
    "manipulation_type", "mask_path", "donor_path", "metadata_json",
]


def list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def detect_col(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    low = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in low:
            return low[c]
    for f in fieldnames:
        for c in candidates:
            if c in f.lower():
                return f
    return None


# ───────────────────────────── mode: retrieved ──────────────────────────────
def build_retrieved(args) -> int:
    """FF++ ground-truth donors -> retrieved same-identity donors."""
    src = Path(args.source_csv)
    if not src.exists():
        print(f"ERROR: source CSV not found: {src}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rows_out, unresolved, aligned_kept = [], [], 0
    cache: Dict[str, List[Path]] = {}

    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        for line_no, r in enumerate(reader, start=2):
            if str(r.get("label", "1")).strip() not in FAKE_VALUES:
                continue
            donor = r.get("donor_path") or r.get("source_ref_path") or ""
            if not donor:
                unresolved.append((line_no, r.get("image_path"), "no donor_path in source row"))
                continue

            dp = Path(donor)
            ddir = str(dp.parent)
            if ddir not in cache:
                cache[ddir] = list_images(dp.parent)
            candidates = [p for p in cache[ddir] if p.name != dp.name]

            if not candidates:
                # Only one frame for this identity: the retrieved donor would BE
                # the aligned donor. Keeping it would silently make this row
                # identical to the GT arm and bias the control toward "no
                # difference" - exactly the conclusion the control is meant to
                # test. Drop it and report.
                unresolved.append((line_no, r.get("image_path"),
                                   f"only 1 frame in {dp.parent}, no alternative donor"))
                aligned_kept += 1
                continue

            new_donor = rng.choice(candidates)
            out = {k: r.get(k, "") for k in RIFT_FIELDS}
            out["donor_path"] = str(new_donor)
            meta = {}
            try:
                meta = json.loads(r.get("metadata_json") or "{}")
            except Exception:
                meta = {}
            meta.update({"donor_type": "retrieved",
                         "gt_donor_path": str(dp),
                         "retrieved_offset_from": dp.name})
            out["metadata_json"] = json.dumps(meta)
            rows_out.append(out)

    print(f"\n  source rows (fake)   : {len(rows_out) + len(unresolved)}")
    print(f"  resolved w/ retrieved: {len(rows_out)}")
    print(f"  unresolved (dropped) : {len(unresolved)}")
    if aligned_kept:
        print(f"    of which single-frame identities: {aligned_kept}")
    if unresolved[:5]:
        print("  first unresolved:")
        for ln, img, why in unresolved[:5]:
            print(f"    line {ln}: {why}")

    if args.dry_run:
        print("\n  DRY RUN - nothing written.")
        if rows_out:
            print(f"  example: {rows_out[0]['image_path']}")
            print(f"           donor -> {rows_out[0]['donor_path']}")
        return 0

    if not rows_out:
        print("\nERROR: zero rows resolved; refusing to write an empty CSV.", file=sys.stderr)
        return 1

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RIFT_FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  [wrote] {out_p}  ({len(rows_out)} rows)")
    return 0


# ────────────────────────────── mode: generic ───────────────────────────────
PRESETS = {
    # CelebDF-v2 canonical layout:
    #   Celeb-synthesis/id0_id16_0000/0001.png   (fake: source id0 -> target id16)
    #   Celeb-real/id0_0000/0001.png             (real frames of identity id0)
    # The FIRST id in the fake name is the SOURCE (donor) identity.
    "celebdf_v2": {
        "id_regex": r"id(?P<source>\d+)_id(?P<target>\d+)_(?P<vid>\d+)",
        "donor_glob": "id{source}_*",
        "manipulation": "CelebDF-v2",
    },
    # DiffSwap has no canonical public layout - it depends entirely on how the
    # swap set was generated and extracted. Declare it explicitly and DRY-RUN.
    "diffswap": {
        "id_regex": r"(?P<source>\d+)_(?P<target>\d+)",
        "donor_glob": "{source}*",
        "manipulation": "DiffSwap",
    },
}


def build_generic(args) -> int:
    src = Path(args.source_csv)
    if not src.exists():
        print(f"ERROR: source CSV not found: {src}", file=sys.stderr)
        return 1

    preset = PRESETS.get(args.preset, {}) if args.preset else {}
    id_regex = args.id_regex or preset.get("id_regex")
    donor_glob = args.donor_glob or preset.get("donor_glob")
    manip = args.manipulation or preset.get("manipulation", args.preset or "unknown")

    if not id_regex or not donor_glob:
        print("ERROR: need --id-regex and --donor-glob (or a --preset that supplies them).",
              file=sys.stderr)
        return 1
    if not args.donor_root:
        print("ERROR: --donor-root is required (dir holding REAL frames per identity).",
              file=sys.stderr)
        return 1

    donor_root = Path(args.donor_root)
    if not donor_root.exists():
        print(f"ERROR: --donor-root does not exist: {donor_root}", file=sys.stderr)
        return 1

    rx = re.compile(id_regex)
    rng = random.Random(args.seed)

    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        img_col = args.image_col or detect_col(fields, ["image_path", "img_path", "path", "image", "filename"])
        lbl_col = args.label_col or detect_col(fields, ["label", "target", "cls", "is_fake"])
        print(f"\n  source columns  : {fields}")
        print(f"  image column    : {img_col}")
        print(f"  label column    : {lbl_col}")
        if not img_col:
            print("ERROR: could not detect the image path column; pass --image-col.", file=sys.stderr)
            return 1
        rows_in = list(reader)

    print(f"  source rows     : {len(rows_in)}")
    print(f"  id regex        : {id_regex}")
    print(f"  donor glob      : {donor_glob}")
    print(f"  donor root      : {donor_root}")

    cache: Dict[str, List[Path]] = {}
    rows_out, unresolved = [], []
    n_fake = 0

    for line_no, r in enumerate(rows_in, start=2):
        if lbl_col and str(r.get(lbl_col, "1")).strip() not in FAKE_VALUES:
            continue
        n_fake += 1
        img = str(r.get(img_col, "")).strip()
        if not img:
            unresolved.append((line_no, img, "empty image path"))
            continue
        p = Path(img if Path(img).is_absolute() else (Path(args.image_root or ".") / img))
        if not p.exists():
            unresolved.append((line_no, str(p), "image file not found"))
            continue

        m = rx.search(p.as_posix())
        if not m:
            unresolved.append((line_no, str(p), f"id_regex did not match"))
            continue
        gd = m.groupdict()
        source_id = gd.get("source", "")
        target_id = gd.get("target", "")

        pattern = donor_glob.format(**gd)
        if pattern not in cache:
            frames: List[Path] = []
            for d in sorted(donor_root.glob(pattern)):
                if d.is_dir():
                    frames.extend(list_images(d))
                elif d.suffix.lower() in IMAGE_EXTS:
                    frames.append(d)
            cache[pattern] = frames
        frames = cache[pattern]

        if not frames:
            unresolved.append((line_no, str(p), f"no donor frames for glob '{pattern}'"))
            continue

        donor = rng.choice(frames)
        rows_out.append({
            "image_path": str(p),
            "label": 1,
            "source_id": source_id,
            "target_id": target_id,
            "manipulation_type": manip,
            "mask_path": "",
            "donor_path": str(donor),
            "metadata_json": json.dumps({"donor_type": "retrieved", "donor_glob": pattern}),
        })

    print(f"\n  fake rows       : {n_fake}")
    print(f"  resolved        : {len(rows_out)}")
    print(f"  unresolved      : {len(unresolved)}")
    frac = (len(rows_out) / n_fake) if n_fake else 0.0
    print(f"  donor_frac      : {frac:.3f}")

    if unresolved[:8]:
        print("\n  first unresolved (fix --id-regex / --donor-glob / --donor-root):")
        for ln, pth, why in unresolved[:8]:
            print(f"    line {ln}: {why}\n              {pth}")

    if rows_out[:2]:
        print("\n  example resolutions:")
        for ex in rows_out[:2]:
            print(f"    fake  : {ex['image_path']}")
            print(f"    donor : {ex['donor_path']}   (source_id={ex['source_id']})")

    if args.dry_run:
        print("\n  DRY RUN - nothing written.")
        return 0

    if frac < args.min_donor_frac:
        print(f"\nERROR: donor_frac {frac:.3f} < --min-donor-frac {args.min_donor_frac}.",
              file=sys.stderr)
        print("Refusing to write. A partially-resolved CSV silently biases the audit "
              "toward whichever identities happened to resolve.", file=sys.stderr)
        return 1

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RIFT_FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n  [wrote] {out_p}  ({len(rows_out)} rows)")
    return 0



# ────────────────────────────── mode: probe ─────────────────────────────────
def build_probe(args) -> int:
    """Discover a dataset's layout instead of guessing at it.

    A detection split CSV normally contains BOTH real (label=0) and fake
    (label=1) rows. The real rows' paths ARE the donor source: they are frames
    of genuine identities. So --donor-root does not need to be guessed - it can
    be read straight off the CSV.

    Prints what it found plus the exact --donor-root / --id-regex / --donor-glob
    to pass to `--mode generic`. Writes nothing.
    """
    from collections import Counter

    src = Path(args.source_csv)
    if not src.exists():
        print(f"ERROR: source CSV not found: {src}", file=sys.stderr)
        return 1

    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    print(f"\n  columns   : {fields}")
    print(f"  rows      : {len(rows)}")

    img_col = args.image_col or detect_col(
        fields, ["image_path", "img_path", "path", "image", "filename"]
    )
    lbl_col = args.label_col or detect_col(fields, ["label", "target", "cls", "is_fake"])
    print(f"  image col : {img_col}")
    print(f"  label col : {lbl_col}")

    if not img_col:
        print("\n  Could not detect an image column. Re-run with --image-col <name>.")
        print(f"  First row: {rows[0] if rows else '<empty>'}")
        return 1

    if lbl_col:
        dist = dict(Counter(str(r.get(lbl_col)) for r in rows))
        print(f"\n  label distribution: {dist}")

    def paths_for(is_fake):
        out = []
        for r in rows:
            if not lbl_col:
                continue
            v = str(r.get(lbl_col, "")).strip()
            if (v in FAKE_VALUES) == is_fake:
                out.append(str(r.get(img_col, "")))
        return out

    fakes = paths_for(True)
    reals = paths_for(False)
    print(f"  fake rows : {len(fakes)}")
    print(f"  real rows : {len(reals)}")

    print("\n  -- sample FAKE paths --")
    for x in fakes[:4]:
        print(f"    {x}")

    print("\n  -- sample REAL paths (this is your donor source) --")
    for x in reals[:4]:
        print(f"    {x}")

    if reals:
        parents = [Path(x).parent for x in reals[:400] if x]
        roots = Counter(str(d.parent) for d in parents)
        print("\n  -- candidate --donor-root --")
        for root, n in roots.most_common(3):
            print(f"    {root}    ({n} sampled frames)")
        print("\n  -- real-frame dir names (shape your --donor-glob) --")
        seen = list(dict.fromkeys(str(x.name) for x in parents))[:6]
        for d in seen:
            print(f"    {d}")

    if fakes:
        print("\n  -- fake dir names (shape your --id-regex) --")
        seen = list(dict.fromkeys(Path(x).parent.name for x in fakes))[:6]
        for d in seen:
            print(f"    {d}")

        rx_s = args.id_regex or (PRESETS.get(args.preset, {}).get("id_regex", "") if args.preset else "")
        if rx_s:
            rx = re.compile(rx_s)
            hits = sum(1 for x in fakes if rx.search(Path(x).as_posix()))
            pct = 100.0 * hits / max(1, len(fakes))
            print(f"\n  -- id_regex test: {rx_s}")
            print(f"     matched {hits}/{len(fakes)} fake paths ({pct:.1f}%)")
            for x in fakes[:3]:
                m = rx.search(Path(x).as_posix())
                print(f"       {Path(x).name} -> {m.groupdict() if m else 'NO MATCH'}")

    print("\n  Next: pass the chosen root/glob to --mode generic --dry-run.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["retrieved", "generic", "probe"])
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=3407)
    # generic-only
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--donor-root")
    ap.add_argument("--image-root")
    ap.add_argument("--image-col")
    ap.add_argument("--label-col")
    ap.add_argument("--id-regex")
    ap.add_argument("--donor-glob")
    ap.add_argument("--manipulation")
    ap.add_argument("--min-donor-frac", type=float, default=0.95)
    args = ap.parse_args()

    if args.mode == "probe":
        print("=" * 74)
        print(f"PROBE  {args.source_csv}")
        print("=" * 74)
        return build_probe(args)

    if not args.dry_run and not args.out:
        print("ERROR: --out is required unless --dry-run.", file=sys.stderr)
        return 1

    print("=" * 74)
    print(f"BUILD RIFT CSV  mode={args.mode}  preset={args.preset or '-'}")
    print("=" * 74)
    return build_retrieved(args) if args.mode == "retrieved" else build_generic(args)


if __name__ == "__main__":
    raise SystemExit(main())
