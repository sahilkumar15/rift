# Path: iganer/rift/gates/gate1_validity.py
# Status: NEW
"""
gate1_validity.py  --  PHASE-0 GATE 1: intervention validity (the precondition).

Question it decides: when you mask the cited region of a forged image, does the
CIFT donor-grounded delta actually collapse MORE than when you mask a random region
of equal area? If not, every necessity/sufficiency number RIFT reports is
confounded by "masking anything changes the signal" and the paper does not hold.

Reads:  a CSV slice of <=100 forged images with cited masks + donor references.
Prints: mean(cited_drop) - mean(random_drop)  and a single PASS/FAIL + STOP/GO line.
Verdict: PASS if separation >= 0.15, else FAIL -> PIVOT.

Run (Katz, real CIFT):
  PYTHONPATH=. python iganer/rift/gates/gate1_validity.py \
      --csv slices/ffpp_forged_50.csv \
      --cift-root /scratch/sahil/projects/img_deepfake/code/ImageDifussionFake \
      --ckpt /path/to/cift.ckpt --backbone convnextv2_base \
      --device cuda --n 50 --min-sep 0.15

  READ: the line "separation=..." and the final "VERDICT:" line.
  GO  -> separation>=0.15: interventions are valid; proceed to Gate 2.
  STOP-> separation<0.15 : pivot. Do NOT train. Do NOT report downstream numbers.

Offline wiring check (no torch, no ckpt):
  python iganer/rift/gates/gate1_validity.py --selftest
"""
from __future__ import annotations
import argparse
import csv
import os
import sys


def _decide(separation: float, min_sep: float) -> str:
    if separation >= min_sep:
        return (f"VERDICT: PASS (separation={separation:.3f} >= {min_sep}). "
                f"GO -> interventions move delta specifically; proceed to Gate 2.")
    return (f"VERDICT: FAIL (separation={separation:.3f} < {min_sep}). "
            f"STOP -> cited masks do not move delta more than random. Every downstream "
            f"necessity/sufficiency number would be confounded. PIVOT to audit-only or "
            f"re-examine the cited-mask source / the delta readout.")


def _selftest() -> int:
    # Exercises the decision logic with two synthetic regimes; no torch needed.
    print("[selftest] gate1 decision logic")
    for sep in (0.31, 0.12):
        print(f"  separation={sep:.3f} -> {_decide(sep, 0.15)}")
    # sanity on the real Gate1Report dataclass (pure-logic part)
    from iganer.rift.interventions.interventions import Gate1Report
    r = Gate1Report(n=50, mean_necessity_drop=0.62, mean_sufficiency_retained=0.58,
                    mean_random_drop=0.21, separation=0.41, verdict="")
    assert r.passed(0.15) is True
    print("  Gate1Report.passed(0.15) on sep=0.41 ->", r.passed(0.15))
    return 0


def _load_slice(csv_path, n):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if str(r.get("label", "1")).strip() in ("1", "fake", "forged"):
                rows.append(r)
            if len(rows) >= n:
                break
    if not rows:
        raise SystemExit(f"No forged rows in {csv_path} (need label==1 rows).")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Gate 1: intervention validity")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--cift-root", dest="cift_root")
    ap.add_argument("--ckpt")
    ap.add_argument("--backbone", default="convnextv2_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-sep", type=float, default=0.15)
    ap.add_argument("--mode", default="blur", choices=["blur", "mean", "zero"])
    ap.add_argument("--topk-frac", type=float, default=0.1)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    for req in ("csv", "cift_root", "ckpt"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_','-')} is required for the real run "
                             f"(or use --selftest for the offline wiring check).")

    import torch
    from iganer.rift.adapters.cift_adapter import CIFTAdapter
    from iganer.rift.interventions.interventions import gate1_intervention_validity
    from iganer.rift.gates._io import load_image_minus1_1, load_mask

    rows = _load_slice(args.csv, args.n)
    adapter = CIFTAdapter(ckpt_path=args.ckpt, device=args.device, backbone=args.backbone,
                          strict_identity_gap=True, cift_root=args.cift_root).load_detector()

    images, masks, donors = [], [], []
    for r in rows:
        img = load_image_minus1_1(r["image_path"], device=args.device)            # (1,3,H,W)
        cited = load_mask(r["mask_path"], like=img)                                # (1,1,H,W)
        donor_path = r.get("donor_path") or r.get("source_ref_path")
        if not donor_path:
            raise SystemExit("Gate 1 needs a TRUE delta -> every forged row must carry a donor "
                             "reference (donor_path / retrieved same-identity ref). Found none for "
                             f"{r['image_path']}. Without a donor the gap is proxy and Gate 1 is invalid.")
        donor = load_image_minus1_1(donor_path, device=args.device)
        images.append(img); masks.append(cited); donors.append(donor)

    # evidence_fn closes over a per-call donor; gate1 iterates (img, mask) pairs in order,
    # so we pair donors by index via a small stateful closure.
    _idx = {"i": 0}
    def evidence_fn(x):
        d = donors[min(_idx["i"], len(donors) - 1)]
        res = adapter.identity_gap(x, donor=d)
        res.assert_mechanism_valid("Gate-1 delta evidence")   # refuses proxy silently passing
        return res.value

    # We must advance the donor index in lockstep with gate1's per-pair loop; simplest is to
    # run gate1 one pair at a time and aggregate, which also lets us bind the right donor.
    from iganer.rift.interventions.interventions import (
        apply_necessity, apply_sufficiency,
    )
    nec_drops, suf_rets, rand_drops, n = [], [], [], 0
    torch.manual_seed(0)
    for img, cm, donor in zip(images, masks, donors):
        def ev(x, _d=donor):
            res = adapter.identity_gap(x, donor=_d)
            res.assert_mechanism_valid("Gate-1 delta evidence")
            return res.value
        e0 = ev(img)
        if abs(e0) < 1e-6:
            continue
        e_nec = ev(apply_necessity(img, cm, args.mode, args.topk_frac))
        e_suf = ev(apply_sufficiency(img, cm, args.mode, args.topk_frac))
        rand = torch.rand_like(cm)
        e_rand = ev(apply_necessity(img, rand, args.mode, args.topk_frac))
        nec_drops.append((e0 - e_nec) / (abs(e0) + 1e-8))
        suf_rets.append(e_suf / (e0 + 1e-8))
        rand_drops.append((e0 - e_rand) / (abs(e0) + 1e-8))
        n += img.shape[0]

    if not nec_drops:
        print("RESULT: no valid samples (all delta ~ 0). This itself is a STOP signal: the "
              "donor-grounded delta is not active on this slice. Check DIMF / donor pairing.")
        print(_decide(0.0, args.min_sep))
        sys.exit(2)

    mean = lambda a: sum(a) / len(a)
    nd, sr, rd = mean(nec_drops), mean(suf_rets), mean(rand_drops)
    sep = nd - rd
    print(f"n={n}  cited_necessity_drop={nd:.3f}  sufficiency_retained={sr:.3f}  "
          f"random_drop={rd:.3f}")
    print(f"separation={sep:.3f}  (cited_drop - random_drop)")
    verdict = _decide(sep, args.min_sep)
    print(verdict)
    sys.exit(0 if sep >= args.min_sep else 3)


if __name__ == "__main__":
    main()
