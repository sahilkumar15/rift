# Path: src/gates/gate3_correlation.py
# Status: NEW
"""
gate3_correlation.py  --  PHASE-0 GATE 3: the significance test (the 4.5/5 result).

Question it decides: across checkpoints, does FAITHFULNESS predict zero-shot AUC
while IN-DOMAIN accuracy does NOT? That dissociation is the paper's headline.

Input: a CSV with one row per checkpoint and columns:
    ckpt, faithfulness, in_domain_auc, plausibility, zero_shot_auc
(You produce faithfulness/plausibility per checkpoint from the audit leaderboard,
and the two AUCs from CIFT eval. This script only does the statistics + verdict.)

Computes Spearman (+ Pearson + bootstrap CI) of each predictor vs zero_shot_auc.

Verdict HEADLINE if:
    faithfulness has the strongest |Spearman| with zero_shot_auc
  AND in_domain_auc does NOT significantly correlate (CI straddles 0)
  AND n >= 15 (headline-reportable; n<15 is a trend, not a headline)
Else: fall back to the audit-only contribution and say so.

Run:
  PYTHONPATH=. python -m src.gates.gate3_correlation.py --csv checkpoints_metrics.csv
  python -m src.gates.gate3_correlation --selftest     # offline, synthetic
"""
from __future__ import annotations
import argparse
import csv
import sys

HEADLINE_MIN_N = 15      # prompt: flag n<15 as NOT reportable as a headline


def _decide(results, n):
    by = {r.predictor: r for r in results}
    faith = by.get("faithfulness")
    indom = by.get("in_domain_auc")
    if faith is None or indom is None:
        return "VERDICT: INCONCLUSIVE (need both 'faithfulness' and 'in_domain_auc' columns)."
    # strongest predictor by |spearman| among reportable ones
    ranked = sorted(results, key=lambda r: (abs(r.spearman) if r.spearman == r.spearman else -1),
                    reverse=True)
    strongest = ranked[0].predictor
    faith_lo, faith_hi = faith.spearman_ci
    indom_lo, indom_hi = indom.spearman_ci
    faith_sig = not (faith_lo <= 0 <= faith_hi) if faith_lo == faith_lo else False
    indom_sig = not (indom_lo <= 0 <= indom_hi) if indom_lo == indom_lo else False

    if n < HEADLINE_MIN_N:
        return (f"VERDICT: TREND-ONLY (n={n} < {HEADLINE_MIN_N}). faithfulness Spearman="
                f"{faith.spearman:.3f} CI=({faith_lo:.2f},{faith_hi:.2f}); in_domain="
                f"{indom.spearman:.3f}. Report as a trend, NOT a headline. Gather more checkpoints.")
    if strongest == "faithfulness" and faith_sig and not indom_sig:
        return (f"VERDICT: HEADLINE (n={n}). faithfulness is the strongest predictor of zero-shot "
                f"AUC (Spearman={faith.spearman:.3f}, CI=({faith_lo:.2f},{faith_hi:.2f})) while "
                f"in-domain AUC does not significantly predict it (CI straddles 0). The "
                f"dissociation holds -> lead with it.")
    return (f"VERDICT: NO HEADLINE (n={n}). Either in-domain AUC also predicts zero-shot "
            f"(in_domain Spearman={indom.spearman:.3f}, sig={indom_sig}) or faithfulness is not "
            f"the strongest/cleanest predictor (strongest={strongest}). Fall back to the "
            f"audit-only contribution (leaderboard that exposes unfaithful explanations).")


def _selftest() -> int:
    print("[selftest] gate3 decision logic")
    from src.metrics.correlation_metrics import correlate_predictors
    # Synthetic: faithfulness tracks zero-shot, in-domain saturates (no signal).
    import random
    rng = random.Random(0)
    rows = []
    for _ in range(18):
        zs = rng.uniform(0.78, 0.93)
        rows.append({
            "faithfulness": zs - 0.05 + rng.uniform(-0.02, 0.02),     # correlated
            "in_domain_auc": 0.992 + rng.uniform(-0.003, 0.003),       # saturated, no signal
            "plausibility": rng.uniform(0.5, 0.7),                     # noise
            "zero_shot_auc": zs,
        })
    res = correlate_predictors(rows, min_n=5)
    for r in res:
        print(f"  {r.predictor:14s} spearman={r.spearman:+.3f} CI={tuple(round(c,2) for c in r.spearman_ci)} "
              f"n={r.n} reportable={r.reportable}")
    print("  " + _decide(res, n=len(rows)))
    print("  --- now with n=8 (should be TREND-ONLY) ---")
    print("  " + _decide(correlate_predictors(rows[:8], min_n=5), n=8))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Gate 3: correlation headline test")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--target", default="zero_shot_auc")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.csv:
        raise SystemExit("--csv required (or --selftest).")

    from src.metrics.correlation_metrics import correlate_predictors

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            row = {}
            for k in ("faithfulness", "in_domain_auc", "plausibility", args.target):
                if k in r and r[k] not in ("", None):
                    row[k] = float(r[k])
            if args.target in row:
                rows.append(row)

    if not rows:
        raise SystemExit(f"No usable rows (need a '{args.target}' column and >=1 predictor).")

    res = correlate_predictors(rows, target=args.target,
                               predictors=("faithfulness", "in_domain_auc", "plausibility"))
    for r in res:
        flag = "" if r.reportable else "  [n<min: not reportable]"
        print(f"{r.predictor:14s} spearman={r.spearman:+.3f} pearson={r.pearson:+.3f} "
              f"CI=({r.spearman_ci[0]:.2f},{r.spearman_ci[1]:.2f}) n={r.n}{flag}")
    print(_decide(res, n=len(rows)))


if __name__ == "__main__":
    main()
