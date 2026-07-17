#!/usr/bin/env python3
# Path: ablations/format_table1_paper.py
# Status: REWRITTEN - now a thin CLI wrapper over ablations/lib/combine_table1.py
# so the pivot logic has exactly one implementation, shared with the
# auto-combine step in run_table1_parallel.sh.
"""Manually (re)build the combined Table 1 from whatever dataset results exist
on disk right now. Safe to run at any time, including while other datasets are
still evaluating - missing datasets are simply omitted, not treated as errors.
"""
from __future__ import annotations

import argparse

from ablations.lib.combine_table1 import (
    build_combined_table, load_config_variants, load_dataset_results, write_combined,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ablations/configs/table1.yaml")
    ap.add_argument("--root", default="experiments/ablations/table1")
    ap.add_argument("--mode", default="blur")
    ap.add_argument("--datasets", default="ffpp_c23,celebdf_v2,diffswap")
    ap.add_argument("--display", default="ffpp_c23=FF++,celebdf_v2=Celeb-DF,"
                                          "diffswap=DiffSwap,ffpp_c23_retrieved=FF++ (retr.)")
    ap.add_argument("--metrics", default="faith_delta,rift_score")
    args = ap.parse_args()

    display = dict(p.split("=", 1) for p in args.display.split(",") if "=" in p)
    dataset_order = [d.strip() for d in args.datasets.split(",")]
    metrics = [m.strip() for m in args.metrics.split(",")]

    variants_cfg = load_config_variants(args.config)
    df = load_dataset_results(args.root, args.mode, dataset_order)
    piv = build_combined_table(df, variants_cfg, dataset_order, display, metrics)
    paths = write_combined(piv, args.root, args.mode)

    print(f"\n[wrote] {paths['csv']}")
    print(f"[wrote] {paths['md']}\n")
    from ablations.lib.combine_table1 import to_markdown
    has_fb = piv[("", "Method")].astype(str).str.endswith("†").any()
    print(to_markdown(piv, has_fallback_footnote=has_fb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
