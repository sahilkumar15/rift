# Path: iganer/rift/eval/eval_rift.py
# Status: NEW
"""Evaluate a trained policy / explainer: audit on a dataset, write leaderboard."""
from __future__ import annotations
from ..audit.audit_runner import audit_one, aggregate
from ..audit.leaderboard import build_leaderboard
from ..utils.io import save_csv, save_json
def evaluate(cfg, adapter, explainers, dataloader):
    import torch
    all_agg=[]; per_sample=[]
    for expl in explainers:
        rows=[]
        for batch in dataloader:
            for item in batch:
                img=item["image"].unsqueeze(0).to(cfg.get("device","cuda"))
                row,_,_,_=audit_one(img, adapter, expl)
                rows.append(row); per_sample.append(row)
        all_agg.append(aggregate(rows))
    cols,table=build_leaderboard(all_agg)
    out=cfg.get("out_dir","outputs/eval")
    save_csv(per_sample, f"{out}/per_sample.csv")
    save_json({"columns":cols,"rows":table}, f"{out}/leaderboard.json")
    return cols, table
