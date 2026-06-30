# Path: correlate_rift.py
# Status: NEW
"""Entrypoint: the headline faithfulness-vs-generalization correlation experiment."""
import argparse, json
from iganer.rift.utils.config import load_config
from iganer.rift.utils.io import save_json, save_csv
from iganer.rift.audit.correlation_runner import run_correlation
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    ap.add_argument("--rows_json", help="precomputed per-checkpoint rows JSON")
    args=ap.parse_args(); cfg=load_config(args.config)
    if args.rows_json:
        rows=json.load(open(args.rows_json))
    else:
        raise SystemExit("Provide --rows_json with per-checkpoint "
                         "{faithfulness,in_domain_auc,plausibility,zero_shot_auc} rows, "
                         "or compute them via eval_rift first.")
    summary=run_correlation(rows, min_n=cfg.get("min_n",5))
    out=cfg.get("out_dir","outputs/correlation")
    save_json(summary, f"{out}/correlation_summary.json")
    save_csv(rows, f"{out}/correlation_results.csv")
    print(json.dumps(summary, indent=2, default=str))
if __name__=="__main__": main()
