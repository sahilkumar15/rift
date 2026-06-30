# Path: eval_rift.py
# Status: NEW
"""Entrypoint: evaluate / leaderboard a trained policy or explainers."""
import argparse
from iganer.rift.utils.config import load_config
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=load_config(args.config)
    print("Wire adapter+explainers, then call iganer.rift.eval.eval_rift.evaluate(cfg,...).")
if __name__=="__main__": main()
