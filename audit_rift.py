# Path: audit_rift.py
# Status: NEW
"""Entrypoint: run the faithfulness audit + leaderboard."""
import argparse
from iganer.rift.utils.config import load_config
from iganer.rift.utils.logging import get_logger
log=get_logger("audit_rift")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=load_config(args.config)
    log.info("Audit requires a WIRED CIFTAdapter (see adapters/cift_adapter.py WIRE 1-4).")
    log.info("Build adapter+explainers+dataloader per cfg, then call "
             "iganer.rift.eval.eval_rift.evaluate(...). See README_RIFT.md quickstart.")
if __name__=="__main__": main()
