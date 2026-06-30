# Path: train_rift_rl.py
# Status: NEW
"""Entrypoint: train the RIFT-RL explanation-repair policy."""
import argparse
from iganer.rift.utils.config import load_config
from iganer.rift.utils.logging import get_logger
log=get_logger("train_rift_rl")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True)
    args=ap.parse_args(); cfg=load_config(args.config)
    from iganer.rift.adapters.cift_adapter import CIFTAdapter
    from iganer.rift.data.datamodule import build_dataloaders
    from iganer.rift.train.train_rift_rl import train
    adapter=CIFTAdapter(cfg.get("cift_ckpt"), device=cfg.get("device","cuda"),
                        strict_identity_gap=cfg.get("strict_identity_gap",False),
                        cift_root=cfg.get("cift_root")).load_detector()
    dls=build_dataloaders(cfg.get("data",{}))
    train(cfg, adapter, dls)
if __name__=="__main__": main()
