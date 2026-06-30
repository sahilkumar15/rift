# Path: train_rift_rl.py
# Status: MODIFIED (accepts -c/--config, --cift-root, dotted key=value overrides)
"""Entrypoint: train the RIFT-RL explanation-repair policy (Block-4 cells).

Accepts the same CLI shape as ablate_rift.py so scripts/run_rift.sh can drive it:
  python train_rift_rl.py -c configs/rift_general.yaml --cift-root /path/CIFT \
      detector.cift_ckpt=/path/cift.ckpt rl.horizon=4 rl.epochs=30
"""
import argparse
import sys

from iganer.rift.utils.config import load_config, merge_overrides
from iganer.rift.audit.ablation_runner import parse_overrides


def main():
    ap = argparse.ArgumentParser(description="Train RIFT-RL repair policy")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--cift-root", dest="cift_root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ov = parse_overrides(args.overrides)
    if args.cift_root:
        ov["detector.cift_root"] = args.cift_root
    if args.device:
        ov["device"] = args.device
    cfg = merge_overrides(cfg, ov)

    try:
        import torch  # noqa
    except Exception:
        print("[train_rift_rl] torch not installed; install on Katz:  pip install -e \".[runtime]\"",
              file=sys.stderr)
        return 2

    ckpt = cfg.get_dotted("detector.cift_ckpt")
    cift_root = cfg.get_dotted("detector.cift_root")
    if not ckpt or not cift_root:
        print("[train_rift_rl] need detector.cift_ckpt and detector.cift_root (or --cift-root).",
              file=sys.stderr)
        return 2

    from iganer.rift.adapters.cift_adapter import CIFTAdapter
    from iganer.rift.data.datamodule import build_dataloaders
    from iganer.rift.train.train_rift_rl import train

    adapter = CIFTAdapter(
        ckpt_path=ckpt, device=cfg.get_dotted("device", "cuda"),
        backbone=cfg.get_dotted("detector.backbone", "convnextv2_base"),
        strict_identity_gap=bool(cfg.get_dotted("detector.strict_identity_gap", True)),
        cift_root=cift_root,
        config_path=cfg.get_dotted("detector.cift_config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    dls = build_dataloaders(cfg.get_dotted("data", {}) or {})
    train(cfg, adapter, dls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
