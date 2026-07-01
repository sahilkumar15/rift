# Path: train_rift_rl.py
# Status: MODIFIED
"""Entrypoint: train the RIFT-RL explanation-repair policy.

Accepts:
  python train_rift_rl.py -c configs/rift_general.yaml --cift-root /path/CIFT \
      detector.cift_ckpt=/path/cift.ckpt rl.horizon=4 rl.epochs=30
"""
import argparse
import sys

from src.utils.config import Config, load_config, merge_overrides
from src.audit.ablation_runner import parse_overrides


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "on")


def _flat_train_cfg(cfg):
    """Translate nested rift_general.yaml config into flat training config."""
    data_cfg = Config(dict(cfg.get_dotted("data", {}) or {}))

    split_csv = cfg.get_dotted("dataset.split_csv")
    if split_csv:
        data_cfg.setdefault("train_csv", split_csv)
        data_cfg.setdefault("val_csv", split_csv)

    data_cfg.setdefault("image_size", cfg.get_dotted("transform.image_size", 256))
    data_cfg.setdefault("batch_size", cfg.get_dotted("train.batch_size", 8))
    data_cfg.setdefault("num_workers", cfg.get_dotted("data.num_workers", 0))

    # RIFT PATCH: let --smoke / dataset.max_items actually limit train/val.
    max_items = cfg.get_dotted("data.max_items", None)
    if max_items is None:
        max_items = cfg.get_dotted("dataset.max_items", None)
    if max_items is not None:
        data_cfg["max_items"] = int(max_items)

    data_cfg["strict_identity_gap"] = _as_bool(
        cfg.get_dotted("detector.strict_identity_gap", cfg.get("strict_identity_gap", False))
    )

    return Config({
        "device": cfg.get_dotted("device", "cuda"),
        "seed": cfg.get_dotted("seed", 42),

        "algo": cfg.get_dotted("rl.algo", cfg.get("algo", "ppo")),
        "lr": cfg.get_dotted("rl.lr", cfg.get("lr", 3e-4)),
        "epochs": cfg.get_dotted("rl.epochs", cfg.get("epochs", 50)),
        "grid": cfg.get_dotted("rl.grid", cfg.get("grid", 8)),
        "horizon": cfg.get_dotted("rl.horizon", cfg.get("horizon", 4)),
        "clip": cfg.get_dotted("rl.clip", cfg.get("clip", 0.2)),
        "ppo_epochs": cfg.get_dotted("rl.ppo_epochs", cfg.get("ppo_epochs", 4)),
        "entropy_coef": cfg.get_dotted("rl.entropy_coef", cfg.get("entropy_coef", 0.01)),
        "value_coef": cfg.get_dotted("rl.value_coef", cfg.get("value_coef", 0.5)),
        "lagrangian": cfg.get_dotted("rl.lagrangian", cfg.get("lagrangian", False)),
        "constraint_budget": cfg.get_dotted("rl.constraint_budget", cfg.get("constraint_budget", 0.0)),
        "reward_preset": cfg.get_dotted("rl.reward_preset", cfg.get("reward_preset", "full_rift")),

        "intervention_mode": cfg.get_dotted("intervention.mode", "blur"),
        "topk_frac": cfg.get_dotted("intervention.topk_frac", 0.12),

        "out_dir": cfg.get_dotted("checkpoint.dir", cfg.get("out_dir", "outputs/ckpt")),
        "monitor": cfg.get_dotted("checkpoint.metric", cfg.get("monitor", "rift_score")),
        "mode": cfg.get_dotted("checkpoint.mode", cfg.get("mode", "max")),
        "top_k": cfg.get_dotted("checkpoint.top_k", cfg.get("top_k", 3)),
        "interval": cfg.get_dotted("checkpoint.save_every", cfg.get("interval", 10)),
        "resume": cfg.get_dotted("resume", cfg.get("resume", "auto")),

        "wandb": _as_bool(cfg.get_dotted("wandb.enabled", cfg.get("wandb", False))),
        "wandb_project": cfg.get_dotted("wandb.project", cfg.get("wandb_project", "rift")),
        "exp_name": (
            cfg.get_dotted("wandb.name", None)
            or cfg.get_dotted("output.name", cfg.get("exp_name", "rift_rl"))
        ),

        "data": data_cfg,
    })


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

    # torchrun/DDP: bind each process to its own GPU before loading CIFT.
    import os

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        try:
            import torch

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            ov["device"] = f"cuda:{local_rank}"
            ov["distributed.enabled"] = True
            ov["distributed.local_rank"] = local_rank
            ov["distributed.rank"] = int(os.environ.get("RANK", "0"))
            ov["distributed.world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        except Exception as e:
            print(f"[train_rift_rl] failed to set DDP CUDA device: {e}", file=sys.stderr)
            return 2

    cfg = merge_overrides(cfg, ov)

    try:
        import torch  # noqa: F401
    except Exception:
        print(
            '[train_rift_rl] torch not installed; install on Katz: pip install -e ".[runtime]"',
            file=sys.stderr,
        )
        return 2

    ckpt = cfg.get_dotted("detector.cift_ckpt", cfg.get("cift_ckpt"))
    cift_root = cfg.get_dotted("detector.cift_root", cfg.get("cift_root"))

    if not ckpt or not cift_root:
        print(
            "[train_rift_rl] need detector.cift_ckpt and detector.cift_root or --cift-root.",
            file=sys.stderr,
        )
        return 2

    from src.adapters.cift_adapter import CIFTAdapter
    from src.data.datamodule import build_dataloaders
    from src.train.train_rift_rl import train

    train_cfg = _flat_train_cfg(cfg)
    dls = build_dataloaders(train_cfg["data"])

    adapter = CIFTAdapter(
        ckpt_path=ckpt,
        device=train_cfg.get("device", "cuda"),
        backbone=cfg.get_dotted("detector.backbone", "convnextv2_base"),
        strict_identity_gap=_as_bool(cfg.get_dotted("detector.strict_identity_gap", True)),
        cift_root=cift_root,
        config_path=cfg.get_dotted("detector.cift_config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    train(train_cfg, adapter, dls)
    return 0


if __name__ == "__main__":
    sys.exit(main())