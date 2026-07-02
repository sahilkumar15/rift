# Path: train_rift_rl.py
"""Entrypoint: train the RIFT-RL explanation-repair policy.

Example:
  bash scripts/run_rift.sh --gpus 0,1,2,3 --mode train --horizon 4

This file converts nested configs/rift_general.yaml into the flat config expected
by src.train.train_rift_rl.train(). The nested YAML follows the CIFT-style layout:
experiment, checkpoint, early_stopping, and wandb are first-class sections.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.audit.ablation_runner import parse_overrides
from src.utils.config import Config, load_config, merge_overrides


_FALSE_STRINGS = {"", "0", "false", "no", "off", "none", "null", "disabled"}
_TRUE_STRINGS = {"1", "true", "yes", "on", "online", "offline"}


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in _FALSE_STRINGS:
        return False
    if s in _TRUE_STRINGS:
        return True

    return default


def _maybe_int(v):
    if v is None:
        return None

    if isinstance(v, str) and v.strip().lower() in _FALSE_STRINGS:
        return None

    try:
        return int(v)
    except Exception:
        return None


def _cfg_get(cfg, dotted, default=None):
    return cfg.get_dotted(dotted, default)


def _path_join(*parts):
    clean = [str(p).strip() for p in parts if p is not None and str(p).strip()]
    if not clean:
        return ""
    return os.path.join(*clean)


def _experiment_name(cfg):
    return (
        _cfg_get(cfg, "experiment.name")
        or _cfg_get(cfg, "output.name")
        or cfg.get("exp_name")
        or "RIFT_rl"
    )


def _experiment_root(cfg):
    return (
        _cfg_get(cfg, "experiment.root_dir")
        or _cfg_get(cfg, "output.root")
        or "experiments"
    )


def _experiment_dir(cfg):
    return _path_join(_experiment_root(cfg), _experiment_name(cfg))


def _checkpoint_dir(cfg):
    explicit = _cfg_get(cfg, "checkpoint.dir")
    if explicit:
        return explicit

    legacy_out_dir = cfg.get("out_dir")
    if legacy_out_dir:
        return legacy_out_dir

    return _path_join(_experiment_dir(cfg), "ckpt")


def _wandb_enabled(cfg):
    mode = str(_cfg_get(cfg, "wandb.mode", "disabled")).strip().lower()

    if mode == "disabled":
        return False

    return _as_bool(_cfg_get(cfg, "wandb.enabled", None), default=True)


def _flat_train_cfg(cfg):
    """Translate nested rift_general.yaml into flat training config."""
    data_cfg = Config(dict(_cfg_get(cfg, "data", {}) or {}))

    # For train, data.train_csv / data.val_csv own the split.
    # dataset.split_csv is only a fallback for audit/gates/legacy.
    split_csv = _cfg_get(cfg, "dataset.split_csv")
    if split_csv:
        data_cfg.setdefault("train_csv", split_csv)
        data_cfg.setdefault("val_csv", split_csv)

    data_cfg.setdefault("image_size", _cfg_get(cfg, "transform.image_size", 256))
    data_cfg.setdefault("batch_size", _cfg_get(cfg, "data.batch_size", 8))
    data_cfg.setdefault("num_workers", _cfg_get(cfg, "data.num_workers", 0))
    data_cfg.setdefault("prefetch_factor", _cfg_get(cfg, "data.prefetch_factor", 2))
    data_cfg.setdefault("pin_memory", _cfg_get(cfg, "data.pin_memory", True))
    data_cfg.setdefault("persistent_workers", _cfg_get(cfg, "data.persistent_workers", True))

    max_items = _cfg_get(cfg, "data.max_items", None)
    if max_items is None:
        max_items = _cfg_get(cfg, "dataset.max_items", None)

    max_items = _maybe_int(max_items)
    if max_items is not None:
        data_cfg["max_items"] = max_items

    val_max_items = _maybe_int(_cfg_get(cfg, "data.val_max_items", None))
    if val_max_items is not None:
        data_cfg["val_max_items"] = val_max_items

    data_cfg["strict_identity_gap"] = _as_bool(
        _cfg_get(cfg, "detector.strict_identity_gap", cfg.get("strict_identity_gap", False))
    )

    experiment_name = _experiment_name(cfg)
    experiment_root = _experiment_root(cfg)
    experiment_dir = _experiment_dir(cfg)
    checkpoint_dir = _checkpoint_dir(cfg)

    monitor = (
        _cfg_get(cfg, "checkpoint.monitor")
        or _cfg_get(cfg, "checkpoint.metric")
        or cfg.get("monitor")
        or "val/rift_score"
    )

    return Config(
        {
            "device": _cfg_get(cfg, "device", "cuda"),
            "seed": _cfg_get(cfg, "seed", 42),

            "experiment_root": experiment_root,
            "experiment_name": experiment_name,
            "experiment_dir": experiment_dir,

            "algo": _cfg_get(cfg, "rl.algo", cfg.get("algo", "ppo")),
            "lr": _cfg_get(cfg, "rl.lr", cfg.get("lr", 3e-4)),
            "epochs": _cfg_get(cfg, "rl.epochs", cfg.get("epochs", 50)),
            "grid": _cfg_get(cfg, "rl.grid", cfg.get("grid", 8)),
            "horizon": _cfg_get(cfg, "rl.horizon", cfg.get("horizon", 4)),
            "hidden": _cfg_get(cfg, "rl.hidden", cfg.get("hidden", 256)),
            "feat_dim": _cfg_get(cfg, "rl.feat_dim", cfg.get("feat_dim", 1024)),
            "clip": _cfg_get(cfg, "rl.clip", cfg.get("clip", 0.2)),
            "ppo_epochs": _cfg_get(cfg, "rl.ppo_epochs", cfg.get("ppo_epochs", 4)),
            "entropy_coef": _cfg_get(cfg, "rl.entropy_coef", cfg.get("entropy_coef", 0.01)),
            "value_coef": _cfg_get(cfg, "rl.value_coef", cfg.get("value_coef", 0.5)),
            "lagrangian": _cfg_get(cfg, "rl.lagrangian", cfg.get("lagrangian", False)),
            "constraint_budget": _cfg_get(
                cfg,
                "rl.constraint_budget",
                cfg.get("constraint_budget", 0.0),
            ),
            "reward_preset": _cfg_get(
                cfg,
                "rl.reward_preset",
                cfg.get("reward_preset", "full_rift"),
            ),
            "val_every": _cfg_get(cfg, "rl.val_every", cfg.get("val_every", 1)),
            "val_max_batches": _cfg_get(
                cfg,
                "rl.val_max_batches",
                cfg.get("val_max_batches", 0),
            ),
            "cache_features": _cfg_get(
                cfg,
                "rl.cache_features",
                cfg.get("cache_features", True),
            ),
            "allow_stop": _as_bool(
                _cfg_get(cfg, "rl.allow_stop", cfg.get("allow_stop", False)),
                default=False,
            ),
            "forbid_revisit": _as_bool(
                _cfg_get(cfg, "rl.forbid_revisit", cfg.get("forbid_revisit", True)),
                default=True,
            ),

            "intervention_mode": _cfg_get(cfg, "intervention.mode", "blur"),
            "topk_frac": _cfg_get(cfg, "intervention.topk_frac", 0.12),

            "out_dir": checkpoint_dir,
            "monitor": monitor,
            "mode": _cfg_get(cfg, "checkpoint.mode", cfg.get("mode", "max")),
            "top_k": _cfg_get(
                cfg,
                "checkpoint.save_top_k",
                _cfg_get(cfg, "checkpoint.top_k", cfg.get("top_k", 3)),
            ),
            "interval": _cfg_get(cfg, "checkpoint.save_every", cfg.get("interval", 1)),
            "save_last": _as_bool(
                _cfg_get(cfg, "checkpoint.save_last", True),
                default=True,
            ),
            "best_filename": _cfg_get(
                cfg,
                "checkpoint.best_filename",
                "rift-best-score-epoch={epoch:02d}",
            ),
            "every_filename": _cfg_get(
                cfg,
                "checkpoint.every_filename",
                "rift-epoch={epoch:02d}",
            ),
            "save_epochs": _cfg_get(cfg, "checkpoint.save_epochs", []),

            "early_stopping_monitor": _cfg_get(cfg, "early_stopping.monitor", monitor),
            "early_stopping_mode": _cfg_get(
                cfg,
                "early_stopping.mode",
                _cfg_get(cfg, "checkpoint.mode", "max"),
            ),
            "early_stopping_patience": _maybe_int(
                _cfg_get(cfg, "early_stopping.patience", None)
            ),
            "early_stopping_min_delta": float(
                _cfg_get(cfg, "early_stopping.min_delta", 0.0) or 0.0
            ),

            "resume": _cfg_get(cfg, "resume", cfg.get("resume", "auto")),

            "wandb": _wandb_enabled(cfg),
            "wandb_project": _cfg_get(
                cfg,
                "wandb.project",
                cfg.get("wandb_project", "RIFT_ICLR27"),
            ),
            "wandb_entity": _cfg_get(cfg, "wandb.entity", None),
            "wandb_group": _cfg_get(cfg, "wandb.group", None),
            "wandb_name": _cfg_get(cfg, "wandb.name", None) or experiment_name,
            "wandb_tags": _cfg_get(cfg, "wandb.tags", []),
            "wandb_notes": _cfg_get(cfg, "wandb.notes", None),
            "wandb_mode": _cfg_get(cfg, "wandb.mode", "disabled"),
            "wandb_save_code": _as_bool(
                _cfg_get(cfg, "wandb.save_code", False),
                default=False,
            ),
            "wandb_log_model": _as_bool(
                _cfg_get(cfg, "wandb.log_model", False),
                default=False,
            ),

            "exp_name": _cfg_get(cfg, "wandb.name", None) or experiment_name,
            "data": data_cfg,
        }
    )


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

    ckpt = _cfg_get(cfg, "detector.cift_ckpt", cfg.get("cift_ckpt"))
    cift_root = _cfg_get(cfg, "detector.cift_root", cfg.get("cift_root"))

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

    if int(os.environ.get("RANK", "0")) == 0:
        Path(train_cfg["experiment_dir"]).mkdir(parents=True, exist_ok=True)
        Path(train_cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
        print(f"[experiment] name={train_cfg['experiment_name']} dir={train_cfg['experiment_dir']}")
        print(f"[checkpoint] dir={train_cfg['out_dir']} monitor={train_cfg['monitor']} mode={train_cfg['mode']}")

    dls = build_dataloaders(train_cfg["data"])

    adapter = CIFTAdapter(
        ckpt_path=ckpt,
        device=train_cfg.get("device", "cuda"),
        backbone=_cfg_get(cfg, "detector.backbone", "convnextv2_base"),
        strict_identity_gap=_as_bool(_cfg_get(cfg, "detector.strict_identity_gap", True)),
        cift_root=cift_root,
        config_path=_cfg_get(cfg, "detector.cift_config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    train(train_cfg, adapter, dls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
