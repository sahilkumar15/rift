from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any, Dict

import yaml


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def policy_row_dir(m: Dict[str, Any], key: str) -> str:
    return str(Path(m["root_dir"]) / m["policies"][key]["run_name"])


def policy_ckpt(m: Dict[str, Any], key: str) -> str:
    p = m["policies"][key]
    ckpt = str(p.get("ckpt", "auto"))
    if ckpt and ckpt != "auto":
        return ckpt
    return str(Path(policy_row_dir(m, key)) / "ckpt" / "latest.pth")


def _emit(k: str, v: Any) -> None:
    if v is None:
        v = ""
    print(f"{k}={shlex.quote(str(v))}")


def emit_bash_train(m: Dict[str, Any], key: str) -> None:
    if key not in m.get("policies", {}):
        raise SystemExit(f"Unknown row {key}. Available: {list(m.get('policies', {}))}")

    p = m["policies"][key]
    tr = m.get("train", {}) or {}
    data = m.get("data", {}) or {}
    cift = m.get("cift", {}) or {}
    wb = m.get("wandb", {}) or {}

    pairs = {
        "ROW": key,
        "CONFIG": m.get("base_config", "configs/rift_general.yaml"),
        "ROOT_DIR": m.get("root_dir", "experiments/ablations/rift_table123"),
        "ROW_DIR": policy_row_dir(m, key),
        "RUN_NAME": p["run_name"],
        "REWARD_PRESET": p["reward_preset"],
        "HORIZON": p["horizon"],
        "POLICY_CKPT": policy_ckpt(m, key),
        "CIFT_ROOT": cift.get("root", ""),
        "CIFT_CKPT": cift.get("ckpt", ""),
        "CIFT_CONFIG": cift.get("config", "configs/diffusionfake_mixed.yaml"),
        "TRAIN_CSV": data.get("train_csv", ""),
        "VAL_CSV": data.get("val_csv", ""),
        "EVAL_CSV": data.get("eval_csv", data.get("val_csv", "")),
        "GPUS_DEFAULT": tr.get("gpus", "0"),
        "BATCH_DEFAULT": tr.get("batch_size", 64),
        "EPOCHS_DEFAULT": tr.get("epochs", 30),
        "WORKERS_DEFAULT": tr.get("workers", 4),
        "TRAIN_MAX_ITEMS_DEFAULT": tr.get("max_items", ""),
        "VAL_MAX_ITEMS_DEFAULT": tr.get("val_max_items", ""),
        "VAL_EVERY": tr.get("val_every", 2),
        "VAL_MAX_BATCHES": tr.get("val_max_batches", 8),
        "RESUME_DEFAULT": tr.get("resume", "auto"),
        "WANDB_ENABLED": "true" if bool(wb.get("enabled", True)) else "false",
        "WANDB_PROJECT": wb.get("project", "RIFT_ICLR27"),
        "WANDB_GROUP": wb.get("group", "rift_table123_ablation"),
        "WANDB_MODE_DEFAULT": tr.get("wandb_mode", "online"),
    }

    for k, v in pairs.items():
        _emit(k, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ablations/configs/table123_rift.yaml")
    ap.add_argument("--row")
    ap.add_argument("--list-train-rows", action="store_true")
    ap.add_argument("--emit-bash-train", action="store_true")
    ap.add_argument("--get", choices=["ckpt", "run_name", "row_dir"])
    args = ap.parse_args()

    m = load_manifest(args.config)

    if args.list_train_rows:
        for k in m.get("policies", {}):
            print(k)
        return 0

    if not args.row:
        raise SystemExit("--row required unless --list-train-rows")

    if args.row not in m.get("policies", {}):
        raise SystemExit(f"Unknown row {args.row}. Available: {list(m.get('policies', {}))}")

    if args.emit_bash_train:
        emit_bash_train(m, args.row)
    elif args.get == "ckpt":
        print(policy_ckpt(m, args.row))
    elif args.get == "run_name":
        print(m["policies"][args.row]["run_name"])
    elif args.get == "row_dir":
        print(policy_row_dir(m, args.row))
    else:
        raise SystemExit("Specify --emit-bash-train or --get")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
