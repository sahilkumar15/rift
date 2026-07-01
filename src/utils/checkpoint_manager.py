# Path: src/utils/checkpoint_manager.py
"""Top-k + latest + interval checkpointing with resume.

Fixes:
  1. PyTorch >=2.6 torch.load defaults to weights_only=True.
     Old RIFT checkpoints include src.utils.config.Config, so auto-resume can fail.
  2. Future checkpoints are sanitized into plain Python containers before saving.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_plain(obj: Any):
    """Recursively convert Config/dict/list objects into safe Python containers."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    # Config in this project behaves like dict but may not subclass dict safely.
    if hasattr(obj, "items"):
        try:
            return {str(k): _to_plain(v) for k, v in obj.items()}
        except Exception:
            pass

    # Keep torch tensors/modules/state_dict items as-is.
    if _HAS_TORCH and torch.is_tensor(obj):
        return obj

    return obj


def _safe_state(state: Dict):
    """Sanitize checkpoint state before torch.save."""
    clean = {}

    for k, v in state.items():
        if k == "config":
            clean[k] = _to_plain(v)
        else:
            clean[k] = _to_plain(v)

    return clean


class CheckpointManager:
    def __init__(
        self,
        out_dir,
        monitor="val/rift_score",
        mode="max",
        top_k=3,
        interval=10,
    ):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.interval = interval
        self.best: List = []

    def _better(self, a, b):
        return a > b if self.mode == "max" else a < b

    def save(self, state: Dict, epoch: int, metrics: Dict):
        if not _HAS_TORCH:
            raise RuntimeError("torch needed to save ckpt.")

        state = _safe_state(state)

        latest = os.path.join(self.out_dir, "latest.pth")
        torch.save(state, latest)

        paths = {"latest": latest}

        if self.interval and epoch % self.interval == 0:
            p = os.path.join(self.out_dir, f"epoch_{epoch:04d}.pth")
            torch.save(state, p)
            paths["interval"] = p

        score = metrics.get(self.monitor)

        if score is not None:
            p = os.path.join(self.out_dir, f"best_e{epoch:04d}_{score:.4f}.pth")
            torch.save(state, p)

            self.best.append((score, p))
            self.best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            for _, old in self.best[self.top_k:]:
                if os.path.exists(old):
                    os.remove(old)

            self.best = self.best[: self.top_k]

            paths["best"] = self.best[0][1]
            paths["top_k"] = [p for _, p in self.best]

        return paths

    def resume(self, mode="auto"):
        if mode in (None, "none", "false", "False", "0", "no", "No"):
            return None

        if mode == "auto":
            latest = os.path.join(self.out_dir, "latest.pth")
            return latest if os.path.exists(latest) else None

        return mode if os.path.exists(mode) else None

    def load(self, path):
        if not _HAS_TORCH:
            raise RuntimeError("torch needed.")

        # PyTorch 2.6 changed torch.load default to weights_only=True.
        # RIFT checkpoints are created locally by this training script and may
        # contain Config metadata, so we explicitly allow full load.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions do not have weights_only.
            return torch.load(path, map_location="cpu")
