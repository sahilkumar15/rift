# Path: src/utils/checkpoint_manager.py
"""Top-k + latest/last + selected-epoch checkpointing with robust resume.

Supports CIFT-style YAML keys:
  checkpoint.monitor
  checkpoint.mode
  checkpoint.save_top_k
  checkpoint.save_last
  checkpoint.best_filename
  checkpoint.every_filename
  checkpoint.save_epochs
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_plain(obj: Any):
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    if hasattr(obj, "items"):
        try:
            return {str(k): _to_plain(v) for k, v in obj.items()}
        except Exception:
            pass

    if _HAS_TORCH and torch.is_tensor(obj):
        return obj

    return obj


def _safe_state(state: Dict):
    return {str(k): _to_plain(v) for k, v in state.items()}


def _safe_metric_name(name: str) -> str:
    name = str(name or "metric")
    name = name.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name)


def _format_filename(template: str, *, epoch: int, monitor: str, score: Optional[float]) -> str:
    metric_name = _safe_metric_name(monitor)

    kwargs = {
        "epoch": epoch,
        "monitor": metric_name,
        "score": float(score) if score is not None else 0.0,
    }

    try:
        name = str(template).format(**kwargs)
    except Exception:
        name = f"rift-epoch={epoch:02d}"

    if not os.path.splitext(name)[1]:
        name += ".pth"

    return name


def _parse_epochs(save_epochs: Optional[Iterable]) -> set[int]:
    if not save_epochs:
        return set()

    out = set()

    for e in save_epochs:
        try:
            out.add(int(e))
        except Exception:
            pass

    return out


class CheckpointManager:
    def __init__(
        self,
        out_dir,
        monitor="val/rift_score",
        mode="max",
        top_k=3,
        interval=1,
        save_last=True,
        best_filename="rift-best-score-epoch={epoch:02d}",
        every_filename="rift-epoch={epoch:02d}",
        save_epochs=None,
    ):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.monitor = str(monitor or "val/rift_score")
        self.mode = str(mode or "max").lower()
        self.top_k = int(top_k if top_k is not None else 3)
        self.interval = int(interval if interval is not None else 1)
        self.save_last = bool(save_last)
        self.best_filename = best_filename or "rift-best-score-epoch={epoch:02d}"
        self.every_filename = every_filename or "rift-epoch={epoch:02d}"
        self.save_epochs = _parse_epochs(save_epochs)
        self.best: List = []

    def _should_save_interval(self, epoch_num: int) -> bool:
        if epoch_num in self.save_epochs:
            return True

        return bool(self.interval and epoch_num % self.interval == 0)

    def save(self, state: Dict, epoch: int, metrics: Dict):
        if not _HAS_TORCH:
            raise RuntimeError("torch needed to save ckpt.")

        state = _safe_state(state)
        epoch_num = int(epoch) + 1
        paths = {}

        if self.save_last:
            latest = os.path.join(self.out_dir, "latest.pth")
            last = os.path.join(self.out_dir, "last.pth")
            torch.save(state, latest)
            torch.save(state, last)
            paths["latest"] = latest
            paths["last"] = last

        if self._should_save_interval(epoch_num):
            filename = _format_filename(
                self.every_filename,
                epoch=epoch_num,
                monitor=self.monitor,
                score=None,
            )
            p = os.path.join(self.out_dir, filename)
            torch.save(state, p)
            paths["interval"] = p

        score = metrics.get(self.monitor)

        if score is None and "/" in self.monitor:
            score = metrics.get(self.monitor.split("/", 1)[-1])

        if score is not None and self.top_k != 0:
            score = float(score)

            filename = _format_filename(
                self.best_filename,
                epoch=epoch_num,
                monitor=self.monitor,
                score=score,
            )

            p = os.path.join(self.out_dir, filename)
            torch.save(state, p)

            if self.top_k < 0:
                paths["best"] = p
                return paths

            self.best.append((score, p))
            self.best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

            for _, old in self.best[self.top_k :]:
                if os.path.exists(old):
                    os.remove(old)

            self.best = self.best[: self.top_k]

            if self.best:
                paths["best"] = self.best[0][1]
                paths["top_k"] = [p for _, p in self.best]

        return paths

    def resume(self, mode="auto"):
        if mode in (None, "none", "false", "False", "0", "no", "No"):
            return None

        if mode == "auto":
            for name in ("latest.pth", "last.pth"):
                path = os.path.join(self.out_dir, name)
                if os.path.exists(path):
                    return path
            return None

        return mode if os.path.exists(str(mode)) else None

    def load(self, path):
        if not _HAS_TORCH:
            raise RuntimeError("torch needed.")

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
