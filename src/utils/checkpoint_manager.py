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
    if _HAS_TORCH and torch.is_tensor(obj):
        return obj
    return obj


def _safe_state(state: Dict):
    return {str(k): _to_plain(v) for k, v in state.items()}


def _format_filename(template: str, *, epoch: int, monitor: str, score: Optional[float]) -> str:
    kwargs = {
        "epoch": int(epoch),
        "monitor": str(monitor).replace("/", "_"),
        "score": float(score) if score is not None else 0.0,
    }
    try:
        name = str(template).format(**kwargs)
    except Exception:
        name = f"rift-epoch={epoch:02d}"
    if not name.endswith(".pth"):
        name += ".pth"
    return name


def _parse_epochs(save_epochs: Optional[Iterable]) -> set[int]:
    out = set()
    if not save_epochs:
        return out
    for e in save_epochs:
        try:
            out.add(int(e))
        except Exception:
            pass
    return out


def _epoch_num(path: str) -> int:
    name = os.path.basename(str(path))
    m = re.search(r"epoch=([0-9]+)", name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _score_num(path: str):
    name = os.path.basename(str(path))
    m = re.search(r"score=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_remove(path: str) -> bool:
    try:
        if os.path.exists(path) and not os.path.islink(path):
            os.remove(path)
            return True
    except Exception:
        return False
    return False


class CheckpointManager:
    def __init__(
        self,
        out_dir,
        monitor="val/rift_score",
        mode="max",
        top_k=3,
        interval=1,
        save_last=True,
        best_filename="rift-best-score={score:.4f}-epoch={epoch:02d}",
        every_filename="rift-epoch={epoch:02d}",
        save_epochs=None,
        keep_last_n=3,
        prune_ckpt=True,
    ):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.monitor = str(monitor or "val/rift_score")
        self.mode = str(mode or "max").lower()
        self.top_k = int(top_k if top_k is not None else 3)
        self.interval = int(interval if interval is not None else 1)
        self.save_last = bool(save_last)
        self.best_filename = best_filename or "rift-best-score={score:.4f}-epoch={epoch:02d}"
        self.every_filename = every_filename or "rift-epoch={epoch:02d}"
        self.save_epochs = _parse_epochs(save_epochs)

        if keep_last_n in (None, "", "none", "None"):
            self.keep_last_n = None
        else:
            self.keep_last_n = int(keep_last_n)

        self.prune_ckpt = bool(prune_ckpt)
        self.best: List = []

    def _should_save_interval(self, epoch_num: int) -> bool:
        if epoch_num in self.save_epochs:
            return True
        return bool(self.interval and epoch_num % self.interval == 0)

    def _score_files(self):
        files = []
        for name in os.listdir(self.out_dir):
            if not name.endswith(".pth"):
                continue
            if name in ("latest.pth", "last.pth"):
                continue
            path = os.path.join(self.out_dir, name)
            if os.path.islink(path):
                continue
            score = _score_num(path)
            if score is not None:
                files.append((score, path))
        files.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        return files

    def _epoch_files(self):
        files = []
        for name in os.listdir(self.out_dir):
            if not name.endswith(".pth"):
                continue
            if name in ("latest.pth", "last.pth"):
                continue
            if not name.startswith("rift-epoch="):
                continue
            path = os.path.join(self.out_dir, name)
            if os.path.islink(path):
                continue
            ep = _epoch_num(path)
            if ep >= 0:
                files.append(path)
        files.sort(key=lambda p: (_epoch_num(p), os.path.getmtime(p)))
        return files

    def _prune_score_checkpoints(self):
        if not self.prune_ckpt:
            return []
        if self.top_k < 0:
            return []

        files = self._score_files()
        keep = set(path for _, path in files[: max(0, self.top_k)])
        removed = []

        for _, path in files:
            if path not in keep and _safe_remove(path):
                removed.append(path)

        self.best = files[: max(0, self.top_k)]
        return removed

    def _prune_epoch_checkpoints(self):
        if not self.prune_ckpt:
            return []
        if self.keep_last_n is None or self.keep_last_n < 0:
            return []

        files = self._epoch_files()
        keep = set(files[-int(self.keep_last_n):])
        removed = []

        for path in files:
            if path not in keep and _safe_remove(path):
                removed.append(path)

        return removed

    def _prune_all_checkpoints(self):
        removed = []
        removed.extend(self._prune_score_checkpoints())
        removed.extend(self._prune_epoch_checkpoints())
        return removed

    def save(self, state: Dict, epoch: int, metrics: Dict):
        if not _HAS_TORCH:
            raise RuntimeError("torch needed to save checkpoint.")

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
            ep_name = _format_filename(
                self.every_filename,
                epoch=epoch_num,
                monitor=self.monitor,
                score=None,
            )
            ep_path = os.path.join(self.out_dir, ep_name)
            torch.save(state, ep_path)
            paths["interval"] = ep_path

        score = metrics.get(self.monitor)
        if score is None and "/" in self.monitor:
            score = metrics.get(self.monitor.split("/", 1)[-1])

        if score is not None and self.top_k != 0:
            score = float(score)
            best_name = _format_filename(
                self.best_filename,
                epoch=epoch_num,
                monitor=self.monitor,
                score=score,
            )
            best_path = os.path.join(self.out_dir, best_name)
            torch.save(state, best_path)
            paths["best_candidate"] = best_path

        removed = self._prune_all_checkpoints()
        if removed:
            paths["pruned"] = removed

        score_files = self._score_files()
        if score_files:
            paths["best"] = score_files[0][1]
            paths["top_k"] = [p for _, p in score_files[: max(0, self.top_k)]]

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
