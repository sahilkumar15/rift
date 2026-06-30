# Path: iganer/rift/utils/checkpoint_manager.py
# Status: NEW
"""Top-k + latest + interval checkpointing with resume. torch-guarded."""
from __future__ import annotations
import os, glob
from typing import Dict, List, Optional
try:
    import torch; _HAS_TORCH = True
except Exception: _HAS_TORCH = False

class CheckpointManager:
    def __init__(self, out_dir, monitor="val/rift_score", mode="max",
                 top_k=3, interval=10):
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.monitor = monitor; self.mode = mode; self.top_k = top_k
        self.interval = interval; self.best: List = []  # (score, path)
    def _better(self, a, b): return a > b if self.mode == "max" else a < b
    def save(self, state: Dict, epoch: int, metrics: Dict):
        if not _HAS_TORCH: raise RuntimeError("torch needed to save ckpt (Katz).")
        latest = os.path.join(self.out_dir, "latest.pth")
        torch.save(state, latest)
        paths = {"latest": latest}
        if self.interval and epoch % self.interval == 0:
            p = os.path.join(self.out_dir, f"epoch_{epoch:04d}.pth")
            torch.save(state, p); paths["interval"] = p
        score = metrics.get(self.monitor)
        if score is not None:
            p = os.path.join(self.out_dir, f"best_e{epoch:04d}_{score:.4f}.pth")
            torch.save(state, p); self.best.append((score, p))
            self.best.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            for _, old in self.best[self.top_k:]:
                if os.path.exists(old): os.remove(old)
            self.best = self.best[:self.top_k]
            paths["best"] = self.best[0][1]; paths["top_k"] = [p for _, p in self.best]
        return paths
    def resume(self, mode="auto"):
        if mode in (None, "none"): return None
        if mode == "auto":
            latest = os.path.join(self.out_dir, "latest.pth")
            return latest if os.path.exists(latest) else None
        return mode if os.path.exists(mode) else None
    def load(self, path):
        if not _HAS_TORCH: raise RuntimeError("torch needed.")
        return torch.load(path, map_location="cpu")
