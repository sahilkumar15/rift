# Path: iganer/rift/utils/config.py
# Status: NEW
"""YAML config loader with dotted access + override merge."""
from __future__ import annotations
import copy, os
from typing import Any, Dict
try:
    import yaml
except Exception:
    yaml = None

class Config(dict):
    """dict with attribute + dotted-key access."""
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    __setattr__ = dict.__setitem__
    def get_dotted(self, dotted: str, default=None):
        cur = self
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur: cur = cur[part]
            else: return default
        return cur

def _wrap(d):
    if isinstance(d, dict): return Config({k: _wrap(v) for k, v in d.items()})
    if isinstance(d, list): return [_wrap(v) for v in d]
    return d

def load_config(path: str) -> Config:
    if yaml is None:
        raise RuntimeError("pyyaml not installed: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _wrap(raw)

def merge_overrides(cfg: Config, overrides: Dict[str, Any]) -> Config:
    out = copy.deepcopy(cfg)
    for dotted, val in overrides.items():
        parts = dotted.split("."); cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, Config())
        cur[parts[-1]] = val
    return out
