# Path: iganer/rift/utils/seed.py
# Status: NEW
"""Global seeding for reproducibility; captures RNG states for checkpoints."""
from __future__ import annotations
import os, random
def seed_everything(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception: pass
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception: pass
    return seed
def get_rng_states():
    states = {"python": random.getstate()}
    try:
        import numpy as np; states["numpy"] = np.random.get_state()
    except Exception: pass
    try:
        import torch; states["torch"] = torch.get_rng_state()
        if torch.cuda.is_available(): states["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception: pass
    return states
def set_rng_states(states):
    if "python" in states: random.setstate(states["python"])
    try:
        import numpy as np
        if "numpy" in states: np.random.set_state(states["numpy"])
    except Exception: pass
    try:
        import torch
        if "torch" in states: torch.set_rng_state(states["torch"])
        if "torch_cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])
    except Exception: pass
