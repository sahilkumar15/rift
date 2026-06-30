# Path: src/utils/distributed.py
# Status: NEW
"""Tiny DDP helpers; single-process safe defaults."""
import os
def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0
def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))
