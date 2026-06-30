# Path: iganer/rift/utils/logging.py
# Status: NEW
"""Minimal stdout logger with consistent prefix."""
import logging, sys
def get_logger(name="rift", level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        lg.addHandler(h); lg.setLevel(level)
    return lg
