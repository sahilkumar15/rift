# Path: iganer/rift/eval/eval_correlation.py
# Status: NEW
"""Thin wrapper around audit.correlation_runner for the eval entrypoint."""
from ..audit.correlation_runner import run_correlation
def evaluate_correlation(rows, min_n=5): return run_correlation(rows, min_n=min_n)
