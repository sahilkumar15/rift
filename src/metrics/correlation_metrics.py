# Path: src/metrics/correlation_metrics.py
# Status: NEW
"""
correlation_metrics.py — the headline "does faithfulness predict generalization?" math.

Pure stats, no torch. Given per-checkpoint rows of
(faithfulness, in_domain_auc, plausibility, zero_shot_auc), compute Spearman &
Pearson of each predictor against zero_shot_auc, with bootstrap CIs and a small-n
honesty guard (correlations on <5 points are not reportable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import random


def _rank(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def bootstrap_ci(
    x: Sequence[float], y: Sequence[float], fn, n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        xb = [x[i] for i in idx]
        yb = [y[i] for i in idx]
        s = fn(xb, yb)
        if not math.isnan(s):
            stats.append(s)
    if not stats:
        return (float("nan"), float("nan"))
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return (lo, hi)


@dataclass
class CorrelationResult:
    predictor: str
    spearman: float
    pearson: float
    spearman_ci: Tuple[float, float]
    n: int
    reportable: bool          # False if n<5 — guards against headline on too few points
    note: str = ""


def correlate_predictors(
    rows: List[Dict[str, float]],
    target: str = "zero_shot_auc",
    predictors: Sequence[str] = ("faithfulness", "in_domain_auc", "plausibility"),
    min_n: int = 5,
) -> List[CorrelationResult]:
    """
    rows: list of dicts, each one checkpoint, e.g.
        {"faithfulness":0.71,"in_domain_auc":0.99,"plausibility":0.62,"zero_shot_auc":0.88}
    Returns one CorrelationResult per predictor. The PAPER's claim is that
    'faithfulness' has the highest |spearman| with target AND in_domain_auc does not.
    """
    out: List[CorrelationResult] = []
    y = [r[target] for r in rows if target in r]
    for p in predictors:
        x = [r[p] for r in rows if p in r and target in r]
        yy = [r[target] for r in rows if p in r and target in r]
        n = len(x)
        rep = n >= min_n
        out.append(CorrelationResult(
            predictor=p,
            spearman=spearman(x, yy),
            pearson=pearson(x, yy),
            spearman_ci=bootstrap_ci(x, yy, spearman),
            n=n,
            reportable=rep,
            note="" if rep else f"n={n} < {min_n}: NOT reportable as a headline; gather more checkpoints.",
        ))
    return out
