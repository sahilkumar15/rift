# Path: src/metrics/binary_metrics.py
# Status: NEW
"""AUC / EER / AP without sklearn dependency (pure python)."""
from __future__ import annotations
from typing import Sequence, Tuple
def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    # rank-sum (Mann-Whitney U) AUC, average ranks for ties
    ranks = [0.0]*len(pairs); i = 0
    srt = sorted(range(len(pairs)), key=lambda k: pairs[k][0])
    while i < len(pairs):
        j = i
        while j+1 < len(pairs) and pairs[srt[j+1]][0] == pairs[srt[i]][0]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): ranks[srt[k]] = avg
        i = j+1
    sum_pos = sum(r for r,(_,l) in zip(ranks, pairs) if l == 1)
    return (sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
def eer(labels, scores) -> float:
    thr = sorted(set(scores))
    best = 1.0
    P = sum(labels); N = len(labels)-P
    if P==0 or N==0: return float("nan")
    for t in thr:
        fp = sum(1 for s,l in zip(scores,labels) if s>=t and l==0)
        fn = sum(1 for s,l in zip(scores,labels) if s< t and l==1)
        far = fp/N; frr = fn/P
        if abs(far-frr) < best: best = abs(far-frr); val = (far+frr)/2
    return float(val)
def average_precision(labels, scores) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp=0; fp=0; P=sum(labels); ap=0.0; prev_rec=0.0
    if P==0: return float("nan")
    for i in order:
        if labels[i]==1: tp+=1
        else: fp+=1
        rec = tp/P; prec = tp/(tp+fp)
        ap += prec*(rec-prev_rec); prev_rec=rec
    return float(ap)
