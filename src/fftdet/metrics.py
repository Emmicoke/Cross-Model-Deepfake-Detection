from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(y, score, thr: float = 0.5) -> dict:
    """score = P(fake). Accuracy/F1 use thr; AUC is threshold-free."""
    y = np.asarray(y)
    pred = (np.asarray(score) >= thr).astype(int)
    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan")
    return {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0), "auc": auc}


def ood_gap(auc_in: float, auc_ood: float) -> float:
    """Gap = AUC_in-dist - AUC_OOD (lower is better)."""
    return auc_in - auc_ood


def timed(fn, n_items: int, repeat: int = 3):
    """Return ms per item of fn() (best of `repeat`)."""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return 1000 * best / max(n_items, 1)
