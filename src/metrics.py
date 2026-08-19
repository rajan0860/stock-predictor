from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class MetricResult:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    high_conf_directional_accuracy: float
    roc_auc: float
    baseline_accuracy: float
    per_fold_results: Optional[List[Dict[str, float]]] = None
    per_ticker_results: Optional[Dict[str, Dict[str, float]]] = None


def mae(y_true, y_pred) -> float:
    return float(((y_true - y_pred).abs()).mean())


def rmse(y_true, y_pred) -> float:
    return float(((y_true - y_pred) ** 2).mean() ** 0.5)


def mape(y_true, y_pred) -> float:
    # Avoid division by zero
    nonzero = y_true != 0
    if not nonzero.any():
        return float('nan')
    return float(((y_true[nonzero] - y_pred[nonzero]).abs() / y_true[nonzero].abs()).mean() * 100)


def directional_accuracy(y_true, y_pred) -> float:
    # Convert returns to sign (positive/negative)
    true_sign = (y_true > 0).astype(int)
    pred_sign = (y_pred > 0).astype(int)
    return float((true_sign == pred_sign).mean() * 100)


def high_conf_directional_accuracy(y_true, y_prob, threshold=0.55) -> float:
    """Evaluate accuracy only on predictions where probability is >= threshold or <= (1 - threshold)."""
    true_sign = (y_true > 0).astype(int)
    pred_sign = (y_prob >= 0.5).astype(int)
    mask = (y_prob >= threshold) | (y_prob <= (1 - threshold))
    if mask.sum() == 0:
        return directional_accuracy(y_true, y_prob)
    return float((true_sign[mask] == pred_sign[mask]).mean() * 100)


def compute_roc_auc(y_true, y_prob) -> float:
    true_sign = (y_true > 0).astype(int)
    if len(np.unique(true_sign)) < 2:
        return float('nan')
    return float(roc_auc_score(true_sign, y_prob))


def baseline_directional_accuracy(y_true) -> float:
    """Naïve baseline: Always predicting upward trend (bullish bias)."""
    true_sign = (y_true > 0).astype(int)
    return float((true_sign == 1).mean() * 100)

