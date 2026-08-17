from dataclasses import dataclass


@dataclass
class MetricResult:
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    baseline_accuracy: float


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


def baseline_directional_accuracy(y_true) -> float:
    # Naive baseline: use previous day's return as prediction
    # For simplicity we approximate with a one-step lagged return
    lagged = y_true.shift(1)
    return directional_accuracy(y_true[1:], lagged[1:])
