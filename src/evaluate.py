import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repository root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd

from src.dataset import load_data, get_walk_forward_splits
from src.metrics import mae, rmse, mape, directional_accuracy, baseline_directional_accuracy, MetricResult


def evaluate(split_count: int = 3, test_size_days: int = 252) -> MetricResult:
    """Run walk‑forward evaluation on the pooled dataset.

    Returns a MetricResult aggregating metrics across all splits (average).
    """
    df = load_data()
    splits = get_walk_forward_splits(df, n_splits=split_count, test_size_days=test_size_days)
    if not splits:
        raise ValueError("No train/test splits generated. Check dataset size or parameters.")

    # Containers for per‑split metrics
    mae_vals, rmse_vals, mape_vals, dir_vals, base_vals = [], [], [], [], []

    for train_mask, test_mask in splits:
        train_df = df[train_mask]
        test_df = df[test_mask]
        # Target column expected to be 'ret_5d' (5‑day forward return)
        y_true = test_df["ret_5d"]
        # Predict using the model – reuse predict function for batch mode
        from src.predict import batch_predict  # we will create a helper later; fallback to single predict loop
        # If batch_predict is not available, fall back to simple loop
        try:
            y_pred = batch_predict(test_df.index.get_level_values(0).unique())
        except Exception:
            # Fallback: iterate ticker list
            y_pred = []
            for ts in test_df.index.get_level_values(0).unique():
                pred = __import__("src.predict", fromlist=["predict"]).predict(ts, force_refresh=False)
                y_pred.append(pred["pred_return"]) if pred else y_pred.append(float('nan'))
            y_pred = pd.Series(y_pred, index=test_df.index.get_level_values(0).unique())

        # Align predictions with true values
        y_pred = y_pred.reindex(y_true.index)
        # Drop NaNs that may arise from missing predictions
        mask = y_true.notna() & y_pred.notna()
        y_true, y_pred = y_true[mask], y_pred[mask]

        mae_vals.append(mae(y_true, y_pred))
        rmse_vals.append(rmse(y_true, y_pred))
        mape_vals.append(mape(y_true, y_pred))
        dir_vals.append(directional_accuracy(y_true, y_pred))
        base_vals.append(baseline_directional_accuracy(y_true))

    # Average across splits
    return MetricResult(
        mae=sum(mae_vals) / len(mae_vals),
        rmse=sum(rmse_vals) / len(rmse_vals),
        mape=sum(mape_vals) / len(mape_vals),
        directional_accuracy=sum(dir_vals) / len(dir_vals),
        baseline_accuracy=sum(base_vals) / len(base_vals),
    )


def generate_markdown_report(result: MetricResult, output_path: Path) -> None:
    """Write a simple markdown report summarising the evaluation metrics."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# Model Evaluation Report

*Generated on {now}*

| Metric | Value |
|---|---|
| MAE | {result.mae:.4f} |
| RMSE | {result.rmse:.4f} |
| MAPE (%) | {result.mape:.2f} |
| Directional Accuracy (%) | {result.directional_accuracy:.2f} |
| Naïve Baseline Directional Accuracy (%) | {result.baseline_accuracy:.2f} |

> **Note:** Higher directional accuracy than the naive baseline indicates the model captures some predictive signal.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main():
    # Default locations – can be overridden via env vars if needed
    output_dir = ROOT / "reports"
    output_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    result = evaluate()
    generate_markdown_report(result, output_file)
    print(f"Evaluation report written to {output_file}")


if __name__ == "__main__":
    main()
