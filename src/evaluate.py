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
    import lightgbm as lgb
    
    df = load_data()
    # Filter rows with target available
    clean_df = df.dropna(subset=['target_5d']).copy()
    clean_df['ticker'] = clean_df['ticker'].astype('category')
    
    exclude_cols = ['target_5d', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in clean_df.columns if c not in exclude_cols]
    
    splits = get_walk_forward_splits(clean_df, n_splits=split_count, test_size_days=test_size_days)
    if not splits:
        raise ValueError("No train/test splits generated. Check dataset size or parameters.")

    params = {
        'n_estimators': 300,
        'max_depth': 5,
        'num_leaves': 15,
        'learning_rate': 0.03,
        'min_child_samples': 30,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Containers for per‑split metrics
    mae_vals, rmse_vals, mape_vals, dir_vals, base_vals = [], [], [], [], []

    for train_mask, test_mask in splits:
        train_sub = clean_df[train_mask]
        test_sub = clean_df[test_mask]
        
        X_train, y_train = train_sub[feature_cols], train_sub['target_5d']
        X_test, y_test = test_sub[feature_cols], test_sub['target_5d']
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        y_pred = pd.Series(preds, index=y_test.index)

        # Drop NaNs if any
        mask = y_test.notna() & y_pred.notna()
        y_t, y_p = y_test[mask], y_pred[mask]

        mae_vals.append(mae(y_t, y_p))
        rmse_vals.append(rmse(y_t, y_p))
        mape_vals.append(mape(y_t, y_p))
        dir_vals.append(directional_accuracy(y_t, y_p))
        base_vals.append(baseline_directional_accuracy(y_t))

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
