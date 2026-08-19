import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# Ensure repository root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dataset import load_data, get_walk_forward_splits
from src.metrics import (
    mae, rmse, mape, directional_accuracy, 
    high_conf_directional_accuracy, compute_roc_auc,
    baseline_directional_accuracy, MetricResult
)


def evaluate(split_count: int = 3, test_size_days: int = 252, purge_days: int = 5) -> MetricResult:
    """Run walk‑forward evaluation with dual models on the pooled dataset."""
    df = load_data()
    clean_df = df.dropna(subset=['target_5d']).copy()
    clean_df['ticker'] = clean_df['ticker'].astype('category')
    
    exclude_cols = [
        'target_5d', 'target_direction', 
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Dividends', 'Stock Splits',
        'nifty_close', 'vix_close'
    ]
    feature_cols = [c for c in clean_df.columns if c not in exclude_cols]
    
    splits = get_walk_forward_splits(clean_df, n_splits=split_count, test_size_days=test_size_days, purge_days=purge_days)
    if not splits:
        raise ValueError("No train/test splits generated. Check dataset size or parameters.")

    reg_params = {
        'n_estimators': 150,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.02,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    clf_params = {
        'n_estimators': 150,
        'max_depth': 3,
        'num_leaves': 8,
        'learning_rate': 0.02,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    mae_vals, rmse_vals, mape_vals, dir_vals, high_conf_vals, roc_vals, base_vals = [], [], [], [], [], [], []
    per_fold_results = []
    
    # Store out-of-fold predictions per ticker for per-ticker analysis
    all_ticker_preds = []

    fold_idx = 1
    for train_mask, test_mask in splits:
        train_sub = clean_df[train_mask]
        test_sub = clean_df[test_mask]
        
        X_train, y_train = train_sub[feature_cols], train_sub['target_5d']
        y_train_dir = (y_train > 0).astype(int)
        
        X_test, y_test = test_sub[feature_cols], test_sub['target_5d']
        y_test_dir = (y_test > 0).astype(int)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        # 1. Regressor
        model_reg = lgb.LGBMRegressor(**reg_params)
        model_reg.fit(X_train, y_train)
        preds_reg = pd.Series(model_reg.predict(X_test), index=y_test.index)
        
        # 2. Classifier
        model_clf = lgb.LGBMClassifier(**clf_params)
        model_clf.fit(X_train, y_train_dir)
        preds_prob = pd.Series(model_clf.predict_proba(X_test)[:, 1], index=y_test.index)
        preds_dir = (preds_prob >= 0.5).astype(int)

        # Record metrics for fold
        f_mae = mae(y_test, preds_reg)
        f_rmse = rmse(y_test, preds_reg)
        f_mape = mape(y_test, preds_reg)
        f_dir = directional_accuracy(y_test, preds_reg)
        f_high_conf = high_conf_directional_accuracy(y_test, preds_prob, threshold=0.55)
        f_roc = compute_roc_auc(y_test, preds_prob)
        f_base = baseline_directional_accuracy(y_test)

        mae_vals.append(f_mae)
        rmse_vals.append(f_rmse)
        mape_vals.append(f_mape)
        dir_vals.append(f_dir)
        high_conf_vals.append(f_high_conf)
        roc_vals.append(f_roc)
        base_vals.append(f_base)

        per_fold_results.append({
            'fold': fold_idx,
            'train_start': X_train.index.min().strftime('%Y-%m'),
            'train_end': X_train.index.max().strftime('%Y-%m'),
            'test_start': X_test.index.min().strftime('%Y-%m'),
            'test_end': X_test.index.max().strftime('%Y-%m'),
            'mae': f_mae,
            'rmse': f_rmse,
            'directional_accuracy': f_dir,
            'high_conf_accuracy': f_high_conf,
            'roc_auc': f_roc,
            'baseline_accuracy': f_base
        })

        # Keep predictions for ticker breakdown
        fold_eval_df = pd.DataFrame({
            'ticker': test_sub['ticker'],
            'y_true': y_test,
            'y_pred_reg': preds_reg,
            'y_prob': preds_prob
        })
        all_ticker_preds.append(fold_eval_df)
        fold_idx += 1

    # Per-ticker breakdown
    per_ticker_results = {}
    if all_ticker_preds:
        combined_preds = pd.concat(all_ticker_preds)
        for t, group in combined_preds.groupby('ticker', observed=True):
            t_str = str(t)
            t_mae = mae(group['y_true'], group['y_pred_reg'])
            t_dir = directional_accuracy(group['y_true'], group['y_pred_reg'])
            t_high_conf = high_conf_directional_accuracy(group['y_true'], group['y_prob'], threshold=0.55)
            t_base = baseline_directional_accuracy(group['y_true'])
            per_ticker_results[t_str] = {
                'mae': t_mae,
                'directional_accuracy': t_dir,
                'high_conf_accuracy': t_high_conf,
                'baseline_accuracy': t_base,
                'samples': len(group)
            }

    return MetricResult(
        mae=float(np.mean(mae_vals)),
        rmse=float(np.mean(rmse_vals)),
        mape=float(np.mean(mape_vals)),
        directional_accuracy=float(np.mean(dir_vals)),
        high_conf_directional_accuracy=float(np.mean(high_conf_vals)),
        roc_auc=float(np.mean(roc_vals)),
        baseline_accuracy=float(np.mean(base_vals)),
        per_fold_results=per_fold_results,
        per_ticker_results=per_ticker_results
    )


def generate_markdown_report(result: MetricResult, output_path: Path) -> None:
    """Write an enriched markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fold_rows = ""
    if result.per_fold_results:
        for f in result.per_fold_results:
            fold_rows += f"| Fold {f['fold']} ({f['test_start']} -> {f['test_end']}) | {f['mae']:.4f} | {f['rmse']:.4f} | {f['directional_accuracy']:.1f}% | {f['high_conf_accuracy']:.1f}% | {f['roc_auc']:.3f} | {f['baseline_accuracy']:.1f}% |\n"
            
    ticker_rows = ""
    if result.per_ticker_results:
        for t, m in result.per_ticker_results.items():
            ticker_rows += f"| `{t}` | {m['mae']:.4f} | {m['directional_accuracy']:.1f}% | {m['high_conf_accuracy']:.1f}% | {m['baseline_accuracy']:.1f}% | {m['samples']} |\n"

    content = f"""# Walk-Forward Model Evaluation Report

*Generated on {now}*

---

## 1. Executive Summary

| Metric | Model Performance | Naïve Baseline (Always Up) |
|---|---|---|
| **MAE (5-Day Return Error)** | **`{result.mae:.4f}`** | — |
| **RMSE** | **`{result.rmse:.4f}`** | — |
| **Directional Accuracy (%)** | **`{result.directional_accuracy:.2f}%`** | **`{result.baseline_accuracy:.2f}%`** |
| **High-Confidence Directional Accuracy (>55% conviction)** | **`{result.high_conf_directional_accuracy:.2f}%`** | **`{result.baseline_accuracy:.2f}%`** |
| **ROC-AUC (Directional Classifier)** | **`{result.roc_auc:.3f}`** | `0.500` |

---

## 2. Walk-Forward Fold-by-Fold Performance

| Fold Window | MAE | RMSE | Dir Acc (%) | High-Conf Acc (%) | ROC-AUC | Naïve Baseline (%) |
|---|---|---|---|---|---|---|
{fold_rows}
---

## 3. Per-Ticker Breakdown

| Ticker | MAE | Dir Acc (%) | High-Conf Acc (%) | Naïve Baseline (%) | Evaluated Samples |
|---|---|---|---|---|---|
{ticker_rows}
---

> **Methodology Notes**:
> - Expanding window walk-forward validation with **5-day purge/embargo gap** to prevent lookahead leakage.
> - Dual architecture: Regularized LightGBM Regressor for return magnitude + LightGBM Classifier for directional probability.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main():
    output_dir = ROOT / "reports"
    output_file = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    result = evaluate()
    generate_markdown_report(result, output_file)
    print(f"Evaluation report written to {output_file}")


if __name__ == "__main__":
    main()

