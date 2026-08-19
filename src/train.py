import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import joblib

from src.dataset import load_data, get_walk_forward_splits

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_stock_model.txt")

def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading dataset...")
    df = load_data()
    
    # Drop rows where target is NaN
    train_df = df.dropna(subset=['target_5d']).copy()
    print(f"Total training samples: {len(train_df)}")
    
    # Exclude targets and raw non-feature columns
    exclude_cols = [
        'target_5d', 'target_direction', 
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Dividends', 'Stock Splits',
        'nifty_close', 'vix_close'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}\n")
    
    # Convert 'ticker' to categorical type for LightGBM
    train_df['ticker'] = train_df['ticker'].astype('category')
    
    # Regularized hyperparameters for financial signal extraction
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
    
    # Walk-forward validation with 5-day embargo/purge
    print("--- Purged Walk-Forward Validation ---")
    splits = get_walk_forward_splits(train_df, n_splits=3, purge_days=5)
    
    fold = 1
    maes = []
    accs = []
    high_conf_accs = []
    naive_accs = []
    
    for train_mask, test_mask in splits:
        X_train = train_df[train_mask][feature_cols]
        y_train = train_df[train_mask]['target_5d']
        y_train_dir = (y_train > 0).astype(int)
        
        X_test = train_df[test_mask][feature_cols]
        y_test = train_df[test_mask]['target_5d']
        y_test_dir = (y_test > 0).astype(int)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        # Fit Regressor
        model_reg = lgb.LGBMRegressor(**reg_params)
        model_reg.fit(X_train, y_train)
        preds_reg = model_reg.predict(X_test)
        
        # Fit Classifier
        model_clf = lgb.LGBMClassifier(**clf_params)
        model_clf.fit(X_train, y_train_dir)
        preds_prob = model_clf.predict_proba(X_test)[:, 1]
        preds_dir = (preds_prob >= 0.5).astype(int)
        
        # Metrics
        mae_val = mean_absolute_error(y_test, preds_reg)
        acc_val = accuracy_score(y_test_dir, preds_dir)
        naive_val = accuracy_score(y_test_dir, np.ones_like(y_test_dir))
        
        # High confidence predictions (>55% or <45%)
        high_conf_mask = (preds_prob >= 0.55) | (preds_prob <= 0.45)
        if high_conf_mask.sum() > 0:
            high_conf_acc = accuracy_score(y_test_dir[high_conf_mask], preds_dir[high_conf_mask])
            high_conf_count = high_conf_mask.sum()
        else:
            high_conf_acc = acc_val
            high_conf_count = 0
            
        train_start = X_train.index.min().strftime('%Y-%m')
        train_end = X_train.index.max().strftime('%Y-%m')
        test_start = X_test.index.min().strftime('%Y-%m')
        test_end = X_test.index.max().strftime('%Y-%m')
        
        print(f"Fold {fold} | Train: {train_start} -> {train_end} | Test: {test_start} -> {test_end}")
        print(f"  MAE: {mae_val:.4f} | Dir Acc: {acc_val:.1%} | High-Conf Acc: {high_conf_acc:.1%} ({high_conf_count} samples) | Naive: {naive_val:.1%}")
        
        maes.append(mae_val)
        accs.append(acc_val)
        high_conf_accs.append(high_conf_acc)
        naive_accs.append(naive_val)
        fold += 1
        
    print("\n--- Cross-Validation Summary ---")
    if maes:
        print(f"Avg MAE: {np.mean(maes):.4f}")
        print(f"Avg Overall Dir Acc: {np.mean(accs):.1%}")
        print(f"Avg High-Conf Dir Acc: {np.mean(high_conf_accs):.1%}")
        print(f"Avg Naive Baseline: {np.mean(naive_accs):.1%}")
    
    # Train final models on ALL data
    print("\nTraining final models on full dataset...")
    X_all = train_df[feature_cols]
    y_all = train_df['target_5d']
    y_all_dir = (y_all > 0).astype(int)
    
    final_reg = lgb.LGBMRegressor(**reg_params)
    final_reg.fit(X_all, y_all)
    
    final_clf = lgb.LGBMClassifier(**clf_params)
    final_clf.fit(X_all, y_all_dir)
    
    # Save model package
    model_data = {
        'model': final_reg,
        'regressor': final_reg,
        'classifier': final_clf,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Optimized model package saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()

