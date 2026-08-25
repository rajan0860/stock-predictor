import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import joblib
import optuna

# Suppress Optuna info logs during optimization
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.dataset import load_data, get_walk_forward_splits

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_stock_model.txt")


def optimize_hyperparameters(train_df, feature_cols, n_trials=30):
    """Run Optuna Bayesian optimization across purged walk-forward splits."""
    splits = get_walk_forward_splits(train_df, n_splits=3, purge_days=5)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 250, step=25),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'num_leaves': trial.suggest_int('num_leaves', 6, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.045, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 35, 80),
            'subsample': trial.suggest_float('subsample', 0.65, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.05, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        fold_maes = []
        for train_mask, test_mask in splits:
            X_train, y_train = train_df[train_mask][feature_cols], train_df[train_mask]['target_5d']
            X_test, y_test = train_df[test_mask][feature_cols], train_df[test_mask]['target_5d']

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, preds))

        return np.mean(fold_maes)

    print("Running Bayesian Hyperparameter Optimization (Optuna)...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Optuna Best Validation MAE: {study.best_value:.4f}")
    return study.best_params


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading dataset...")
    df = load_data()
    
    # Drop rows where target is NaN
    train_df = df.dropna(subset=['target_5d']).copy()
    print(f"Total training samples: {len(train_df)}")
    
    # Exclude targets and raw non-feature price columns
    exclude_cols = [
        'target_5d', 'target_direction', 'target_alpha_5d', 'target_alpha_dir',
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'Dividends', 'Stock Splits',
        'nifty_close', 'vix_close', 'usdinr_close', 'crude_close', 'dxy_close', 'us10y_close', 'nifty_fwd_5d'
    ]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"Predictive Features ({len(feature_cols)}): {feature_cols}\n")
    
    # Convert 'ticker' to categorical type for LightGBM
    train_df['ticker'] = train_df['ticker'].astype('category')
    
    # Run Optuna tuning
    best_params = optimize_hyperparameters(train_df, feature_cols, n_trials=30)
    best_params.update({'random_state': 42, 'n_jobs': -1, 'verbose': -1})
    print(f"\nOptimal Tuned Hyperparameters: {best_params}\n")

    # Walk-forward validation with 5-day embargo/purge
    print("--- Purged Walk-Forward Cross-Validation ---")
    splits = get_walk_forward_splits(train_df, n_splits=3, purge_days=5)
    
    fold = 1
    maes = []
    accs = []
    alpha_accs = []
    high_conf_accs = []
    naive_accs = []
    
    for train_mask, test_mask in splits:
        train_sub = train_df[train_mask]
        test_sub = train_df[test_mask]

        X_train = train_sub[feature_cols]
        y_train = train_sub['target_5d']
        y_train_alpha = train_sub['target_alpha_5d']
        
        # Noise deadband filter: remove tiny moves (|return| < 0.3%) for directional classification training
        clean_mask = y_train.abs() >= 0.003
        X_train_clf = X_train[clean_mask]
        y_train_clf = (y_train[clean_mask] > 0).astype(int)

        X_test = test_sub[feature_cols]
        y_test = test_sub['target_5d']
        y_test_dir = (y_test > 0).astype(int)
        y_test_alpha_dir = (test_sub['target_alpha_5d'] > 0).astype(int)
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        # 1. Fit Tuned Regressor (Nominal Return)
        model_reg = lgb.LGBMRegressor(**best_params)
        model_reg.fit(X_train, y_train)
        preds_reg = model_reg.predict(X_test)

        # 2. Fit Alpha Regressor (Excess Return over Nifty 50)
        model_alpha = lgb.LGBMRegressor(**best_params)
        model_alpha.fit(X_train, y_train_alpha)
        preds_alpha = model_alpha.predict(X_test)
        
        # 3. Fit Deadband-Tuned Classifier
        model_clf = lgb.LGBMClassifier(**best_params)
        model_clf.fit(X_train_clf, y_train_clf)
        preds_prob = model_clf.predict_proba(X_test)[:, 1]
        preds_dir = (preds_prob >= 0.5).astype(int)
        
        # Metrics
        mae_val = mean_absolute_error(y_test, preds_reg)
        acc_val = accuracy_score(y_test_dir, preds_dir)
        alpha_acc_val = accuracy_score(y_test_alpha_dir, (preds_alpha > 0).astype(int))
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
        print(f"  MAE: {mae_val:.4f} | Dir Acc: {acc_val:.1%} | Alpha Acc: {alpha_acc_val:.1%} | High-Conf: {high_conf_acc:.1%} ({high_conf_count} samples) | Naive: {naive_val:.1%}")
        
        maes.append(mae_val)
        accs.append(acc_val)
        alpha_accs.append(alpha_acc_val)
        high_conf_accs.append(high_conf_acc)
        naive_accs.append(naive_val)
        fold += 1
        
    print("\n--- Cross-Validation Summary ---")
    if maes:
        print(f"Avg MAE: {np.mean(maes):.4f}")
        print(f"Avg Overall Dir Acc: {np.mean(accs):.1%}")
        print(f"Avg Alpha Outperformance Acc: {np.mean(alpha_accs):.1%}")
        print(f"Avg High-Conf Dir Acc: {np.mean(high_conf_accs):.1%}")
        print(f"Avg Naive Baseline: {np.mean(naive_accs):.1%}")
    
    # Train final models on ALL data
    print("\nTraining final models on full dataset...")
    X_all = train_df[feature_cols]
    y_all = train_df['target_5d']
    y_all_alpha = train_df['target_alpha_5d']
    
    clean_all = y_all.abs() >= 0.003
    X_all_clf = X_all[clean_all]
    y_all_clf = (y_all[clean_all] > 0).astype(int)
    
    final_reg = lgb.LGBMRegressor(**best_params)
    final_reg.fit(X_all, y_all)

    final_alpha = lgb.LGBMRegressor(**best_params)
    final_alpha.fit(X_all, y_all_alpha)
    
    final_clf = lgb.LGBMClassifier(**best_params)
    final_clf.fit(X_all_clf, y_all_clf)
    
    # Save model package
    model_data = {
        'model': final_reg,
        'regressor': final_reg,
        'alpha_model': final_alpha,
        'classifier': final_clf,
        'best_params': best_params,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Optimized model package saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()


