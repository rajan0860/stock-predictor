import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib

from dataset import load_data, get_walk_forward_splits

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_stock_model.txt")

def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading dataset...")
    df = load_data()
    
    # Drop rows where target is NaN (we can't train on them)
    train_df = df.dropna(subset=['target_5d']).copy()
    print(f"Total training samples: {len(train_df)}")
    
    # Features to use for training
    # Exclude target and raw price columns that shouldn't be features
    exclude_cols = ['target_5d', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Convert 'ticker' to categorical type for LightGBM
    train_df['ticker'] = train_df['ticker'].astype('category')
    
    # Hyperparameters from ARCHITECTURE.md
    params = {
        'n_estimators': 300,
        'max_depth': 5,
        'num_leaves': 15,
        'learning_rate': 0.03,
        'min_child_samples': 30,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Walk-forward validation
    print("\n--- Walk-Forward Validation ---")
    splits = get_walk_forward_splits(train_df, n_splits=3)
    
    fold = 1
    maes = []
    accs = []
    naive_accs = []
    
    for train_mask, test_mask in splits:
        X_train, y_train = train_df[train_mask][feature_cols], train_df[train_mask]['target_5d']
        X_test, y_test = train_df[test_mask][feature_cols], train_df[test_mask]['target_5d']
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        
        # Directional Accuracy (Up or Down)
        actual_dir = np.sign(y_test)
        pred_dir = np.sign(preds)
        
        actual_dir = np.where(actual_dir == 0, 1, actual_dir)
        pred_dir = np.where(pred_dir == 0, 1, pred_dir)
        
        acc = accuracy_score(actual_dir, pred_dir)
        
        # Naive Baseline (assume positive drift)
        naive_acc = accuracy_score(actual_dir, np.ones_like(actual_dir))
        
        train_start = X_train.index.min().strftime('%Y-%m')
        train_end = X_train.index.max().strftime('%Y-%m')
        test_start = X_test.index.min().strftime('%Y-%m')
        test_end = X_test.index.max().strftime('%Y-%m')
        
        print(f"Fold {fold} | Train: {train_start} -> {train_end} | Test: {test_start} -> {test_end}")
        print(f"  MAE: {mae:.4f} | Dir Acc: {acc:.1%} | Naive Baseline: {naive_acc:.1%}")
        
        maes.append(mae)
        accs.append(acc)
        naive_accs.append(naive_acc)
        fold += 1
        
    print("\n--- Cross-Validation Summary ---")
    if maes:
        print(f"Avg MAE: {np.mean(maes):.4f}")
        print(f"Avg Dir Acc: {np.mean(accs):.1%}")
        print(f"Avg Naive Acc: {np.mean(naive_accs):.1%}")
    
    # Train final model on ALL data
    print("\nTraining final model on all data...")
    X_all, y_all = train_df[feature_cols], train_df['target_5d']
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X_all, y_all)
    
    # Save the model and feature columns list so predict.py knows exactly what features to expect
    model_data = {
        'model': final_model,
        'feature_cols': feature_cols
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
