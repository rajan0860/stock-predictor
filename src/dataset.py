import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.config import TICKERS
DATA_DIR = "data"

def load_data():
    """Loads feature data for all tickers and combines them into one DataFrame."""
    dfs = []
    for ticker in TICKERS:
        path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_features.parquet")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Run features.py first.")
            continue
        
        df = pd.read_parquet(path)
        dfs.append(df)
        
    if not dfs:
        raise FileNotFoundError("No feature data found. Please run fetch_data.py and features.py")
        
    combined = pd.concat(dfs)
    
    # Sort by date
    combined = combined.sort_index()
    return combined

def get_walk_forward_splits(df, n_splits=3, test_size_days=252, purge_days=5):
    """
    Yields (train_mask, test_mask) for expanding window walk-forward validation
    with an embargo / purge gap to eliminate label lookahead leakage.
    
    Args:
        df (pd.DataFrame): Combined dataset sorted by DatetimeIndex.
        n_splits (int): Number of walk-forward folds.
        test_size_days (int): Number of trading days in each test window.
        purge_days (int): Number of days purged between train and test windows (matches target horizon).
    """
    unique_dates = df.index.unique().sort_values()
    total_days = len(unique_dates)
    
    splits = []
    
    # We want `n_splits` at the end of the dataset
    for i in range(n_splits):
        # The test window for this split
        test_end_idx = total_days - (n_splits - 1 - i) * test_size_days
        test_start_idx = test_end_idx - test_size_days
        
        if test_start_idx <= purge_days:
            continue
            
        # Purge gap: train ends `purge_days` before test starts
        train_end_idx = test_start_idx - purge_days
        
        train_dates = unique_dates[0:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]
        
        # Get boolean masks
        train_mask = df.index.isin(train_dates)
        test_mask = df.index.isin(test_dates)
        
        splits.append((train_mask, test_mask))
        
    return splits
