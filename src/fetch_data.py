import os
import yfinance as yf
import pandas as pd
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TICKERS

DATA_DIR = "data"

BENCHMARK_TICKERS = {
    "^NSEI": "nifty50_price.parquet",
    "^INDIAVIX": "vix_price.parquet",
    "USDINR=X": "usdinr_price.parquet",
    "BZ=F": "crude_price.parquet",
    "DX-Y.NYB": "dxy_price.parquet",
    "^TNX": "us10y_price.parquet"
}

def update_ticker_price(ticker: str, filename: str, lookback_days: int = 15) -> bool:
    """Incrementally updates price history for a single ticker.
    
    If the file exists, fetches only recent days and merges with existing history.
    If the file does not exist, performs a full 5-year fetch.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        stock = yf.Ticker(ticker)
        if os.path.exists(filepath):
            existing_df = pd.read_parquet(filepath)
            if existing_df.index.tz is not None:
                existing_df.index = existing_df.index.tz_localize(None)
            
            # Fetch small delta window (e.g. 1 month)
            delta_df = stock.history(period="1mo", interval="1d")
            if not delta_df.empty:
                if delta_df.index.tz is not None:
                    delta_df.index = delta_df.index.tz_localize(None)
                
                # Combine, deduplicate on index (keeping latest delta bar), and sort
                combined = pd.concat([existing_df, delta_df])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                combined.to_parquet(filepath)
                return True
        else:
            # Full fetch if no local cache exists
            hist = stock.history(period="5y", interval="1d")
            if not hist.empty:
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                hist.to_parquet(filepath)
                return True
    except Exception as e:
        print(f"  Warning: Incremental fetch failed for {ticker}: {e}")
    return False

def update_benchmarks_incremental(lookback_days: int = 15, cache_seconds: int = 300):
    """Updates all benchmark indices incrementally if not updated within cache_seconds."""
    nifty_path = os.path.join(DATA_DIR, "nifty50_price.parquet")
    if os.path.exists(nifty_path):
        try:
            mtime = os.path.getmtime(nifty_path)
            now_ts = datetime.datetime.now().timestamp()
            if (now_ts - mtime) < cache_seconds:
                return # Skip redundant network calls
        except Exception:
            pass
            
    for bench_ticker, filename in BENCHMARK_TICKERS.items():
        update_ticker_price(bench_ticker, filename, lookback_days=lookback_days)


def update_stock_incremental(ticker: str, lookback_days: int = 15) -> bool:
    """Updates a single stock's price history incrementally."""
    ticker_safe = ticker.replace('.', '_')
    return update_ticker_price(ticker, f"{ticker_safe}_price.parquet", lookback_days=lookback_days)

def fetch_data(full: bool = False):
    """Fetches benchmark and stock data.
    
    Args:
        full (bool): If True, downloads full 5-year history and quarterly fundamentals.
                     If False, updates price history incrementally.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not full:
        print("Running incremental data update...")
        update_benchmarks_incremental()
        for ticker in TICKERS:
            update_stock_incremental(ticker)
        print("Incremental update complete.")
        return

    # 1. Fetch Benchmark & Macro Indices (Full 5Y)
    for bench_ticker, filename in BENCHMARK_TICKERS.items():
        print(f"Fetching benchmark data for {bench_ticker}...")
        try:
            bench_stock = yf.Ticker(bench_ticker)
            bench_hist = bench_stock.history(period="5y", interval="1d")
            if not bench_hist.empty:
                if bench_hist.index.tz is not None:
                    bench_hist.index = bench_hist.index.tz_localize(None)
                bench_hist.to_parquet(os.path.join(DATA_DIR, filename))
                print(f"  Saved benchmark {bench_ticker}: {len(bench_hist)} days")
            else:
                print(f"  Warning: No price history found for {bench_ticker}")
        except Exception as e:
            print(f"  Warning: Failed to fetch {bench_ticker}: {e}")

    # 2. Fetch Individual Stock Tickers (Full 5Y + Fundamentals)
    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        hist = stock.history(period="5y", interval="1d")
        if hist.empty:
            print(f"Warning: No price history found for {ticker}")
        else:
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            hist.to_parquet(os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_price.parquet"))
            print(f"  Saved price history: {len(hist)} days")
            
        print(f"  Fetching fundamentals for {ticker}...")
        try:
            financials = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            
            if financials is not None and not financials.empty:
                financials.T.to_parquet(os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_financials.parquet"))
                print(f"  Saved financials: {financials.shape[1]} quarters")
            else:
                print(f"  Warning: No quarterly financials found for {ticker}")
                
            if balance_sheet is not None and not balance_sheet.empty:
                balance_sheet.T.to_parquet(os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_balance_sheet.parquet"))
                print(f"  Saved balance sheet: {balance_sheet.shape[1]} quarters")
            else:
                print(f"  Warning: No quarterly balance sheet found for {ticker}")
        except Exception as e:
            print(f"  Warning: Fundamentals fetch failed for {ticker}: {e}")

if __name__ == "__main__":
    fetch_data(full=True)


