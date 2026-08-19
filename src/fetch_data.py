import os
import yfinance as yf
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TICKERS

DATA_DIR = "data"

BENCHMARK_TICKERS = {
    "^NSEI": "nifty50_price.parquet",
    "^INDIAVIX": "vix_price.parquet"
}

def fetch_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Fetch Benchmark & Macro Indices
    for bench_ticker, filename in BENCHMARK_TICKERS.items():
        print(f"Fetching benchmark data for {bench_ticker}...")
        try:
            bench_stock = yf.Ticker(bench_ticker)
            bench_hist = bench_stock.history(period="5y", interval="1d")
            if not bench_hist.empty:
                bench_hist.to_parquet(os.path.join(DATA_DIR, filename))
                print(f"  Saved benchmark {bench_ticker}: {len(bench_hist)} days")
            else:
                print(f"  Warning: No price history found for {bench_ticker}")
        except Exception as e:
            print(f"  Warning: Failed to fetch {bench_ticker}: {e}")

    # 2. Fetch Individual Stock Tickers
    for ticker in TICKERS:
        print(f"Fetching data for {ticker}...")
        stock = yf.Ticker(ticker)
        
        # 1. Fetch daily price history (5 years)
        hist = stock.history(period="5y", interval="1d")
        if hist.empty:
            print(f"Warning: No price history found for {ticker}")
        else:
            hist.to_parquet(os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}_price.parquet"))
            print(f"  Saved price history: {len(hist)} days")
            
        # 2. Fetch fundamentals
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
    fetch_data()

