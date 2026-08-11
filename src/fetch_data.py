import os
import yfinance as yf
import pandas as pd

TICKERS = ["RELIANCE.NS", "TCS.NS"]
DATA_DIR = "data"

def fetch_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    
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

if __name__ == "__main__":
    fetch_data()
