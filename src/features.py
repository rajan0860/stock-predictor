import os
import pandas as pd
import pandas_ta as ta

from src.config import TICKERS

DATA_DIR = "data"

def compute_features():
    for ticker in TICKERS:
        ticker_safe = ticker.replace('.', '_')
        print(f"Computing features for {ticker}...")
        
        # Load price data
        price_path = os.path.join(DATA_DIR, f"{ticker_safe}_price.parquet")
        if not os.path.exists(price_path):
            print(f"  Error: Missing price data for {ticker}. Run fetch_data.py first.")
            continue
            
        df = pd.read_parquet(price_path)
        
        # yfinance price index is tz-aware DatetimeIndex. Make tz-naive for easier joining.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        # --- Technical Features ---
        # Returns
        df['ret_1d'] = df['Close'].pct_change(1)
        df['ret_5d'] = df['Close'].pct_change(5)
        df['ret_20d'] = df['Close'].pct_change(20)
        
        # Moving Averages
        df['ma_5'] = ta.sma(df['Close'], length=5)
        df['ma_20'] = ta.sma(df['Close'], length=20)
        df['ma_50'] = ta.sma(df['Close'], length=50)
        df['price_vs_ma20'] = df['Close'] / df['ma_20'] - 1
        
        # RSI & MACD
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        if macd is not None and not macd.empty:
            df['macd'] = macd.iloc[:, 0]  # MACD line
            df['macd_signal'] = macd.iloc[:, 1]
            df['macd_hist'] = macd.iloc[:, 2]
            
        # Volatility
        df['volatility_20d'] = df['ret_1d'].rolling(window=20).std()
        
        # Target: 5-day forward return
        # Shift -5 means today's target is the return from today to 5 days in the future
        df['target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        
        # --- Fundamental Features ---
        fin_path = os.path.join(DATA_DIR, f"{ticker_safe}_financials.parquet")
        bs_path = os.path.join(DATA_DIR, f"{ticker_safe}_balance_sheet.parquet")
        
        if os.path.exists(fin_path) and os.path.exists(bs_path):
            fin_df = pd.read_parquet(fin_path)
            bs_df = pd.read_parquet(bs_path)
            
            # Make tz-naive
            if fin_df.index.tz is not None:
                fin_df.index = fin_df.index.tz_localize(None)
            if bs_df.index.tz is not None:
                bs_df.index = bs_df.index.tz_localize(None)
                
            # Compute fundamental ratios
            fund = pd.DataFrame(index=fin_df.index)
            
            if 'Total Revenue' in fin_df.columns:
                fund['revenue'] = fin_df['Total Revenue']
            
            if 'Net Income' in fin_df.columns and 'Total Revenue' in fin_df.columns:
                fund['net_margin'] = fin_df['Net Income'] / fin_df['Total Revenue']
                
            if 'Total Debt' in bs_df.columns and 'Stockholders Equity' in bs_df.columns:
                # Merge BS into fund
                fund = fund.join(bs_df[['Total Debt', 'Stockholders Equity']], how='outer')
                fund['debt_to_equity'] = fund['Total Debt'] / fund['Stockholders Equity']
            
            # Sort fund index chronologically (oldest to newest) to calculate YoY correctly
            fund = fund.sort_index()
            if 'revenue' in fund.columns:
                fund['revenue_yoy'] = fund['revenue'].pct_change(periods=4) # 4 quarters = 1 year
                
            # Merge with price data using forward-fill
            # 1. Combine indexes to ensure fundamental dates exist
            combined_index = df.index.union(fund.index).sort_values()
            fund_reindexed = fund.reindex(combined_index).ffill()
            
            # 2. Extract only the dates present in the price dataframe
            fund_daily = fund_reindexed.reindex(df.index)
            
            # 3. Add to main dataframe
            for col in ['revenue_yoy', 'net_margin', 'debt_to_equity']:
                if col in fund_daily.columns:
                    df[col] = fund_daily[col]
                else:
                    df[col] = pd.NA
                    
        else:
            print(f"  Warning: Fundamentals missing for {ticker}, filling with NA.")
            df['revenue_yoy'] = pd.NA
            df['net_margin'] = pd.NA
            df['debt_to_equity'] = pd.NA
            
        # Add ticker column
        df['ticker'] = ticker
            
        # Save features
        out_path = os.path.join(DATA_DIR, f"{ticker_safe}_features.parquet")
        df.to_parquet(out_path)
        print(f"  Saved features: {out_path} ({df.shape[1]} columns)")

if __name__ == "__main__":
    compute_features()
