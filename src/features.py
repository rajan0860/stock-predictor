import os
import pandas as pd
import pandas_ta_classic as ta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TICKERS

DATA_DIR = "data"

def load_benchmark_data():
    """Load and process Nifty 50, India VIX, USD/INR, and Brent Crude benchmarks."""
    nifty_path = os.path.join(DATA_DIR, "nifty50_price.parquet")
    vix_path = os.path.join(DATA_DIR, "vix_price.parquet")
    usdinr_path = os.path.join(DATA_DIR, "usdinr_price.parquet")
    crude_path = os.path.join(DATA_DIR, "crude_price.parquet")
    
    benchmarks = pd.DataFrame()
    
    if os.path.exists(nifty_path):
        nifty = pd.read_parquet(nifty_path)
        if nifty.index.tz is not None:
            nifty.index = nifty.index.tz_localize(None)
        benchmarks['nifty_close'] = nifty['Close']
        benchmarks['nifty_ret_1d'] = nifty['Close'].pct_change(1)
        benchmarks['nifty_ret_5d'] = nifty['Close'].pct_change(5)
        benchmarks['nifty_vol_20d'] = benchmarks['nifty_ret_1d'].rolling(20).std()
        benchmarks['nifty_fwd_5d'] = nifty['Close'].shift(-5) / nifty['Close'] - 1
    
    if os.path.exists(vix_path):
        vix = pd.read_parquet(vix_path)
        if vix.index.tz is not None:
            vix.index = vix.index.tz_localize(None)
        benchmarks['vix_close'] = vix['Close']
        benchmarks['vix_ret_1d'] = vix['Close'].pct_change(1)
        
    if os.path.exists(usdinr_path):
        usdinr = pd.read_parquet(usdinr_path)
        if usdinr.index.tz is not None:
            usdinr.index = usdinr.index.tz_localize(None)
        benchmarks['usdinr_close'] = usdinr['Close']
        benchmarks['usdinr_ret_1d'] = usdinr['Close'].pct_change(1)
        benchmarks['usdinr_ret_5d'] = usdinr['Close'].pct_change(5)

    if os.path.exists(crude_path):
        crude = pd.read_parquet(crude_path)
        if crude.index.tz is not None:
            crude.index = crude.index.tz_localize(None)
        benchmarks['crude_close'] = crude['Close']
        benchmarks['crude_ret_1d'] = crude['Close'].pct_change(1)
        benchmarks['crude_ret_5d'] = crude['Close'].pct_change(5)
        benchmarks['crude_vol_20d'] = benchmarks['crude_ret_1d'].rolling(20).std()
        
    return benchmarks

def compute_features():
    benchmarks = load_benchmark_data()
    
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
        # Returns & Momentum
        df['ret_1d'] = df['Close'].pct_change(1)
        df['ret_5d'] = df['Close'].pct_change(5)
        df['ret_20d'] = df['Close'].pct_change(20)
        
        # Lagged Return Features (Autocorrelation signals)
        df['ret_1d_lag1'] = df['ret_1d'].shift(1)
        df['ret_1d_lag2'] = df['ret_1d'].shift(2)
        df['ret_5d_lag5'] = df['ret_5d'].shift(5)
        
        # Moving Averages & Ratios
        df['ma_5'] = ta.sma(df['Close'], length=5)
        df['ma_20'] = ta.sma(df['Close'], length=20)
        df['ma_50'] = ta.sma(df['Close'], length=50)
        df['price_vs_ma20'] = df['Close'] / df['ma_20'] - 1
        df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20'] - 1
        
        # Bollinger Bands & Bandwidth
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None and not bb.empty:
            bbl = bb.iloc[:, 0]  # lower
            bbm = bb.iloc[:, 1]  # mid
            bbu = bb.iloc[:, 2]  # upper
            df['bb_pct'] = (df['Close'] - bbl) / (bbu - bbl + 1e-8)
            df['bb_width'] = (bbu - bbl) / (bbm + 1e-8)
        else:
            df['bb_pct'] = 0.5
            df['bb_width'] = 0.0
            
        # Normalized ATR (relative volatility)
        if 'High' in df.columns and 'Low' in df.columns:
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['atr_pct'] = atr / (df['Close'] + 1e-8)
        else:
            df['atr_pct'] = 0.0
            
        # Volume Dynamics
        if 'Volume' in df.columns:
            vol_ma20 = df['Volume'].rolling(window=20).mean()
            df['vol_vs_ma20'] = df['Volume'] / (vol_ma20 + 1e-8) - 1
        else:
            df['vol_vs_ma20'] = 0.0
        
        # RSI & MACD
        df['rsi_14'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        if macd is not None and not macd.empty:
            df['macd'] = macd.iloc[:, 0]  # MACD line
            df['macd_signal'] = macd.iloc[:, 1]
            df['macd_hist'] = macd.iloc[:, 2]
            
        # Historical Volatility
        df['volatility_20d'] = df['ret_1d'].rolling(window=20).std()
        
        # Calendar Seasonality Features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # --- Merge Benchmark Features ---
        if not benchmarks.empty:
            df = df.join(benchmarks, how='left')
            # Forward-fill benchmark columns in case of slight holiday mismatch
            for bcol in benchmarks.columns:
                df[bcol] = df[bcol].ffill()
            if 'nifty_ret_5d' in df.columns:
                df['excess_ret_5d'] = df['ret_5d'] - df['nifty_ret_5d']
        
        # --- Targets ---
        # Target 1: 5-day forward continuous percentage return
        df['target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        # Target 2: 5-day forward binary direction (1 if return > 0 else 0)
        df['target_direction'] = (df['target_5d'] > 0).astype('Int64')
        # Target 3: 5-day forward excess alpha return over Nifty 50
        if 'nifty_fwd_5d' in df.columns:
            df['target_alpha_5d'] = df['target_5d'] - df['nifty_fwd_5d']
            df['target_alpha_dir'] = (df['target_alpha_5d'] > 0).astype('Int64')
        else:
            df['target_alpha_5d'] = df['target_5d']
            df['target_alpha_dir'] = df['target_direction']
        
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
                fund['revenue_yoy'] = fund['revenue'].pct_change(periods=4, fill_method=None) # 4 quarters = 1 year
                
            # Merge with price data using forward-fill
            combined_index = df.index.union(fund.index).sort_values()
            fund_reindexed = fund.reindex(combined_index).ffill()
            fund_daily = fund_reindexed.reindex(df.index)
            
            # Add to main dataframe
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
