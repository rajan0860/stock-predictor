import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import joblib
import warnings

# Suppress lightgbm warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_stock_model.txt")

def predict(ticker, force_refresh=True):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")
        return None
        
    if force_refresh:
        # Fast incremental delta update for target stock and macro benchmarks
        try:
            from src.fetch_data import update_stock_incremental, update_benchmarks_incremental
            from src.features import compute_single_ticker_features
            
            # 1. Delta fetch recent days
            update_benchmarks_incremental(lookback_days=15)
            update_stock_incremental(ticker, lookback_days=15)
            
            # 2. Recompute features for this single stock
            compute_single_ticker_features(ticker, verbose=False)
        except Exception as e:
            print(f"Warning: Live incremental refresh failed ({e}), using cached data.")

    ticker_safe = ticker.replace('.', '_')
    feature_path = os.path.join("data", f"{ticker_safe}_features.parquet")
    
    if os.path.exists(feature_path):
        ticker_df = pd.read_parquet(feature_path)
    else:
        from src.dataset import load_data
        df = load_data()
        ticker_df = df[df['ticker'] == ticker].copy()
    
    if len(ticker_df) == 0:
        print(f"Error: No data found for {ticker}.")
        return None
        
    try:
        model_data = joblib.load(MODEL_PATH)

        model = model_data.get('regressor', model_data.get('model'))
        alpha_model = model_data.get('alpha_model')
        classifier = model_data.get('classifier')
        feature_cols = model_data['feature_cols']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    # Get the very last row
    latest_row = ticker_df.iloc[-1:]
    latest_date = latest_row.index[0]
    
    # Make prediction
    X = latest_row[feature_cols].copy()
    
    for col in feature_cols:
        if col == 'ticker':
            X[col] = X[col].astype('category')
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

        
    pred_return = float(model.predict(X)[0])
    pred_alpha = float(alpha_model.predict(X)[0]) if alpha_model is not None else None
    latest_price = float(latest_row['Close'].values[0])
    implied_price = float(latest_price * (1 + pred_return))
    
    prob_up = None
    if classifier is not None:
        try:
            prob_up = float(classifier.predict_proba(X)[0, 1])
        except Exception:
            prob_up = None

    date_str = latest_date.strftime('%Y-%m-%d')
    print(f"\nAs of {date_str}, {ticker} closed at Rs.{latest_price:.2f}.")
    print(f"The model predicts a {pred_return*100:+.1f}% return over the next 5 trading days,")
    print(f"implying a price around Rs.{implied_price:.2f}.")
    if pred_alpha is not None:
        print(f"Expected Alpha vs Nifty 50: {pred_alpha*100:+.2f}%")
    if prob_up is not None:
        direction_label = "BULLISH (UP)" if prob_up >= 0.5 else "BEARISH (DOWN)"
        print(f"Directional Signal: {direction_label} (Confidence: {prob_up*100:.1f}%)")
    print("This is a model estimate, not financial advice.\n")
    
    # Compute empirical forecast quality and reliability on comparable historical signals
    try:
        from src.signal_quality import get_forecast_quality_metrics
        quality_metrics = get_forecast_quality_metrics(ticker, pred_return, prob_up)
    except Exception:
        quality_metrics = None

    # Extract technical levels & indicators
    last_row = ticker_df.iloc[-1]
    ma_20 = float(last_row.get('ma_20', latest_price))
    ma_50 = float(last_row.get('ma_50', latest_price))
    rsi_14 = float(last_row.get('rsi_14', 50.0))
    vol_vs_ma20 = float(last_row.get('vol_vs_ma20', 1.0))
    
    lookback_window = min(20, len(ticker_df))
    support_20d = float(ticker_df['Low'].iloc[-lookback_window:].min() if 'Low' in ticker_df.columns else latest_price * 0.96)
    resistance_20d = float(ticker_df['High'].iloc[-lookback_window:].max() if 'High' in ticker_df.columns else latest_price * 1.04)
    
    revenue_yoy = float(last_row.get('revenue_yoy', float('nan')))
    net_margin = float(last_row.get('net_margin', float('nan')))
    debt_to_equity = float(last_row.get('debt_to_equity', float('nan')))
    
    sector_profiles = {
        "RELIANCE.NS": {
            "sector": "Energy, Oil & Telecom Conglomerate",
            "drivers": "Crude refining margins, petrochemical spreads, Jio ARPU growth, retail expansion.",
            "crude_impact": "Dual impact: Downstream refining benefits from crack spreads; upstream benefits from higher realized crude prices.",
            "macro_sens": "High Nifty beta (1.1x). Highly sensitive to FII equity flows."
        },
        "TCS.NS": {
            "sector": "Information Technology / Software Services",
            "drivers": "BFSI & North American enterprise IT deal total contract value (TCV), hiring trends.",
            "crude_impact": "Low direct impact; lower global inflation supports corporate IT spending budgets.",
            "macro_sens": "Direct beneficiary of USD/INR depreciation (forex tailwind). Sensitive to US 10Y yields."
        },
        "INFY.NS": {
            "sector": "Information Technology / Digital Services",
            "drivers": "Cloud transformation deals, generative AI consulting pipeline, wage margins.",
            "crude_impact": "Negligible direct sensitivity; benefits when lower energy prices stabilize global enterprise opex.",
            "macro_sens": "High USD sensitivity (~85% revenue in USD/EUR). FII portfolio favorite."
        },
        "HDFCBANK.NS": {
            "sector": "Financials / Private Sector Banking",
            "drivers": "Deposit growth, Net Interest Margin (NIM) trajectory, credit quality / GNPA.",
            "crude_impact": "Lower crude lowers domestic inflation, enabling RBI repo rate easing cycles.",
            "macro_sens": "Heaviest weight in Nifty 50. Primary vehicle for foreign institutional capital flows."
        },
        "HINDPETRO.NS": {
            "sector": "Oil Refining & Marketing (OMC)",
            "drivers": "Retail fuel marketing margins (petrol/diesel), Gross Refining Margins (GRMs).",
            "crude_impact": "Sharp crude drops significantly expand retail fuel marketing margins, a major net tailwind.",
            "macro_sens": "High volatility around Brent crude and government excise / retail price regulations."
        },
        "BPCL.NS": {
            "sector": "Oil Refining & Marketing (OMC)",
            "drivers": "Marketing margins, Kochi & Bina refinery throughput, dividend yield.",
            "crude_impact": "Brent decline lowers crude procurement costs and improves marketing profitability on auto fuels.",
            "macro_sens": "Strong inverse correlation with spikes in Brent crude; benefits from stable INR."
        },
        "KFINTECH.NS": {
            "sector": "Financial Services / Capital Market Infrastructure",
            "drivers": "Domestic Mutual Fund industry AUM, monthly SIP inflows, alternative asset registrar services.",
            "crude_impact": "Macro stability supports domestic retail equity participation and SIP inflow momentum.",
            "macro_sens": "Tied to Indian equity market capitalization and retail mutual fund volume expansion."
        },
        "JYOTICNC.NS": {
            "sector": "Capital Goods / Industrial CNC Machinery",
            "drivers": "Aerospace, defense, automotive capex cycle, multi-year order book execution.",
            "crude_impact": "Lower energy costs reduce manufacturing input costs and logistics overhead.",
            "macro_sens": "Driven by domestic manufacturing PMI and private sector capital expenditure."
        }
    }
    
    sector_info = sector_profiles.get(ticker, {
        "sector": "Diversified Equities",
        "drivers": "Corporate earnings and domestic demand growth.",
        "crude_impact": "Energy cost sensitivity.",
        "macro_sens": "General market beta."
    })

    try:
        from src.corporate_actions import get_corporate_actions
        corp_actions = get_corporate_actions(ticker)
    except Exception:
        corp_actions = None

    return {
        'date': date_str,
        'ticker': ticker,
        'latest_price': latest_price,
        'pred_return': pred_return,
        'pred_alpha': pred_alpha,
        'implied_price': implied_price,
        'prob_up': prob_up,
        'quality_metrics': quality_metrics,
        'technical_levels': {
            'ma_20': ma_20,
            'ma_50': ma_50,
            'rsi_14': rsi_14,
            'vol_vs_ma20': vol_vs_ma20,
            'support_20d': support_20d,
            'resistance_20d': resistance_20d,
            'dist_ma20_pct': ((latest_price / ma_20) - 1.0) * 100.0 if ma_20 > 0 else 0.0,
            'dist_ma50_pct': ((latest_price / ma_50) - 1.0) * 100.0 if ma_50 > 0 else 0.0,
            'dist_support_pct': ((latest_price / support_20d) - 1.0) * 100.0 if support_20d > 0 else 0.0,
            'dist_resistance_pct': ((resistance_20d / latest_price) - 1.0) * 100.0 if latest_price > 0 else 0.0,
        },
        'fundamentals': {
            'revenue_yoy': revenue_yoy,
            'net_margin': net_margin,
            'debt_to_equity': debt_to_equity
        },
        'sector_context': sector_info,
        'corporate_actions': corp_actions
    }




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <TICKER>")
        sys.exit(1)
        
    ticker = sys.argv[1]
    predict(ticker)


def batch_predict(tickers):
    """Return a pandas Series of predictions for a list of tickers.

    Args:
        tickers (Iterable[str]): Iterable of ticker symbols.
    Returns:
        pd.Series: Series indexed by ticker with predicted returns.
    """
    import pandas as pd
    preds = []
    idx = []
    for t in tickers:
        result = predict(t, force_refresh=False)
        if result and isinstance(result, dict) and 'pred_return' in result:
            preds.append(result['pred_return'])
            idx.append(t)
        else:
            preds.append(float('nan'))
            idx.append(t)
    return pd.Series(preds, index=idx)
