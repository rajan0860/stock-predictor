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

    if quality_metrics:
        print(f"📊 Historical Reliability: In comparable prior signals ({quality_metrics['comparable_samples']} setups), the model was directionally correct {quality_metrics['directional_accuracy_pct']:.1f}% of the time, with median absolute 5-day error of {quality_metrics['median_abs_error_pct']:.2f}%.")

    return {
        'date': date_str,
        'ticker': ticker,
        'latest_price': latest_price,
        'pred_return': pred_return,
        'pred_alpha': pred_alpha,
        'implied_price': implied_price,
        'prob_up': prob_up,
        'quality_metrics': quality_metrics
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
