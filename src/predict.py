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
        # Re-run fetch and features for this specific ticker to get live data
        try:
            from src.fetch_data import TICKERS as FETCH_TICKERS
            from src.features import TICKERS as FEAT_TICKERS
            import src.fetch_data as fetch_data
            import src.features as features
            
            # Temporarily override TICKERS to just this one so it's fast
            original_fetch_tickers = FETCH_TICKERS.copy()
            original_feat_tickers = FEAT_TICKERS.copy()
            
            fetch_data.TICKERS = [ticker]
            features.TICKERS = [ticker]
            
            # Suppress prints for cleaner output
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                fetch_data.fetch_data()
                features.compute_features()
            finally:
                sys.stdout = old_stdout
                fetch_data.TICKERS = original_fetch_tickers
                features.TICKERS = original_feat_tickers
        except Exception as e:
            print(f"Warning: Live refresh failed ({e}), using cached data.")

    from src.dataset import load_data
    df = load_data()
    ticker_df = df[df['ticker'] == ticker].copy()
    
    if len(ticker_df) == 0:
        print(f"Error: No data found for {ticker}.")
        return None
        
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data.get('regressor', model_data.get('model'))
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
    
    if 'ticker' in X.columns:
        X['ticker'] = X['ticker'].astype('category')
        
    pred_return = model.predict(X)[0]
    latest_price = latest_row['Close'].values[0]
    implied_price = latest_price * (1 + pred_return)
    
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
    if prob_up is not None:
        direction_label = "BULLISH (UP)" if prob_up >= 0.5 else "BEARISH (DOWN)"
        print(f"Directional Signal: {direction_label} (Confidence: {prob_up*100:.1f}%)")
    print("This is a model estimate, not financial advice.\n")
    
    return {
        'date': date_str,
        'ticker': ticker,
        'latest_price': float(latest_price),
        'pred_return': float(pred_return),
        'implied_price': float(implied_price),
        'prob_up': prob_up
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
