import yfinance as yf
import pandas as pd
from datetime import datetime, date

def get_corporate_actions(ticker: str) -> dict:
    """
    Fetches upcoming and recent corporate actions (Earnings, Dividends, Splits)
    for a given stock ticker via yfinance.
    """
    result = {
        "earnings_date": "TBD / Not Announced",
        "earnings_est_eps": "N/A",
        "earnings_est_rev": "N/A",
        "ex_dividend_date": "N/A",
        "dividend_rate": "N/A",
        "dividend_yield": "N/A",
        "last_split": "None in recent period",
        "events_summary": []
    }
    
    try:
        t = yf.Ticker(ticker)
        
        # 1. Calendar
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                # Earnings Date
                e_date = cal.get('Earnings Date')
                if e_date:
                    if isinstance(e_date, list) and len(e_date) > 0:
                        first_d = e_date[0]
                        if isinstance(first_d, (datetime, date)):
                            result["earnings_date"] = first_d.strftime("%d %b %Y")
                        else:
                            result["earnings_date"] = str(first_d)
                    elif isinstance(e_date, (datetime, date)):
                        result["earnings_date"] = e_date.strftime("%d %b %Y")
                
                # EPS Estimates
                eps_avg = cal.get('Earnings Average')
                if eps_avg is not None:
                    result["earnings_est_eps"] = f"₹{eps_avg:.2f}"
                    
                # Ex-Dividend from Calendar
                ex_div = cal.get('Ex-Dividend Date')
                if ex_div:
                    if isinstance(ex_div, (datetime, date)):
                        result["ex_dividend_date"] = ex_div.strftime("%d %b %Y")
                    else:
                        result["ex_dividend_date"] = str(ex_div)
        except Exception:
            pass
            
        # 2. Info for Dividends & Splits
        try:
            info = t.info
            div_rate = info.get('dividendRate')
            curr_px = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose')))
            
            if div_rate is not None:
                result["dividend_rate"] = f"₹{div_rate:.2f} / share"
                if curr_px and curr_px > 0:
                    exact_yield = (div_rate / curr_px) * 100
                    result["dividend_yield"] = f"{exact_yield:.2f}%"
                else:
                    result["dividend_yield"] = f"{info.get('dividendYield', 0):.2f}%"
            elif info.get('dividendYield') is not None:
                dy = info.get('dividendYield')
                result["dividend_yield"] = f"{dy * 100 if dy < 0.20 else dy:.2f}%"
                
            ex_div_ts = info.get('exDividendDate')
            if ex_div_ts and result["ex_dividend_date"] == "N/A":
                try:
                    result["ex_dividend_date"] = datetime.fromtimestamp(ex_div_ts).strftime("%d %b %Y")
                except Exception:
                    pass
        except Exception:
            pass

            
        # 3. Recent Splits
        try:
            splits = t.splits
            if not splits.empty:
                last_split_date = splits.index[-1].strftime("%d %b %Y")
                last_split_ratio = splits.iloc[-1]
                result["last_split"] = f"{last_split_ratio:.0f}:1 Split on {last_split_date}"
        except Exception:
            pass
            
    except Exception as e:
        print(f"Warning: Could not fetch corporate actions for {ticker}: {e}")
        
    return result

if __name__ == "__main__":
    for sym in ["RELIANCE.NS", "BPCL.NS", "TCS.NS"]:
        print(f"\nCorporate Actions for {sym}:")
        actions = get_corporate_actions(sym)
        for k, v in actions.items():
            print(f"  {k}: {v}")
