SUPPORTED_STOCKS = {
    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "aliases": ["reliance", "reliance industries"]
    },
    "TCS.NS": {
        "name": "Tata Consultancy Services (TCS)",
        "aliases": ["tcs", "tata consultancy", "tata consultancy services"]
    },
    "HDFCBANK.NS": {
        "name": "HDFC Bank",
        "aliases": ["hdfc", "hdfc bank"]
    },
    "INFY.NS": {
        "name": "Infosys",
        "aliases": ["infy", "infosys"]
    },
    "HINDPETRO.NS": {
        "name": "Hindustan Petroleum",
        "aliases": ["hpcl", "hindustan petroleum", "hindpetro"]
    }
}

TICKERS = list(SUPPORTED_STOCKS.keys())

TICKER_MAP = {}
for ticker, info in SUPPORTED_STOCKS.items():
    for alias in info["aliases"]:
        TICKER_MAP[alias] = ticker

DISPLAY_TO_TICKER = {info["name"]: ticker for ticker, info in SUPPORTED_STOCKS.items()}

MACRO_INDICATORS = {
    "Nifty 50": {"ticker": "^NSEI", "prefix": "", "suffix": "", "desc": "Benchmark Index"},
    "India VIX": {"ticker": "^INDIAVIX", "prefix": "", "suffix": "", "desc": "Market Volatility"},
    "Brent Crude": {"ticker": "BZ=F", "prefix": "$", "suffix": "/bbl", "desc": "Global Energy Cost"},
    "Dollar Index (DXY)": {"ticker": "DX-Y.NYB", "prefix": "", "suffix": "", "desc": "USD Global Strength"},
    "USD / INR": {"ticker": "USDINR=X", "prefix": "₹", "suffix": "", "desc": "Forex Rate"},
}

