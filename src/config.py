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
