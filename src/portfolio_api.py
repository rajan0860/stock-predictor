import os
import json
import uuid
import datetime
from typing import List, Literal, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel
from src.predict import batch_predict

router = APIRouter()

class PortfolioRequest(BaseModel):
    tickers: List[str]
    allocation: Literal["equal", "mean_variance", "risk_parity"] = "equal"

class TickerPrediction(BaseModel):
    ticker: str
    latest_price: float
    pred_return: float
    implied_price: float
    prob_up: Optional[float] = None
    volatility: Optional[float] = None

class PortfolioResponse(BaseModel):
    predictions: List[TickerPrediction]
    allocation: Dict[str, float]

def compute_allocation(tickers: List[str], method: str) -> Dict[str, float]:
    if method == "equal":
        weight = 1.0 / len(tickers) if tickers else 0.0
        return {t: weight for t in tickers}
    # placeholders for future methods
    return {t: 1.0 / len(tickers) for t in tickers}

@router.post("/portfolio", response_model=PortfolioResponse)
def portfolio(req: PortfolioRequest):
    batch = batch_predict(req.tickers)
    predictions = []
    for t in req.tickers:
        res = batch.get(t)
        if isinstance(res, dict):
            predictions.append(TickerPrediction(
                ticker=t,
                latest_price=res.get('latest_price'),
                pred_return=res.get('pred_return'),
                implied_price=res.get('implied_price'),
                prob_up=res.get('prob_up'),
                volatility=None
            ))
    allocation_map = compute_allocation(req.tickers, req.allocation)
    return PortfolioResponse(predictions=predictions, allocation=allocation_map)

PORTFOLIO_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_portfolios.json"))

class PortfolioSaveRequest(PortfolioResponse):
    name: Optional[str] = None

class PortfolioSaveResponse(BaseModel):
    id: str
    message: str

@router.post("/portfolio/save", response_model=PortfolioSaveResponse)
def save_portfolio(payload: PortfolioSaveRequest):
    if os.path.exists(PORTFOLIO_DB):
        try:
            with open(PORTFOLIO_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    else:
        data = []
    entry_id = str(uuid.uuid4())
    entry = {
        "id": entry_id,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "name": payload.name or f"Portfolio {entry_id[:8]}",
        "tickers": payload.tickers,
        "allocation": payload.allocation,
        "predictions": [p.dict() for p in payload.predictions]
    }
    data.append(entry)
    with open(PORTFOLIO_DB, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return PortfolioSaveResponse(id=entry_id, message="Portfolio saved successfully.")
