import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Assuming Ollama is running and model is loaded
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.predict import predict

# Map common names to tickers (same logic as nl_agent.py)
TICKER_MAP = {
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS"
}

# Define the tool manually for the API to avoid circular imports
from langchain_core.tools import tool

@tool
def get_stock_prediction(company_name: str) -> str:
    """Gets the 5-day stock return prediction for a given company.
    
    Args:
        company_name: The name of the company (e.g. "reliance" or "tcs")
    """
    company_name = company_name.lower().strip()
    
    ticker = TICKER_MAP.get(company_name)
    if not ticker:
        for key, val in TICKER_MAP.items():
            if company_name in key or key in company_name:
                ticker = val
                break
                
    if not ticker:
        return f"Sorry, I don't know the ticker for '{company_name}'. I only track Reliance and TCS right now."
        
    print(f"\n[API calling predict.py for {ticker}]")
    result = predict(ticker, force_refresh=True)
    
    if not result:
        return f"Failed to generate prediction for {ticker}."
        
    return (
        f"As of {result['date']}, {result['ticker']} closed at Rs.{result['latest_price']:.2f}. "
        f"The model predicts a {result['pred_return']*100:+.1f}% return over the next 5 trading days, "
        f"implying a price around Rs.{result['implied_price']:.2f}. "
        f"Remember, this is a model estimate, not financial advice."
    )

app = FastAPI(title="Stock Predictor API")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize LLM Agent
try:
    llm = ChatOllama(model="llama3.1:8b")
    llm_with_tools = llm.bind_tools([get_stock_prediction])
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm_with_tools = None

system_prompt = SystemMessage(content=(
    "You are a stock market prediction assistant. Your only job is to provide stock predictions "
    "using the `get_stock_prediction` tool. Never guess or hallucinate predictions. If the tool "
    "gives you an answer, relay it naturally to the user. Always remind the user that this is a "
    "model estimate, not financial advice."
))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not llm_with_tools:
        raise HTTPException(status_code=500, detail="LLM not initialized properly. Is Ollama running?")
        
    try:
        messages = [
            system_prompt,
            HumanMessage(content=request.message)
        ]
        
        ai_msg = llm_with_tools.invoke(messages)
        
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                if tool_call['name'] == 'get_stock_prediction':
                    tool_result = get_stock_prediction.invoke(tool_call['args'])
                    messages.append({
                        "role": "tool",
                        "name": tool_call['name'],
                        "content": tool_result,
                        "tool_call_id": tool_call['id']
                    })
            
            final_response = llm_with_tools.invoke(messages)
            return ChatResponse(response=final_response.content)
        else:
            return ChatResponse(response=ai_msg.content)
            
    except Exception as e:
        print(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
