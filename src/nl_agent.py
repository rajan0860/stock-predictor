import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from src.predict import predict

from src.config import TICKER_MAP, SUPPORTED_STOCKS

@tool
def get_stock_prediction(company_name: str) -> str:
    """Gets the 5-day stock return prediction for a given company.
    
    Args:
        company_name: The name of the company (e.g. "reliance" or "tcs")
    """
    company_name = company_name.lower().strip()
    
    # Try to find the exact ticker, or fall back to string matching
    ticker = TICKER_MAP.get(company_name)
    if not ticker:
        for key, val in TICKER_MAP.items():
            if company_name in key or key in company_name:
                ticker = val
                break
                
    if not ticker:
        supported_names = ", ".join([info["name"] for info in SUPPORTED_STOCKS.values()])
        return f"Sorry, I don't know the ticker for '{company_name}'. I currently track: {supported_names}."
        
    print(f"\n[Tool calling predict.py for {ticker} - fetching live data...]")
    result = predict(ticker, force_refresh=True)
    
    if not result:
        return f"Failed to generate prediction for {ticker}."
        
    return (
        f"As of {result['date']}, {result['ticker']} closed at Rs.{result['latest_price']:.2f}. "
        f"The model predicts a {result['pred_return']*100:+.1f}% return over the next 5 trading days, "
        f"implying a price around Rs.{result['implied_price']:.2f}. "
        f"Remember, this is a model estimate, not financial advice."
    )

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a stock market prediction assistant. Your only job is to provide stock predictions "
    "using the `get_stock_prediction` tool. Never guess or hallucinate predictions. If the tool "
    "gives you an answer, relay it naturally to the user. Always remind the user that this is a "
    "model estimate, not financial advice."
))

def run_agent():
    print("Initializing Ollama Agent...")
    try:
        llm = ChatOllama(model="llama3.1:8b")
        llm_with_tools = llm.bind_tools([get_stock_prediction])
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        print("Please ensure Ollama is running and the model is pulled (`ollama pull llama3.1:8b`).")
        return
    
    print("\n" + "="*50)
    print("Stock Predictor Natural Language Agent")
    supported_names_short = ", ".join([ticker.split('.')[0] for ticker in SUPPORTED_STOCKS.keys()])
    print(f"Ask about {supported_names_short} predictions in plain English.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input.strip():
                continue
                
            messages = [
                SYSTEM_PROMPT,
                HumanMessage(content=user_input)
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
                print(f"\n{final_response.content}")
            else:
                print(f"\n{ai_msg.content}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError processing query: {e}")

if __name__ == "__main__":
    run_agent()
