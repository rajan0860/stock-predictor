import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from src.nl_agent import get_stock_prediction, SYSTEM_PROMPT

# Configure the Streamlit page
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 25px;
        font-size: 1.1rem;
    }
    .macro-container {
        background: #1E222D;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 25px;
        border: 1px solid #2A2E39;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1>📈 Stock Predictor Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by LightGBM & yfinance</div>", unsafe_allow_html=True)

# Import mapping for UI display
from src.config import DISPLAY_TO_TICKER, MACRO_INDICATORS

@st.cache_data(ttl=300)
def fetch_macro_summary():
    data = {}
    for name, info in MACRO_INDICATORS.items():
        try:
            t = yf.Ticker(info["ticker"])
            hist = t.history(period="5d")
            if len(hist) >= 2:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                chg = curr - prev
                pct = (chg / prev) * 100
                data[name] = {
                    "price": curr,
                    "delta": f"{chg:+.2f} ({pct:+.2f}%)",
                    "delta_val": chg,
                    "prefix": info.get("prefix", ""),
                    "suffix": info.get("suffix", ""),
                    "desc": info.get("desc", "")
                }
            elif len(hist) == 1:
                curr = hist['Close'].iloc[-1]
                data[name] = {
                    "price": curr,
                    "delta": "0.00%",
                    "delta_val": 0,
                    "prefix": info.get("prefix", ""),
                    "suffix": info.get("suffix", ""),
                    "desc": info.get("desc", "")
                }
        except Exception:
            pass
    return data

# Display Macro Market Pulse at top
macro_data = fetch_macro_summary()
if macro_data:
    st.markdown("### 🌍 Market Pulse & Macro Indicators")
    cols = st.columns(len(macro_data))
    for col, (name, val) in zip(cols, macro_data.items()):
        prefix = val["prefix"]
        suffix = val["suffix"]
        price_str = f"{prefix}{val['price']:,.2f}{(' ' + suffix) if suffix else ''}"
        col.metric(
            label=name,
            value=price_str,
            delta=val["delta"]
        )
        col.caption(f"<span style='color:#718096; font-size:0.75rem;'>{val['desc']}</span>", unsafe_allow_html=True)
    
    with st.expander("📊 View Macro Trends (30-Day History)", expanded=False):
        selected_macro = st.selectbox(
            "Select Macro Indicator to Chart",
            options=list(MACRO_INDICATORS.keys()),
            index=2 # Default to Brent Crude
        )
        macro_ticker = MACRO_INDICATORS[selected_macro]["ticker"]
        try:
            m_end = datetime.datetime.now()
            m_start = m_end - datetime.timedelta(days=45)
            m_hist = yf.download(macro_ticker, start=m_start, end=m_end, progress=False)
            if not m_hist.empty:
                m_chart = pd.DataFrame(m_hist['Close'])
                if isinstance(m_chart.columns, pd.MultiIndex):
                    m_chart.columns = [col[0] for col in m_chart.columns]
                st.line_chart(m_chart, use_container_width=True)
            else:
                st.info("Macro historical data unavailable.")
        except Exception as e:
            st.warning(f"Unable to load macro chart: {e}")

    st.markdown("---")


tab1, tab2 = st.tabs(["Dashboard", "AI Assistant"])

with tab1:
    # User Input
    c_left, col2, c_right = st.columns([1, 2, 1])
    with col2:
        selected_name = st.selectbox(
            "Select a Stock to Predict",
            options=list(DISPLAY_TO_TICKER.keys()),
            index=0
        )
        ticker = DISPLAY_TO_TICKER[selected_name]

        st.markdown("<br>", unsafe_allow_html=True)

        # Main prediction logic
        predict_clicked = st.button("Predict 5-Day Return", use_container_width=True)

    if predict_clicked:
        with st.spinner(f"Fetching live data and running model for {ticker}..."):
            # Wrap predict in try/except in case of issues
            try:
                result = predict(ticker, force_refresh=True)
                
                if result:
                    st.success(f"Prediction generated successfully for {ticker}!")
                    
                    # Metrics Display
                    st.markdown("### Model Forecast")
                    m1, m2, m3 = st.columns(3)
                    
                    # Format metrics
                    latest_price = result['latest_price']
                    implied_price = result['implied_price']
                    pred_return = result['pred_return']
                    
                    delta_price = implied_price - latest_price
                    delta_price_str = f"-₹{abs(delta_price):,.2f}" if delta_price < 0 else f"₹{delta_price:,.2f}"
                    
                    m1.metric(
                        label="Latest Close", 
                        value=f"₹{latest_price:,.2f}"
                    )
                    m2.metric(
                        label="Predicted Return", 
                        value=f"{pred_return*100:+.2f}%",
                        delta=f"{pred_return*100:.2f}%",
                        delta_color="normal"
                    )
                    m3.metric(
                        label="Implied 5-Day Target", 
                        value=f"₹{implied_price:,.2f}",
                        delta=delta_price_str,
                        delta_color="normal"
                    )
                    
                    st.info("📅 **As of:** " + result['date'])
                    
                    st.markdown("---")
                    
                    # Historical Chart
                    st.markdown("### Recent Price History (30 Days)")
                    try:
                        end_date = datetime.datetime.now()
                        start_date = end_date - datetime.timedelta(days=45) # 45 calendar days ~ 30 trading days
                        hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        
                        if not hist_data.empty:
                            # Streamlit line chart requires a flat index and specific columns
                            chart_data = pd.DataFrame(hist_data['Close'])
                            # Handle multi-level columns if yfinance returns them
                            if isinstance(chart_data.columns, pd.MultiIndex):
                                chart_data.columns = [col[0] for col in chart_data.columns]
                                
                            st.line_chart(chart_data, use_container_width=True)
                        else:
                            st.warning("Could not fetch historical data for chart.")
                    except Exception as e:
                        st.warning(f"Could not load chart: {e}")
                        
                    # Disclaimer
                    st.caption("⚠️ **Disclaimer:** This is a machine learning model estimate, not financial advice. Do not trade based solely on these predictions.")
                    
                else:
                    st.error("Model returned None. Have you trained the model by running `python src/train.py`?")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

with tab2:
    st.markdown("### 💬 AI Agent Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "tool":
            pass # Skip showing raw tool outputs
        elif hasattr(msg, "content") and msg.content:
            st.chat_message("assistant").write(msg.content)
            
    if prompt := st.chat_input("Ask about stock predictions... (e.g. 'How does TCS look this week?')"):
        st.chat_message("user").write(prompt)
        
        st.session_state.messages.append(HumanMessage(content=prompt))
        messages = [SYSTEM_PROMPT] + st.session_state.messages
        
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    llm = ChatOllama(model="llama3.1:8b")
                    llm_with_tools = llm.bind_tools([get_stock_prediction])
                    
                    ai_msg = llm_with_tools.invoke(messages)
                    
                    if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                        messages.append(ai_msg)
                        st.session_state.messages.append(ai_msg)
                        
                        for tool_call in ai_msg.tool_calls:
                            if tool_call['name'] == 'get_stock_prediction':
                                tool_result = get_stock_prediction.invoke(tool_call['args'])
                                tool_message = {
                                    "role": "tool",
                                    "name": tool_call['name'],
                                    "content": tool_result,
                                    "tool_call_id": tool_call['id']
                                }
                                messages.append(tool_message)
                                st.session_state.messages.append(tool_message)
                        
                        final_response = llm_with_tools.invoke(messages)
                        st.write(final_response.content)
                        st.session_state.messages.append(final_response)
                    else:
                        st.write(ai_msg.content)
                        st.session_state.messages.append(ai_msg)
                        
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}")
