import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

@st.cache_data(ttl=120)
def get_cached_prediction(stock_ticker: str):
    return predict(stock_ticker, force_refresh=True)

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

        # Initialize session state for active prediction
        if "prediction_result" not in st.session_state:
            st.session_state.prediction_result = None
        if "prediction_ticker" not in st.session_state:
            st.session_state.prediction_ticker = None

        # Main prediction logic
        predict_clicked = st.button("Predict 5-Day Return", use_container_width=True)

    if predict_clicked:
        with st.spinner(f"Fetching live data and running model for {ticker}..."):
            try:
                res = get_cached_prediction(ticker)
                if res:
                    st.session_state.prediction_result = res
                    st.session_state.prediction_ticker = ticker
                else:
                    st.session_state.prediction_result = None
                    st.error("Model returned None. Have you trained the model by running `python src/train.py`?")
            except Exception as e:
                st.session_state.prediction_result = None
                st.error(f"An error occurred during prediction: {e}")

    # Render persisted prediction and chart if available for selected ticker
    if st.session_state.prediction_result and st.session_state.prediction_ticker == ticker:
        result = st.session_state.prediction_result
        st.success(f"Prediction generated successfully for {ticker}!")


                    
                    # Metrics Display
                    st.markdown("### 🎯 Model Forecast & Signal Analysis")
                    m1, m2, m3 = st.columns(3)
                    
                    # Format metrics
                    latest_price = result['latest_price']
                    implied_price = result['implied_price']
                    pred_return = result['pred_return']
                    pred_alpha = result.get('pred_alpha')
                    prob_up = result.get('prob_up')
                    
                    delta_price = implied_price - latest_price
                    delta_price_str = f"-₹{abs(delta_price):,.2f}" if delta_price < 0 else f"₹{delta_price:,.2f}"
                    
                    m1.metric(
                        label="Latest Close", 
                        value=f"₹{latest_price:,.2f}"
                    )
                    m2.metric(
                        label="Predicted 5-Day Return", 
                        value=f"{pred_return*100:+.2f}%",
                        delta=f"{pred_return*100:.2f}%",
                        delta_color="normal"
                    )
                    m3.metric(
                        label="Implied Target Price", 
                        value=f"₹{implied_price:,.2f}",
                        delta=delta_price_str,
                        delta_color="normal"
                    )
                    
                    # Secondary Metrics Row: Direction & Alpha
                    s1, s2, s3 = st.columns(3)
                    if prob_up is not None:
                        is_bullish = prob_up >= 0.5
                        dir_text = "🟢 Bullish (UP)" if is_bullish else "🔴 Bearish (DOWN)"
                        conf_str = f"Confidence: {prob_up*100:.1f}%" if is_bullish else f"Confidence: {(1-prob_up)*100:.1f}%"
                        s1.metric(label="Directional Signal", value=dir_text, delta=conf_str, delta_color="off")
                    else:
                        s1.metric(label="Directional Signal", value="Neutral")
                        
                    if pred_alpha is not None:
                        s2.metric(
                            label="Expected Alpha vs Nifty 50",
                            value=f"{pred_alpha*100:+.2f}%",
                            delta=f"{pred_alpha*100:+.2f}% vs Nifty",
                            delta_color="normal"
                        )
                    else:
                        s2.metric(label="Expected Alpha", value="N/A")
                        
                    s3.metric(label="Data As Of", value=result['date'])
                    
                    st.markdown("---")
                    
                    # Interactive Candlestick & Volume Chart
                    st.markdown("### 📊 Interactive Technical Price Chart")
                    
                    # Timeframe selection
                    tf_col1, tf_col2 = st.columns([1, 1])
                    with tf_col1:
                        timeframe = st.radio(
                            "Select Timeframe:",
                            options=["1 Month (30D)", "3 Months (90D)", "6 Months", "1 Year"],
                            index=1,
                            horizontal=True
                        )
                    
                    tf_days_map = {
                        "1 Month (30D)": 45,
                        "3 Months (90D)": 110,
                        "6 Months": 200,
                        "1 Year": 390
                    }
                    lookback_days = tf_days_map.get(timeframe, 110)
                    
                    try:
                        end_date = datetime.datetime.now()
                        start_date = end_date - datetime.timedelta(days=lookback_days)
                        hist_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                        
                        if not hist_data.empty:
                            if isinstance(hist_data.columns, pd.MultiIndex):
                                hist_data.columns = [col[0] for col in hist_data.columns]
                                
                            hist_data['SMA_20'] = hist_data['Close'].rolling(20).mean()
                            hist_data['SMA_50'] = hist_data['Close'].rolling(50).mean()
                            
                            # Create Subplots: Candlestick (Top) & Volume (Bottom)
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.04,
                                row_heights=[0.72, 0.28],
                                subplot_titles=(f"{ticker} Price & Moving Averages", "Daily Trading Volume")
                            )
                            
                            # 1. Candlestick
                            fig.add_trace(
                                go.Candlestick(
                                    x=hist_data.index,
                                    open=hist_data['Open'],
                                    high=hist_data['High'],
                                    low=hist_data['Low'],
                                    close=hist_data['Close'],
                                    name="Price",
                                    increasing_line_color='#26a69a',
                                    decreasing_line_color='#ef5350'
                                ),
                                row=1, col=1
                            )
                            
                            # 2. Moving Average Overlays
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data['SMA_20'],
                                    name="20-Day SMA",
                                    line=dict(color="#FFD700", width=1.5)
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=hist_data.index,
                                    y=hist_data['SMA_50'],
                                    name="50-Day SMA",
                                    line=dict(color="#29B6F6", width=1.5)
                                ),
                                row=1, col=1
                            )
                            
                            # 3. 5-Day Model Target Forward Line
                            last_date = hist_data.index[-1]
                            target_date = last_date + datetime.timedelta(days=7) # ~5 business days
                            fig.add_trace(
                                go.Scatter(
                                    x=[last_date, target_date],
                                    y=[latest_price, implied_price],
                                    mode="lines+markers",
                                    name="5-Day Target Forecast",
                                    line=dict(color="#AB47BC", width=2.5, dash="dash"),
                                    marker=dict(size=8, symbol="star")
                                ),
                                row=1, col=1
                            )
                            
                            # 4. Volume Bar Chart
                            vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(hist_data['Close'], hist_data['Open'])]
                            fig.add_trace(
                                go.Bar(
                                    x=hist_data.index,
                                    y=hist_data['Volume'],
                                    name="Volume",
                                    marker_color=vol_colors,
                                    showlegend=False
                                ),
                                row=2, col=1
                            )
                            
                            # Layout styling
                            fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="#0E1117",
                                plot_bgcolor="#161B22",
                                xaxis_rangeslider_visible=False,
                                height=550,
                                margin=dict(l=10, r=10, t=30, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                hovermode="x unified"
                            )
                            fig.update_yaxes(title_text="Price (₹)", row=1, col=1, gridcolor="#21262D")
                            fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#21262D")
                            fig.update_xaxes(gridcolor="#21262D")
                            
                            st.plotly_chart(fig, width='stretch')
                            
                            # Quick Stats summary below chart
                            p_high = hist_data['High'].max()
                            p_low = hist_data['Low'].min()
                            avg_vol = hist_data['Volume'].mean()
                            sma20_last = hist_data['SMA_20'].iloc[-1]
                            dist_sma20 = ((latest_price / sma20_last) - 1) * 100 if pd.notna(sma20_last) else 0
                            
                            qs1, qs2, qs3, qs4 = st.columns(4)
                            qs1.metric(label="Period High", value=f"₹{p_high:,.2f}")
                            qs2.metric(label="Period Low", value=f"₹{p_low:,.2f}")
                            qs3.metric(label="Avg Daily Volume", value=f"{avg_vol:,.0f}")
                            qs4.metric(label="vs 20-Day SMA", value=f"{dist_sma20:+.2f}%", delta=f"{dist_sma20:.2f}%", delta_color="normal")
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
