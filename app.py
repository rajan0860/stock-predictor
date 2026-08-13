import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict

# Configure the Streamlit page
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="centered",
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
        margin-bottom: 40px;
        font-size: 1.1rem;
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
from src.config import DISPLAY_TO_TICKER

# User Input
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_name = st.selectbox(
        "Select a Stock",
        options=list(DISPLAY_TO_TICKER.keys()),
        index=0
    )
    ticker = DISPLAY_TO_TICKER[selected_name]

st.markdown("<br>", unsafe_allow_html=True)

# Main prediction logic
if st.button("Predict 5-Day Return", use_container_width=True):
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
                
                m1.metric(
                    label="Latest Close", 
                    value=f"₹{latest_price:,.2f}"
                )
                m2.metric(
                    label="Predicted Return", 
                    value=f"{pred_return*100:+.2f}%",
                    delta=f"{pred_return*100:+.2f}%",
                    delta_color="normal"
                )
                m3.metric(
                    label="Implied 5-Day Target", 
                    value=f"₹{implied_price:,.2f}",
                    delta=f"₹{implied_price - latest_price:+,.2f}",
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

