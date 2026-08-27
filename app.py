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

@st.cache_data(ttl=600)
def fetch_macro_history(ticker_symbol: str, days: int = 90):
    try:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

from src.fii_dii import fetch_fii_dii_live

@st.cache_data(ttl=300)
def get_cached_fii_dii():
    return fetch_fii_dii_live()

# Display Macro Market Pulse at top
macro_data = fetch_macro_summary()
if macro_data:
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'><span style='font-size:1.1rem; font-weight:600; color:#E2E8F0;'>🌍 Global Macro Tape</span><span style='font-size:0.75rem; color:#718096;'>Live Benchmark Rates (5m TTL)</span></div>", unsafe_allow_html=True)
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
        col.caption(f"<span style='color:#718096; font-size:0.72rem;'>{val['desc']}</span>", unsafe_allow_html=True)

    
    with st.expander("📊 Interactive Macro Trends & Comparative Analysis", expanded=False):
        view_mode = st.radio(
            "Analysis View Mode:",
            options=["Single Indicator Deep Dive", "Multi-Asset Relative Performance (% Change)"],
            horizontal=True
        )

        macro_tf_cols = st.columns([1, 1])
        with macro_tf_cols[0]:
            macro_tf = st.radio(
                "Macro Timeframe:",
                options=["1 Month (30D)", "3 Months (90D)", "6 Months", "1 Year"],
                index=1,
                horizontal=True,
                key="macro_tf_radio"
            )
        
        tf_map = {"1 Month (30D)": 45, "3 Months (90D)": 110, "6 Months": 200, "1 Year": 390}
        days_to_fetch = tf_map.get(macro_tf, 110)

        if view_mode == "Single Indicator Deep Dive":
            with macro_tf_cols[1]:
                selected_macro = st.selectbox(
                    "Select Macro Indicator:",
                    options=list(MACRO_INDICATORS.keys()),
                    index=2 # Brent Crude
                )
            
            macro_info = MACRO_INDICATORS[selected_macro]
            m_df = fetch_macro_history(macro_info["ticker"], days=days_to_fetch)
            
            if not m_df.empty:
                m_df['SMA_20'] = m_df['Close'].rolling(20).mean()
                latest_m_val = m_df['Close'].iloc[-1]
                p_high = m_df['High'].max() if 'High' in m_df.columns else m_df['Close'].max()
                p_low = m_df['Low'].min() if 'Low' in m_df.columns else m_df['Close'].min()
                
                # Single Area Chart
                fig_m = go.Figure()
                fig_m.add_trace(
                    go.Scatter(
                        x=m_df.index,
                        y=m_df['Close'],
                        mode='lines',
                        name=selected_macro,
                        line=dict(color='#00E676', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(0, 230, 118, 0.08)'
                    )
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=m_df.index,
                        y=m_df['SMA_20'],
                        mode='lines',
                        name='20-Day SMA',
                        line=dict(color='#FFD700', width=1.5, dash='dot')
                    )
                )
                
                prefix = macro_info.get("prefix", "")
                suffix = macro_info.get("suffix", "")
                unit_label = f" ({prefix}...{suffix})" if prefix or suffix else ""
                
                fig_m.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161B22",
                    height=380,
                    margin=dict(l=10, r=10, t=25, b=10),
                    yaxis=dict(title=f"{selected_macro}{unit_label}", gridcolor="#21262D"),
                    xaxis=dict(gridcolor="#21262D"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_m, width='stretch')
                
                # Quick Stats
                ms1, ms2, ms3 = st.columns(3)
                ms1.metric(label="Period High", value=f"{prefix}{p_high:,.2f}{suffix}")
                ms2.metric(label="Period Low", value=f"{prefix}{p_low:,.2f}{suffix}")
                ms3.metric(label="Latest Level", value=f"{prefix}{latest_m_val:,.2f}{suffix}")
            else:
                st.warning("Could not fetch historical data for this macro indicator.")

        else: # Multi-Asset Relative Performance
            selected_multi = st.multiselect(
                "Select Macro Indicators to Compare:",
                options=list(MACRO_INDICATORS.keys()),
                default=["Brent Crude", "India VIX", "US 10Y Yield", "Dollar Index (DXY)"]
            )
            
            if selected_multi:
                fig_multi = go.Figure()
                palette = ['#00E676', '#FF5252', '#FFD700', '#29B6F6', '#AB47BC', '#FF9800']
                
                for idx, m_name in enumerate(selected_multi):
                    t_sym = MACRO_INDICATORS[m_name]["ticker"]
                    m_df = fetch_macro_history(t_sym, days=days_to_fetch)
                    if not m_df.empty:
                        first_valid = m_df['Close'].dropna().iloc[0]
                        norm_returns = ((m_df['Close'] / first_valid) - 1) * 100
                        fig_multi.add_trace(
                            go.Scatter(
                                x=m_df.index,
                                y=norm_returns,
                                mode='lines',
                                name=m_name,
                                line=dict(color=palette[idx % len(palette)], width=2)
                            )
                        )
                
                fig_multi.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161B22",
                    height=420,
                    margin=dict(l=10, r=10, t=25, b=10),
                    yaxis=dict(title="Normalized % Change", gridcolor="#21262D", ticksuffix="%"),
                    xaxis=dict(gridcolor="#21262D"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_multi, width='stretch')
            else:
                st.info("Select at least one macro indicator above to plot.")

    st.markdown("---")

# Display FII / DII Institutional Flow Banner
fii_dii = get_cached_fii_dii()
if fii_dii:
    st.markdown("### 🏦 Institutional Market Flows (FII / DII)")
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1.5])
    
    f_net = fii_dii.get("fii_net", 0.0)
    d_net = fii_dii.get("dii_net", 0.0)
    t_net = fii_dii.get("total_net", 0.0)
    
    fc1.metric(
        label="FII / FPI Net Flow",
        value=f"{'+' if f_net >= 0 else ''}₹{f_net:,.2f} Cr",
        delta=f"Buy: ₹{fii_dii.get('fii_buy', 0):,.0f} | Sell: ₹{fii_dii.get('fii_sell', 0):,.0f}",
        delta_color="normal" if f_net >= 0 else "inverse"
    )
    fc2.metric(
        label="DII Net Flow",
        value=f"{'+' if d_net >= 0 else ''}₹{d_net:,.2f} Cr",
        delta=f"Buy: ₹{fii_dii.get('dii_buy', 0):,.0f} | Sell: ₹{fii_dii.get('dii_sell', 0):,.0f}",
        delta_color="normal" if d_net >= 0 else "inverse"
    )
    fc3.metric(
        label="Combined Net Flow",
        value=f"{'+' if t_net >= 0 else ''}₹{t_net:,.2f} Cr",
        delta=f"Net Institutional Flow",
        delta_color="normal" if t_net >= 0 else "inverse"
    )
    
    fc4.markdown(f"**Institutional Regime:**<br>{fii_dii.get('sentiment', 'Neutral')}", unsafe_allow_html=True)
    fc4.caption(f"{fii_dii.get('sentiment_desc', '')} <br> *Provisional Data as of: {fii_dii.get('date', 'Recent')}*", unsafe_allow_html=True)
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
        predict_clicked = st.button("Predict 5-Day Return", width='stretch')


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

        latest_price = result['latest_price']
        implied_price = result['implied_price']
        pred_return = result['pred_return']
        pred_alpha = result.get('pred_alpha')
        prob_up = result.get('prob_up')
        q_metrics = result.get('quality_metrics')

        med_err_pct = q_metrics['median_abs_error_pct'] if q_metrics else 2.50
        med_err = med_err_pct / 100.0
        target_low = latest_price * (1.0 + pred_return - med_err)
        target_high = latest_price * (1.0 + pred_return + med_err)
        snr = q_metrics['snr'] if q_metrics else (abs(pred_return) * 100 / med_err_pct)
        
        is_bullish = (prob_up >= 0.5) if prob_up is not None else (pred_return >= 0)
        
        if is_bullish:
            if snr < 0.40:
                dir_text = "🟡 Slight Bullish Bias (Low Conviction)"
                summary_banner = f"💡 **Executive Summary:** Slight bullish bias, but forecast magnitude ({pred_return*100:+.2f}%) is below typical 5-day error (±{med_err_pct:.2f}%) — **treat as watchlist information, not a high-conviction signal.**"
            elif snr < 0.70:
                dir_text = "🟢 Moderate Bullish Bias"
                summary_banner = f"💡 **Executive Summary:** Moderate bullish bias ({pred_return*100:+.2f}%) with positive risk/reward asymmetry — monitor key support levels."
            else:
                dir_text = "🟢 Strong Bullish Signal"
                summary_banner = f"💡 **Executive Summary:** High-conviction bullish forecast ({pred_return*100:+.2f}%) with strong statistical separation from baseline noise."
            prob_label = f"Model prob. of positive 5-day return: {prob_up*100:.1f}%" if prob_up is not None else "Neutral"
        else:
            if snr < 0.40:
                dir_text = "🟡 Slight Bearish Bias (Low Conviction)"
                summary_banner = f"💡 **Executive Summary:** Slight bearish bias, but forecast magnitude ({pred_return*100:+.2f}%) is below typical 5-day error (±{med_err_pct:.2f}%) — **treat as watchlist information, not a high-conviction signal.**"
            elif snr < 0.70:
                dir_text = "🔴 Moderate Bearish Bias"
                summary_banner = f"💡 **Executive Summary:** Moderate bearish pressure ({pred_return*100:+.2f}%) — exercise caution on long entries."
            else:
                dir_text = "🔴 Strong Bearish Signal"
                summary_banner = f"💡 **Executive Summary:** High-conviction bearish signal ({pred_return*100:+.2f}%) with strong downward momentum."
            prob_label = f"Model prob. of negative 5-day return: {(1-prob_up)*100:.1f}%" if prob_up is not None else "Neutral"

        # Executive Summary Alert Banner
        st.info(summary_banner, icon="📌")

        # Metrics Display
        st.markdown("### 🎯 Model Forecast & Target Range")
        m1, m2, m3 = st.columns(3)
        
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
            label="Implied 5-Day Target (Point & Range)", 
            value=f"₹{implied_price:,.2f}",
            delta=f"Range: ₹{target_low:,.1f} – ₹{target_high:,.1f} (±{med_err_pct:.2f}%)",
            delta_color="normal"
        )
        
        # Secondary Metrics Row: Direction & Alpha
        s1, s2, s3 = st.columns(3)
        s1.metric(label="Directional Stance", value=dir_text, delta=prob_label, delta_color="off")
            
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

        # Transparent Forecast-Quality & Empirical Reliability Panel
        if q_metrics:
            st.markdown("#### 🛡️ Forecast Quality & Empirical Reliability")
            st.markdown(
                f"""
                <div style="background: rgba(33, 150, 243, 0.08); border-left: 4px solid #2196F3; padding: 12px 16px; border-radius: 6px; margin-bottom: 15px; font-size: 0.95rem; line-height: 1.5;">
                    🔍 <b>Historical Validation Benchmark:</b><br>
                    In <b>{q_metrics['comparable_samples']} comparable prior signals</b> ({q_metrics['scope']}), 
                    the model was directionally correct <b>{q_metrics['directional_accuracy_pct']:.1f}%</b> of the time, 
                    with a median absolute 5-day error of <b>±{q_metrics['median_abs_error_pct']:.2f}%</b>.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            qc1, qc2, qc3, qc4 = st.columns(4)
            qc1.metric(
                label="Historical Win Rate",
                value=f"{q_metrics['directional_accuracy_pct']:.1f}%",
                delta=f"{q_metrics['comparable_samples']} setups",
                delta_color="off"
            )
            qc2.metric(
                label="Median 5-Day Error",
                value=f"±{q_metrics['median_abs_error_pct']:.2f}%",
                delta=f"Mean: ±{q_metrics['mean_abs_error_pct']:.2f}%",
                delta_color="off"
            )
            
            snr_val = q_metrics['snr']
            qc3.metric(
                label="Signal-to-Noise Ratio (SNR)",
                value=f"{snr_val:.2f}x",
                delta="|Forecast| / Med Error",
                delta_color="off"
            )
            
            qc4.markdown(f"**Economic Conviction:**<br>{q_metrics['significance']}", unsafe_allow_html=True)
            qc4.caption(f"{q_metrics['significance_desc']}<br>Avg Win: +{q_metrics['avg_win_pct']:.2f}% | Avg Loss: -{q_metrics['avg_loss_pct']:.2f}%", unsafe_allow_html=True)
            
            with st.popover("ℹ️ How 'Comparable Prior Signals' are Calculated"):
                st.markdown("""
                **Empirical Similarity & Audit Criteria:**
                1. **Historical Walk-Forward Folds:** Benchmarked across 5,900+ out-of-sample prediction instances using strict expanding window validation with 5-day embargo purging (zero lookahead leakage).
                2. **Directional Polarity:** Filtered for historical setups with the same forecast direction (Bullish / Bearish).
                3. **Confidence Score Binning:** Matches historical predictions within ±8% of current model probability.
                4. **Asset-Calibrated Error:** Benchmarks the forecast magnitude against the stock's actual median 5-day price deviation.
                """)

        # 1. Actionable Decision Card & Trade-Quality Filter
        tech = result.get('technical_levels', {})
        sup_val = tech.get('support_20d', latest_price * 0.96)
        res_val = tech.get('resistance_20d', latest_price * 1.04)
        ma20_val = tech.get('ma_20', latest_price)
        ma50_val = tech.get('ma_50', latest_price)
        rsi_val = tech.get('rsi_14', 50.0)
        vol_ratio = tech.get('vol_vs_ma20', 1.0)
        
        risk_per_share = max(0.01, latest_price - sup_val)
        reward_per_share = max(0.01, implied_price - latest_price) if is_bullish else max(0.01, latest_price - implied_price)
        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 1.0

        st.markdown("#### 📋 Tactical Playbook & Trade Decision Architecture")
        
        trade_status_color = "#FFC107" if snr < 0.40 else ("#4CAF50" if is_bullish else "#F44336")
        trade_status_title = "⏸️ NO TRADE / WATCHLIST ONLY (Low SNR)" if snr < 0.40 else ("🟢 ACTIVE TRADE SETUP (BULLISH)" if is_bullish else "🔴 ACTIVE TRADE SETUP (BEARISH)")
        
        st.markdown(
            f"""
            <div style="background: rgba(30, 34, 45, 0.9); border: 1px solid {trade_status_color}; border-left: 5px solid {trade_status_color}; padding: 14px 18px; border-radius: 8px; margin-bottom: 18px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: {trade_status_color}; margin-bottom: 8px;">{trade_status_title}</div>
                <div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.6;">
                    • <b>Action for 5-Day Swing:</b> {"Avoid chasing for a 5-day swing: expected gain (" + f"{pred_return*100:+.2f}%" + ") is smaller than normal forecast error (±" + f"{med_err_pct:.2f}%" + "). Transaction costs, slippage, and ordinary daily noise could overwhelm the statistical edge." if snr < 0.40 else "Statistically meaningful edge present relative to baseline error. Consider risk-managed positioning."}<br>
                    • <b>Confirmation Rule:</b> Consider action only if price breaks and holds above resistance at <b>₹{res_val:,.2f}</b> on clearly above-average volume (>1.5x 20D average), rather than buying solely on a mild model forecast.<br>
                    • <b>If Already Holding:</b> Treat this as a <b>Hold/Watch</b> signal rather than an aggressive add. Define risk below technical support at <b>₹{sup_val:,.2f}</b> (-{tech.get('dist_support_pct', 0):.1f}%).<br>
                    • <b>Risk-to-Reward Architecture:</b> Entry: <b>₹{latest_price:,.2f}</b> | Stop-Loss / Invalidation: <b>₹{sup_val:,.2f}</b> (-₹{risk_per_share:,.2f}) | Target: <b>₹{implied_price:,.2f}</b> | <b>R:R = 1 : {rr_ratio:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 2. Live Technical Levels Panel
        st.markdown("#### 📐 Live Technical Levels & Regime Analysis")
        tl1, tl2, tl3, tl4 = st.columns(4)
        tl1.metric(
            label="20-Day SMA",
            value=f"₹{ma20_val:,.2f}",
            delta=f"{tech.get('dist_ma20_pct', 0):+.2f}% from price",
            delta_color="normal"
        )
        tl2.metric(
            label="50-Day SMA",
            value=f"₹{ma50_val:,.2f}",
            delta=f"{tech.get('dist_ma50_pct', 0):+.2f}% from price",
            delta_color="normal"
        )
        tl3.metric(
            label="20D Key Support (Low)",
            value=f"₹{sup_val:,.2f}",
            delta=f"-{tech.get('dist_support_pct', 0):.2f}% below",
            delta_color="inverse"
        )
        tl4.metric(
            label="20D Key Resistance (High)",
            value=f"₹{res_val:,.2f}",
            delta=f"+{tech.get('dist_resistance_pct', 0):.2f}% above",
            delta_color="normal"
        )
        
        rsi_state = "Overbought (>70)" if rsi_val >= 70 else ("Oversold (<30)" if rsi_val <= 30 else "Neutral Momentum (30-70)")
        vol_state = "Heavy / Surge (>1.5x)" if vol_ratio >= 1.5 else ("Below Average (<0.8x)" if vol_ratio < 0.8 else "Normal Volume (0.8x - 1.5x)")
        st.caption(f"📊 **Momentum & Volume Regimes:** RSI (14) = **{rsi_val:.1f}** ({rsi_state}) | Volume vs 20D Avg = **{vol_ratio:.2f}x** ({vol_state})")
        st.markdown("---")

        # 3. Sector Intelligence & Macro Sensitivity Overlay
        sector_info = result.get('sector_context', {})
        funds = result.get('fundamentals', {})
        st.markdown(f"#### 🏢 Sector Context: {sector_info.get('sector', 'Equities')}")
        
        sec_col1, sec_col2 = st.columns([1.5, 1])
        with sec_col1:
            st.markdown(
                f"""
                <div style="background: rgba(22, 27, 34, 0.8); border: 1px solid #30363D; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; line-height: 1.5;">
                    🛢️ <b>Crude & Macro Drivers:</b> {sector_info.get('crude_impact', 'N/A')}<br>
                    📈 <b>Primary Growth Catalysts:</b> {sector_info.get('drivers', 'N/A')}<br>
                    🌐 <b>Macro & Flow Sensitivity:</b> {sector_info.get('macro_sens', 'N/A')}
                </div>
                """,
                unsafe_allow_html=True
            )
        with sec_col2:
            rev_str = f"{funds.get('revenue_yoy', 0)*100:+.1f}%" if pd.notna(funds.get('revenue_yoy')) else "N/A"
            margin_str = f"{funds.get('net_margin', 0)*100:.1f}%" if pd.notna(funds.get('net_margin')) else "N/A"
            debt_str = f"{funds.get('debt_to_equity', 0):.2f}" if pd.notna(funds.get('debt_to_equity')) else "N/A"
            st.markdown(
                f"""
                <div style="background: rgba(22, 27, 34, 0.8); border: 1px solid #30363D; padding: 12px 16px; border-radius: 6px; font-size: 0.88rem; line-height: 1.5;">
                    💼 <b>Fundamental Metrics:</b><br>
                    • Revenue YoY: <b>{rev_str}</b><br>
                    • Net Profit Margin: <b>{margin_str}</b><br>
                    • Debt-to-Equity: <b>{debt_str}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("---")

        # 4. Upcoming Corporate Actions & Event Horizon
        corp_act = result.get('corporate_actions', {})
        if corp_act:
            st.markdown("#### 📅 Upcoming Corporate Actions & Event Horizon")
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.metric(
                label="Next Earnings Date",
                value=corp_act.get('earnings_date', 'TBD'),
                delta=f"Est. EPS: {corp_act.get('earnings_est_eps', 'N/A')}",
                delta_color="off"
            )
            ca2.metric(
                label="Dividend Yield",
                value=corp_act.get('dividend_yield', 'N/A'),
                delta=f"Rate: {corp_act.get('dividend_rate', 'N/A')}",
                delta_color="normal"
            )
            ca3.metric(
                label="Ex-Dividend Date",
                value=corp_act.get('ex_dividend_date', 'N/A'),
                delta="Recent / Upcoming",
                delta_color="off"
            )
            ca4.metric(
                label="Stock Splits & Bonuses",
                value=corp_act.get('last_split', 'None'),
                delta="Capital Actions",
                delta_color="off"
            )
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
            
        # 5. Out-of-Sample Historical Prediction Audit Log
        if os.path.exists("models/oof_predictions.parquet"):
            with st.expander(f"📜 View Out-of-Sample Historical Predictions for {ticker} (Last 10 Setups)", expanded=False):
                try:
                    oof_full = pd.read_parquet("models/oof_predictions.parquet")
                    oof_t = oof_full[oof_full['ticker'] == ticker].sort_values('date', ascending=False).head(10).copy()
                    if not oof_t.empty:
                        table_df = pd.DataFrame({
                            'Date': pd.to_datetime(oof_t['date']).dt.strftime('%Y-%m-%d'),
                            'Predicted 5D Return': oof_t['y_pred'].apply(lambda x: f"{x*100:+.2f}%"),
                            'Actual 5D Return': oof_t['y_true'].apply(lambda x: f"{x*100:+.2f}%"),
                            'Absolute Error': oof_t['abs_error'].apply(lambda x: f"{x*100:.2f}%"),
                            'Model Directional Prob': oof_t['prob_up'].apply(lambda x: f"{x*100:.1f}%"),
                            'Outcome': oof_t['correct_dir'].apply(lambda x: "🟢 WIN" if x == 1 else "🔴 LOSS")
                        })
                        st.dataframe(table_df, width='stretch', hide_index=True)
                        st.caption("🔍 *Walk-forward out-of-fold historical records. Predictions were generated strictly on expanding training windows prior to each date without lookahead leakage.*")
                except Exception as e:
                    st.warning(f"Could not load prediction history: {e}")

        # Disclaimer
        st.caption("⚠️ **Disclaimer:** This is a machine learning model estimate, not financial advice. Do not trade based solely on these predictions.")


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
