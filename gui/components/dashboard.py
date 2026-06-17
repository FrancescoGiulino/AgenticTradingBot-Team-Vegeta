import streamlit as st
import pandas as pd
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.utils.db_reader import fetch_portfolio_history, fetch_trade_journal
from trading_bot.tools import get_portfolio_status

def render_dashboard():
    st.header("Dashboard")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("Refresh from Alpaca", help="Fetch live data from Alpaca instead of DB"):
            with st.spinner("Fetching..."):
                portfolio_data = get_portfolio_status.invoke({})
                if "error" not in portfolio_data:
                    st.success("Refreshed!")
                    st.session_state.live_portfolio = portfolio_data
                else:
                    st.error(portfolio_data["error"])

    df_history = fetch_portfolio_history(limit=50)
    
    current_value = 0.0
    current_cash = 0.0
    
    if "live_portfolio" in st.session_state:
        current_value = st.session_state.live_portfolio.get("portfolio_value", 0.0)
        current_cash = st.session_state.live_portfolio.get("cash", 0.0)
    elif not df_history.empty:
        current_value = df_history.iloc[0]["total_value"]
        current_cash = df_history.iloc[0]["cash"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Valore Portfolio", f"${current_value:,.2f}")
    with col2:
        st.metric("Liquidità", f"${current_cash:,.2f}")

    st.markdown("---")
    
    st.subheader("Performance History")
    chart_col1, chart_col2 = st.columns(2)
    
    if not df_history.empty:
        df_plot = df_history.sort_values(by="id", ascending=True).copy()
        
        with chart_col1:
            st.markdown("**Total Value**")
            st.line_chart(df_plot.set_index("timestamp")["total_value"])
            
        with chart_col2:
            st.markdown("**Liquidity (Cash)**")
            st.line_chart(df_plot.set_index("timestamp")["cash"])
    else:
        st.info("No portfolio history found in database. Wait for the bot to run.")

    st.markdown("---")
    
    st.subheader("Recent Trade Events")
    df_trades = fetch_trade_journal(limit=10)
    if not df_trades.empty:
        df_actions = df_trades[df_trades["action"].isin(["BUY", "SELL"])]
        if not df_actions.empty:
            for _, row in df_actions.iterrows():
                color = "green" if row["action"] == "BUY" else "red"
                st.markdown(
                    f"""
                    <div style="padding:10px; border-left: 5px solid {color}; background-color: rgba(255,255,255,0.05); margin-bottom: 10px;">
                        <strong>{row['timestamp']}</strong> | 
                        <span style="color:{color}"><strong>{row['action']}</strong></span> {row['quantity']} shares of <strong>{row['ticker']}</strong> 
                        at ${row['price']} <br>
                        <em>Rationale: {row['rationale']}</em><br>
                        <em>Outcome: {row['outcome']}</em>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.info("No recent BUY/SELL actions. Bot is holding or hasn't traded.")
    else:
        st.info("No trade events yet.")

    st.markdown("---")
    
    st.subheader("Current Configuration")
    config_path = os.path.join(project_root, "configuration.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            st.json(config_data)
    else:
        st.warning("configuration.json not found.")
