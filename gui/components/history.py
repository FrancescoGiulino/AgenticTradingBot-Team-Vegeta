import streamlit as st
from gui.utils.db_reader import fetch_trade_journal, fetch_market_observations, fetch_system_logs

def render_history():
    st.header("History & Logs")
    
    st.markdown("Select a database table to view:")
    
    col1, col2, col3 = st.columns(3)
    
    if "selected_table" not in st.session_state:
        st.session_state.selected_table = "Trade Journal"
        
    with col1:
        if st.button("Trade Journal", use_container_width=True):
            st.session_state.selected_table = "Trade Journal"
    with col2:
        if st.button("Market Observations", use_container_width=True):
            st.session_state.selected_table = "Market Observations"
    with col3:
        if st.button("System Logs", use_container_width=True):
            st.session_state.selected_table = "System Logs"
            
    st.markdown("---")
    
    st.subheader(st.session_state.selected_table)
    
    if st.session_state.selected_table == "Trade Journal":
        df = fetch_trade_journal(limit=100)
    elif st.session_state.selected_table == "Market Observations":
        df = fetch_market_observations(limit=100)
    else:
        df = fetch_system_logs(limit=100)
        
    if not df.empty:
        # Provide built-in filtering by displaying dataframe with interactive toggles
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data found in this table.")
