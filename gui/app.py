import streamlit as st

st.set_page_config(page_title="Agentic Trading Bot", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.title("Trading Agent")
    st.markdown("---")
    if st.button("Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
    if st.button("Worker", use_container_width=True):
        st.session_state.page = "Worker"
    if st.button("History", use_container_width=True):
        st.session_state.page = "History"

if st.session_state.page == "Dashboard":
    from components.dashboard import render_dashboard
    render_dashboard()
elif st.session_state.page == "Worker":
    from components.worker import render_worker
    render_worker()
elif st.session_state.page == "History":
    from components.history import render_history
    render_history()
