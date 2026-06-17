import streamlit as st
import os
import sys
import time
from streamlit_autorefresh import st_autorefresh

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.config_agent import process_user_prompt
from gui.utils.db_reader import fetch_trade_journal

def render_worker():
    st.header("Worker Chat & Events")
    
    # Auto-refresh the page every 3 seconds to pull new events and stream
    # Disable autorefresh while we are processing a prompt to prevent interruptions
    is_processing = st.session_state.get("processing_prompt", False)
    if not is_processing:
        st_autorefresh(interval=3000, limit=None, key="worker_autorefresh")
    
    # --- Chat Interface Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Configuration Agent. You can give me instructions to change the trading rules, and I will also stream live trade events here."}
        ]
        
    if "last_event_id" not in st.session_state:
        # Ignore old events on first load by setting last_event_id to the max ID currently in DB
        df_initial = fetch_trade_journal(limit=1)
        if not df_initial.empty:
            st.session_state.last_event_id = int(df_initial.iloc[0]["id"])
        else:
            st.session_state.last_event_id = 0

    # --- Fetch New Events ---
    # We fetch the latest trades and inject them into the chat if they are new.
    df_trades = fetch_trade_journal(limit=20)
    if not df_trades.empty:
        # Sort ascending so the oldest of the recent 20 is processed first
        df_trades_sorted = df_trades.sort_values("id")
        
        for _, row in df_trades_sorted.iterrows():
            event_id = row["id"]
            if event_id > st.session_state.last_event_id:
                st.session_state.last_event_id = event_id
                
                # Omit HOLD or 0 quantity events
                if row["action"] == "HOLD" or row["quantity"] == 0:
                    continue

                # Format the event as a chat message
                color = "green" if row["action"] == "BUY" else "red" if row["action"] == "SELL" else "gray"
                event_msg = (
                    f"**🚨 TRADE EVENT:**\n"
                    f"- **Action:** {row['action']} {row['quantity']} shares of {row['ticker']}\n"
                    f"- **Price:** ${row['price']}\n"
                    f"- **Outcome:** {row['outcome']}"
                )
                st.session_state.messages = st.session_state.messages + [{"role": "assistant", "content": event_msg, "is_event": True}]

    # --- Display Chat History ---
    AVATARS = {
        "INIT_PORTFOLIO": "💼",
        "DECISOR": "🧠",
        "CHECKER": "✅",
        "EXECUTER": "⚡",
        "SUMMARIZER": "📝"
    }

    for message in st.session_state.messages:
        if message.get("is_event"):
            avatar = "📈"
        elif "avatar_type" in message:
            avatar = AVATARS.get(message["avatar_type"], "🤖")
        else:
            avatar = None
            
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # --- Handle Chat Input ---
    if prompt := st.chat_input("Enter configuration instructions (e.g., 'Don't buy tech stocks')..."):
        # Add user message to chat history
        st.session_state.messages = st.session_state.messages + [{"role": "user", "content": prompt}]
        st.session_state.pending_prompt = prompt
        st.session_state.processing_prompt = True
        st.rerun()

    # Process pending prompt if it exists
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pending_prompt
        
        # Display assistant response placeholder
        with st.chat_message("assistant"):
            with st.spinner("Processing configuration update..."):
                success, thoughts = process_user_prompt(prompt)
                
                if thoughts:
                    st.markdown(f"**Agent Thoughts:**\n{thoughts}")
                    st.session_state.messages = st.session_state.messages + [{"role": "assistant", "content": f"**Agent Thoughts:**\n{thoughts}"}]

                if success:
                    response = "I have successfully updated the configuration based on your instructions!"
                else:
                    response = "I'm sorry, I encountered an error while trying to update the configuration."
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages = st.session_state.messages + [{"role": "assistant", "content": response}]
        
        # Clear state and resume autorefresh
        st.session_state.pending_prompt = None
        st.session_state.processing_prompt = False
        st.rerun()

