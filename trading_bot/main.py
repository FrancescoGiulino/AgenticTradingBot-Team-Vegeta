import time
from .graph import app
from .db import init_db
from datetime import datetime
from dotenv import load_dotenv
from .rate_limiter import rate_limiter

# Load API keys from .env just to be safe, though tools.py already does it
load_dotenv()
from .graph import app

def print_header():
    print("=" * 60)
    print("STARTING AGENTIC AI TRADING BOT")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target Ambition: Level 3 (Autonomous Loop)")
    print("=" * 60)

def main():
    print_header()
    
    # Initialize the SQLite database
    init_db()
    
    # Load Rate Limits Configuration
    rate_limiter.load_config("rate_limits.json")
    
    # 1. Initialize the starting state
    # According to the rules, the agent starts with a simulated portfolio.
    # Alpaca already handles the 100k USD paper money, but we define the tickers we care about.
    initial_state = {
        "portfolio": {}, 
        "target_tickers": ["AAPL", "MSFT", "GOOGL"], # You can change these to any valid US tickers
        "proposed_decision": None,
        "is_decision_valid": False,
        "last_n_actions": [],
        "journal": [],
        "error_message": None,
        "cycle_id": None
    }
    
    # We maintain the 'current_state' outside the loop so memory persists across cycles
    current_state = initial_state
    
    # Set the frequency of market checks.
    # NOTE: Be mindful of API rate limits (especially free tier yfinance or news APIs).
    # 30-60 seconds is usually a safe bet for a live demo.
    CYCLE_DELAY_SECONDS = 5 
    
    cycle_count = 1
    
    try:
        # 2. Start the Autonomous Loop
        print(f"\n[SYSTEM] Entering Autonomous Mode. Press Ctrl+C to stop.\n")
        
        while True:
            print(f"\n--- STARTING CYCLE {cycle_count} ---")
            
            # Since target_tickers is a list, we can rotate through them to analyze a different stock each cycle
            # This makes the demo look much more dynamic!
            current_ticker = current_state["target_tickers"][0]
            print(f"[SYSTEM] Focusing analysis on: {current_ticker}")
            
            # Generate a unique cycle ID for this iteration
            current_state["cycle_id"] = f"cycle-{cycle_count}-{int(time.time())}"
            
            # 3. Invoke the LangGraph workflow
            # We pass the current_state, and the graph returns the updated state after all nodes finish
            updated_state = app.invoke(current_state)
            
            # 4. Prepare state for the next cycle (Rotation logic)
            # Move the ticker we just analyzed to the back of the list
            target_tickers = updated_state.get("target_tickers", ["AAPL"])
            rotated_tickers = target_tickers[1:] + [target_tickers[0]]
            updated_state["target_tickers"] = rotated_tickers
            
            # Update our main state tracker
            current_state = updated_state
            
            print(f"--- CYCLE {cycle_count} COMPLETE ---")
            print(f"[SYSTEM] Sleeping for {CYCLE_DELAY_SECONDS} seconds before next cycle...\n")
            
            # 5. Wait before the next loop
            time.sleep(CYCLE_DELAY_SECONDS)
            cycle_count += 1

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("AGENT STOPPED BY USER")
        print("Finalizing logs and shutting down safely.")
        print("=" * 60)

if __name__ == "__main__":
    main()
