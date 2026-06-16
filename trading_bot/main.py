import time
import logging
from .graph import app
from .db import init_db
from datetime import datetime
from dotenv import load_dotenv
from .rate_limiter import rate_limiter

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[93m', # Yellow
        'INFO': '\033[94m',    # Blue
        'DEBUG': '\033[90m',   # Grey
        'CRITICAL': '\033[91m',# Red
        'ERROR': '\033[91m',   # Red
    }
    EMOJIS = {
        'WARNING': '⚠️',
        'INFO': 'ℹ️',
        'DEBUG': '🐛',
        'CRITICAL': '🚨',
        'ERROR': '❌',
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        emoji = self.EMOJIS.get(record.levelname, '')
        formatted_msg = super().format(record)
        return f"{color}{emoji} {formatted_msg}{self.RESET}"

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Load API keys from .env just to be safe, though tools.py already does it
load_dotenv()

def print_header():
    logger.info("=" * 60)
    logger.info("STARTING AGENTIC AI TRADING BOT")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Target Ambition: Level 3 (Autonomous Loop)")
    logger.info("=" * 60)

def main():
    print_header()
    
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
        "error_message": None
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
        logger.info("[SYSTEM] Entering Autonomous Mode. Press Ctrl+C to stop.")
        
        while True:
            logger.info(f" STARTING CYCLE {cycle_count} ")
            
            # Since target_tickers is a list, we can rotate through them to analyze a different stock each cycle
            # This makes the demo look much more dynamic!
            current_ticker = current_state["target_tickers"][0]
            logger.info(f"[SYSTEM] Focusing analysis on: {current_ticker}")
            
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
            
            logger.info(f"CYCLE {cycle_count} COMPLETE ")
            logger.info(f"[SYSTEM] Sleeping for {CYCLE_DELAY_SECONDS} seconds before next cycle...")
            
            # 5. Wait before the next loop
            time.sleep(CYCLE_DELAY_SECONDS)
            cycle_count += 1

    except KeyboardInterrupt:
        logger.warning("=" * 60)
        logger.warning("AGENT STOPPED BY USER")
        logger.warning("Finalizing logs and shutting down safely.")
        logger.warning("=" * 60)

if __name__ == "__main__":
    main()
