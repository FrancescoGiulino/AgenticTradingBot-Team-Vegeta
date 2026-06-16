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
    
    # Initialize Database
    init_db()
    
    # Load Rate Limits Configuration
    rate_limiter.load_config("rate_limits.json")
    
    # Load Initial Configuration
    import json
    import os
    from .db import log_portfolio_history
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configuration.json")
    user_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)
            
    # 1. Initialize the starting state
    initial_state = {
        "portfolio": {}, 
        "target_tickers": user_config.get("target_tickers", ["AAPL", "MSFT", "GOOGL"]),
        "proposed_decision": None,
        "is_decision_valid": False,
        "last_n_actions": [],
        "journal": [],
        "error_message": None,
        "cycle_id": None
    }
    
    current_state = initial_state
    
    CYCLE_DELAY_SECONDS = 5 
    cycle_count = 1
    
    try:
        logger.info("[SYSTEM] Entering Autonomous Mode. Press Ctrl+C to stop.")
        
        while True:
            logger.info(f"STARTING CYCLE {cycle_count} ")
            
            # Refresh config dynamically each loop
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    
            # Sync target_tickers with config, adding new ones if they aren't there
            current_tickers = current_state.get("target_tickers", [])
            config_tickers = user_config.get("target_tickers", ["AAPL", "MSFT", "GOOGL"])
            # Ensure we only track tickers in config
            current_tickers = [t for t in current_tickers if t in config_tickers]
            for t in config_tickers:
                if t not in current_tickers:
                    current_tickers.append(t)
                    
            if not current_tickers:
                current_tickers = ["AAPL"]
                
            current_state["target_tickers"] = current_tickers
            current_ticker = current_state["target_tickers"][0]
            
            # Generate a unique cycle ID
            current_state["cycle_id"] = f"cycle-{cycle_count}-{int(time.time())}"
            
            logger.info(f"[SYSTEM] Focusing analysis on: {current_ticker}")
            
            updated_state = app.invoke(current_state)
            
            target_tickers = updated_state.get("target_tickers", ["AAPL"])
            if len(target_tickers) > 1:
                rotated_tickers = target_tickers[1:] + [target_tickers[0]]
            else:
                rotated_tickers = target_tickers
                
            updated_state["target_tickers"] = rotated_tickers
            current_state = updated_state
            
            # Log Portfolio History
            portfolio = current_state.get("portfolio", {})
            total_value = portfolio.get("portfolio_value", 0.0)
            cash = portfolio.get("cash", 0.0)
            if total_value > 0 or cash > 0:
                log_portfolio_history(current_state["cycle_id"], total_value, cash)
            
            logger.info(f"CYCLE {cycle_count} COMPLETE ")
            logger.info(f"[SYSTEM] Sleeping for {CYCLE_DELAY_SECONDS} seconds before next cycle...")
            
            time.sleep(CYCLE_DELAY_SECONDS)
            cycle_count += 1

    except KeyboardInterrupt:
        logger.warning("=" * 60)
        logger.warning("AGENT STOPPED BY USER")
        logger.warning("Finalizing logs and shutting down safely.")
        logger.warning("=" * 60)
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
