import time
import logging
from .graph import app
from .db import init_db
from datetime import datetime
from dotenv import load_dotenv
from .rate_limiter import rate_limiter
# import threading
from .config import shared_config
import json
import os
from .db import log_portfolio_history

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

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s'))
logging.basicConfig(level=logging.INFO, handlers=[handler])

load_dotenv()

def print_header():
    logger.info("=" * 60)
    logger.info("STARTING AGENTIC AI TRADING BOT")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Target Ambition: Level 3 (Autonomous Loop)")
    logger.info("=" * 60)

def user_input_thread():
    """Runs in the background, listening for user commands."""
    while True:
        try:
            new_focus = input()
            if new_focus.strip():
                shared_config.update_focus(new_focus.strip())
                print(f"\n✅ [COMMAND RECEIVED] Market focus will change to '{new_focus.strip()}' at the start of the next cycle!\n")
        except EOFError:
            time.sleep(0.5)
        except Exception:
            break

def main():
    print_header()
    init_db()
    
    rate_limiter.load_config("rate_limits.json")

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configuration.json")
    
    initial_state = {
        "portfolio": {},
        "market_focus": "Innovative Tech and EV",
        "target_tickers": [], 
        "analyzed_tickers": [],
        "cycle_count": 1,
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
        # 🟢 Start the background thread for real-time user input
        # input_thread = threading.Thread(target=user_input_thread, daemon=True)
        # input_thread.start()
        
        logger.info("[SYSTEM] Entering Autonomous Mode. Press Ctrl+C to stop.")
        print("\n" + "="*60)
        print("🤖 AGENT IS RUNNING! Type a new market sector at any time and press ENTER to steer the agent.")
        print("="*60 + "\n")

        while True:
            logger.info(f"STARTING CYCLE {cycle_count} ")
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        user_config = json.load(f)
                        preferred_sectors = user_config.get("preferred_sectors", [])
                        if preferred_sectors:
                            new_focus = preferred_sectors[0]
                            if new_focus != shared_config.market_focus:
                                logger.info(f"[GUI SYNC] Syncing focus to: {new_focus}")
                                shared_config.update_focus(new_focus)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to read configuration.json: {e}")

            current_state["cycle_id"] = f"cycle-{cycle_count}-{int(time.time())}"
            current_state["cycle_count"] = cycle_count

            tickers = current_state.get("target_tickers", [])
            if tickers:
                current_ticker = tickers[0]
                logger.info(f"[SYSTEM] Focusing analysis on: {current_ticker}")
            else:
                logger.info("[SYSTEM] Watchlist is empty. Waiting for EXPLORER to generate target tickers...")
            
            updated_state = app.invoke(current_state)
            
            current_state = updated_state
            
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
