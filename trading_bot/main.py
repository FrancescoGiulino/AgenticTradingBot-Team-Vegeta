import time
import logging
from .graph import app
from trading_bot.services.database_instance import DbInstance
from datetime import datetime
from dotenv import load_dotenv
from .rate_limiter import rate_limiter
import threading
from .config import shared_config

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

def user_input_thread():
    """Runs in the background, listening for user commands."""
    while True:
        try:
            new_action = input()
            if new_action.strip():
                shared_config.update_action(new_action.strip())
                print(f"\n✅ [COMMAND RECEIVED] The agent will focus on the following action: '{new_action.strip()}' at the start of the next cycle!\n")
        except EOFError:
            # Ignoriamo il segnale "sporco" inviato dai terminali Windows
            import time
            time.sleep(0.5)
        except Exception:
            break

def main():
    print_header()
    
    db = DbInstance()
    
    # Load Rate Limits Configuration
    rate_limiter.load_config("rate_limits.json")
    
    initial_state = {
        "portfolio": {},
        "user_action": None,
        "target_tickers": [], 
        "proposed_decision": None,
        "is_decision_valid": False,
        "last_n_actions": [],
        "journal": [],
        "error_message": None
    }
    
    current_state = initial_state
    CYCLE_DELAY_SECONDS = 5 
    cycle_count = 1
    
    try:
        # 🟢 Start the background thread for real-time user input
        input_thread = threading.Thread(target=user_input_thread, daemon=True)
        input_thread.start()
        
        logger.info("[SYSTEM] Entering Autonomous Mode. Press Ctrl+C to stop.")
        print("\n" + "="*60)
        print("🤖 AGENT IS RUNNING! Type a new market sector at any time and press ENTER to steer the agent.")
        print("="*60 + "\n")

        while True:
            logger.info(f"STARTING CYCLE {cycle_count} ")
            
            tickers = current_state.get("target_tickers", [])
            if tickers:
                current_ticker = tickers[0]
                logger.info(f"[SYSTEM] Focusing analysis on: {current_ticker}")
            else:
                logger.info("[SYSTEM] Watchlist is empty. Waiting for EXPLORER to generate target tickers...")
            
            # 3. Invoke the LangGraph workflow
            updated_state = app.invoke(current_state)
            
            # 4. Update our main state tracker (NESSUNA ROTAZIONE NECESSARIA)
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
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
