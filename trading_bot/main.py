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
            new_action = input()
            if new_action.strip():
                shared_config.update_action(new_action.strip())
                print(f"\n✅ [COMMAND RECEIVED] The agent will focus on the following action: '{new_action.strip()}' at the start of the next cycle!\n")
        except EOFError:
            time.sleep(0.5)
        except Exception:
            break

def main():
    print_header()
    
    db = DbInstance()
    
    # Load Rate Limits Configuration
    rate_limiter.load_config("rate_limits.json")
    
    # We now import and run the Textual TUI
    from trading_bot.tui import TradingApp
    TradingApp().run()

if __name__ == "__main__":
    main()
