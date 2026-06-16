import sqlite3
import os
import sys
import atexit
import logging

logger = logging.getLogger(__name__)

# Name of the database file
DB_NAME = "trading_agent.db"

# Singleton connection object
_conn = None

def get_connection():
    """
    Returns the singleton SQLite database connection.
    Initializes it if it doesn't exist yet.
    """
    global _conn
    if _conn is None:
        try:
            # check_same_thread=False allows sharing the connection across threads if needed
            _conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        except Exception as e:
            logger.error(f"[DATABASE ERROR] Failed to connect to SQLite: {str(e)}")
            raise e
    return _conn

@atexit.register
def close_connection():
    """
    Closes the singleton database connection cleanly upon program termination.
    Registered via atexit.
    """
    global _conn
    if _conn is not None:
        try:
            _conn.close()
            logger.info("[DATABASE] Singleton connection closed cleanly.")
        except Exception as e:
            logger.error(f"[DATABASE ERROR] Failed to close connection cleanly: {str(e)}")
        finally:
            _conn = None

def init_db():
    """
    Initializes the SQLite database, creating the necessary tables if they do not exist.
    This operation is wrapped in a try-except block to prevent crashes.
    """
    logger.info(f"[DATABASE] Initializing database '{DB_NAME}'")
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # 1. Market Observations Table (Researcher/Analyst Agent logs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cycle_id TEXT,
                ticker TEXT NOT NULL,
                current_price REAL,
                news_summary TEXT,
                sentiment_score TEXT
            )
        """)

        # 2. Trade Journal Table (Decision Maker Agent / Hackathon requirement)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cycle_id TEXT,
                agent_type TEXT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER,
                price REAL,
                news_summary TEXT,
                rationale TEXT,
                outcome TEXT,
                data_sources TEXT
            )
        """)

        # 3. System Logs Table (Error handling, node triggers, and tool failures)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                agent_node TEXT,
                event_type TEXT,
                message TEXT
            )
        """)

        # 4. Portfolio History Table (For Dashboard Charts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                cycle_id TEXT,
                total_value REAL,
                cash REAL
            )
        """)

        conn.commit()
        cursor.close()
        logger.info("[DATABASE] Tables verified/created successfully.")
    except Exception as e:
        logger.error(f"[DATABASE ERROR] Failed to initialize database: {str(e)}")


def log_market_observation(cycle_id: str, ticker: str, current_price: float, news_summary: str, sentiment_score: str) -> bool:
    """
    Logs a market observation.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO market_observations (cycle_id, ticker, current_price, news_summary, sentiment_score)
            VALUES (?, ?, ?, ?, ?)
        """, (cycle_id, ticker, current_price, news_summary, sentiment_score))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"[DATABASE ERROR] Failed to log market observation: {str(e)}")
        return False


def log_trade_journal(cycle_id: str, agent_type: str, ticker: str, action: str, quantity: int, 
                      price: float, news_summary: str, rationale: str, outcome: str, data_sources: str) -> bool:
    """
    Logs a decision and execution outcome to the Trade Journal.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trade_journal (cycle_id, agent_type, ticker, action, quantity, price, news_summary, rationale, outcome, data_sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cycle_id, agent_type, ticker, action, quantity, price, news_summary, rationale, outcome, data_sources))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"[DATABASE ERROR] Failed to log trade journal: {str(e)}")
        return False


def log_system_event(agent_node: str, event_type: str, message: str) -> bool:
    """
    Logs a system level event, failure, or recovery attempt.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_logs (agent_node, event_type, message)
            VALUES (?, ?, ?)
        """, (agent_node, event_type, message))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"[DATABASE ERROR] Failed to log system event: {str(e)}")
        return False

def log_portfolio_history(cycle_id: str, total_value: float, cash: float) -> bool:
    """
    Logs the portfolio value and cash balance.
    Returns True if successful, False otherwise.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO portfolio_history (cycle_id, total_value, cash)
            VALUES (?, ?, ?)
        """, (cycle_id, total_value, cash))
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"[DATABASE ERROR] Failed to log portfolio history: {str(e)}")
        return False
