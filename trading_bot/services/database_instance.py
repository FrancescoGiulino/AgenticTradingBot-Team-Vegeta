import sqlite3
import os
import atexit
import logging
from dotenv import load_dotenv
from trading_bot.utils.singleton import singleton

load_dotenv()

logger = logging.getLogger(__name__)

@singleton
class DbInstance():
    def __init__(self):
        self.db_name = os.getenv("DB_NAME", "trading_agent.db")
        self._conn = None
        self._init_db()
        atexit.register(self.close_connection)

    def get_connection(self):
        """Returns the singleton connection"""
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self.db_name, check_same_thread=False)
                self._conn.row_factory = sqlite3.Row 
            except Exception as e:
                logger.error(f"[DATABASE ERROR] Failed to connect to SQLite: {str(e)}")
                raise e
        return self._conn

    def close_connection(self):
        """Closes the connection"""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("[DATABASE] Singleton connection closed cleanly.")
            except Exception as e:
                logger.error(f"[DATABASE ERROR] Failed to close: {str(e)}")
            finally:
                self._conn = None

    def _init_db(self):
        """Private method for creating tables at the beginning"""
        logger.info(f"[DATABASE] Initializing '{self.db_name}'")
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # TODO check if these market_observation and system_logs is useful
                cursor.execute("""
                    CREATE TABLE market_observations (
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
                    CREATE TABLE trade_journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cycle_id TEXT,
                        agent_type TEXT,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        rationale TEXT,
                        outcome TEXT,
                        data_sources TEXT
                    )
                """)

                # 3. System Logs Table (Error handling, node triggers, and tool failures)
                cursor.execute("""
                    CREATE TABLE system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        agent_node TEXT,
                        event_type TEXT,
                        message TEXT
                    )
                """)   

                conn.commit()
                cursor.close()
            logger.info("[DATABASE] Tables verified/created successfully.")
        except Exception as e:
            logger.error(f"[DATABASE ERROR] Failed to initialize database: {str(e)}")


    def log_trade_journal(self, cycle_id: str, agent_type: str, ticker: str, action: str, 
                          quantity: int, rationale: str, outcome: str, data_sources: str) -> bool:
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trade_journal (cycle_id, agent_type, ticker, action, quantity, rationale, outcome, data_sources)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (cycle_id, agent_type, ticker, action, quantity, rationale, outcome, data_sources))
            return True
        except Exception as e:
            logger.error(f"[DATABASE ERROR] Failed to log trade: {str(e)}")
            return False

    def get_recent_trades(self, limit: int = 50) -> list[dict]:
        """Retrieves paginated trading info"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trade_journal ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[DATABASE ERROR] Failed to fetch trades: {str(e)}")
            return []