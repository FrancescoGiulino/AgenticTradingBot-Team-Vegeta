import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trading_agent.db")

def get_db_connection():
    # Use a short timeout and connect cleanly for reading
    return sqlite3.connect(DB_PATH, timeout=5.0)

def fetch_table_data(table_name: str, limit: int = 100) -> pd.DataFrame:
    try:
        conn = get_db_connection()
        query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

def fetch_portfolio_history(limit: int = 100) -> pd.DataFrame:
    return fetch_table_data("portfolio_history", limit)

def fetch_trade_journal(limit: int = 50) -> pd.DataFrame:
    return fetch_table_data("trade_journal", limit)

def fetch_market_observations(limit: int = 50) -> pd.DataFrame:
    return fetch_table_data("market_observations", limit)

def fetch_system_logs(limit: int = 50) -> pd.DataFrame:
    return fetch_table_data("system_logs", limit)
