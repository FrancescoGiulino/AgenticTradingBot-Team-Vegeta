from trading_bot.utils.singleton import singleton
from pydantic import BaseModel
from alpaca.trading.client import TradingClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient

import logging
import os

logger = logging.getLogger(__name__)

@singleton
class AlpacaService:
    def __init__(self):
        logger.info("[ALPACA] Initializing Singleton Trading Client...")
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        is_paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"
        
        
        if not api_key or not secret_key:
            raise ValueError("CRITICAL: Alpaca API keys missing in .env file!")
            
        self._client = TradingClient(api_key, secret_key, paper=is_paper)
        self._news_client = NewsClient(api_key, secret_key)
        self._historical_client = StockHistoricalDataClient(api_key, secret_key)
        
        # TODO add ratelimiter here?

    @property
    def client(self) -> TradingClient:
        """Returns the alpaca client instance"""
        return self._client
    
    @property
    def news_client(self) -> NewsClient:
        """Returns the alpaca news instance"""
        return self._news_client
    
    @property
    def historical_client(self) -> StockHistoricalDataClient:
        """Returns the alpaca news instance"""
        return self._historical_client