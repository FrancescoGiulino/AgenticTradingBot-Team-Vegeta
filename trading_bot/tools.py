import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

# Load environment variables from the .env file
load_dotenv()

# Retrieve Alpaca keys
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

news_client = NewsClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_SECRET_KEY
)
from .rate_limiter import rate_limiter

# Initialize Alpaca Trading Client (paper=True is mandatory for the simulation)
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

@tool
def get_portfolio_status() -> dict:
    """
    Fetches the current portfolio status from Alpaca, including cash balance
    and current open positions. Use this to know how much money is available 
    and if there are positions in loss.
    """
    try:
        rate_limiter.acquire("alpaca_rpm")
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        
        # Format positions into a readable dictionary
        portfolio_positions = {}
        for pos in positions:
            portfolio_positions[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl) # Crucial for the DECISOR logic (sell if in loss)
            }
        
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "positions": portfolio_positions
        }
    except Exception as e:
        # Returning the error as a string allows the LLM to gracefully handle the failure
        return {"error": f"Failed to retrieve portfolio: {str(e)}"}

@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetches the current market price for a given stock ticker using yfinance.
    """
    try:
        rate_limiter.acquire("yfinance_rpm")
        stock = yf.Ticker(ticker)
        # Fetch the latest available price (1 day period)
        todays_data = stock.history(period='1d')
        
        if todays_data.empty:
            return {"error": f"No price data found for {ticker}."}
        
        current_price = todays_data['Close'].iloc[-1]
        return {"ticker": ticker, "price": float(current_price)}
    except Exception as e:
        return {"error": f"Failed to fetch price for {ticker}: {str(e)}"}

@tool
def execute_trade(ticker: str, action: str, quantity: int) -> dict:
    """
    Executes a real market order on Alpaca Paper Trading.
    Expects action to be either 'BUY' or 'SELL'.
    """
    try:
        # Map string action to Alpaca's OrderSide enum
        side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC # Good Till Cancelled
        )
        
        # Submit the order to Alpaca
        rate_limiter.acquire("alpaca_rpm")
        order = trading_client.submit_order(order_data=market_order_data)
        return {"status": "success", "order_id": str(order.id)}
        
    except Exception as e:
        # Graceful failure if the API rejects the order (e.g., market is closed)
        return {"error": f"Order execution failed: {str(e)}"}

@tool
def get_stock_news(ticker: str) -> dict:
    """
    Fetches the latest real financial news headlines for a given ticker 
    using the official Alpaca News API (Benzinga integration).
    """
    try:
        rate_limiter.acquire("yfinance_rpm")
        # Convert hyphen (used for Yahoo) back to the original slash for Alpaca
        clean_ticker = ticker.replace("-", "/")
        
        # Prepare the request for the latest 3 news articles for this ticker
        request_params = NewsRequest(
            symbols=clean_ticker,
            limit=3,
            include_content=False # Fetch only headlines to avoid overwhelming the LLM
        )
        
        # Fetch the actual news using the API
        news_response = news_client.get_news(request_params)
        
        # Extract the list of articles
        news_items = news_response.news
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        if not news_items:
            return {"news": f"No recent news found for {clean_ticker}."}
        
        # Extract only the headlines
        titles = [item.headline for item in news_items]
        combined_news = " | ".join(titles)
        
        return {"news": combined_news}
        
    except Exception as e:
        return {"error": f"Failed to fetch Alpaca news for {ticker}: {str(e)}"}
        return {"error": f"Failed to fetch news for {ticker}: {str(e)}"}
