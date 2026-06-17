import os
import yfinance as yf
import logging
from dotenv import load_dotenv
from langchain_core.tools import tool
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from .rate_limiter import rate_limiter
import urllib.request
from html.parser import HTMLParser
import re
from ddgs import DDGS

logger = logging.getLogger(__name__)

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

news_client = NewsClient(
    api_key=ALPACA_API_KEY, 
    secret_key=ALPACA_SECRET_KEY
)

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
        
        portfolio_positions = {}
        for pos in positions:
            portfolio_positions[pos.symbol] = {
                "qty": float(pos.qty),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl) 
            }
        
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "positions": portfolio_positions
        }
    except Exception as e:
        return {"error": f"Failed to retrieve portfolio: {str(e)}"}

@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetches the current market price for a given stock ticker using yfinance.
    """
    try:
        rate_limiter.acquire("yfinance_rpm")
        stock = yf.Ticker(ticker)
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
        side = OrderSide.BUY if action.upper() == "BUY" else OrderSide.SELL
        
        market_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=side,
            time_in_force=TimeInForce.GTC
        )
        
        rate_limiter.acquire("alpaca_rpm")
        order = trading_client.submit_order(order_data=market_order_data)
        return {"status": "success", "order_id": str(order.id)}
        
    except Exception as e:
        return {"error": f"Order execution failed: {str(e)}"}

@tool
def get_stock_news(ticker: str) -> str:
    """
    Fetches the latest real financial news headlines for a given ticker 
    using the official Alpaca News API.
    Returns a clean string formatted specifically for the LLM.
    """
    try:
        rate_limiter.acquire("yfinance_rpm")
        clean_ticker = ticker.replace("-", "/")
        
        request_params = NewsRequest(
            symbols=clean_ticker,
            limit=3,
            include_content=False 
        )
        
        news_response = news_client.get_news(request_params)
        
        news_items = []
        
        if isinstance(news_response, list):
            news_items = news_response
        elif hasattr(news_response, 'news'):
            news_items = news_response.news
        elif hasattr(news_response, 'articles'):
            news_items = news_response.articles
        else:
            try:
                resp_dict = dict(news_response)
                news_items = resp_dict.get('news', resp_dict.get('articles', []))
            except Exception:
                pass

        if not news_items or not isinstance(news_items, list) or len(news_items) == 0:
            return f"[NESSUNA NOTIZIA] Nessun evento rilevante recente per {clean_ticker}."
        
        titles = []
        for item in news_items:
            if hasattr(item, 'headline'):
                titles.append(f"- {item.headline}")
            elif isinstance(item, dict) and 'headline' in item:
                titles.append(f"- {item['headline']}")
                
        if not titles:
            return f"[NESSUNA NOTIZIA] Impossibile decodificare il formato delle notizie per {clean_ticker}."
            
        combined_news = "\n".join(titles)
        return f"[ULTIME NOTIZIE {clean_ticker}]:\n{combined_news}"
        
    except Exception as e:
        logger.error(f"[TOOL ERROR] Alpaca News API failed per {ticker}: {str(e)}")
        return f"[NESSUNA NOTIZIA] Errore tecnico nel fetch. Considera l'ambiente informativo NEUTRO. Procedi analizzando i dati di Prezzo e Portfolio."

class _SimpleTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.ignore_tags = {'script', 'style', 'noscript', 'meta', 'head', 'title'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.ignore_tags:
            cleaned = data.strip()
            if cleaned:
                self.text.append(cleaned)

@tool
def web_search(query: str) -> str:
    """
    Performs a web search using DuckDuckGo to find recent news.
    It attempts to scrape the text of the first couple of sites to provide rich context.
    Useful for researching companies, finding stock tickers, or getting the latest market context.
    """
    try:
        with DDGS() as ddgs:
            # Try news first
            try:
                results = list(ddgs.news(query, max_results=3))
            except Exception as e:
                logger.error(f"[web_search] News error: {e}")
                results = []
                
            if not results:
                logger.info("[web_search] Falling back to text search")
                results = list(ddgs.text(query, max_results=3))
                
            if not results:
                return "No results found."
            
            formatted_results = []
            for r in results:
                title = r.get('title', '')
                snippet = r.get('body', r.get('snippet', ''))
                url = r.get('url', r.get('href', ''))
                
                content = snippet
                if url:
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                        with urllib.request.urlopen(req, timeout=5) as response:
                            html = response.read().decode('utf-8', errors='ignore')
                            extractor = _SimpleTextExtractor()
                            extractor.feed(html)
                            full_text = ' '.join(extractor.text)
                            full_text = re.sub(r'\s+', ' ', full_text).strip()
                            if full_text:
                                content += f"\nScraped Text: {full_text[:2000]}..."
                    except Exception as e:
                        logger.warning(f"[web_search] Failed to scrape {url}: {e}")
                
                formatted_results.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search failed: {str(e)}"