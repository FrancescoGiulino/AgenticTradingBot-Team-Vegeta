import os
import logging

from langchain_openai import ChatOpenAI

from trading_bot.knowledge.knowledge_manager import knowledge_base

logger = logging.getLogger(__name__)
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState, MarketDiscovery, TradeDecision, DynamicWatchlist, ValidationResult
from .tools import get_portfolio_status, get_stock_price, execute_trade, get_stock_news, web_search
from .rate_limiter import rate_limiter
from datetime import datetime, timedelta
import time
from pydantic import BaseModel, Field
from .config import shared_config
from trading_bot.utils.cache import DiscoveryCache
from alpaca.data.requests import NewsRequest, StockBarsRequest
import pandas as pd
from alpaca.data.timeframe import TimeFrame
from trading_bot.services.alpaca_service import AlpacaService
from trading_bot.state import AgentState
from alpaca.data.enums import DataFeed
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from trading_bot.services.database_instance import DbInstance

# Initialize the LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0.15,
    timeout=60.0,
    max_retries=1
)

llm_local = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0.15,
    timeout=60.0,
    max_retries=1
)

def init_portfolio_node(state: AgentState) -> dict:
    """
    INITIAL NODE: Retrieves Alpaca's user current state.
    """
    try:
        # Giustissimo l'uso del rate limiter!
        rate_limiter.acquire("alpaca_rpm") 
        trading_client = AlpacaService().client

        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        
        portfolio_positions = {}
        for pos in positions:
            portfolio_positions[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price), 
                "current_price": float(pos.current_price),    
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),    
                "unrealized_plpc": float(pos.unrealized_plpc)  
            }
        order_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(order_request)
        pending_tickers = [order.symbol for order in open_orders]

        portfolio_data = {
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),       
            "portfolio_value": float(account.portfolio_value),
            "daytrade_count": int(account.daytrade_count),     
            "positions": portfolio_positions,
            "pending_orders": pending_tickers
        }
        
        action, changed = shared_config.get_action_and_reset_flag()
        current_action = action if changed else state.get("user_action")
        
        logger.info(f"[INIT_PORTFOLIO]: obtained initial data: {portfolio_data}")
        return {
            "portfolio": portfolio_data,
            "user_action": current_action,
            "cycle_logs": [{"node": "init_portfolio", "event": "Portfolio synced successfully."}]
        }

    except Exception as e:
        return {
            "error_message": f"CRITICAL - Failed to retrieve portfolio: {str(e)}",
            "cycle_logs": [{"node": "init_portfolio", "event": "ERROR", "message": str(e)}]
        }

def user_input_validator_node(state: AgentState) -> dict:
    """
    VALIDATOR NODE: Checks the user input, performs web research to see if it's a valid
    and relevant market topic/sector, and refines it or rejects it.
    """
    user_action = state.get("user_action")
    if not user_action:
        return {}
    
    logger.info(f"[VALIDATOR] Validating user action: {user_action}")
    try:
        search_results = web_search.invoke({"query": f"{user_action} market sector news stocks"})

        prompt = f"""
        You are a quantitative research analyst validator.
        The user wants the trading agent to focus on the following request: "{user_action}".
        
        Determine the intent of the user request:
        - DIRECT_ACTION: Explicit trades on specific tickers (e.g. "sell AAPL", "buy 10 shares of TSLA"). Set extracted_tickers and extracted_action.
        - THEMATIC_ACTION: A broad portfolio command (e.g. "balance my portfolio", "sell my tech stocks"). Set extracted_action.
        - MARKET_EXPLORATION: A general request to look into a sector (e.g. "focus on AI", "renewable energy").
        
        If it is a MARKET_EXPLORATION, you must determine if it is a valid and relevant market topic. 
        If it is, set is_valid to True and provide a refined_action. If it's garbage, set is_valid to False.
        DIRECT_ACTION and THEMATIC_ACTION are always valid.

        Here the result of a web search of the user input: {search_results}
        """
        
        structured_llm = llm.with_structured_output(ValidationResult)
        validation = structured_llm.invoke(prompt)
        
        if validation.intent == "MARKET_EXPLORATION" and validation.is_valid:
            # Quick web search to refine the exploration topic
            search_results = web_search.invoke({"query": f"{validation.refined_action} market sector news stocks"})
            # We just use the search results implicitly or trust the LLM's validation
            logger.info(f"[VALIDATOR] Did web search for MARKET_EXPLORATION: {validation.refined_action}")
            
        if validation.is_valid:
            logger.info(f"[VALIDATOR] Input VALID ({validation.intent}): {validation.refined_action}")
            return {
                "user_action": validation.refined_action,
                "action_intent": validation.intent,
                "action_tickers": validation.extracted_tickers,
                "action_type": validation.extracted_action,
                "cycle_logs": [{"node": "validator", "event": f"Validated user input ({validation.intent}): {validation.refined_action}"}]
            }
        else:
            logger.warning(f"[VALIDATOR] Input REJECTED: {validation.refined_action}")
            return {
                "user_action": None, # Reset user action
                "action_intent": None,
                "action_tickers": [],
                "action_type": None,
                "error_message": f"User action rejected: {validation.refined_action}",
                "cycle_logs": [{"node": "validator", "event": f"Rejected user input: {validation.refined_action}"}]
            }
    except Exception as e:
        logger.error(f"[VALIDATOR] Error: {str(e)}")
        return {
            "error_message": f"Validator node failed: {str(e)}",
            "cycle_logs": [{"node": "validator", "event": "ERROR", "message": str(e)}]
        }

def load_history_node(state: AgentState) -> dict:
    """
    MEMORY NODE: Retrieves the last operations on the local database 
    to provide the agent some context about it's recent actions
    """
    try:
        db = DbInstance()
        recent_trades = db.get_recent_trades()
        
        logger.info(f"[HISTORY]: obtained recent trading data")

        return {
            "recent_history": recent_trades,
            "cycle_logs": [{"node": "load_history", "event": f"Loaded {len(recent_trades)} trades."}]
        }
    except Exception as e:
        return {
            "recent_history": [],
            "error_message": f"CRITICAL - Failed to retrieve recent history: {str(e)}",
            }

def discovery_node(state: AgentState) -> dict:
    """
    DISCOVERY NODE: Reads Alpaca News and uses LLM to find 
    daily trends and tickers to analyze
    """
    try:
        intent = state.get("action_intent")
        if intent == "DIRECT_ACTION":
            target_tickers = state.get("action_tickers", [])
            clean_candidates = {ticker.upper().strip(): "User explicitly requested a trade on this ticker." for ticker in target_tickers}
            logger.info(f"[DISCOVERY] Bypassing general news for DIRECT_ACTION. Extracted tickers: {clean_candidates}")
            return {
                "market_themes": ["User Directed Action"],
                "candidate_tickers": clean_candidates,
                "cycle_logs": [{"node": "discovery", "event": f"Bypassed news. Directly added user targets: {list(clean_candidates.keys())}"}]
            }

        cache_manager = DiscoveryCache()
        cached_data = cache_manager.get_cached_discovery()

        if cached_data:
            logger.info("[DISCOVERY] CACHE HIT! Loading candidates from memory.")
            return {
                "market_themes": cached_data["market_themes"],
                "candidate_tickers": cached_data["candidate_tickers"],
                "cycle_logs": [{"node": "discovery", "event": "CACHE HIT: Skipped API and LLM calls."}]
            }
        
        alpaca = AlpacaService()
        news_client = alpaca.news_client
        
        request_params = NewsRequest(limit=30)
        response = news_client.get_news(request_params)
        
        if hasattr(response, "news"):
            raw_news = response.news 
        elif hasattr(response, "data"):
            raw_news = response.data
        elif isinstance(response, dict):
            raw_news = response.get("news", [])
        else:
            raw_news = response

        if isinstance(raw_news, dict):
            raw_news = raw_news.get("news", list(raw_news.values()))
        
        news_entries = []
        for item in raw_news:
            if isinstance(item, dict):
                headline = item.get("headline", "No Title")
                summary = item.get("summary", "No Summary Available")
                symbols_list = item.get("symbols", [])
            else:
                headline = getattr(item, "headline", "No Title")
                summary = getattr(item, "summary", "No Summary Available")
                symbols_list = getattr(item, "symbols", [])

            if symbols_list is None:
                symbols_list = []
            
            symbols_str = ", ".join(symbols_list)
            
            entry = f"TITLE: {headline}\nSUMMARY: {summary}\nRELATED SYMBOLS: {symbols_str}"
            news_entries.append(entry)
            
        news_text = "\n\n".join(news_entries)
        logger.info(f"[DISCOVERY]: Successfully parsed {len(news_entries)} news entries.")
        
        
        prompt = f"""
        You are an elite quantitative research analyst.
        Read the following recent market news and identify the driving macroeconomic themes.
        Then, select the top 5 to 10 stock tickers that represent the most volatile or interesting 
        opportunities based on these news.
        
        MARKET NEWS:
        {news_text}
        """

        structured_llm = llm.with_structured_output(MarketDiscovery)
        discovery_result = structured_llm.invoke(prompt)

        clean_candidates = {
            ticker.upper().strip(): rationale 
            for ticker, rationale in discovery_result.candidate_tickers.items()
        }

        cache_manager.set_discovery(discovery_result.market_themes, clean_candidates)

        logger.info(f"{discovery_result.market_themes} : {clean_candidates}")

        return {
            "market_themes": discovery_result.market_themes,
            "candidate_tickers": clean_candidates,
            "cycle_logs": [{"node": "discovery", "event": f"Found {len(clean_candidates)} candidates."}]
        }
        
    except Exception as e:
        logger.error(f"[ERROR] Discovery node failed: {str(e)}")
        return {
            "error_message": f"Discovery node failed: {str(e)}",
            "cycle_logs": [{"node": "discovery", "event": "ERROR", "message": str(e)}]
        }
decisor_llm = llm.with_structured_output(TradeDecision)

def quant_enrichment_node(state: AgentState) -> dict:
    """
    NODO QUANTITATIVO: Scarica lo storico prezzi e calcola SMA, ATR e Liquidità.
    Nessun LLM coinvolto, solo matematica pura.
    """
    logger.info("[NODE] QUANT: Calculating technical metrics...")
    
    try:
        alpaca = AlpacaService()
        hist_client = alpaca.historical_client
        
        portfolio_tickers = list(state.get("portfolio", {}).get("positions", {}).keys())
        candidate_tickers = list(state.get("candidate_tickers", {}).keys())
        all_tickers = list(set(portfolio_tickers + candidate_tickers))
        
        if not all_tickers:
            return {"quant_data": {}}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        quant_results = {}

        for ticker in all_tickers:
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=[ticker],
                    timeframe=TimeFrame.Day,
                    start=start_date,
                    end=end_date,
                    feed=DataFeed.IEX
                )
                
                bars_response = hist_client.get_stock_bars(request_params)
                if not bars_response or bars_response.df.empty:
                    logger.warning(f"[QUANT] No data returned for {ticker}. Skipping.")
                    continue
                    
                bars = bars_response.df
                
                try:
                    ticker_df = bars.loc[ticker].copy()
                except KeyError:
                    ticker_df = bars.copy()
                
                if ticker_df.empty or len(ticker_df) < 50:
                    logger.warning(f"[QUANT] Not enough data for {ticker}. Skipping math.")
                    continue
                
                current_price = float(ticker_df['close'].iloc[-1])
                avg_volume = int(ticker_df['volume'].tail(20).mean())
                
                # --- CALCOLO SMA 50 ---
                ticker_df['SMA_50'] = ticker_df['close'].rolling(window=50).mean()
                sma_50 = float(ticker_df['SMA_50'].iloc[-1])
                
                # --- CALCOLO ATR 14 ---
                high_low = ticker_df['high'] - ticker_df['low']
                high_close = (ticker_df['high'] - ticker_df['close'].shift()).abs()
                low_close = (ticker_df['low'] - ticker_df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr_14 = float(true_range.rolling(window=14).mean().iloc[-1])
                
                # --- RISK SIZING ---
                risk_budget_usd = 100.0 
                suggested_qty = int(risk_budget_usd / atr_14) if atr_14 > 0 else 0

                # --- LIQUIDITY FLAG (Blair Hull) ---
                if avg_volume < 500000:
                    liquidity_status = "Insufficient liquidity / Excessive slippage risk"
                else:
                    liquidity_status = "Liquidity OK"

                # --- COMPRESSION FLAG (Bollinger / Basic ATR) ---
                compression_ratio = (atr_14 / current_price) * 100 if current_price > 0 else 0
                if compression_ratio < 1.5:
                    volatility_status = "Compressed Volatility (Imminent Breakout Risk)"
                elif compression_ratio > 5.0:
                    volatility_status = "Extreme/Dangerous Volatility (Reduce size)"
                else:
                    volatility_status = "Normal Volatility"

                # --- DICTIONARY ASSIGNMENT ---
                quant_results[ticker] = {
                    "current_price": round(current_price, 2),
                    "trend": "Bearish trend (Price < SMA)" if current_price < sma_50 else "Bullish trend (Price > SMA)",
                    "atr_14": round(atr_14, 2),
                    "volatility_status": volatility_status,
                    "suggested_max_qty_based_on_volatility": suggested_qty,
                    "liquidity": liquidity_status
                }
                
            except Exception as inner_e:
                logger.warning(f"[QUANT] Failed to process {ticker}: {str(inner_e)}")
                continue
        logger.info(f"[QUANT] Successfully calculated metrics for {len(quant_results)} tickers.")
        
        return {
            "quant_data": quant_results,
            "cycle_logs": [{"node": "quant_enrichment", "event": f"Calculated math for {len(quant_results)} assets."}]
        }

    except Exception as e:
        logger.error(f"[ERROR] Quant node failed: {str(e)}")
        return {
            "error_message": f"Quant math failed: {str(e)}",
            "cycle_logs": [{"node": "quant_enrichment", "event": "ERROR", "message": str(e)}]
        }

def decisor_node(state: AgentState) -> dict:
    """Node: DECISOR"""
    logger.info("[NODE] DECISOR: Analyzing market and reasoning ")
    
    portfolio = state.get("portfolio", {})
    recent_history = state.get("recent_history", {})
    market_themes = state.get("market_themes",[])
    candidate_tickers = state.get("candidate_tickers", {})
    quant_data = state.get("quant_data", {})
    pending_orders = state.get("pending_orders",[])
    user_action = state.get("user_action")
    # TODO change location?
    
    error_message = state.get("error_message")
    
    if error_message:
        logger.warning(f" [WARNING] DECISOR detected an error: {error_message}. Defaulting to HOLD. ")
        decision = TradeDecision(
            ticker="none", action="HOLD", quantity=0.0, confidence_score=1.0,
            rationale=f"Fallback forced due to previous node error: {error_message}"
        )
        return {"proposed_decision": decision}    
    
    logger.info(f"[DECISOR] Fetching news for {candidate_tickers}... ")
    action_intent = state.get("action_intent")
    action_type = state.get("action_type")
    action_tickers = state.get("action_tickers")

    # TODO add info properly!
    prompt_info = knowledge_base.get_knowledge("the_intelligent_investor.txt")
    quantity_info = knowledge_base.get_knowledge("quantities.txt")

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an active and strategic AI Trading Agent.
        
        {prompt_info}
        {quantity_info}

        Provide a detailed, explicit 'rationale' explaining exactly why you chose this action (e.g., explicitly mention that you are buying the ticker X to deploy initial capital if rule Y triggers)."""),
        ("human", """
        Portfolio Status: {portfolio}
        Recent History: {recent_history}
        User Action Requested: {user_action}
        Action Intent: {action_intent}
        Action Type: {action_type}
        Action Tickers: {action_tickers}
        Hot Market Themes: {market_themes} 
        Candidate Tickers: {candidate_tickers} note: keep the portfolio balanced over different markets basen on your portfolio and history
        Pending Orders: {pending_orders}
        Quantities: {quant_data} 
        
        You must ALWAYS check the 'pending_orders' list inside the Portfolio Status.
        If a ticker has a pending order, the market is either closed or the order is awaiting execution. 
            - You MUST NOT propose a BUY or SELL for any ticker listed in 'pending_orders'.
            - If your top candidate is in the pending list, you must skip it and evaluate the next best candidate, or propose HOLD.
            - DO NOT stack multiple orders on the same asset.
         
        [DIRECTIVE FOR DIRECT OR THEMATIC ACTIONS]
        If Action Intent is DIRECT_ACTION or THEMATIC_ACTION, your primary goal is to propose ONE trade that moves closer to fulfilling the User Action Requested.
        - For DIRECT_ACTION, you MUST propose the requested action on the requested tickers, unless the quant metrics show extreme danger.
        - Propose exactly one logical next step.

        **Now hunt. Capital is meant to grow, not to be preserved.**
        [RATIONALE DIRECTIVE]
        When filling out the rationale field, you must explicitly state which module made the decision and why it was prioritized over others.
            - Example BUY: "Priority 2 (Graham module). Portfolio is safe. Exploited irrational panic on a solid asset. Ignored other candidates lacking margin of safety."
            - Example SELL: "Priority 1 (Schwager module). Executed mechanical stop-loss to limit damage. Ignored BUY candidates to focus on capital preservation.""")
    ])
    
    reasoning_chain = prompt | decisor_llm
    
    try:
        #estimated_tokens = len(prompt_str) // 4
        #rate_limiter.acquire("google_genai_rpm", 1)
        #rate_limiter.acquire("google_genai_tpm", estimated_tokens)
        
        decision = reasoning_chain.invoke({
            "prompt_info": prompt_info,
            "quantity_info": quantity_info,
            "portfolio": portfolio, "recent_history": recent_history, "user_action": user_action, "market_themes": market_themes, "candidate_tickers": candidate_tickers, "quant_data": quant_data,
            "pending_orders": pending_orders,
            "action_intent": action_intent,
            "action_type": action_type,
            "action_tickers": action_tickers
        })
    except Exception as e:
        logger.error(f" [ERROR] DECISOR LLM failed: {str(e)} ")
        # TODO change location?
        
        decision = TradeDecision(
            ticker="NO-TICK", action="HOLD", quantity=0, confidence_score=1.0,
            rationale=f"LLM processing error: {str(e)}."
        )
        
    return {"proposed_decision": decision}

def executer_node(state: AgentState) -> dict:
    """Node: EXECUTER"""
    logger.info("[NODE] EXECUTER: Executing the trade ")
    
    decision = state.get("proposed_decision")
    
    if decision.action.upper() == "HOLD":
        logger.info("[EXECUTER] Action is HOLD, skipping order execution.")
        return {"error_message": None}
        
    if decision.quantity <= 0:
        error_msg = f"Quantity must be > 0, got {decision.quantity}"
        logger.error(f"[ERROR] EXECUTER failed: {error_msg}")
        return {"error_message": error_msg}
    
    try:
        rate_limiter.acquire("alpaca_rpm")
        trading_client = AlpacaService().client
        order_request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = trading_client.get_orders(order_request)
        
        for order in open_orders:
            if order.symbol == decision.ticker and decision.action.upper() in str(order.side).upper():
                logger.info(f"[EXECUTER] Pending {decision.action} order already exists for {decision.ticker}. Skipping execution.")
                return {"error_message": None}
    except Exception as e:
        logger.warning(f"[EXECUTER] Could not check pending orders: {e}")
    
    logger.info(f"[EXECUTER] Sending order to Alpaca: {decision.action} {decision.quantity} {decision.ticker} ")
    
    execution_result = execute_trade.invoke({
        "ticker": decision.ticker, "action": decision.action, "quantity": decision.quantity
    })
    
    if "error" in execution_result:
        logger.error(f"[ERROR] EXECUTER failed: {execution_result['error']} ")
        return {"error_message": execution_result["error"]}
        
    logger.info(f"[EXECUTER] Order successful! Order ID: {execution_result.get('order_id')} ")
    return {"error_message": None}

# unused for now!
def summarizer(state: AgentState) -> dict:
    """Node: SUMMARIZER"""
    logger.info("[NODE] SUMMARIZER: Logging journal and cleaning state ")
    db = DbInstance()

    cycle_id = state.get("cycle_id") or f"cycle-{int(time.time())}"
    decision = state.get("proposed_decision")
    
    if state.get("error_message"):
        logger.error(f"[NODE] SUMMARIZER: Unable to summarize ({state.get("error_message")})")
        return{
            "error_message": None,
            "proposed_decision": None
        }
    
    success = db.log_trade_journal(
        cycle_id=cycle_id, 
        agent_type="decisor", 
        ticker=decision.ticker,
        action=decision.action, 
        quantity=decision.quantity,
        rationale=decision.rationale, 
        outcome="success", # <--- TODO change!!!
        data_sources="Alpaca, yfinance"
    )

class SatisfactionResult(BaseModel):
    is_satisfied: bool = Field(description="True if the user's action has been completely satisfied.")
    rationale: str = Field(description="Why it is satisfied or not.")

def satisfaction_checker_node(state: AgentState) -> dict:
    """
    SATISFACTION CHECKER NODE: Checks if the ongoing user action has been fully satisfied.
    If satisfied, clears the action from the state.
    """
    user_action = state.get("user_action")
    if not user_action:
        return {}
        
    logger.info(f"[SATISFACTION_CHECKER] Checking if user action is satisfied: {user_action}")
    try:
        portfolio = state.get("portfolio", {})
        recent_history = state.get("recent_history", [])
        
        prompt = f"""
        You are an AI Trading Assistant. The user originally requested: "{user_action}".
        
        Look at the current portfolio state:
        {portfolio}
        
        Look at the recent trading history (which includes trades just executed):
        {recent_history}
        
        Determine if the user's request has been completely satisfied.
        If it requires multiple trades (e.g. "Sell AAPL and MSFT") and only one was done, it is NOT satisfied yet.
        If it was a general MARKET_EXPLORATION and the bot made a decision based on it, consider it satisfied so we can move on.
        """
        
        structured_llm = llm.with_structured_output(SatisfactionResult)
        result = structured_llm.invoke(prompt)
        
        if result.is_satisfied:
            logger.info(f"[SATISFACTION_CHECKER] Action SATISFIED: {result.rationale}")
            return {
                "user_action": None,
                "action_intent": None,
                "action_tickers": [],
                "action_type": None,
                "cycle_logs": [{"node": "satisfaction_checker", "event": f"Goal satisfied: {result.rationale}"}]
            }
        else:
            logger.info(f"[SATISFACTION_CHECKER] Action PENDING: {result.rationale}")
            return {
                "cycle_logs": [{"node": "satisfaction_checker", "event": f"Goal pending (will continue next cycle): {result.rationale}"}]
            }
    except Exception as e:
        logger.error(f"[SATISFACTION_CHECKER] Error: {e}")
        return {}