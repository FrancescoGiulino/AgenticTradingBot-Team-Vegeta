import os
import logging

from trading_bot.knowledge_manager import knowledge_base

logger = logging.getLogger(__name__)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState, TradeDecision, DynamicWatchlist
from .tools import get_portfolio_status, get_stock_price, execute_trade, get_stock_news
from .rate_limiter import rate_limiter
import json
from datetime import datetime
import time
from .db import log_trade_journal
from .config import shared_config

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.15,
    timeout=60.0,
    max_retries=1
)

def init_portfolio(state: AgentState) -> dict:
    """
    Node: INIT_PORTFOLIO
    Fetches portfolio, checks for user interruptions, and generates a watchlist.
    """
    logger.info("[NODE] INIT_PORTFOLIO: Fetching portfolio data ")
    
    portfolio_data = get_portfolio_status.invoke({})
    error_msg = portfolio_data.get("error")
    
    if error_msg:
        logger.warning(f" [WARNING] INIT_PORTFOLIO error: {error_msg} ")
        return {"portfolio": {}, "error_message": error_msg}

    # 1. Leggiamo la memoria condivisa per vedere se l'utente ha inserito comandi
    current_focus, has_changed = shared_config.get_focus_and_reset_flag()
    target_tickers = list(state.get("target_tickers", []))
    
    # 2. Se l'utente ha cambiato focus, FORZIAMO lo svuotamento della watchlist
    if has_changed:
        logger.info(f"--- [SYSTEM ALERT] User requested new focus: '{current_focus}'. Clearing old watchlist! ---")
        target_tickers = [] # Questo costringerà l'Esploratore ad attivarsi
        
    # 3. DYNAMIC WATCHLIST GENERATION
    if not target_tickers:
        logger.info(f"[EXPLORER] Watchlist empty. Generating new dynamic tickers for sector: {current_focus}...")
        explorer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert financial researcher. Based on the given market sector/focus, output a list of 2 or 3 highly liquid, well-known US stock tickers to analyze. Output MUST follow the JSON schema."),
            ("human", "Generate tickers for this focus: {focus}")
        ])
        
        explorer_chain = explorer_prompt | llm.with_structured_output(DynamicWatchlist)
        try:
            watchlist_result = explorer_chain.invoke({"focus": current_focus})
            target_tickers = watchlist_result.tickers
            logger.info(f"[EXPLORER] New Watchlist Created: {target_tickers}. Rationale: {watchlist_result.rationale}")
        except Exception as e:
            logger.error(f"[ERROR] EXPLORER failed to generate watchlist: {e}")
            target_tickers = ["AAPL", "MSFT"] # Fallback

    return {
        "portfolio": portfolio_data,
        "target_tickers": target_tickers,
        "market_focus": current_focus, # Salviamo il focus aggiornato nello stato
        "error_message": None 
    }

decisor_llm = llm.with_structured_output(TradeDecision)

def decisor(state: AgentState) -> dict:
    """Node: DECISOR"""
    logger.info("[NODE] DECISOR: Analyzing market and reasoning ")
    
    portfolio = state.get("portfolio", {})
    target_tickers = state.get("target_tickers", ["AAPL"])
    current_ticker = target_tickers[0] if target_tickers else "AAPL"
    
    error_message = state.get("error_message")
    
    if error_message:
        logger.warning(f" [WARNING] DECISOR detected an error: {error_message}. Defaulting to HOLD. ")
        fallback_decision = TradeDecision(
            ticker=current_ticker, action="HOLD", quantity=0, news_summary="N/A",
            current_price=0.0, rationale=f"System error upstream: {error_message}."
        )
        return {"proposed_decision": fallback_decision}

    price_data = get_stock_price.invoke(current_ticker)
    
    logger.info(f"[DECISOR] Fetching news for {current_ticker}... ")
    news_data = get_stock_news.invoke(current_ticker)

    prompt_info = knowledge_base.get_knowledge("the_intelligent_investor.txt")

    # Prepare the system prompt enforcing your exact business rules
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an active and strategic AI Trading Agent.
        Your output MUST be based ONLY on the provided data. Do not hallucinate prices or news.
        
        {prompt_info}

        Provide a detailed, explicit 'rationale' explaining exactly why you chose this action (e.g., explicitly mention that you are buying to deploy initial capital if rule 2 triggers)."""),
        ("human", """
        Portfolio Status: {portfolio}
        Target Ticker: {ticker}
        Current Price Data: {price_data}
        Recent News: {news_data}
        
        Formulate your final decision.""")
    ])
    
    reasoning_chain = prompt | decisor_llm
    
    try:
        # Rate limiting logic
        prompt_str = prompt.format(portfolio=portfolio, ticker=current_ticker, price_data=price_data, news_data=news_data)
        estimated_tokens = len(prompt_str) // 4
        rate_limiter.acquire("google_genai_rpm", 1)
        rate_limiter.acquire("google_genai_tpm", estimated_tokens)
        
        decision = reasoning_chain.invoke({
            "portfolio": portfolio, "ticker": current_ticker, "price_data": price_data, "news_data": news_data
        })
    except Exception as e:
        logger.error(f" [ERROR] DECISOR LLM failed: {str(e)} ")
        decision = TradeDecision(
            ticker=current_ticker, action="HOLD", quantity=0, news_summary="N/A",
            current_price=0.0, rationale=f"LLM processing error: {str(e)}."
        )

    return {"proposed_decision": decision}

def checker(state: AgentState) -> dict:
    """Node: CHECKER"""
    logger.info("[NODE] CHECKER: Validating feasibility of the decision ")
    
    decision = state.get("proposed_decision")
    portfolio = state.get("portfolio", {})
    
    if not decision:
        logger.error("[ERROR] CHECKER: No decision found to validate. ")
        return {"is_decision_valid": False}
        
    action = decision.action.upper()
    ticker = decision.ticker
    quantity = decision.quantity
    
    if action in ["BUY", "SELL"] and quantity <= 0:
        logger.info(f"[CHECKER] REJECTED: Quantity must be > 0. Proposed quantity is {quantity}. ")
        return {"is_decision_valid": False}
    
    if action == "HOLD":
        logger.info("[CHECKER] Action is HOLD. Accepted automatically. ")
        return {"is_decision_valid": True}
        
    elif action == "SELL":
        positions = portfolio.get("positions", {})
        if ticker in positions and positions[ticker]["qty"] >= quantity:
            logger.info(f"[CHECKER] Sufficient shares owned to SELL {quantity} {ticker}. Accepted. ")
            return {"is_decision_valid": True}
        else:
            logger.info(f"[CHECKER] REJECTED: Not enough shares of {ticker} to sell. ")
            return {"is_decision_valid": False}
            
    elif action == "BUY":
        cash_available = portfolio.get("cash", 0.0)
        estimated_cost = quantity * decision.current_price
        if cash_available >= estimated_cost:
            logger.info(f"[CHECKER] Sufficient cash to BUY {quantity} {ticker}. Accepted. ")
            return {"is_decision_valid": True}
        else:
            logger.info(f"[CHECKER] REJECTED: Insufficient cash. ")
            return {"is_decision_valid": False}
            
    return {"is_decision_valid": False}

def executer(state: AgentState) -> dict:
    """Node: EXECUTER"""
    logger.info("[NODE] EXECUTER: Executing the trade ")
    
    decision = state.get("proposed_decision")
    is_valid = state.get("is_decision_valid", False)
    
    if not is_valid or not decision or decision.action.upper() == "HOLD":
        logger.info("[EXECUTER] No execution required (Action is HOLD or Invalid). ")
        return {} 
        
    logger.info(f"[EXECUTER] Sending order to Alpaca: {decision.action} {decision.quantity} {decision.ticker} ")
    
    execution_result = execute_trade.invoke({
        "ticker": decision.ticker, "action": decision.action, "quantity": decision.quantity
    })
    
    if "error" in execution_result:
        logger.error(f"[ERROR] EXECUTER failed: {execution_result['error']} ")
        return {"error_message": execution_result["error"]}
        
    logger.info(f"[EXECUTER] Order successful! Order ID: {execution_result.get('order_id')} ")
    return {"error_message": None}

def summarizer(state: AgentState) -> dict:
    """Node: SUMMARIZER"""
    logger.info("[NODE] SUMMARIZER: Logging journal and cleaning state ")
    
    decision = state.get("proposed_decision")
    is_valid = state.get("is_decision_valid", False)
    error_message = state.get("error_message")
    last_n = state.get("last_n_actions", [])
    
    # DYNAMIC WATCHLIST MANAGEMENT: Remove the ticker we just processed
    target_tickers = list(state.get("target_tickers", []))
    if target_tickers:
        target_tickers.pop(0) # Pop the first item off the list
    
    MAX_N = 5
    
    if error_message: outcome = f"FAILED: {error_message}"
    elif not is_valid and decision: outcome = "REJECTED BY CHECKER"
    elif decision: outcome = f"SUCCESSFULLY EXECUTED {decision.action}"
    else: outcome = "UNKNOWN / NO DECISION"

    journal_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": decision.ticker if decision else "N/A",
        "action": decision.action if decision else "HOLD",
        "quantity": decision.quantity if decision else 0,
        "price": decision.current_price if decision else 0.0,
        "news_summary": decision.news_summary if decision else "N/A",
        "rationale": decision.rationale if decision else "No decision generated due to system constraints.",
        "outcome": outcome
    }
    
    cycle_id = state.get("cycle_id") or f"cycle-{int(time.time())}"
    success = log_trade_journal(
        cycle_id=cycle_id, agent_type="decisor", ticker=journal_entry["ticker"],
        action=journal_entry["action"], quantity=journal_entry["quantity"],
        price=journal_entry["price"], news_summary=journal_entry["news_summary"],
        rationale=journal_entry["rationale"], outcome=journal_entry["outcome"],
        data_sources="Alpaca, yfinance"
    )
    
    if success: logger.info(f"[SUMMARIZER] Successfully appended log to SQLite database ")
    
    updated_last_n = list(last_n)
    updated_last_n.append({"ticker": journal_entry["ticker"], "action": journal_entry["action"], "outcome": journal_entry["outcome"]})
    if len(updated_last_n) > MAX_N: updated_last_n.pop(0)

    return {
        "target_tickers": target_tickers, # 🟢 Restituiamo la lista aggiornata (accorciata) per il prossimo ciclo
        "last_n_actions": updated_last_n,
        "journal": [journal_entry], 
        "proposed_decision": None,  
        "is_decision_valid": False, 
        "error_message": None       
    }