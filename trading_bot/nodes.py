import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .state import AgentState, TradeDecision
from .tools import get_portfolio_status, get_stock_price, execute_trade, get_stock_news
from .db import log_trade_journal
import json
import time
from datetime import datetime

# Initialize the LLM with Google API Key
# We use temperature=0.0 to make the agent's decisions deterministic and strictly logical
llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0
)

def init_portfolio(state: AgentState) -> dict:
    """
    Node: INIT_PORTFOLIO
    Responsible for fetching the current portfolio status from Alpaca
    and injecting it into the agent's state before the DECISOR runs.
    """
    print("--- [NODE] INIT_PORTFOLIO: Fetching portfolio data ---")
    
    portfolio_data = get_portfolio_status.invoke({})
    
    if "error" in portfolio_data:
        print(f"--- [WARNING] INIT_PORTFOLIO encountered an error: {portfolio_data['error']} ---")
        return {
            "portfolio": {}, 
            "error_message": portfolio_data["error"]
        }
    
    return {
        "portfolio": portfolio_data,
        "error_message": None 
    }

# Force the LLM to return data exactly matching our Pydantic schema
# This ensures we never get plain text when we expect a JSON
decisor_llm = llm.with_structured_output(TradeDecision)

def decisor(state: AgentState) -> dict:
    """
    Node: DECISOR
    Executes the core reasoning logic defined in the hackathon schema:
    1. Checks portfolio for losses (Proposes SELL).
    2. Evaluates market data and news (Proposes BUY).
    3. Otherwise, proposes HOLD.
    """
    print("--- [NODE] DECISOR: Analyzing market and reasoning ---")
    
    portfolio = state.get("portfolio", {})
    # For now, we pick the first target ticker to analyze
    target_tickers = state.get("target_tickers", ["AAPL"])
    current_ticker = target_tickers[0] if target_tickers else "AAPL"
    
    error_message = state.get("error_message")
    
    # ANTI-FRAGILE LOGIC: If a previous node failed, we gracefully HOLD
    if error_message:
        print(f"--- [WARNING] DECISOR detected an error: {error_message}. Defaulting to HOLD. ---")
        fallback_decision = TradeDecision(
            ticker=current_ticker,
            action="HOLD",
            quantity=0,
            news_summary="N/A",
            current_price=0.0,
            rationale=f"System error detected upstream: {error_message}. Safety protocol engaged: proposing HOLD to protect capital."
        )
        return {"proposed_decision": fallback_decision}

    # Fetch fresh market data using our tool (simulating the 'get finance data' arrow in your diagram)
    price_data = get_stock_price.invoke(current_ticker)
    
    # Fetch real news using the tool
    print(f"--- [DECISOR] Fetching news for {current_ticker}... ---")
    news_data = get_stock_news.invoke(current_ticker)
    
    # Prepare the system prompt enforcing your exact business rules
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an active and strategic AI Trading Agent.
        Your output MUST be based ONLY on the provided data. Do not hallucinate prices or news.
        
        Follow these strict sequential rules to make your decision:
        1. PORTFOLIO PROTECTION (SELL): If the 'target_ticker' is currently owned and is in loss (unrealized_pl < 0), you MUST propose "SELL".
        2. INITIAL CAPITAL DEPLOYMENT (BUY): Look at the Portfolio Status. If your portfolio is empty (0 positions) and you have a lot of cash, your primary mandate is to enter the market. If the news for the current ticker is neutral, slightly positive, or just informational, you MUST propose "BUY" (choose a quantity between 5 and 15). Do not sit on 100% cash.
        3. OPPORTUNISTIC BUY: If you already have positions in the market, propose "BUY" only if the recent news is distinctly positive and encouraging.
        4. HOLD: If you already have positions and the news is neutral/mixed, or if the news is strictly negative, propose "HOLD".
        
        Provide a detailed, explicit 'rationale' explaining exactly why you chose this action (e.g., explicitly mention that you are buying to deploy initial capital if rule 2 triggers)."""),
        ("human", """
        Portfolio Status: {portfolio}
        Target Ticker: {ticker}
        Current Price Data: {price_data}
        Recent News: {news_data}
        
        Formulate your final decision.
        """)
    ])
    
    # Chain the prompt with the structured LLM
    reasoning_chain = prompt | decisor_llm
    
    try:
        # Invoke the chain with the gathered data
        decision = reasoning_chain.invoke({
            "portfolio": portfolio,
            "ticker": current_ticker,
            "price_data": price_data,
            "news_data": news_data
        })
    except Exception as e:
        # GRACEFUL RECOVERY: If the LLM fails to parse the JSON or crashes
        print(f"--- [ERROR] DECISOR LLM failed: {str(e)} ---")
        decision = TradeDecision(
            ticker=current_ticker,
            action="HOLD",
            quantity=0,
            news_summary="N/A",
            current_price=0.0,
            rationale=f"LLM processing error encountered: {str(e)}. Defaulting to safe action: HOLD."
        )

    return {"proposed_decision": decision}

def checker(state: AgentState) -> dict:
    """
    Node: CHECKER
    Validates the proposed decision against hard portfolio constraints:
    - BUY: Checks if there is enough cash to cover the purchase.
    - SELL: Checks if the portfolio actually holds enough quantity of the ticker.
    - HOLD: Automatically accepted.
    """
    print("--- [NODE] CHECKER: Validating feasibility of the decision ---")
    
    decision = state.get("proposed_decision")
    portfolio = state.get("portfolio", {})
    
    # If for some reason there is no decision, we reject and stop.
    if not decision:
        print("--- [ERROR] CHECKER: No decision found to validate. ---")
        return {"is_decision_valid": False}
        
    action = decision.action.upper()
    ticker = decision.ticker
    quantity = decision.quantity
    
    # HOLD is always feasible
    if action == "HOLD":
        print("--- [CHECKER] Action is HOLD. Accepted automatically. ---")
        return {"is_decision_valid": True}
        
    # SELL logic: Do we actually own the shares?
    elif action == "SELL":
        positions = portfolio.get("positions", {})
        if ticker in positions and positions[ticker]["qty"] >= quantity:
            print(f"--- [CHECKER] Sufficient shares owned to SELL {quantity} {ticker}. Accepted. ---")
            return {"is_decision_valid": True}
        else:
            print(f"--- [CHECKER] REJECTED: Not enough shares of {ticker} to sell. ---")
            return {"is_decision_valid": False}
            
    # BUY logic: Do we have enough cash?
    elif action == "BUY":
        cash_available = portfolio.get("cash", 0.0)
        # We estimate the cost using the current price provided by the DECISOR
        estimated_cost = quantity * decision.current_price
        
        if cash_available >= estimated_cost:
            print(f"--- [CHECKER] Sufficient cash to BUY {quantity} {ticker}. Accepted. ---")
            return {"is_decision_valid": True}
        else:
            print(f"--- [CHECKER] REJECTED: Insufficient cash. Need {estimated_cost}, have {cash_available}. ---")
            return {"is_decision_valid": False}
            
    # Fallback for unexpected actions
    print(f"--- [WARNING] CHECKER: Unrecognized action '{action}'. Rejected. ---")
    return {"is_decision_valid": False}

def executer(state: AgentState) -> dict:
    """
    Node: EXECUTER
    Executes the validated trade decision using the Alpaca API.
    Bypasses execution if the decision was HOLD or if the CHECKER rejected it.
    """
    print("--- [NODE] EXECUTER: Executing the trade ---")
    
    decision = state.get("proposed_decision")
    is_valid = state.get("is_decision_valid", False)
    
    # 1. Skip execution if invalid, missing, or just a HOLD
    if not is_valid or not decision or decision.action.upper() == "HOLD":
        print("--- [EXECUTER] No execution required (Action is HOLD or Invalid). ---")
        return {} # No changes to the state
        
    print(f"--- [EXECUTER] Sending order to Alpaca: {decision.action} {decision.quantity} {decision.ticker} ---")
    
    # 2. Call the execution tool
    execution_result = execute_trade.invoke({
        "ticker": decision.ticker,
        "action": decision.action,
        "quantity": decision.quantity
    })
    
    # 3. Handle execution errors (Anti-Fragile logic)
    if "error" in execution_result:
        print(f"--- [ERROR] EXECUTER failed: {execution_result['error']} ---")
        # Save the error in the state so the Summarizer/Journal can log the failure
        return {"error_message": execution_result["error"]}
        
    print(f"--- [EXECUTER] Order successful! Order ID: {execution_result.get('order_id')} ---")
    
    # Clear any previous errors if successful
    return {"error_message": None}

def summarizer(state: AgentState) -> dict:
    """
    Node: SUMMARIZER
    Manages short-term memory and logs every cycle into a persistent Trade Journal.
    Steps:
    1. Formats a structured log entry based on the current state and execution outcomes.
    2. Appends the entry to a local JSON file (the persistent journal).
    3. Updates the 'last_n_actions' rolling list (memory layer).
    4. Clears temporary/heavy data to provide a clean state for the next iteration.
    """
    print("--- [NODE] SUMMARIZER: Logging journal and cleaning state ---")
    
    decision = state.get("proposed_decision")
    is_valid = state.get("is_decision_valid", False)
    error_message = state.get("error_message")
    last_n = state.get("last_n_actions", [])
    
    # Define max memory size for rolling actions
    MAX_N = 5
    
    # 1. Determine the final status/outcome of the cycle
    if error_message:
        outcome = f"FAILED: {error_message}"
    elif not is_valid and decision:
        outcome = "REJECTED BY CHECKER"
    elif decision:
        outcome = f"SUCCESSFULLY EXECUTED {decision.action}"
    else:
        outcome = "UNKNOWN / NO DECISION"

    # 2. Construct the journal entry for state memory
    # Using fallback values if no decision was formulated due to upstream crashes
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
    
    # 3. Persist the journal entry into the SQLite database instead of JSON
    cycle_id = state.get("cycle_id") or f"cycle-{int(time.time())}"
    success = log_trade_journal(
        cycle_id=cycle_id,
        agent_type="decisor",
        ticker=journal_entry["ticker"],
        action=journal_entry["action"],
        quantity=journal_entry["quantity"],
        price=journal_entry["price"],
        news_summary=journal_entry["news_summary"],
        rationale=journal_entry["rationale"],
        outcome=journal_entry["outcome"],
        data_sources="Alpaca, yfinance"
    )
    if success:
        print("--- [SUMMARIZER] Successfully logged trade to SQLite database. ---")
    else:
        print("--- [WARNING] SUMMARIZER failed to log trade to SQLite database. ---")

    # 4. Update the rolling memory (last_n_actions)
    # Create a shallow copy, append the new summary, and keep only the latest MAX_N items
    updated_last_n = list(last_n)
    updated_last_n.append({
        "ticker": journal_entry["ticker"],
        "action": journal_entry["action"],
        "outcome": journal_entry["outcome"]
    })
    if len(updated_last_n) > MAX_N:
        updated_last_n.pop(0) # Remove the oldest action from the head of the list

    # 5. State Cleanup: return updates. 
    # Returning None for temporary flags resets them for the next loop iteration.
    return {
        "last_n_actions": updated_last_n,
        "journal": [journal_entry], # Appended automatically via operator.add
        "proposed_decision": None,  # Resetting for a clean state next turn
        "is_decision_valid": False, # Resetting for a clean state next turn
        "error_message": None       # Clearing error state for the next cycle
    }
