import os
import logging
import json
import time
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import Literal, List
from langchain_groq import ChatGroq

from trading_bot.knowledge_manager import knowledge_base
from .state import AgentState, TradeDecision
from .tools import get_portfolio_status, get_stock_price, execute_trade, get_stock_news, web_search
from .db import log_trade_journal
from .config import shared_config

logger = logging.getLogger(__name__)

llm_google = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.15,
    timeout=60.0,
    max_retries=1
)

llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    timeout=60.0,
    max_retries=1
)

locallm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="local-model",
    temperature=0.15,
    timeout=60.0,
    max_retries=1
)

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
llm = locallm if USE_LOCAL_LLM else llm_groq

def init_portfolio(state: AgentState) -> dict:
    logger.info("[NODE] INIT_PORTFOLIO: Fetching portfolio data ")
    portfolio_data = get_portfolio_status.invoke({})
    error_msg = portfolio_data.get("error")
    
    if error_msg:
        logger.warning(f" [WARNING] INIT_PORTFOLIO error: {error_msg} ")
        return {"portfolio": {}, "error_message": error_msg}

    cash = portfolio_data.get("cash", 0.0)
    positions = portfolio_data.get("positions", {})
    pos_str = []
    for ticker, data in positions.items():
        pos_str.append(f"{data['qty']} {ticker} (Market Value: ${data['market_value']:.2f}, Unrealized P&L: ${data['unrealized_pl']:.2f})")
    
    pos_text = ", ".join(pos_str) if pos_str else "No open positions."
    portfolio_summary = f"Cash Available: ${cash:.2f}\nPositions: {pos_text}"

    current_focus, has_changed = shared_config.get_focus_and_reset_flag()
    target_tickers = list(state.get("target_tickers", []))
    analyzed_tickers = list(state.get("analyzed_tickers", []))
    analyzed_portfolio_tickers = list(state.get("analyzed_portfolio_tickers", []))
    cycles_since_portfolio_analysis = state.get("cycles_since_portfolio_analysis", 0) + 1
    
    open_position_tickers = list(positions.keys())
    if open_position_tickers and all(ticker in analyzed_portfolio_tickers for ticker in open_position_tickers):
        logger.info("[INIT_PORTFOLIO] All open positions have been analyzed recently. Resetting portfolio memory.")
        analyzed_portfolio_tickers = []
    
    if has_changed:
        logger.info(f"[SYSTEM ALERT] User requested new focus: '{current_focus}'. Clearing old watchlist and analyzed memory!")
        target_tickers = []
        analyzed_tickers = []
        state["focus_iteration_count"] = 0

    focus_iteration_count = state.get("focus_iteration_count", 0)

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configuration.json")
    user_command = state.get("user_command") or ""
    new_command_found = False
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)
            wanted_action = user_config.get("wanted_action", "").strip()
            if wanted_action:
                user_command = wanted_action
                new_command_found = True
                try:
                    user_config["wanted_action"] = ""
                    with open(config_path, "w") as fw:
                        json.dump(user_config, fw, indent=4)
                    logger.info(f"[INIT_PORTFOLIO] Found new user command: '{wanted_action}'. Cleared from config.")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to clear wanted_action from config: {e}")

    if new_command_found:
        target_tickers = []

    if not has_changed and not user_command:
        if focus_iteration_count >= 5:
            import random
            alt_markets = ["Food and Agriculture", "Transportation and Logistics", "Healthcare and Pharmaceuticals", "Energy and Utilities", "Financial Services", "Retail and E-commerce"]
            alt_markets = [m for m in alt_markets if m.lower() not in current_focus.lower()]
            if not alt_markets:
                alt_markets = ["Technology", "Retail"]
            new_focus = random.choice(alt_markets)
            logger.info(f"[SYSTEM ALERT] Rotating market focus to '{new_focus}' to variate searches (completed 5 iterations on '{current_focus}').")
            
            shared_config.update_focus(new_focus)
            current_focus = new_focus
            target_tickers = []
            analyzed_tickers = []
            focus_iteration_count = 0
        else:
            focus_iteration_count += 1

    return_dict = {
        "portfolio": portfolio_data,
        "portfolio_summary": portfolio_summary,
        "user_command": user_command,
        "market_focus": current_focus,
        "target_tickers": target_tickers,
        "analyzed_tickers": analyzed_tickers,
        "analyzed_portfolio_tickers": analyzed_portfolio_tickers,
        "cycles_since_portfolio_analysis": cycles_since_portfolio_analysis,
        "focus_iteration_count": focus_iteration_count,
        "error_message": None 
    }
    
    if new_command_found:
        return_dict["research_context"] = None
        
    return return_dict

class SupervisorDecision(BaseModel):
    next_node: Literal["researcher", "portfolio_analyzer", "decisor", "checker", "summarizer", "FINISH"] = Field(
        description="The next node to route to. 'portfolio_analyzer' to review existing holdings. 'researcher' to gather context on user commands. 'decisor' to make a trading decision. 'checker' to validate a proposed decision and automatically execute it if valid. 'summarizer' to wrap up after a decision is processed or rejected. 'FINISH' to end the graph cycle."
    )
    rationale: str = Field(description="Why this node was chosen.")

def supervisor_node(state: AgentState) -> dict:
    logger.info("[NODE] SUPERVISOR: Deciding next action based on state")
    
    if state.get("error_message"):
        logger.warning(f"[SUPERVISOR] Error detected: {state['error_message']}. Routing to summarizer.")
        return {"next_node": "summarizer"}

    system_prompt = """You are the Supervisor of an AI Trading Bot.
You control the execution flow between specialized worker nodes.
Available Nodes:
- 'portfolio_analyzer': Call this if 'cycles_since_portfolio_analysis' is >= 5 and there are open positions in the portfolio. This prioritizes reviewing current holdings.
- 'researcher': Call this if there is a 'user_command' (prioritize this!) but 'research_context' is empty. Alternatively, if there is NO 'user_command', act autonomously: check the 'market_focus' and if 'target_tickers' is empty, call 'researcher' to generate a watchlist.
- 'decisor': Call this to propose a trade (BUY/SELL/HOLD). Needs 'portfolio_summary' and either 'target_tickers' or 'user_command'.
- 'checker': Call this after the decisor has proposed a decision, to validate it against the portfolio limits and automatically execute if valid.
- 'summarizer': Call this if there is an error to log, or to wrap up.
- 'FINISH': Call this to end the cycle (e.g., after summarizing, or if there's nothing to do).

Analyze the current state and determine the next step.
IMPORTANT: Output ONLY valid JSON. Do not include markdown formatting like ```json. Do not include any explanations outside the JSON object.
"""

    state_desc_template = """
User Command: {user_command}
Market Focus: {market_focus}
Research Context: {research_context}
Target Tickers: {target_tickers}
Proposed Decision: {proposed_decision}
Is Decision Valid: {is_decision_valid}
Portfolio Summary: {portfolio_summary}
Cycles Since Portfolio Analysis: {cycles_since_portfolio_analysis}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", state_desc_template)
    ])

    chain = prompt | llm.with_structured_output(SupervisorDecision)
    
    try:
        decision = chain.invoke({
            "user_command": state.get("user_command") or "None",
            "market_focus": state.get("market_focus") or "None",
            "research_context": state.get("research_context") or "None",
            "target_tickers": state.get("target_tickers", []),
            "proposed_decision": state.get("proposed_decision", "None"),
            "is_decision_valid": state.get("is_decision_valid", "False"),
            "portfolio_summary": state.get("portfolio_summary") or "None",
            "cycles_since_portfolio_analysis": state.get("cycles_since_portfolio_analysis", 0)
        })
        logger.info(f"[SUPERVISOR] Decided: {decision.next_node}. Rationale: {decision.rationale}")
        return {"next_node": decision.next_node}
    except Exception as e:
        logger.error(f"[ERROR] Supervisor LLM failed: {e}")
        return {"next_node": "FINISH"}

class ResearchOutput(BaseModel):
    target_tickers: List[str] = Field(description="List of stock tickers identified from the research to focus on.")
    research_context: str = Field(description="Summary of the research findings to help the decisor.")

def portfolio_analyzer(state: AgentState) -> dict:
    logger.info("[NODE] PORTFOLIO ANALYZER: Reviewing open positions")
    
    portfolio = state.get("portfolio", {})
    positions = portfolio.get("positions", {})
    analyzed_portfolio_tickers = list(state.get("analyzed_portfolio_tickers", []))
    
    open_tickers = [t for t in positions.keys() if t not in analyzed_portfolio_tickers]
    
    if not open_tickers:
        logger.info("[PORTFOLIO ANALYZER] All open positions have been analyzed. Resetting cycles.")
        return {"cycles_since_portfolio_analysis": 0}
        
    tickers_to_evaluate = open_tickers[:5]
    logger.info(f"[PORTFOLIO ANALYZER] Selected for evaluation: {tickers_to_evaluate}")
    
    target_tickers = list(state.get("target_tickers", []))
    for t in reversed(tickers_to_evaluate):
        if t in target_tickers:
            target_tickers.remove(t)
        target_tickers.insert(0, t)
        
    analyzed_portfolio_tickers.extend(tickers_to_evaluate)
    
    return {
        "target_tickers": target_tickers,
        "analyzed_portfolio_tickers": analyzed_portfolio_tickers,
        "cycles_since_portfolio_analysis": 0,
        "research_context": "Periodic portfolio review triggered. Focus on evaluating whether to hold or sell these positions based on current market conditions and performance."
    }

def researcher_node(state: AgentState) -> dict:
    logger.info("[NODE] RESEARCHER: Gathering information")
    
    command = state.get("user_command", "")
    market_focus = state.get("market_focus", "technology")
    analyzed_tickers = state.get("analyzed_tickers", [])
    
    #TODO: This is an atonomous exploration based on Market Focus, we want to allow teh AI to search non focused markets
    if not command:
        logger.info(f"[RESEARCHER] Autonomous mode. Generating watchlist for suggested focus: {market_focus}. Avoiding: {analyzed_tickers}")
        explorer_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert financial researcher. The given market sector/focus is just a SUGGESTION. You are encouraged to explore other promising fields or unrelated high-potential tickers if you see fit. Output a list of 2 or 3 highly liquid, well-known US stock tickers to analyze. Do NOT suggest any tickers from this list: {analyzed_tickers}. Provide a brief rationale in 'research_context', explaining your choice of fields.\nIMPORTANT: Output ONLY valid JSON. Do not include markdown formatting like ```json. Do not include any explanations outside the JSON object."),
            ("human", "Suggested focus: {focus}\nGenerate tickers based on this suggestion or explore other high-potential fields.")
        ])
        
        explorer_chain = explorer_prompt | llm.with_structured_output(ResearchOutput)
        try:
            result = explorer_chain.invoke({"focus": market_focus, "analyzed_tickers": analyzed_tickers})
            logger.info(f"[RESEARCHER] New Watchlist Created: {result.target_tickers}")
            return {
                "target_tickers": result.target_tickers,
                "research_context": result.research_context
            }
        except Exception as e:
            logger.error(f"[ERROR] EXPLORER failed to generate watchlist: {e}")
            return {"research_context": "Fallback due to error.", "target_tickers": ["AAPL", "MSFT"]}

    logger.info(f"[RESEARCHER] User command detected. Searching web for: {command}")
    search_result = ""
    try:
        search_result = web_search.invoke({"query": command})
    except Exception as e:
        search_result = f"Search failed: {e}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial researcher. Based on the user command and search results, identify the relevant US stock tickers (e.g., AAPL, MSFT) and summarize the context that a trader would need to know.\nIMPORTANT: Output ONLY valid JSON. Do not include markdown formatting like ```json. Do not include any explanations outside the JSON object."),
        ("human", "User Command: {command}\nSearch Results: {search_result}")
    ])
    
    chain = prompt | llm.with_structured_output(ResearchOutput)
    
    try:
        result = chain.invoke({"command": command, "search_result": search_result})
        logger.info(f"[RESEARCHER] Found Tickers: {result.target_tickers}")
        
        target_tickers = list(state.get("target_tickers", []))
        for t in reversed(result.target_tickers):
            if t in target_tickers:
                target_tickers.remove(t)
            target_tickers.insert(0, t)
            
        return {
            "target_tickers": target_tickers,
            "research_context": result.research_context
        }
    except Exception as e:
        logger.error(f"[ERROR] Researcher LLM failed: {e}")
        return {"research_context": "Failed to analyze research data.", "target_tickers": state.get("target_tickers", [])}

def decisor(state: AgentState) -> dict:
    logger.info("[NODE] DECISOR: Analyzing market and reasoning ")
    
    portfolio = state.get("portfolio", {})
    target_tickers = state.get("target_tickers", [])
    current_ticker = target_tickers[0] if target_tickers else "AAPL"
    
    price_data = get_stock_price.invoke(current_ticker)
    news_data = get_stock_news.invoke(current_ticker)
    prompt_info = knowledge_base.get_knowledge("the_intelligent_investor.txt")
    
    user_command = state.get("user_command", "")
    research_context = state.get("research_context", "")

    system_prompt_template = """You are an active and strategic AI Trading Agent.
Your output MUST be based ONLY on the provided data. Do not hallucinate prices or news.
{prompt_info}

USER COMMAND: "{user_command}"
RESEARCH CONTEXT: "{research_context}"
PORTFOLIO SUMMARY:
{portfolio_summary}

Follow these strict rules:
1. If there is a USER COMMAND, treat it as a SUGGESTION, NOT an absolute order. You MUST validate it for safety and financial viability. If it is deemed unsafe, extremely risky, or lacking logical financial backing (e.g., buying a crashing stock without good reason), you MUST abort by proposing "HOLD", and clearly log the safety concerns in your rationale to avoid doing damages.
2. If there is NO user command, use your discretion to buy (if lots of cash and positive news), sell (if holding in loss or bad news), or hold.
3. CRITICAL CONSTRAINT: You cannot "SELL" 0 shares or "BUY" 0 shares. If your calculated quantity for a BUY or SELL is 0, you MUST propose "HOLD" instead.

Provide a detailed, explicit 'rationale' explaining exactly why you chose this action based on the user's command and market data.
IMPORTANT: Output ONLY valid JSON. Do not include markdown formatting like ```json. Do not include any explanations outside the JSON object."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", """
        Target Ticker: {ticker}
        Current Price Data: {price_data}
        Recent News: {news_data}
        
        Formulate your final decision.""")
    ])
    
    decisor_llm = llm.with_structured_output(TradeDecision)
    reasoning_chain = prompt | decisor_llm
    
    try:
        decision = reasoning_chain.invoke({
            "ticker": current_ticker, "price_data": price_data, "news_data": news_data,
            "prompt_info": prompt_info, "user_command": user_command, "research_context": research_context,
            "portfolio_summary": state.get("portfolio_summary", "None")
        })
    except Exception as e:
        logger.error(f" [ERROR] DECISOR LLM failed: {str(e)} ")
        decision = TradeDecision(
            ticker=current_ticker, action="HOLD", quantity=0, news_summary="N/A",
            current_price=0.0, rationale=f"LLM processing error: {str(e)}.",
            cleared_wanted_action=False
        )

    return {"proposed_decision": decision}

def checker(state: AgentState) -> dict:
    logger.info("[NODE] CHECKER: Validating feasibility of the decision ")
    
    decision = state.get("proposed_decision")
    portfolio = state.get("portfolio", {})
    
    if not decision:
        logger.error("[ERROR] CHECKER: No decision found to validate. ")
        return {"is_decision_valid": False, "checker_reason": "No decision found to validate."}
        
    action = decision.action.upper()
    ticker = decision.ticker
    quantity = decision.quantity
    
    pending_orders = portfolio.get("pending_orders", [])
    for order in pending_orders:
        if order["ticker"] == ticker and order["action"] == action:
            msg = f"There is already a pending {action} order for {ticker}."
            logger.info(f"[CHECKER] REJECTED: {msg}")
            return {"is_decision_valid": False, "checker_reason": msg}
            
    if action in ["BUY", "SELL"] and quantity <= 0:
        msg = f"Quantity must be > 0. Proposed quantity is {quantity}."
        logger.info(f"[CHECKER] REJECTED: {msg}")
        return {"is_decision_valid": False, "checker_reason": msg}
    
    if action == "HOLD":
        msg = "Action is HOLD. Accepted automatically."
        logger.info(f"[CHECKER] {msg}")
        return {"is_decision_valid": True, "checker_reason": msg}
        
    elif action == "SELL":
        positions = portfolio.get("positions", {})
        if ticker in positions and positions[ticker]["qty"] >= quantity:
            msg = f"Sufficient shares owned to SELL {quantity} {ticker}."
            logger.info(f"[CHECKER] {msg} Accepted.")
            return {"is_decision_valid": True, "checker_reason": msg}
        else:
            msg = f"Not enough shares of {ticker} to sell."
            logger.info(f"[CHECKER] REJECTED: {msg}")
            return {"is_decision_valid": False, "checker_reason": msg}
            
    elif action == "BUY":
        cash_available = portfolio.get("cash", 0.0)
        estimated_cost = quantity * decision.current_price
        if cash_available >= estimated_cost:
            msg = f"Sufficient cash to BUY {quantity} {ticker}."
            logger.info(f"[CHECKER] {msg} Accepted.")
            return {"is_decision_valid": True, "checker_reason": msg}
        else:
            msg = f"Insufficient cash to BUY. Available: {cash_available}, Required: {estimated_cost}."
            logger.info(f"[CHECKER] REJECTED: {msg}")
            return {"is_decision_valid": False, "checker_reason": msg}
            
    logger.error(f"Unknown action type: {action}")
    return {"is_decision_valid": False, "checker_reason": "Unknown action type."}

def executer(state: AgentState) -> dict:
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
    logger.info("[NODE] SUMMARIZER: Logging journal and cleaning state ")
    
    decision = state.get("proposed_decision")
    is_valid = state.get("is_decision_valid", False)
    error_message = state.get("error_message")
    checker_reason = state.get("checker_reason", "")
    last_n = state.get("last_n_actions", [])
    
    target_tickers = list(state.get("target_tickers", []))
    if target_tickers:
        target_tickers.pop(0) 
    
    MAX_N = 5
    
    if error_message: outcome = f"FAILED: {error_message}"
    elif not is_valid and decision: outcome = f"REJECTED BY CHECKER: {checker_reason}"
    elif decision: outcome = f"SUCCESSFULLY EXECUTED {decision.action} ({checker_reason})"
    else: outcome = "UNKNOWN / NO DECISION"

    journal_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": decision.ticker if decision else "N/A",
        "action": decision.action if decision else "HOLD",
        "quantity": decision.quantity if decision else 0,
        "price": decision.current_price if decision else 0.0,
        "news_summary": decision.news_summary if decision else "N/A",
        "rationale": decision.rationale if decision else "No decision generated.",
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

    analyzed_tickers = list(state.get("analyzed_tickers", []))
    if decision and decision.ticker and decision.ticker != "N/A":
        if decision.ticker not in analyzed_tickers:
            analyzed_tickers.append(decision.ticker)
            if len(analyzed_tickers) > 30:
                analyzed_tickers.pop(0)

    user_command = state.get("user_command")
    research_context = state.get("research_context")

    if not target_tickers:
        logger.info("[SUMMARIZER] Target tickers empty. Clearing user_command and research_context state.")
        user_command = None
        research_context = None

    return {
        "target_tickers": target_tickers,
        "analyzed_tickers": analyzed_tickers,
        "last_n_actions": updated_last_n,
        "journal": [journal_entry], 
        "proposed_decision": None,  
        "is_decision_valid": False, 
        "error_message": None,
        "research_context": research_context,
        "user_command": user_command
    }