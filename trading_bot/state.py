from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import operator

class DynamicWatchlist(BaseModel):
    tickers: List[str] = Field(description="List of 2 to 3 valid US stock ticker symbols to monitor (e.g. ['TSLA', 'F', 'GM']).")
    rationale: str = Field(description="Why you chose these specific tickers based on the current sector focus.")

class TradeDecision(BaseModel):
    ticker: str = Field(description="The ticker symbol of the stock")
    action: str = Field(description="The action to take: 'BUY', 'SELL', or 'HOLD'")
    quantity: int = Field(description="The number of shares to trade. Set to 0 if HOLD.")
    news_summary: str = Field(description="Brief summary of the news considered for this decision")
    current_price: float = Field(description="The current market price of the asset")
    rationale: str = Field(description="Explicit and detailed reasoning for the decision (Why buy/sell/hold?)")
    cleared_wanted_action: bool = Field(default=False, description="Set to true ONLY if you successfully executed or fully addressed the 'wanted_action' in this cycle, so it can be cleared.")

class AgentState(TypedDict):
    portfolio: Dict[str, Any]
    market_focus: Optional[str]
    target_tickers: List[str]
    analyzed_tickers: List[str]
    cycles_since_portfolio_analysis: int
    analyzed_portfolio_tickers: List[str]
    cycle_count: int
    focus_iteration_count: int
    next_node: Optional[str]
    portfolio_summary: Optional[str]
    user_command: Optional[str]
    research_context: Optional[str]
    proposed_decision: Optional[TradeDecision]
    is_decision_valid: bool
    last_n_actions: List[Dict[str, Any]]
    journal: Annotated[List[Dict[str, Any]], operator.add]
    error_message: Optional[str]
    cycle_id: Optional[str]
    wanted_action_context: Optional[str]