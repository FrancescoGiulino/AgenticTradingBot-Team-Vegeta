from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import operator

# Output Schema for the Dynamic Watchlist (EXPLORER phase)
class DynamicWatchlist(BaseModel):
    tickers: List[str] = Field(description="List of 2 to 3 valid US stock ticker symbols to monitor (e.g. ['TSLA', 'F', 'GM']).")
    rationale: str = Field(description="Why you chose these specific tickers based on the current sector focus.")

# Output Schema for the DECISOR node
class TradeDecision(BaseModel):
    ticker: str = Field(description="The ticker symbol of the stock")
    action: str = Field(description="The action to take: 'BUY', 'SELL', or 'HOLD'")
    quantity: int = Field(description="The number of shares to trade. Set to 0 if HOLD.")
    news_summary: str = Field(description="Brief summary of the news considered for this decision")
    current_price: float = Field(description="The current market price of the asset")
    rationale: str = Field(description="Explicit and detailed reasoning for the decision (Why buy/sell/hold?)")

# Graph State definition
class AgentState(TypedDict):
    # Portfolio data fetched from Alpaca
    portfolio: Dict[str, Any]
    
    # The sector or market trend we want the agent to focus on (e.g., "automotive", "tech", "healthcare")
    market_focus: Optional[str]
    
    # Tickers we are currently analyzing (Dynamically populated)
    target_tickers: List[str]
    
    proposed_decision: Optional[TradeDecision]
    is_decision_valid: bool
    last_n_actions: List[Dict[str, Any]]
    journal: Annotated[List[Dict[str, Any]], operator.add]
    error_message: Optional[str]
    cycle_id: Optional[str]