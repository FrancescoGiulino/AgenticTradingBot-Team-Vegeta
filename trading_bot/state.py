from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import operator
from typing import Literal 

# Output Schema for the Dynamic Watchlist (EXPLORER phase)
class DynamicWatchlist(BaseModel):
    tickers: List[str] = Field(description="List of 2 to 3 valid US stock ticker symbols to monitor (e.g. ['TSLA', 'F', 'GM']).")
    rationale: str = Field(description="Why you chose these specific tickers based on the current sector focus.")

class MarketDiscovery(BaseModel):
    market_themes: List[str] = Field(
        description="3-5 macro themes driving the market today based on the news (e.g., 'AI chip shortage', 'Inflation fears')."
    )
    candidate_tickers: Dict[str, str] = Field(
        description="A list of 5-10 specific stock tickers mentioned in the news that represent the best trading opportunities: Key represents the ticker itself, represented in UPPERCASE. The value contains the reason of why it's an opportunity"
    )

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the user input represents a valid and relevant market sector or trading topic, False otherwise.")
    refined_action: str = Field(description="The cleaned and refined market topic based on web search (or the original if already perfect). If is_valid is False, provide a brief reason.")

# Output Schema for the DECISOR node
class TradeDecision(BaseModel):
    ticker: str = Field(description="The ticker symbol of the stock")
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="The action to take: 'BUY', 'SELL', or 'HOLD'")
    quantity: int
    
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in this trade (0.0 to 1.0)")
    rationale: str = Field(description="Detailed reasoning for the decision. Must include relevant macro/micro news and technical factors considered.")

class AgentState(TypedDict):
    cycle_id: str
    
    portfolio: Dict[str, Any]
    recent_history: List[Dict[str, Any]]  # Le ultime N operazioni lette dal DB all'inizio del ciclo
    
    user_action: Optional[str]            # Azione esplicita richiesta dall'utente
    
    market_themes: List[str]              # Es: ["AI regulation", "Oil supply shortage"]
    candidate_tickers: Dict[str, str]          # I ticker scoperti dall'agente da analizzare

    pending_orders: List[str]
    
    quant_data: Dict[str, Dict[str, Any]]

    proposed_decision: TradeDecision  # Operazione proposta (può valutare più azioni alla volta)
    risk_approval_status: bool            # Il nodo di Risk Management ha approvato i trade?
    
    cycle_logs: Annotated[List[Dict[str, Any]], operator.add]
    error_message: Optional[str]