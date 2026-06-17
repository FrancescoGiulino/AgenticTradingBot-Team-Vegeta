import context
from trading_bot import nodes
from trading_bot.state import TradeDecision
from trading_bot.nodes import checker

def test_checker_accepts_hold():
    # Setup state with a HOLD decision
    state = {
        "portfolio": {"cash": 100000.0, "positions": {}},
        "proposed_decision": TradeDecision(
            ticker="AAPL", action="HOLD", quantity=0, news_summary="N/A", current_price=150.0, rationale="Holding"
        )
    }
    
    # Run node
    result = checker(state)
    
    # Verify
    assert result["is_decision_valid"] is True

def test_checker_rejects_buy_insufficient_funds():
    # Setup state with a BUY decision that exceeds cash
    state = {
        "portfolio": {"cash": 100.0, "positions": {}},
        "proposed_decision": TradeDecision(
            ticker="AAPL", action="BUY", quantity=10, news_summary="N/A", current_price=150.0, rationale="Buying"
        )
    }
    
    # Run node
    result = checker(state)
    
    # Verify
    assert result["is_decision_valid"] is False
