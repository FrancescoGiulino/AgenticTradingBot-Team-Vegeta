from textual.message import Message

class AgentStateUpdated(Message):
    """Sent when the agent state has been updated."""
    def __init__(self, state: dict):
        self.state = state
        super().__init__()

class AgentLogMessage(Message):
    """Sent to log a message in the UI."""
    def __init__(self, text: str):
        self.text = text
        super().__init__()

class PortfolioUpdated(Message):
    """Sent when the portfolio has been updated."""
    def __init__(self, portfolio: dict):
        self.portfolio = portfolio
        super().__init__()

class HistoryUpdated(Message):
    """Sent when the history has been updated."""
    def __init__(self, history: list):
        self.history = history
        super().__init__()
