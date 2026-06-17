import time
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, RichLog, Input, Label
from textual.containers import Horizontal, Vertical
from textual import work
import threading

from trading_bot.graph import app as agent_app
from trading_bot.config import shared_config
from trading_bot.utils.ui_messages import AgentLogMessage, PortfolioUpdated, HistoryUpdated
from trading_bot.services.database_instance import DbInstance
import logging
from rich.markup import escape

class TextualLogHandler(logging.Handler):
    def __init__(self, app: App):
        super().__init__()
        self.app = app
        
    def emit(self, record):
        try:
            msg = record.getMessage()
            escaped_msg = escape(msg)
            
            color = 'white'
            emoji = ''
            if record.levelname == 'ERROR' or record.levelname == 'CRITICAL':
                color = 'bold red'
                emoji = '❌'
            elif record.levelname == 'WARNING':
                color = 'yellow'
                emoji = '⚠️'
            elif record.levelname == 'INFO':
                color = 'blue'
                emoji = 'ℹ️'
                
            final_msg = f"[{color}]{emoji} {record.name} | {escaped_msg}[/{color}]"
            self.app.post_message(AgentLogMessage(final_msg))
        except Exception:
            self.handleError(record)


class TradingApp(App):
    CSS = """
    #left-pane {
        width: 40%;
        height: 100%;
        border-right: solid green;
    }
    #right-pane {
        width: 60%;
        height: 100%;
    }
    #balances-label {
        height: 3;
        content-align: center middle;
        border-bottom: solid green;
        background: $boost;
    }
    DataTable {
        height: 1fr;
        border-bottom: solid green;
    }
    RichLog {
        height: 100%;
    }
    Input {
        dock: bottom;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Label("Cash: --- | Buying Power: ---", id="balances-label")
                yield DataTable(id="portfolio-table")
                yield DataTable(id="history-table")
            with Vertical(id="right-pane"):
                yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Input(placeholder="Send a command or market sector to the Agent (e.g. 'Focus on AI')", id="command-input")
        yield Footer()

    def on_mount(self) -> None:
        # Configure the DataTables
        portfolio_table = self.query_one("#portfolio-table", DataTable)
        portfolio_table.add_columns("Ticker", "Qty", "Avg Price", "Current Price", "Market Val", "Unrealized P/L")

        history_table = self.query_one("#history-table", DataTable)
        history_table.add_columns("ID", "Timestamp", "Ticker", "Action", "Qty", "Outcome")

        # Set up logging redirection
        textual_handler = TextualLogHandler(self)
        root_logger = logging.getLogger()
        # Remove old handlers so we don't print to console
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(textual_handler)
        root_logger.setLevel(logging.INFO)

        self.exit_event = threading.Event()
        self.run_agent_loop()

    def on_unmount(self) -> None:
        self.exit_event.set()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        new_action = event.value.strip()
        if new_action:
            shared_config.update_action(new_action)
            self.post_message(AgentLogMessage(f"[bold green]✅ [COMMAND RECEIVED][/bold green] The agent will focus on: '{new_action}' at the start of the next cycle!"))
            event.input.value = ""

    def on_agent_log_message(self, message: AgentLogMessage) -> None:
        log = self.query_one("#agent-log", RichLog)
        log.write(message.text)

    def on_portfolio_updated(self, message: PortfolioUpdated) -> None:
        portfolio_table = self.query_one("#portfolio-table", DataTable)
        portfolio_table.clear()
        
        cash = message.portfolio.get("cash", 0)
        buying_power = message.portfolio.get("buying_power", 0)
        
        balances_label = self.query_one("#balances-label", Label)
        balances_label.update(f"[bold green]Cash: ${cash:,.2f}[/bold green]  |  [bold blue]Buying Power: ${buying_power:,.2f}[/bold blue]")
        
        positions = message.portfolio.get("positions", {})
        for ticker, pos in positions.items():
            portfolio_table.add_row(
                ticker,
                str(pos.get("qty", 0)),
                f"${pos.get('avg_entry_price', 0):.2f}",
                f"${pos.get('current_price', 0):.2f}",
                f"${pos.get('market_value', 0):.2f}",
                f"${pos.get('unrealized_pl', 0):.2f}"
            )

    def on_history_updated(self, message: HistoryUpdated) -> None:
        history_table = self.query_one("#history-table", DataTable)
        history_table.clear()
        for row in message.history:
            history_table.add_row(
                str(row.get("id", "")),
                str(row.get("timestamp", "")),
                str(row.get("ticker", "")),
                str(row.get("action", "")),
                str(row.get("quantity", "")),
                str(row.get("outcome", ""))
            )

    @work(thread=True)
    def run_agent_loop(self) -> None:
        initial_state = {
            "portfolio": {},
            "user_action": None,
            "target_tickers": [], 
            "proposed_decision": None,
            "is_decision_valid": False,
            "last_n_actions": [],
            "journal": [],
            "error_message": None
        }
        
        current_state = initial_state
        CYCLE_DELAY_SECONDS = 1
        cycle_count = 1

        db = DbInstance()
        
        self.post_message(AgentLogMessage("[bold blue]🤖 AGENT IS RUNNING![/bold blue] Type a new market sector at any time and press ENTER to steer the agent."))

        while True:
            self.post_message(AgentLogMessage(f"\n[bold yellow]--- STARTING CYCLE {cycle_count} ---[/bold yellow]"))
            
            try:
                for state_update in agent_app.stream(current_state):
                    for node_name, node_state in state_update.items():
                        self.post_message(AgentLogMessage(f"[dim]Finished node:[/dim] [bold]{node_name}[/bold]"))
                        
                        if node_state is not None:
                            current_state.update(node_state)
                            
                            if "cycle_logs" in node_state and node_state["cycle_logs"]:
                                for log in node_state["cycle_logs"]:
                                    self.post_message(AgentLogMessage(f"  > {log.get('event', '')}"))
                            
                            if "error_message" in node_state and node_state["error_message"]:
                                self.post_message(AgentLogMessage(f"[bold red]ERROR:[/bold red] {node_state['error_message']}"))

                            if node_name == "init_portfolio" and "portfolio" in node_state:
                                self.post_message(PortfolioUpdated(node_state["portfolio"]))
                        
                        if node_name == "load_history" or node_name == "summarizer":
                            recent_trades = db.get_recent_trades()
                            self.post_message(HistoryUpdated(recent_trades))

                self.post_message(AgentLogMessage(f"[bold yellow]--- CYCLE {cycle_count} COMPLETE ---[/bold yellow]"))
                self.post_message(AgentLogMessage(f"[dim]Sleeping for {CYCLE_DELAY_SECONDS} seconds...[/dim]"))
                
                if self.exit_event.wait(CYCLE_DELAY_SECONDS):
                    break
                    
                cycle_count += 1
            except Exception as e:
                self.post_message(AgentLogMessage(f"[bold red]CRITICAL ERROR IN AGENT LOOP:[/bold red] {str(e)}"))
                if self.exit_event.wait(CYCLE_DELAY_SECONDS):
                    break

if __name__ == "__main__":
    TradingApp().run()
