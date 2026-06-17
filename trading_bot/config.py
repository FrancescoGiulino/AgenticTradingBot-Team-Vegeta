import threading

class AgentConfig:
    def __init__(self):
        self.market_focus = "Innovative Tech and EV"
        self.focus_changed = False
        self.lock = threading.Lock()

    def update_focus(self, new_focus: str):
        """Chiamato dal thread dell'utente per aggiornare l'obiettivo."""
        with self.lock:
            self.market_focus = new_focus
            self.focus_changed = True

    def get_focus_and_reset_flag(self) -> tuple:
        """Chiamato dall'agente all'inizio di ogni ciclo."""
        with self.lock:
            changed = self.focus_changed
            self.focus_changed = False
            return self.market_focus, changed

shared_config = AgentConfig()