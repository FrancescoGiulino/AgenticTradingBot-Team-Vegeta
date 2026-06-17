import threading

class AgentConfig:
    def __init__(self):
        self.user_action = None
        self.action_changed = False
        self.lock = threading.Lock()

    def update_action(self, new_action: str):
        """Chiamato dal thread dell'utente per aggiornare l'azione."""
        with self.lock:
            self.user_action = new_action
            self.action_changed = True

    def get_action_and_reset_flag(self) -> tuple:
        """Chiamato dall'agente all'inizio di ogni ciclo."""
        with self.lock:
            changed = self.action_changed
            self.action_changed = False
            return self.user_action, changed

shared_config = AgentConfig()