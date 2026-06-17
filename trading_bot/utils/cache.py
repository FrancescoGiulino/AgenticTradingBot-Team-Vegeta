import time

class DiscoveryCache():
    def __init__(self):
        self._cache = None
        self.ttl_seconds = 3600 #resets each 1 hour 

    def get_cached_discovery(self) -> dict:
        if not self._cache:
            return None

        elapsed_time = time.time() - self._cache.get("timestamp", 0)
        if elapsed_time < self.ttl_seconds:
            return self._cache.get("data")
        
        return None
    
    # TODO perform a bit of refactoring to make it more generic
    def set_discovery(self, market_themes: list, candidate_tickers: list):
        """Salva i nuovi dati con il timestamp attuale."""
        self._cache = {
            "timestamp": time.time(),
            "data": {
                "market_themes": market_themes,
                "candidate_tickers": candidate_tickers
            }
        }
 