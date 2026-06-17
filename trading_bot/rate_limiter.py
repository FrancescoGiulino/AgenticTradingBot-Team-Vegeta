import time
import threading
import json
import os
import logging

logger = logging.getLogger(__name__)

class GlobalRateLimiter:
    """
    A thread-safe global rate limiter utilizing the Token Bucket algorithm.
    It can be configured to manage limits for different APIs and resources.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(GlobalRateLimiter, cls).__new__(cls)
                cls._instance.buckets = {}
                cls._instance.buckets_lock = threading.Lock()
        return cls._instance

    def register_bucket(self, name: str, max_tokens: float, refill_interval: float = 60.0):
        """
        Registers a new bucket.
        :param name: Identifier for the resource (e.g., "alpaca_rpm", "google_genai_tpm").
        :param max_tokens: The maximum number of tokens. -1 means unlimited.
        :param refill_interval: The time in seconds it takes to completely refill the bucket.
        """
        with self.buckets_lock:
            self.buckets[name] = {
                "max_tokens": float(max_tokens),
                "current_tokens": float(max_tokens) if max_tokens != -1 else -1,
                "refill_interval": float(refill_interval),
                "last_refill": time.time()
            }
            logger.info(f"Registered rate limit bucket '{name}' with max_tokens={max_tokens}, refill_interval={refill_interval}s")

    def acquire(self, name: str, amount: float = 1):
        """
        Acquires tokens from the specified bucket. Blocks if necessary until tokens are available.
        :param name: The name of the bucket.
        :param amount: The number of tokens to acquire.
        """
        with self.buckets_lock:
            bucket = self.buckets.get(name)
            if not bucket:
                return

            if bucket["max_tokens"] == -1:
                return

        while True:
            with self.buckets_lock:
                now = time.time()
                elapsed = now - bucket["last_refill"]
                
                # Refill logic
                tokens_to_add = (elapsed / bucket["refill_interval"]) * bucket["max_tokens"]
                if tokens_to_add > 0:
                    bucket["current_tokens"] = min(bucket["max_tokens"], bucket["current_tokens"] + tokens_to_add)
                    bucket["last_refill"] = now
                
                if bucket["current_tokens"] >= amount:
                    bucket["current_tokens"] -= amount
                    remaining = bucket["current_tokens"]
                    # logger.info(f"[RATE LIMITER] Used {amount} token(s) for '{name}'. Remaining tokens: {remaining:.2f}")
                    return
                
                deficit = amount - bucket["current_tokens"]
                time_to_wait = (deficit / bucket["max_tokens"]) * bucket["refill_interval"]
            
            logger.warning(f"[RATE LIMITER] Limit reached for '{name}'. Sleeping for {time_to_wait:.2f} seconds to acquire {amount} tokens.")
            time.sleep(max(0.1, time_to_wait))

    def load_config(self, config_path: str):
        """
        Loads rate limit configuration from a JSON file.
        """
        if not os.path.exists(config_path):
            logger.warning(f"Rate limiter configuration file {config_path} not found. Proceeding with defaults (unlimited).")
            return
            
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            for service, limits in config.items():
                if "requests_per_minute" in limits:
                    self.register_bucket(f"{service}_rpm", limits["requests_per_minute"], 60.0)
                if "tokens_per_minute" in limits:
                    self.register_bucket(f"{service}_tpm", limits["tokens_per_minute"], 60.0)
        except Exception as e:
            logger.error(f"Failed to load rate limiter configuration: {e}")

rate_limiter = GlobalRateLimiter()
