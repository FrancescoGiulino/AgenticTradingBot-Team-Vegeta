import time
import pytest
import threading
from trading_bot.rate_limiter import GlobalRateLimiter

@pytest.fixture
def rate_limiter():
    # Because it's a singleton, we need to reset it for tests
    limiter = GlobalRateLimiter()
    limiter.buckets = {}
    return limiter

def test_rate_limiter_unlimited(rate_limiter):
    # Test -1 for unlimited tokens/requests
    rate_limiter.register_bucket("test_unlimited", -1)
    
    start_time = time.time()
    # Should acquire without blocking
    rate_limiter.acquire("test_unlimited", 1000)
    end_time = time.time()
    
    # Execution should be near instantaneous
    assert end_time - start_time < 0.1

def test_rate_limiter_blocking(rate_limiter):
    # Register bucket with max 2 tokens, refills in 1 second
    rate_limiter.register_bucket("test_block", 2, refill_interval=1.0)
    
    start_time = time.time()
    # Acquire 2 tokens immediately
    rate_limiter.acquire("test_block", 2)
    # Acquire 1 more, should block for about 0.5s
    rate_limiter.acquire("test_block", 1)
    end_time = time.time()
    
    # Should take roughly 0.5 seconds to acquire the 3rd token
    assert 0.4 <= end_time - start_time <= 0.8

def test_rate_limiter_multithreading(rate_limiter):
    # Ensure thread safety
    rate_limiter.register_bucket("test_thread", 10, refill_interval=2.0)
    
    def worker():
        rate_limiter.acquire("test_thread", 1)
        
    threads = []
    start_time = time.time()
    for _ in range(15):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
    end_time = time.time()
    
    # Acquiring 15 tokens with 10 max and 2s refill means we wait for 5 tokens to refill.
    # 5 tokens = 5 * (2.0 / 10) = 1.0 second
    assert end_time - start_time >= 0.9
