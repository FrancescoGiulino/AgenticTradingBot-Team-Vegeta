import os
import sys

# Insert the root directory into sys.path so that tests can import 'trading_bot'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# We can import trading_bot here to verify it works, or let the tests do it.
import trading_bot
