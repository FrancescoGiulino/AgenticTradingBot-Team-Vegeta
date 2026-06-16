# Agentic AI Trading Agent

## Mission Statement
This project aims to build a truly autonomous trading agent, moving beyond simple conversational chatbots. Designed to operate on a simulated stock market, the agent follows a strict **Perceive → Reason → Act** loop. 

Our core philosophy aligns with the hackathon's primary rule: **Zero Data Hallucination**. Every financial decision (BUY, SELL, or HOLD) is backed by real-time market data and explicit, traceable reasoning.

## Core Architecture & Features (Planned)
- **Autonomous Tool Calling:** Retrieves live prices, 5-day historical trends, and latest news using strict API tool calls.
- **Sequential Multi-Agent Workflow:** Utilizes a pipeline consisting of a Coordinator, an Analyst, and a Chief Risk Officer (CRO) to cross-validate every trade proposal.
- **Robust Risk Management:** Implements safety fallbacks to handle missing data or API failures gracefully without crashing.
- **Trade Journaling:** Maintains an automated, structured log of every decision, including timestamp, ticker, rationale, and outcome.
- **Global Rate Limiting:** Enforces rate limits (Requests Per Minute and Tokens Per Minute) across all API dependencies using a Token Bucket algorithm, configurable via `rate_limits.json`.

## Tech Stack
- **Framework:** LangGraph (for stateful, multi-actor LLM orchestration)
- **Intelligence:** Google Gemini Gemma4 (31B / 26B a4b) for cheap structured reasoning
- **Market Data:** yfinance / NewsAPI
- **Broker:** Alpaca Paper Trading API
- **Language:** Python 3.13+