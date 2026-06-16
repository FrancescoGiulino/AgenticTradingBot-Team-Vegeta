import logging
from langgraph.graph import StateGraph, START, END
from .state import AgentState

logger = logging.getLogger(__name__)
from .nodes import init_portfolio, decisor, checker, executer, summarizer

def route_after_checker(state: AgentState) -> str:
    if state.get("is_decision_valid", False):
        logger.info("[ROUTER] Decision is VALID. Routing to EXECUTER.")
        return "executer"
    else:
        logger.info("[ROUTER] Decision is INVALID. Bypassing execution, routing to SUMMARIZER.")
        return "summarizer"

workflow = StateGraph(AgentState)

workflow.add_node("init_portfolio", init_portfolio)
workflow.add_node("decisor", decisor)
workflow.add_node("checker", checker)
workflow.add_node("executer", executer)
workflow.add_node("summarizer", summarizer)

workflow.add_edge(START, "init_portfolio")
workflow.add_edge("init_portfolio", "decisor")
workflow.add_edge("decisor", "checker")

workflow.add_conditional_edges(
    "checker",
    route_after_checker,
    {
        "executer": "executer",
        "summarizer": "summarizer"
    }
)

workflow.add_edge("executer", "summarizer")

# MODIFICA CHIAVE: Invece di tornare a init_portfolio, il grafo finisce qui.
# Sarà il main.py a far ripartire un nuovo grafo dopo 30 secondi.
workflow.add_edge("summarizer", END)

app = workflow.compile()
