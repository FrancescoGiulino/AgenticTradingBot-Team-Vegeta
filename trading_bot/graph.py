import logging
from langgraph.graph import StateGraph, START, END
from .state import AgentState

logger = logging.getLogger(__name__)
from .nodes import init_portfolio_node, load_history_node, discovery_node , quant_enrichment_node,decisor_node, executer_node, summarizer

def check_for_errors(state: AgentState) -> str:
    if state.get("error_message"):
        logger.error(f"[ROUTER] Critical error detected: {state['error_message']}. Routing straight to SUMMARIZER.")
        return "summarizer"
    return "continue"

def route_after_checker(state: AgentState) -> str:
    if state.get("risk_approval_status", False):
        logger.info("[ROUTER] Risk approval is TRUE. Routing to EXECUTER.")
        return "executer"
    else:
        logger.info("[ROUTER] Risk approval is FALSE. Bypassing execution, routing to SUMMARIZER.")
        return "summarizer"

workflow = StateGraph(AgentState)

workflow.add_node("init_portfolio", init_portfolio_node)
workflow.add_node("load_history", load_history_node)
workflow.add_node("discovery", discovery_node)
workflow.add_node("quant_enrichment", quant_enrichment_node)
workflow.add_node("decisor", decisor_node) 
#workflow.add_node("checker", checker)
workflow.add_node("executer", executer_node)
workflow.add_node("summarizer", summarizer)

workflow.add_edge(START, "init_portfolio")
workflow.add_conditional_edges(
    "init_portfolio",
    check_for_errors,
    {
        "continue": "load_history",  
        "summarizer": "summarizer"
    }
)

workflow.add_edge("load_history", "discovery")
workflow.add_edge("discovery", "quant_enrichment")
workflow.add_edge("quant_enrichment", "decisor")
workflow.add_conditional_edges(
    "decisor",
    check_for_errors,
    {
        "continue": "executer",   # <--- change in checker !!!!    
        "summarizer": "summarizer"
    }
)
"""
workflow.add_conditional_edges(
    "checker",
    route_after_checker,
    {
        "executer": "executer",
        "summarizer": "summarizer"
    }
)
"""
workflow.add_edge("executer", "summarizer")

# MODIFICA CHIAVE: Invece di tornare a init_portfolio, il grafo finisce qui.
# Sarà il main.py a far ripartire un nuovo grafo dopo 30 secondi.
workflow.add_edge("summarizer", END)

app = workflow.compile()
