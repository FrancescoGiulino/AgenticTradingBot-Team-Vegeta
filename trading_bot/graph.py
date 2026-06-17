import logging
from langgraph.graph import StateGraph, START, END
from .state import AgentState

logger = logging.getLogger(__name__)

from .nodes import (
    init_portfolio,
    supervisor_node,
    researcher_node,
    decisor,
    checker,
    executer,
    summarizer
)

def supervisor_router(state: AgentState) -> str:
    next_node = state.get("next_node", "FINISH")
    if next_node == "FINISH":
        return END
    return next_node

workflow = StateGraph(AgentState)

workflow.add_node("init_portfolio", init_portfolio)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("decisor", decisor)
workflow.add_node("checker", checker)
workflow.add_node("executer", executer)
workflow.add_node("summarizer", summarizer)

workflow.add_edge(START, "init_portfolio")
workflow.add_edge("init_portfolio", "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "researcher": "researcher",
        "decisor": "decisor",
        "checker": "checker",
        "executer": "executer",
        "summarizer": "summarizer",
        END: END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("decisor", "supervisor")
workflow.add_edge("checker", "supervisor")
workflow.add_edge("executer", "supervisor")
workflow.add_edge("summarizer", "supervisor")

app = workflow.compile()
