import context
from trading_bot import graph
from trading_bot.graph import workflow

def test_executer_edge():
    # Verify that the 'executer' node is routed directly to 'summarizer' 
    # to avoid the infinite loop issue where the supervisor is repeatedly called.
    
    # LangGraph StateGraph edges are stored in `workflow.edges`
    edges = workflow.edges
    
    # In StateGraph, edges is a set of tuples: (start_node_id, end_node_id)
    executer_edges = [edge for edge in edges if edge[0] == "executer"]
    
    # Assert that there is exactly one edge from executer, and it points to summarizer
    assert len(executer_edges) == 1
    assert executer_edges[0][1] == "summarizer"
