"""
agents/graph.py — LangGraph Agentic Workflow Definition
=========================================================

STEP-BY-STEP IMPLEMENTATION PLAN:
-----------------------------------
Step 1:  Import StateGraph, END from langgraph.graph
Step 2:  Import AgentState from agents/state.py
Step 3:  Import all node functions from agents/nodes/

Step 4:  Create the graph:
            graph = StateGraph(AgentState)

Step 5:  Add all nodes to the graph:
            graph.add_node("input_guard",      input_guard_node)
            graph.add_node("query_analyzer",   query_analyzer_node)
            graph.add_node("query_rewriter",   query_rewriter_node)
            graph.add_node("multi_query",      multi_query_node)
            graph.add_node("hyde_generator",   hyde_generator_node)
            graph.add_node("retriever",        retriever_node)
            graph.add_node("reranker",         reranker_node)
            graph.add_node("synthesizer",      synthesizer_node)
            graph.add_node("output_guard",     output_guard_node)
            graph.add_node("citation_formatter", citation_formatter_node)

Step 6:  Set entry point:
            graph.set_entry_point("input_guard")

Step 7:  Add conditional edge after input_guard:
            - If state.is_safe == True  → go to "query_analyzer"
            - If state.is_safe == False → go to END (return rejection message)

Step 8:  Add linear edges for the happy path:
            query_analyzer → query_rewriter → multi_query → hyde_generator
            → retriever → reranker → synthesizer → output_guard

Step 9:  Add conditional edge after output_guard:
            - If state.is_output_safe == True  → go to "citation_formatter" → END
            - If state.is_output_safe == False → go back to "synthesizer" (regenerate)
            - Add a max_retries check to avoid infinite loops

Step 10: Compile the graph:
            agent = graph.compile()

Step 11: Export `agent` for use in API routes

VISUALIZATION:
            To visualize the graph during development:
            print(agent.get_graph().draw_ascii())
"""

