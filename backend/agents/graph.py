"""Compiled research agent with guarded entry, fatal-error exits and bounded retries."""

from functools import partial

from langgraph.graph import END, StateGraph

from agents.state import AgentState
from agents.nodes.input_guard import input_guard_node
from agents.nodes.query_analyzer import query_analyzer_node
from agents.nodes.query_rewriter import query_rewriter_node
from agents.nodes.multi_query import multi_query_node
from agents.nodes.hyde_generator import hyde_generator_node
from agents.nodes.retriever import retriever_node
from agents.nodes.reranker import reranker_node
from agents.nodes.synthesizer import synthesizer_node
from agents.nodes.output_guard import output_guard_node
from agents.nodes.citation_formatter import citation_formatter_node


def _after_input(state: AgentState) -> str:
    return "query_analyzer" if state.get("is_safe") is True and not state.get("error") else END


def _next_or_end(state: AgentState, next_node: str) -> str:
    return END if state.get("error") else next_node


def _after_output(state: AgentState) -> str:
    if state.get("error"):
        return END
    if state.get("is_output_safe") is True:
        return "citation_formatter"
    limit = max(1, min(state.get("max_generation_attempts", 2), 3))
    if state.get("generation_attempts", 0) < limit and state.get("retry_count", 0) < limit - 1:
        return "synthesizer"
    return END


graph = StateGraph(AgentState)
for name, node in (
    ("input_guard", input_guard_node), ("query_analyzer", query_analyzer_node),
    ("query_rewriter", query_rewriter_node), ("multi_query", multi_query_node),
    ("hyde_generator", hyde_generator_node), ("retriever", retriever_node),
    ("reranker", reranker_node), ("synthesizer", synthesizer_node),
    ("output_guard", output_guard_node), ("citation_formatter", citation_formatter_node),
):
    graph.add_node(name, node)
graph.set_entry_point("input_guard")
graph.add_conditional_edges("input_guard", _after_input, {"query_analyzer": "query_analyzer", END: END})

# The successful path is linear; errors terminate before further external calls.
chain = ["query_analyzer", "query_rewriter", "multi_query", "hyde_generator",
         "retriever", "reranker", "synthesizer", "output_guard"]
for source, target in zip(chain, chain[1:]):
    graph.add_conditional_edges(source, partial(_next_or_end, next_node=target), {target: target, END: END})
graph.add_conditional_edges("output_guard", _after_output, {
    "citation_formatter": "citation_formatter", "synthesizer": "synthesizer", END: END,
})
graph.add_edge("citation_formatter", END)
agent = graph.compile()
