"""Node 2: classify the research question."""

from agents.state import AgentState
from agents.nodes._shared import ask, skipped, update


async def query_analyzer_node(state: AgentState) -> AgentState:
    if skipped(state):
        return {}
    try:
        category = (await ask(
            "Classify the question. Return only factual, summary, comparative, definition, or methodology.",
            {"question": state["query"]},
        )).strip(chr(96) + " .\n").lower()
        if category not in {"factual", "summary", "comparative", "definition", "methodology"}:
            raise ValueError("Invalid category")
        return update(f"Query classified as {category}", query_type=category)
    except Exception:
        return update("Classification unavailable; using factual retrieval", query_type="factual")
